import itertools
import os
import time
from typing import Any, Optional, Tuple

import torch
import triton
import triton.language as tl
import triton.testing

from jit_kernel.velox_voice import dgx_mxfp4_gemm

SEED = 42

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


DEBUG = False


BLOCK = 16
CODE_LUT = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device="cuda")

# TODO (yiakwy) : add fp4 blockwise quantization

# TODO (yiakwy) : rewrite per row / col quantization

@triton.jit
def _per_row_col_quantize_kernel(
    X, OutQ, OutS,
    M, K,
    stride_x_m, stride_x_k,
    stride_q_m, stride_q_k,
    BLOCK_KP2: tl.constexpr,
):
    pid = tl.program_id(0)

    cols = tl.arange(0, BLOCK_KP2)
    mask = cols < K // 2

    even = tl.load(X + pid * stride_x_m + (cols * 2) * stride_x_k,
                   mask=mask, other=0.0).to(tl.float32)
    
    odd = tl.load(X + pid * stride_x_m + (cols * 2 + 1) * stride_x_k,
                  mask=mask, other=0.0).to(tl.float32)

    # TODO (yiakwy) : rewrite to compute row_max
    abs_even = tl.abs(even)
    abs_odd = tl.abs(odd)
    local_max = tl.maximum(abs_even, abs_odd)
    row_max = tl.max(local_max, axis=0)

    # NOTE (yiakwy) : clamp per row/col max to fp4 range
    s = row_max / 6.0
    s_safe = tl.maximum(s, 1e-6)

    # Normalize
    n_even = even / s_safe
    n_odd = odd / s_safe

    ax_even = tl.abs(n_even)
    ax_odd = tl.abs(n_odd)

    # TODO (yiakwy) : vectorize compare (least sqrt distance) with CODE_LUT

    # Encode even columns: sequential thresholds, last write wins
    c_even = tl.zeros([BLOCK_KP2], dtype=tl.uint8)
    c_even = tl.where(ax_even > .25, 1, c_even)
    c_even = tl.where(ax_even > .75, 2, c_even)
    c_even = tl.where(ax_even > 1.25, 3, c_even)
    c_even = tl.where(ax_even > 1.75, 4, c_even)
    c_even = tl.where(ax_even > 2.5, 5, c_even)
    c_even = tl.where(ax_even > 3.5, 6, c_even)

    sign_e = tl.where(n_even < 0.0, 8, 0).to(tl.uint8)
    c_even = c_even | sign_e

    # Encode odd columns
    c_odd = tl.zeros([BLOCK_KP2], dtype=tl.uint8)
    c_odd = tl.where(ax_odd > .25, 1, c_odd)
    c_odd = tl.where(ax_odd > .75, 2, c_odd)
    c_odd = tl.where(ax_odd > 1.25, 3, c_odd)
    c_odd = tl.where(ax_odd > 1.75, 4, c_odd)
    c_odd = tl.where(ax_odd > 2.5, 5, c_odd)
    c_odd = tl.where(ax_odd > 3.5, 6, c_odd)

    sign_o = tl.where(n_odd < 0.0, 8, 0).to(tl.uint8)
    c_odd = c_odd | sign_o

    # Pack: lo 4 bits = even, hi 4 bits = odd
    packed = c_even | (c_odd << 4)
    tl.store(OutQ + pid * stride_q_m + cols * stride_q_k, packed, mask=mask)

    # Per-row scale as ue8m0 (trunc log2 + 127 bias)
    safe_s = tl.maximum(s, 1e-20)
    log2x = tl.log2(safe_s)
    truncated = tl.where(log2x >= 0, tl.floor(log2x), tl.ceil(log2x))
    e = tl.clamp(truncated + 127.0, 0.0, 255.0).to(tl.uint8)
    e = tl.where(s > 1e-20, e, 0x80).to(tl.uint8)

    tl.store(OutS + pid, e)


def per_row_col_quantize(w, block_size=None):
    """[M, K] bf16/fp32 quantize (packed u8 [M, K/2], per-row ue8m0 u8 [M, 1])."""
    assert w.ndim == 2
    
    M, K = w.shape
    
    assert K % 2 == 0, "K must be even"
    assert K <= 65536, f"K={K} exceeds max block size"

    # TODO (yiakwy) : remove
    w_c = w.contiguous().view(M, K).to(w.dtype)

    out_q = torch.empty(M, K // 2, device=w.device, dtype=torch.uint8)

    # TODO (yiakwy) : add support 
    out_s = torch.empty(M, device=w.device, dtype=torch.uint8)

    BLOCK_KP2 = triton.next_power_of_2(K // 2)

    _per_row_col_quantize_kernel[(M,)](
        w_c, out_q, out_s,
        M, K,
        w_c.stride(0), w_c.stride(1),
        out_q.stride(0), out_q.stride(1),
        BLOCK_KP2=BLOCK_KP2,
    )

    return out_q, out_s


# TODO (yiakwy) : add support of block quantize with scales [M, K/2/block_size]
def quantize_w(w, block_size=BLOCK, format="nvfp4"):
    """[M, K] bf16/fp32 qunatize to (packed u8 [M, K/2], per-row / per-col ue8m0 u8 [M, 1]))."""
    return per_row_col_quantize(w, block_size)


def unpack(p):
    """packed u8 [M, K/2] -> signed magnitudes [M, K] (e2m1 with sign bit 3)."""
    lo = (p & 0xF)
    hi = (p >> 4)
    c = torch.stack([lo, hi], -1).reshape(p.shape[0], p.shape[1] * 2)
    mag = CODE_LUT.to(p.device)[(c & 7).long()]
    sign = ((c & 8) > 0)  # bit 3 = sign (bool)
    return torch.where(sign, -mag, mag)


def torch_matmul_ref(xq, wq, sa_u8, sb_u8):
    x_fp16 = unpack(xq)
    w_fp16 = unpack(wq)

    raw = x_fp16 @ w_fp16.T

    row_s = (2.0 ** (sa_u8.to(torch.int16).to(torch.float32) - 127)).unsqueeze(1)  # [M, 1]
    col_s = (2.0 ** (sb_u8.to(torch.int16).to(torch.float32) - 127)).unsqueeze(0)  # [1, N]

    return raw * row_s * col_s  


def calculate_diff(M, N, K):
    torch.manual_seed(SEED)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    xq, xs = quantize_w(x)
    wq, ws = quantize_w(w)

    out = dgx_mxfp4_gemm(xq, wq, xs, ws)
    ref = torch_matmul_ref(xq, wq, xs, ws)

    assert out.shape == (M, N), f"shape mismatch: {out.shape}"
    assert out.dtype == torch.float32
    diff = (out - ref).abs().max().item()
    assert diff == 0.0, f"bit-exact fail: max|diff|={diff}"

    print(f"✅ {M}x{N}x{K} dgx_mxfp4_gemm")


configs = [
    (128, 128, 64),      # min tile
    (128, 256, 64),
    (256, 128, 128),
    (512, 512, 2048),    # multi-tile
    (1024, 512, 1024),
    (2048, 2048, 2048),  # large
]


if __name__ == "__main__":
    # Correctness check - simplified for CI
    if IS_CI:
        # Only test one configuration in CI
        test_configs = [configs[0]]
    else:
        test_configs = configs

    for cfg in test_configs:
        print(f"cfg : {cfg}")
        calculate_diff(*cfg)

    print("\n" + "=" * 60)
    if not DEBUG:
        print("Starting performance benchmark...")
        # benchmark.run(print_data=True)
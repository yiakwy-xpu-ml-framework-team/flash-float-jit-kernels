"""A/B probe: on-chip split-K reduce strategy (thread-level remote loads vs bulk DMA).

Isolates the REDUCE strategy cost at a FIXED split_k by driving the TVM-FFI entry
directly (bypasses the wrapper's split_k policy). Run the SAME command twice, with
and without the bulk-reduce compile flag, in separate processes (the env var changes
the JIT cache key -> separate binaries):

    # original strategy (per-thread remote loads)
    python3 sandbox/probes/probe_splitk_reduce_ab.py --B 1 --m 1024 --sk 2

    # bulk-DMA reduce + half2 SIMD adds (USE_BULK_SPLITK_REDUCE)
    FLASH_FLOAT_BULK_SPLITK=1 python3 sandbox/probes/probe_splitk_bulk_reduce_ab.py ...

NOTE: pick m so that the triangular tile count keeps grid_mn < 132 (e.g. m <= 1152);
for larger shapes the host forces split_k = 1 and the reduce never runs.
Known pre-existing issue: nbm in {6, 10, 11} (m = 768/1280/1408) hangs at sk >= 2.
"""
import argparse
import torch

from jit_kernel.thunder_moun import _jit_thunder_moun_module_v2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=1)
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--sk", type=int, default=2)
    p.add_argument("--iters", type=int, default=300)
    args = p.parse_args()

    torch.manual_seed(0)
    module = _jit_thunder_moun_module_v2()
    B, m, sk = args.B, args.m, args.sk

    x = torch.randn(B, m, m, dtype=torch.float16, device="cuda")
    xq = x.to(torch.float8_e4m3fn)
    xs0 = torch.ones((B, m, m // 128), dtype=torch.float32, device="cuda")
    xs1 = torch.ones((B, m // 128, m // 128), dtype=torch.float32, device="cuda")
    out = torch.zeros((B, m, m), dtype=torch.float16, device="cuda")

    def call():
        module.symmetric_gemm_fp8_block_scaled(
            xq, xq, xs0, xs1, out, m, m, m, B,
            xq.stride(-2), xq.stride(-1), xq.stride(-2), xq.stride(-1),
            out.stride(-2), out.stride(-1), sk)

    # correctness spot check vs fp8-quantized fp64 reference
    call(); torch.cuda.synchronize()
    ref = xq.float().double() @ xq.float().double().transpose(-1, -2)
    rel = (out.double() - ref).abs().max().item() / ref.abs().max().item()

    for _ in range(50): call()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(args.iters): call()
    e.record(); torch.cuda.synchronize()
    us = s.elapsed_time(e) / args.iters * 1000

    import os
    strategy = "BULK-DMA+simd_vadd" if os.environ.get("FLASH_FLOAT_BULK_SPLITK", "0") == "1" else "thread-loads"
    print(f"[{strategy}] B={B} m={m} sk={sk}: {us:.1f} us   (rel_max_vs_fp8ref={rel:.4f})")


if __name__ == "__main__":
    main()

from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# TODO (yiakwy) : move to triton 3_7, 3_8 (I didn't veriy on triton 3_5)

# NOTE (yiakwy): DGX Spark (GB10, sm_121) types.
#   - No TMA/TensorDescriptor : F = nfft/2 + 1 is always odd, it is hard to use TMA
#   - mel is tiny and stays hot in L2
from triton.experimental.gluon.language.nvidia.blackwell import mma_v2


@gluon.jit
def swizzle2d(pid, grid_m, grid_n, GROUP_M: gl.constexpr):
    width = GROUP_M * grid_n

    group_id = pid // width
    first_pid_m = group_id * GROUP_M
    group_size = min(grid_m - first_pid_m, GROUP_M)

    gl.assume(group_size >= 0)

    pid_m = first_pid_m + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    return pid_m, pid_n


@gluon.jit
def _tf32_hi(x):
    xi = x.to(gl.int32, bitcast=True)
    hi_i = (xi + 0x0FFF + ((xi >> 13) & 1)) & -8192
    return hi_i.to(gl.float32, bitcast=True)


# NOTE (yiakwy) : decompose FP32 to low precision for full mma thorughput
@gluon.jit
def _dot_tf32x3(a, b, acc):
    a_hi = _tf32_hi(a)
    a_lo = a - a_hi

    b_hi = _tf32_hi(b)
    b_lo = b - b_hi

    acc = mma_v2(a_hi, b_hi, acc, input_precision="tf32")
    acc = mma_v2(a_hi, b_lo, acc, input_precision="tf32")
    acc = mma_v2(a_lo, b_hi, acc, input_precision="tf32")
    return acc


# NOTE (yiakwy) : accuracy verification :
#   ✅ 349012x257x80 power_mel_log: max_diff=0.0073 (all-entries=5.1347, well-conditioned=52.7%) ok=True
#   ✅ 349012x513x80 power_mel_log: max_diff=0.0202 (all-entries=6.2001, well-conditioned=53.1%) ok=True
@gluon.jit
def _dot_fp16x3(a, b, acc):
    # fp16 hi/lo parts, fp16 tc mma is 2x faster than fp32
    a_hi = a.to(gl.float16)
    a_lo = (a - a_hi.to(gl.float32)).to(gl.float16)

    b_hi = b.to(gl.float16)
    b_lo = (b - b_hi.to(gl.float32)).to(gl.float16)

    acc = mma_v2(a_hi, b_hi, acc)
    acc = mma_v2(a_hi, b_lo, acc)
    acc = mma_v2(a_lo, b_hi, acc)
    return acc


@gluon.jit
def power_mel_log_kernel(
    spec_ptr, mel_ptr, out_ptr,
    cmvn_mean_ptr, cmvn_istd_ptr,
    T, F, M,
    BLOCK_T: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_F: gl.constexpr,
    GROUP_M: gl.constexpr, num_warps: gl.constexpr,
    num_block_t: gl.constexpr, num_block_m: gl.constexpr,
    HAS_CMVN: gl.constexpr,
    PRECISION: gl.constexpr,  # "tf32x3" | "fp16x3"
):
    pid = gl.program_id(0)

    if GROUP_M > 1:
        pid_t, pid_m = swizzle2d(pid, num_block_t, num_block_m, GROUP_M)
    else:
        pid_t = pid // num_block_m
        pid_m = pid % num_block_m

    off_t = pid_t * BLOCK_T
    off_m = pid_m * BLOCK_M

    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, 8]
    )

    k_width: gl.constexpr = 2 if PRECISION == "fp16x3" else 1
    a_layout: gl.constexpr = gl.DotOperandLayout(parent=mma_layout, operand_index=0, k_width=k_width)
    b_layout: gl.constexpr = gl.DotOperandLayout(parent=mma_layout, operand_index=1, k_width=k_width)

    # A operand: power tile [BLOCK_T, BLOCK_F], gathered straight out of the
    # interleaved complex spec (viewed as fp32 [T, 2F]).
    rows = off_t + gl.arange(0, BLOCK_T, layout=gl.SliceLayout(1, a_layout))
    fcols = gl.arange(0, BLOCK_F, layout=gl.SliceLayout(0, a_layout))
    row_mask = rows < T
    spec_rows = rows.to(gl.int64) * (2 * F)

    # B operand: mel tile [BLOCK_F, BLOCK_M]
    mcols = off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(0, b_layout))
    fcols_b = gl.arange(0, BLOCK_F, layout=gl.SliceLayout(1, b_layout))
    mel_rows = mcols.to(gl.int64) * F

    acc = gl.zeros((BLOCK_T, BLOCK_M), dtype=gl.float32, layout=mma_layout)
    for f0 in range(0, F, BLOCK_F):
        fcurr = f0 + fcols
        fm1 = fcurr < F
        amask = row_mask[:, None] & fm1[None, :]
        base = spec_rows[:, None] + fcurr[None, :] * 2
        real = gl.load(spec_ptr + base, mask=amask, other=0.0)
        imag = gl.load(spec_ptr + base + 1, mask=amask, other=0.0)
        
        # step 1: power = real^2 + imag^2
        power = real * real + imag * imag

        # step 2: C += power @ mel [M, F] sliced as [BLOCK_F, BLOCK_M]
        fcurb = f0 + fcols_b
        bmask = (fcurb < F)[:, None] & (mcols < M)[None, :]
        mel = gl.load(mel_ptr + mel_rows[None, :] + fcurb[:, None], mask=bmask, other=0.0)
        if PRECISION == "fp16x3":
            acc = _dot_fp16x3(power, mel, acc)
        else:
            acc = _dot_tf32x3(power, mel, acc)

    # step 3: log floor (+ optional CMVN) in registers, masked store
    kLogFloor = 1.1920929e-7
    out = gl.log(gl.maximum(acc, kLogFloor))

    if HAS_CMVN:
        cmvn_cols = off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(0, mma_layout))
        m_mask = cmvn_cols < M
        mean_vec = gl.load(cmvn_mean_ptr + cmvn_cols, mask=m_mask, other=0.0)
        istd_vec = gl.load(cmvn_istd_ptr + cmvn_cols, mask=m_mask, other=1.0)
        out = (out - mean_vec[None, :]) * istd_vec[None, :]

    ro = off_t + gl.arange(0, BLOCK_T, layout=gl.SliceLayout(1, mma_layout))
    co = off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(0, mma_layout))
    omask = (ro < T)[:, None] & (co < M)[None, :]
    gl.store(out_ptr + ro[:, None].to(gl.int64) * M + co[None, :], out, mask=omask)


class GluonPowerMelLog:
    """
    log(( |spec_real|^2 + |spec_imag|^2 ) * mel.T) + CMVN

    Args:
        spec: complex64 [T, F]
        mel: float32 [M, F]
        out: optional float32 [T, M]
        cmvn_mean / cmvn_istd: optional float32 [M]
    """

    def __init__(
        self,
        BLOCK_T: int = 64,
        BLOCK_M: int = 32,
        BLOCK_F: int = 64,

        GROUP_SIZE_M: int = 4,

        STAGES: int = 2,  # kept for API compatibility; regular loads do not need smem stages

        NUM_WARPS: int = 4,

        # "tf32x3" (safe default) or "fp16x3" (2x mma throughput, fp16 range)
        precision: str = "tf32x3",
    ):
        assert precision in ("tf32x3", "fp16x3")
        self.BLOCK_T = BLOCK_T
        self.BLOCK_M = BLOCK_M
        self.BLOCK_F = BLOCK_F

        self.GROUP_SIZE_M = GROUP_SIZE_M

        self.STAGES = STAGES

        self.precision = precision

        self.NUM_WARPS = NUM_WARPS

        # NOTE (yiakwy): we run the kernel both on NVIDIA Hopper GPUs and on DGX Spark
        props = torch.cuda.get_device_properties(0)
        self.NUM_CUs = props.multi_processor_count  # 48 for dgx spark and 132 for hopper

    def __call__(self,
                 spec,  # complex64 [T, F]
                 mel,  # float32 [M, F]
                 out=None,  # optional float32 [T, M]
                 cmvn_mean=None, cmvn_istd=None):
        assert spec.is_cuda and spec.dtype == torch.cfloat
        assert spec.is_contiguous()
        assert mel.dtype == torch.float32 and mel.shape[1] == spec.shape[1]

        T, F = spec.shape
        M, _ = mel.shape

        if out is None:
            out = torch.empty((T, M), device=spec.device, dtype=torch.float32)
        else:
            assert out.shape == (T, M)

        HAS_CMVN = (cmvn_mean is not None) and (cmvn_istd is not None)
        if HAS_CMVN:
            assert cmvn_mean.shape[-1] == M and cmvn_istd.shape[-1] == M
            cmvn_mean = cmvn_mean.contiguous()
            cmvn_istd = cmvn_istd.contiguous()
        else:
            # dummy operands; never dereferenced (HAS_CMVN is constexpr)
            cmvn_mean = mel
            cmvn_istd = mel

        num_block_t = triton.cdiv(T, self.BLOCK_T)
        num_block_m = triton.cdiv(M, self.BLOCK_M)
        grid = (num_block_t * num_block_m,)

        power_mel_log_kernel[grid](
            spec.view(torch.float32), mel.contiguous(), out,
            cmvn_mean, cmvn_istd,
            T, F, M,
            BLOCK_T=self.BLOCK_T,
            BLOCK_M=self.BLOCK_M,
            BLOCK_F=self.BLOCK_F,
            GROUP_M=self.GROUP_SIZE_M,
            num_warps=self.NUM_WARPS,
            num_block_t=num_block_t,
            num_block_m=num_block_m,
            HAS_CMVN=HAS_CMVN,
            PRECISION=self.precision,
        )

        return out

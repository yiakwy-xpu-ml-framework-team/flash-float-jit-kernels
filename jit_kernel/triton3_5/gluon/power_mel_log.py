from typing import List, Optional, Tuple, Union

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# NOTE (yiakwy): DGX Spark (GB10, sm_121) types.
#   - wgmma (warpgroup_mma) is sm_90a-only and tcgen05 is sm_100a-only, so the
#     only tensor-core MMA usable on DGX Spark is the Blackwell-namespace mma_v2
#     (mma.sync).
#   - The host-side TensorDescriptor only exists under gluon.nvidia.hopper and
#     is generic; the blackwell device-side TMA ops consume it.
from triton.experimental.gluon.language.nvidia.blackwell import (
    fence_async_shared,
    mbarrier,
    mma_v2,
    tma,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


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


@gluon.jit
def _dot_fp16x3(a, b, acc):
    a_hi = a.to(gl.float16)
    a_lo = (a - a_hi.to(gl.float32)).to(gl.float16)
    b_hi = b.to(gl.float16)
    b_lo = (b - b_hi.to(gl.float32)).to(gl.float16)
    acc = mma_v2(a_hi, b_hi, acc)
    acc = mma_v2(a_hi, b_lo, acc)
    acc = mma_v2(a_lo, b_hi, acc)
    return acc


@gluon.jit
def _dot(a, b, acc, PRECISION: gl.constexpr):
    if PRECISION == "fp16x3":
        return _dot_fp16x3(a, b, acc)
    else:
        return _dot_tf32x3(a, b, acc)


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
        acc = _dot(power, mel, acc, PRECISION)

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


@gluon.jit
def power_mel_log_tma_kernel(
    real_desc, imag_desc, mel_t_desc,
    out_desc,
    cmvn_mean_ptr, cmvn_istd_ptr,
    T, F, M,
    BLOCK_T: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_F: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    STAGES: gl.constexpr,
    num_warps: gl.constexpr,
    num_block_t: gl.constexpr,
    num_block_m: gl.constexpr,
    HAS_CMVN: gl.constexpr,
    PRECISION: gl.constexpr,  # "tf32x3" | "fp16x3"
):
    pid = gl.program_id(0)

    if GROUP_SIZE_M > 1:
        pid_t, pid_m = swizzle2d(pid, num_block_t, num_block_m, GROUP_SIZE_M)
    else:
        pid_t = pid // num_block_m
        pid_m = pid % num_block_m

    off_t = pid_t * BLOCK_T
    off_m = pid_m * BLOCK_M

    dtype = real_desc.dtype

    tile_real = gl.allocate_shared_memory(
        dtype, [STAGES] + real_desc.block_shape, real_desc.layout
    )
    tile_imag = gl.allocate_shared_memory(
        dtype, [STAGES] + imag_desc.block_shape, imag_desc.layout
    )
    tile_mel = gl.allocate_shared_memory(
        dtype, [STAGES] + mel_t_desc.block_shape, mel_t_desc.layout
    )

    load_ready_bars = mbarrier.allocate_mbarrier(batch=STAGES)
    for i in gl.static_range(STAGES):
        mbarrier.init(load_ready_bars.index(i), count=1)

    num_f_iters = (F + BLOCK_F - 1) // BLOCK_F

    LOAD_BYTES: gl.constexpr = (
        real_desc.block_type.nbytes * 2 + mel_t_desc.block_type.nbytes
    )

    # 1. Ramp-up: fill the pipeline with the first min(STAGES, num_f_iters) F-blocks
    for i in gl.static_range(STAGES):
        off_f = i * BLOCK_F
        if off_f < F:
            bar = load_ready_bars.index(i)
            mbarrier.expect(bar, LOAD_BYTES)
            tma.async_copy_global_to_shared(
                real_desc, [off_t, off_f], bar, tile_real.index(i)
            )
            tma.async_copy_global_to_shared(
                imag_desc, [off_t, off_f], bar, tile_imag.index(i)
            )
            tma.async_copy_global_to_shared(
                mel_t_desc, [off_f, off_m], bar, tile_mel.index(i)
            )

    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, 8]
    )

    k_width: gl.constexpr = 2 if PRECISION == "fp16x3" else 1
    a_layout: gl.constexpr = gl.DotOperandLayout(parent=mma_layout, operand_index=0, k_width=k_width)
    b_layout: gl.constexpr = gl.DotOperandLayout(parent=mma_layout, operand_index=1, k_width=k_width)

    acc = gl.zeros((BLOCK_T, BLOCK_M), dtype=gl.float32, layout=mma_layout)

    # 2. Main loop over F blocks
    for f in range(num_f_iters):
        stage = f % STAGES
        phase = (f // STAGES) & 1

        bar = load_ready_bars.index(stage)
        mbarrier.wait(bar, phase=phase)

        real = tile_real.index(stage).load(a_layout)
        imag = tile_imag.index(stage).load(a_layout)
        mel = tile_mel.index(stage).load(b_layout)

        # step 1: power = real^2 + imag^2
        power = real * real + imag * imag

        # step 2: C += power @ mel^T (mel transposed on host)
        acc = _dot(power, mel, acc, PRECISION)

        # step 3: prefetch the next F-block into the just-consumed slot
        next_f = (f + STAGES) * BLOCK_F
        if next_f < F:
            # make sure every warp has finished reading the slot before the
            # async proxy (TMA) overwrites it
            gl.barrier()
            mbarrier.expect(bar, LOAD_BYTES)
            tma.async_copy_global_to_shared(
                real_desc, [off_t, next_f], bar, tile_real.index(stage)
            )
            tma.async_copy_global_to_shared(
                imag_desc, [off_t, next_f], bar, tile_imag.index(stage)
            )
            tma.async_copy_global_to_shared(
                mel_t_desc, [next_f, off_m], bar, tile_mel.index(stage)
            )

    # 3. Epilogue: log floor (+ optional CMVN) in registers, then TMA store
    kLogFloor = 1.1920929e-7
    out = gl.log(gl.maximum(acc, kLogFloor))

    if HAS_CMVN:
        cmvn_layout: gl.constexpr = gl.SliceLayout(0, mma_layout)
        offs_m = off_m + gl.arange(0, BLOCK_M, layout=cmvn_layout)
        m_mask = offs_m < M
        mean_vec = gl.load(cmvn_mean_ptr + offs_m, mask=m_mask, other=0.0)
        istd_vec = gl.load(cmvn_istd_ptr + offs_m, mask=m_mask, other=1.0)
        out = (out - mean_vec[None, :]) * istd_vec[None, :]

    out_tile = gl.allocate_shared_memory(
        out_desc.dtype, out_desc.block_shape, out_desc.layout, out
    )
    fence_async_shared()

    tma.async_copy_shared_to_global(out_desc, [off_t, off_m], out_tile)
    tma.store_wait(pendings=0)
    out_tile._keep_alive()

    for i in gl.static_range(STAGES):
        mbarrier.invalidate(load_ready_bars.index(i))


class GluonPowerMelLog:
    """
    log(( |spec_real|^2 + |spec_imag|^2 ) * mel.T) + CMVN

    Args:
        spec: complex64 [T, F]  |  (spec_real, spec_imag) each fp32 [T, F]
        mel: float32 [M, F]
        out: optional float32 [T, M]
        cmvn_mean / cmvn_istd: optional float32 [M]
        precision: "tf32x3" (default) or "fp16x3" (2x mma throughput; requires
            |power|, |mel| within fp16 range, safe for velox mel data)
    """

    def __init__(
        self,
        BLOCK_T: int = 64,
        BLOCK_M: int = 32,
        BLOCK_F: int = 64,

        GROUP_SIZE_M: int = 4,

        # NOTE (yiakwy) : 2 for GB10 and 4 for Hopper
        STAGES: int = 2,

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

        self.NUM_WARPS = NUM_WARPS

        self.precision = precision

        # TODO (yiakwy) : check Hopper
        props = torch.cuda.get_device_properties(0)
        self.NUM_CUs = props.multi_processor_count  # 48 for dgx spark and 132 for hopper

    @staticmethod
    def is_tma_16B_aligned(t: torch.Tensor) -> bool:
        return (
            t.dim() == 2
            and t.dtype == torch.float32
            and t.stride(-1) == 1
            and t.stride(0) % 4 == 0
            and t.data_ptr() % 16 == 0
        )

    @staticmethod
    def _block_smem_layout(block_shape: List[int]) -> gl.NVMMASharedLayout:
        return gl.NVMMASharedLayout.get_default_for(block_shape, gl.float32)

    @staticmethod
    def _pad_last_dim(x: torch.Tensor, multiple: int) -> torch.Tensor:
        n = x.shape[-1]
        n_pad = triton.cdiv(n, multiple) * multiple
        if n_pad == n:
            return x
        padded = torch.zeros(*x.shape[:-1], n_pad, device=x.device, dtype=x.dtype)
        padded[..., :n] = x
        return padded

    def dispatch_without_tma(self, spec, mel, out, cmvn_mean, cmvn_istd):
        # interleaved cfloat path: zero host copies
        T, F = spec.shape
        M, _ = mel.shape

        HAS_CMVN = (cmvn_mean is not None) and (cmvn_istd is not None)
        if HAS_CMVN:
            cmvn_mean = cmvn_mean.contiguous()
            cmvn_istd = cmvn_istd.contiguous()
        else:
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

    def dispatch_with_tma(self, spec_real, spec_imag, mel, out, cmvn_mean, cmvn_istd):
        # 16B-aligned path: TMA pipeline
        T, F = spec_real.shape
        M, _ = mel.shape

        # mel transposed once on the host ([F, M]); it is tiny (~165KB).
        mel_t = mel.contiguous().t().contiguous()
        M_pad = M
        if M % 4 != 0:
            mel_t = self._pad_last_dim(mel_t, 4)  # zero-padded mel columns are exact
            M_pad = mel_t.shape[-1]

        out_buf = out
        if M_pad != M or out.stride(0) % 4 != 0:
            out_buf = torch.empty((T, M_pad), device=spec_real.device, dtype=torch.float32)

        spec_block_shape = [self.BLOCK_T, self.BLOCK_F]
        mel_block_shape = [self.BLOCK_F, self.BLOCK_M]
        out_block_shape = [self.BLOCK_T, self.BLOCK_M]

        spec_real_layout = self._block_smem_layout(spec_block_shape)
        spec_imag_layout = self._block_smem_layout(spec_block_shape)
        mel_t_layout = self._block_smem_layout(mel_block_shape)
        out_layout = self._block_smem_layout(out_block_shape)

        real_desc = TensorDescriptor.from_tensor(spec_real, spec_block_shape, spec_real_layout)
        imag_desc = TensorDescriptor.from_tensor(spec_imag, spec_block_shape, spec_imag_layout)
        mel_t_desc = TensorDescriptor.from_tensor(mel_t, mel_block_shape, mel_t_layout)
        out_desc = TensorDescriptor.from_tensor(out_buf, out_block_shape, out_layout)

        HAS_CMVN = (cmvn_mean is not None) and (cmvn_istd is not None)
        if HAS_CMVN:
            cmvn_mean = cmvn_mean.contiguous()
            cmvn_istd = cmvn_istd.contiguous()
        else:
            cmvn_mean = mel_t
            cmvn_istd = mel_t

        num_block_t = triton.cdiv(T, self.BLOCK_T)
        num_block_m = triton.cdiv(M, self.BLOCK_M)
        grid = (num_block_t * num_block_m,)

        power_mel_log_tma_kernel[grid](
            real_desc, imag_desc, mel_t_desc,
            out_desc,
            cmvn_mean, cmvn_istd,
            T, F, M,
            BLOCK_T=self.BLOCK_T,
            BLOCK_M=self.BLOCK_M,
            BLOCK_F=self.BLOCK_F,
            GROUP_SIZE_M=self.GROUP_SIZE_M,
            STAGES=self.STAGES,
            num_warps=self.NUM_WARPS,
            num_block_t=num_block_t,
            num_block_m=num_block_m,
            HAS_CMVN=HAS_CMVN,
            PRECISION=self.precision,
        )

        if out_buf is not out:
            out.copy_(out_buf[:, :M])

    def __call__(self,
                 spec,  # complex64 [T, F]  |  (spec_real, spec_imag) fp32 [T, F]
                 mel,  # float32 [M, F]
                 out=None,  # optional float32 [T, M]
                 cmvn_mean=None, cmvn_istd=None):
        if isinstance(spec, (tuple, list)):
            assert len(spec) == 2, "16B aligned spec must be (spec_real, spec_imag)"

            spec_real, spec_imag = spec

            assert spec_real.shape == spec_imag.shape, "real/imag shape mismatch"
            assert mel.shape[1] == spec_real.shape[1], "mel F mismatch"
            assert self.is_tma_16B_aligned(spec_real), (
                "spec_real is not TMA 16B-aligned: need shape [T, F], "
                "stride(-1) == 1 and stride(0) % 4 == 0; "
                f"got shape={tuple(spec_real.shape)} stride={spec_real.stride()}"
            )
            assert self.is_tma_16B_aligned(spec_imag), (
                "spec_imag is not TMA 16B-aligned: need shape [T, F], "
                "stride(-1) == 1 and stride(0) % 4 == 0; "
                f"got shape={tuple(spec_imag.shape)} stride={spec_imag.stride()}"
            )
        else:
            assert spec.is_cuda and spec.dtype == torch.cfloat, \
                "spec must be complex64 [T, F] or (spec_real, spec_imag) fp32"
            assert spec.is_contiguous()
            assert mel.shape[1] == spec.shape[1], "mel F mismatch"

        assert mel.dtype == torch.float32
        
        T = spec[0].shape[0] if isinstance(spec, (tuple, list)) else spec.shape[0]
        M = mel.shape[0]

        if out is None:
            out = torch.empty((T, M), device=mel.device, dtype=torch.float32)
        else:
            assert out.shape == (T, M)

        HAS_CMVN = (cmvn_mean is not None) and (cmvn_istd is not None)
        if HAS_CMVN:
            assert cmvn_mean.shape[-1] == M and cmvn_istd.shape[-1] == M

        if isinstance(spec, (tuple, list)):
            self.dispatch_with_tma(spec_real, spec_imag, mel, out, cmvn_mean, cmvn_istd)
        else:
            self.dispatch_without_tma(spec, mel, out, cmvn_mean, cmvn_istd)

        return out

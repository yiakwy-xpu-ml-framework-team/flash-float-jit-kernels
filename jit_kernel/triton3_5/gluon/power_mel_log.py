import ctypes
import hashlib
import itertools
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl
from packaging import version
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_init,
    warpgroup_mma_wait,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


def is_hopper():
    if not torch.cuda.is_available():
        return False
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 9


# Ref kernels are adatped from modded-nanogpt
if version.parse(triton.__version__) < version.parse("3.6"):

    # adatped from triton 3.6+
    @gluon.jit
    def xcd_swizzle(pid, domain_size, XCD_SWIZZLE: gl.constexpr):
        """
        Swizzle the program id based on integer XCD_SWIZZLE.
        This is useful for reording how blocks are ordered. A scheduler may, for example,
        assign sequential blocks 0, 1, 2, 3, ..., 8, 9, 10.. to its 8 hardware units 0, 1, 2, 3, ..., 0, 1, 2.
        This pattern may not be ideal for memory access, and it may be better to swizzle so the assignment
        becomes 0, 0, 0, 0, ..., 1, 1, 1, ... In the swizzled arrangement, sequential blocks are assigned to
        the same hardware unit.
        """
        # Number of pids per group in the new arrangement
        pids_per_group = domain_size // XCD_SWIZZLE
        extra_pid_groups = domain_size % XCD_SWIZZLE

        # Compute current current and local pid within the group
        group = pid % XCD_SWIZZLE
        local_pid = pid // XCD_SWIZZLE

        # Calculate new pid based on the new grouping
        new_pid = group * pids_per_group + min(group, extra_pid_groups) + local_pid
        return new_pid

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

    # NOTE (yiakwy) : FIX newer triton API
    setattr(gl, "swizzle2d", swizzle2d)
    setattr(gl, "xcd_swizzle", xcd_swizzle)


@gluon.constexpr_function
def get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps):
    warps_per_cta = [4, 1]
    m = 16
    while warps_per_cta[0] * warps_per_cta[1] != num_warps:
        if BLOCK_M > m * warps_per_cta[0]:
            warps_per_cta[0] *= 2
        else:
            warps_per_cta[1] *= 2
    return warps_per_cta


@gluon.constexpr_function
def get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps):
    m = 16
    mReps = triton.cdiv(BLOCK_M, m)
    nReps = triton.cdiv(num_warps, mReps)
    maxN = max(BLOCK_N // nReps, 8)
    n = 256
    while n > maxN or BLOCK_N % n != 0:
        n -= 8
    assert n >= 8, "expected to find a valid n"
    return n


@gluon.constexpr_function
def pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps):
    m = 16
    k = 256 // dtype.primitive_bitwidth
    n = get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps)
    warps_per_cta = get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps)
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=[int(w) for w in warps_per_cta],
        instr_shape=[int(m), int(n), int(k)],
    )


@gluon.jit
def power_mel_log_kernel(
    spec_real_desc, spec_imag_desc, mel_desc,
    out_desc,
    cmvn_mean_ptr,
    cmvn_istd_ptr,
    T, F, M,
    BLOCK_T: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_F: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    STAGES: gl.constexpr,
    NUM_CUs: gl.constexpr,
    num_warps : gl.constexpr,
    num_block_m: gl.constexpr,
    num_block_n: gl.constexpr,
    num_tiles: gl.constexpr,
    HAS_CMVN: gl.constexpr,
    _cache_plance_holder: gl.constexpr,
):
    _pid = gl.program_id(0)

    dtype = spec_real_desc.dtype

    tile_real = gl.allocate_shared_memory(
        dtype, [STAGES] + spec_real_desc.block.shape, spec_real_desc.layout
    )
    tile_imag = gl.allocate_shared_memory(
        dtype, [STAGES] + spec_imag_desc.bock.shape, spec_imag_desc.layout
    )
    tile_mel = gl.allocate_shared_memory(
        dtype, [STAGES] + mel_desc.block.shape, mel_desc.layout
    )

    # NOTE (yiakwy) : on-chip buffer for output
    out_tile = gl.allocate_shared_memory(
        out_desc.dtype, out_desc.block_type.shape, out_desc.layout
    )

    # TODO (yiakwy) : add support of WASP
    # NOTE (yiakwy) : init TMA outside of persistent loop
    load_ready_bars = mbarrier.allocate_mbarrier(batch=STAGES)
    for i in gl.static_range(STAGES):
        mbarrier.init(load_ready_bars.index(i), count=1)

    for pid in range(_pid, num_tiles, NUM_CUs):
        if GROUP_SIZE_M > 1:
            pid_t, pid_m = gl.swizzle2d(pid, num_block_m, num_block_n, GROUP_SIZE_M)
        else:
            pid_t = pid // num_block_n
            pid_m = pid % num_block_n

        write_stage = 0
        read_stage = 0

        off_t = pid_t * BLOCK_T
        off_m = pid_m * BLOCK_M

        # NOTE (yiakwy) : init acc
        wgmma_layout = pick_wgmma_layout(dtype, BLOCK_T, BLOCK_M, num_warps)

        acc = warpgroup_mma_init(
            gl.zeros((BLOCK_T, BLOCK_M), dtype=gl.float32, layout=wgmma_layout)
        )

        # 1. Ramp Up to fill with F blocks
        for i in gl.static_range(STAGES):
            off_f = i * BLOCK_F
            if off_f < F:
                mbarrier.expect(
                    load_ready_bars.index(write_stage),
                    spec_real_desc.block_type.nbytes * 2 + mel_desc.block_type.nbytes,
                )

                tma.async_load(
                    spec_real_desc,
                    [off_t, off_f],
                    load_ready_bars.index(write_stage),
                    tile_real.index(write_stage),
                )
                tma.async_load(
                    spec_imag_desc,
                    [off_t, off_f],
                    load_ready_bars.index(write_stage),
                    tile_imag.index(write_stage),
                )
                tma.async_load(
                    mel_desc,
                    [off_m, off_f],
                    load_ready_bars.index(write_stage),
                    tile_mel.index(write_stage),
                )
                write_stage = (write_stage + 1) % STAGES

        tma_phase = 0

        # 2. Main loop
        for f in range(gl.cdiv(F, BLOCK_F)):
            off_f = f * BLOCK_F

            mbarrier.wait(load_ready_bars.index(read_stage), phase=tma_phase)

            real = tile_real.index(read_stage)
            imag = tile_imag.index(read_stage)
            mel = tile_mel.index(read_stage)

            # step 1 : power = real^2 + imag^2
            real = real * real + imag * imag

            # step 2 : wgmma：C += power @ mel^T 
            acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
            acc = warpgroup_mma(real, mel, acc, transpose_b=True, is_async=True)

            # issue next load
            next_f = (f + STAGES) * BLOCK_F
            if next_f < F:
                mbarrier.expect(
                    load_ready_bars.index(write_stage),
                    spec_real_desc.block_type.nbytes * 2 + mel_desc.block_type.nbytes,
                )

                tma.async_load(
                    spec_real_desc,
                    [off_t, next_f],
                    load_ready_bars.index(write_stage),
                    tile_real.index(write_stage),
                )
                tma.async_load(
                    spec_imag_desc,
                    [off_t, next_f],
                    load_ready_bars.index(write_stage),
                    tile_imag.index(write_stage),
                )
                tma.async_load(
                    mel_desc,
                    [off_m, next_f],
                    load_ready_bars.index(write_stage),
                    tile_mel.index(write_stage),
                )
                write_stage = (write_stage + 1) % STAGES

            read_stage = (read_stage + 1) % STAGES
            if read_stage == 0:
                tma_phase ^= 1

        # 3. Epilogue log
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

        out_tile.store(acc.to(out_desc.dtype))
        fence_async_shared() # TODO (yiakwy) : remove

        kLogFloor = 1.1920929e-7
        # inplace
        out_tile[:, :] = gl.log(gl.maximum(out_tile, kLogFloor))

        if HAS_CMVN:
            mean_vec = gl.load(cmvn_mean_ptr + off_m + gl.arange(0, BLOCK_M))
            istd_vec = gl.load(cmvn_istd_ptr + off_m + gl.arange(0, BLOCK_M))
            # inplace
            out_tile[:, :] = (out_tile - mean_vec[None, :]) * istd_vec[None, :]

        fence_async_shared()
        tma.async_copy_shared_to_global(out_desc, [off_t, off_m], out_tile)
        tma.store_wait(pendings=0)


class GluonPowerMelLog:
    def __init__(
        self,
        BLOCK_T: int = 64,
        BLOCK_M: int = 32,
        BLOCK_F: int = 64,

        GROUP_SIZE_M: int = 4,

        STAGES: int = 4,

        NUM_WARPS: int = 8,
    ):
        self.BLOCK_T = BLOCK_T
        self.BLOCK_M = BLOCK_M
        self.BLOCK_F = BLOCK_F

        self.GROUP_SIZE_M = GROUP_SIZE_M

        self.STAGES = STAGES

        self.NUM_WARPS = NUM_WARPS

        # NOTE (yiakwy) : we will run the kernel both in NVIDIA Hopper GPU and DGX Spark
        props = torch.cuda.get_device_properties(0)
        self.NUM_CUs = props.multi_processor_count # 48 for dgx spark and 132 for hopper

    def __call__(self, 
                 spec, # complex64 [T, F]
                 mel, # float32 [M, F]
                 out=None, # optional float32 [T, M]
                 cmvn_mean=None, cmvn_istd=None):
        
        T, F = spec.shape
        M, _ = mel.shape

        if out is None:
            out = torch.empty((T, M), device=spec.device, dtype=torch.float32)
            cache_mode = 1
        else:
            assert out.shape == (T, M)
            cache_mode = 0

        spec_real = spec.real.contiguous()
        spec_imag = spec.imag.contiguous()

        spec_block_shape = [self.BLOCK_T, self.BLOCK_F]

        spec_real_layout = gl.NVMMASharedLayout.get_default_for(spec_block_shape, gl.float32)
        spec_imag_layout = gl.NVMMASharedLayout.get_default_for(spec_block_shape, gl.float32)

        real_desc = TensorDescriptor.from_tensor(spec_real, spec_block_shape, spec_real_layout)
        imag_desc = TensorDescriptor.from_tensor(spec_imag, spec_block_shape, spec_imag_layout)

        mel_shape = [self.BLOCK_M, self.BLOCK_F]
        out_shape = [self.BLOCK_T, self.BLOCK_M]

        mel_layout = gl.NVMMASharedLayout.get_default_for(mel_shape, gl.float32)
        out_layout = gl.NVMMASharedLayout.get_default_for(out_shape, gl.float32)

        mel_desc = TensorDescriptor.from_tensor(mel, mel_shape, mel_layout)
        out_desc = TensorDescriptor.from_tensor(out, out_shape, out_layout)

        HAS_CMVN = (cmvn_mean is not None) and \
                   (cmvn_istd is not None)

        if HAS_CMVN:
            cmvn_mean_ptr = cmvn_mean.contiguous().data_ptr()
            cmvn_istd_ptr = cmvn_istd.contiguous().data_ptr()
        else:
            raise Exception("CMVN must be provided.")

        num_block_m = triton.cdiv(T, self.BLOCK_T)
        num_block_n = triton.cdiv(M, self.BLOCK_M)

        num_tiles = num_block_m * num_block_n

        grid_size = min(self.NUM_CUs, num_tiles)

        grid = (grid_size,)

        power_mel_log_kernel[grid](
            real_desc, imag_desc, mel_desc,
            out_desc,
            cmvn_mean_ptr, cmvn_istd_ptr,
            T, F, M,
            BLOCK_T=self.BLOCK_T,
            BLOCK_M=self.BLOCK_M,
            BLOCK_F=self.BLOCK_F,
            GROUP_SIZE_M=self.GROUP_SIZE_M,
            STAGES=self.STAGES,
            NUM_CUs=self.NUM_CUs,
            num_warps=self.NUM_WARPS,
            num_block_m=num_block_m,
            num_block_n=num_block_n,
            num_tiles=num_tiles,
            HAS_CMVN=HAS_CMVN,
            _cache_plance_holder=cache_mode,
        )

        return out
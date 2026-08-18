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


USE_TVM_FFI = False

tvm_ffi_modules = {
    "XXT": None,
}


def is_hopper():
    if not torch.cuda.is_available():
        return False
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 9


# NOTE (yiakwy) : useful for multiple-die chiplet architecture :
#   - Blackwell(2 dies), Rubin(2 dies),
#   - Rubin Ultra(4 dies), MI300(4 dies),
#   - MI300X(8 dies)
#
# Make sure SPLIT-K blocks cooperatively work on the same die to maximize the cache hit re-usage
@gluon.jit
def _remmap_pid(tile_id, tall_xcds, BLOCKS_PER_XCD, NUM_XCDS):
    xcd = tile_id % NUM_XCDS
    local_pid = tile_id // NUM_XCDS  # (local_id, xcd), strides : (NUM_XCDS, 1)
    if xcd < tall_xcds:
        tile_id = (
            xcd * BLOCKS_PER_XCD + local_pid
        )  # (xcd, local_id), strides : (BLOCKS_PER_XCD, 1)
    else:
        tile_id = (
            tall_xcds * BLOCKS_PER_XCD
            + (xcd - tall_xcds) * (BLOCKS_PER_XCD - 1)
            + local_pid
        )
    return tile_id


# NOTE (yiakwy) : see our paper for cuda kernel for swizzle of lower left triangle
# pid = pid_m * (pid_m + 1) / 2 + pid_n
# pid_m*2 + pid_m - 2*pid - pid_n = 0
@gluon.jit
def linear_to_tril(pid):
    # row = floor((sqrt(8*pid + 1) - 1) / 2)
    row = tl.floor((tl.math.sqrt(8.0 * pid + 1.0) - 1.0) / 2.0).to(tl.int32)
    col = pid - (row * (row + 1)) // 2
    return row, col


@gluon.jit
def _compute_pid(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M):
    if GROUP_SIZE_M == 1:
        pid_n = tile_id % num_pid_n
        pid_m = tile_id // num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n

        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


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

else:
    # for hopper tma triton 3.6+

    @gluon.jit
    def allocate_mbarrier(batch: gl.constexpr = None, two_ctas: gl.constexpr = False):
        """
        Helper function to allocate an mbarrier

        Args:
            two_ctas (bool): Whether the barrier should synchronize every other CTA
        """
        num_ctas: gl.constexpr = gl.num_ctas()
        num_elems: gl.constexpr = num_ctas if not two_ctas else num_ctas // 2

        gl.static_assert(batch is None or isinstance(batch.value, int))

        bar = gl.allocate_shared_memory(
            gl.int64, [batch, num_elems], mbarrier.MBarrierLayout()
        )
        return bar

    # NOTE (yiakwy) : hopper does not implement allocate_mbarrier method
    if not hasattr(mbarrier, "allocate_mbarrier"):
        setattr(mbarrier, "allocate_mbarrier", allocate_mbarrier)

    # NOTE (yiakwy) : see https://github.com/triton-lang/triton/pull/10083
    @gl._core.builtin
    def async_load(
        tensor_desc, coord, barrier, result, pred=True, multicast=False, _semantic=None
    ):
        # NOTE (yiakwy) : TMA multicast is not supported in triton 3.6 release
        return tma.async_copy_global_to_shared(
            tensor_desc, coord, barrier, result, pred=pred, _semantic=_semantic
        )

    if not hasattr(tma, "async_load"):
        setattr(tma, "async_load", async_load)


# Adpated from gemm swizzle by @byronxu99 for reference
@gluon.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Split output matrix into blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    # Map PID to a single matrix in batch
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    # Map PID to 2D grid of blocks
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = gl.swizzle2d(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx


def XXT(
    A: torch.Tensor,
    out: torch.Tensor,
    use_gluon: bool = True,
    use_tvm_ffi: Optional[bool] = None,
):
    """
    Launch Triton kernel to compute C = A @ A.T
    """
    assert A.ndim == 2 or A.ndim == 3
    assert out.size(-2) == out.size(-1), "Output matrix has incorrect shape"

    M, K = A.shape[-2:]

    if K == 768:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 64
    else:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 128

    gemm_op = GluonXXT(
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        LOWER_UPPER=1,
    )

    return gemm_op(A, out, use_tvm_ffi=use_tvm_ffi)


@gluon.constexpr_function
def get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps):
    # NOTE (yiakwy) : default to warpgrup 4x1 layout
    warps_per_cta = [4, 1]
    m = 16
    # Tile the atom until we have enough warps.
    while warps_per_cta[0] * warps_per_cta[1] != num_warps:
        # Tile along M only if it would not cause broadcasting.
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


# TODO (yiakwy) : add support of split-k


# NOTE（yiakwy）: the kernel is adapted from
#  - modded-nanogpt XXT_kernel
#  - our new triangular sched algorithm with TMA and multcast support, see our C++ multistage gemm algorithm
#  - gluon wgmma matmul kernel
#    - https://github.com/triton-lang/triton/blob/main/python/tutorials/gluon/05-wgmma.py and
#    - https://github.com/triton-lang/triton/blob/main/python/examples/gluon/03-matmul-multicta.py
@gluon.jit
def XXT_kernel(
    A_desc,
    C_desc,
    M,
    K,
    # a_stride_b, a_stride_r, a_stride_c,
    # c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    LOWER_UPPER: gl.constexpr,
    STAGES: gl.constexpr,
    num_warps: gl.constexpr,
):
    pid = gl.program_id(axis=0)

    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(M, BLOCK_SIZE_N)

    batch_idx = pid // (num_pid_m * num_pid_n)
    pid_local = pid % (num_pid_m * num_pid_n)

    if GROUP_SIZE_M == 1:
        pid_n = pid_local % num_pid_n
        pid_m = pid_local // num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid_local // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid_local % num_pid_in_group) % group_size_m)
        pid_n = (pid_local % num_pid_in_group) // group_size_m

    # TODO (yiakwy) : replace with
    # pid_n, pid_m = _compute_pid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M) # type: ignore

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N

    # TODO (yiakwy) : updated to new scheduler for triangular matrix
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)

    gl.assume(skip_block_above_diag == True)

    if skip_block_below_diag or skip_block_above_diag:
        return

    m_idx = m_idx + batch_idx * M
    n_idx = n_idx + batch_idx * M

    # NOTE (yiakwy) : NV TMA does not need to use gl.DistributedLinearLayout to compute per-thread offset to load data into registers

    # NOTE (yiakwy) : NV TMA does not need the combination of gl.SliceLayout and gl.BlockedLayout to to (buffer) load data into shmem

    dtype: gl.constexpr = A_desc.dtype
    tile_a = gl.allocate_shared_memory(
        dtype, [STAGES] + A_desc.block_type.shape, A_desc.layout
    )

    # TODO (yiakwy) : support BLOCK_SIZE_M != BLOCK_SIZE_N
    gl.assume(
        BLOCK_SIZE_M == BLOCK_SIZE_N
    )  #  "only support symmetric blocking to reduce shared memory usage"

    wgmma_layout: gl.constexpr = pick_wgmma_layout(
        dtype, BLOCK_SIZE_M, BLOCK_SIZE_N, num_warps
    )

    acc_dtype = gl.float32
    acc = warpgroup_mma_init(
        gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=wgmma_layout)
    )

    off_m = m_idx

    # mma_barrier_count : gl.constexpr = 4 # COMPUTE_WARPS
    # load_empty_bars = allocate_mbarrier(batch=STAGES)

    # TODO (yiakwy) : add support of WASP
    load_ready_bars = mbarrier.allocate_mbarrier(batch=STAGES)

    write_stage = 0
    read_stage = 0

    for i in gl.static_range(STAGES):
        # mbarrier.init(load_empty_bars.index(i), count=mma_barrier_count)
        mbarrier.init(load_ready_bars.index(i), count=1)

    # 1. Ramp Up to fill
    for i in gl.static_range(STAGES):
        off_k = i * BLOCK_SIZE_K
        if off_k < K:
            a = tile_a.index(write_stage)

            mbarrier.expect(
                load_ready_bars.index(write_stage), A_desc.block_type.nbytes
            )

            # NOTE (yiakwy) : triton 3.6 does not suppor multicast, please updated to triton 3.7 (after Apri 2026) for the performance boost
            tma.async_load(
                A_desc,
                [off_m, off_k],
                load_ready_bars.index(write_stage),
                a,
                multicast=True,
            )
            # async_copy_global_to_shared(tensor_desc, coord, barrier, result, pred=True, _semantic=None):

            write_stage = (write_stage + 1) % STAGES

    tma_phase = 0

    # 2. Main loop
    ssteps = gl.cdiv(K, BLOCK_SIZE_K)
    for k in range(gl.cdiv(K, BLOCK_SIZE_K)):
        off_k = k * BLOCK_SIZE_K

        mbarrier.wait(load_ready_bars.index(read_stage), phase=tma_phase)

        # TODO (yiakwy) : add wgmma
        a = tile_a.index(read_stage)
        at = a.permute((1, 0))  # gl.transpose(a)

        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))
        acc = warpgroup_mma(a, at, acc, is_async=True)

        # issue next load
        next_k = (k + STAGES) * BLOCK_SIZE_K
        if next_k < K:
            _a = tile_a.index(write_stage)

            mbarrier.expect(
                load_ready_bars.index(write_stage), A_desc.block_type.nbytes
            )

            tma.async_load(
                A_desc,
                [off_m, next_k],
                load_ready_bars.index(write_stage),
                _a,
                multicast=True,
            )

            write_stage = (write_stage + 1) % STAGES

        read_stage = (read_stage + 1) % STAGES
        if read_stage == 0:
            tma_phase ^= 1

    # 3. Epilogue
    acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

    c = gl.allocate_shared_memory(C_desc.dtype, C_desc.block_type.shape, C_desc.layout)
    c.store(acc.to(C_desc.dtype))
    fence_async_shared()

    # NOTE (yiakwy) : store & transpose store
    off_n = n_idx
    tma.async_copy_shared_to_global(C_desc, [off_m, off_n], c)

    ct = c.permute((1, 0))  # gl.transpose(c)
    tma.async_copy_shared_to_global(C_desc, [off_n, off_m], ct)
    tma.store_wait(pendings=0)


def _gluon_xxt_tvm_ffi(A, C, grid, kernel_kwargs, key):
    import tvm_ffi

    from jit_kernel.triton3_5.gluon.tvm_ffi_mod import (
        flatten_tvm_ffi_args,
        generate_tvm_ffi_source,
    )

    global tvm_ffi_modules

    if tvm_ffi_modules["XXT"] is not None and key in tvm_ffi_modules["XXT"]:
        module, kernel_name, compiled_kernel, gx, gy, gz = tvm_ffi_modules["XXT"][
            key
        ]
    else:
        compiled_kernel = XXT_kernel[grid](
            kernel_kwargs["A_desc"],
            kernel_kwargs["C_desc"],
            M=kernel_kwargs["M"],
            K=kernel_kwargs["K"],
            BLOCK_SIZE_M=kernel_kwargs["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=kernel_kwargs["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=kernel_kwargs["BLOCK_SIZE_K"],
            GROUP_SIZE_M=kernel_kwargs["GROUP_SIZE_M"],
            LOWER_UPPER=kernel_kwargs["LOWER_UPPER"],
            STAGES=kernel_kwargs["STAGES"],
            num_warps=kernel_kwargs["num_warps"],
        )
        torch.cuda.synchronize()

        cubin_bytes = compiled_kernel.kernel
        kernel_name = compiled_kernel.name
        sources, _constants = generate_tvm_ffi_source(compiled_kernel, kernel_name)

        module = tvm_ffi.cpp.load_inline(
            "triton3_5_gluon_xxt_loader",
            cuda_sources=sources,
            extra_ldflags=["-lcudart", "-lcuda"],
            embed_cubin={"triton_cubin": cubin_bytes},
        )

        gx = int(grid[0])
        gy = int(grid[1]) if len(grid) > 1 else 1
        gz = int(grid[2]) if len(grid) > 2 else 1

        if tvm_ffi_modules["XXT"] is None:
            tvm_ffi_modules["XXT"] = {}

        tvm_ffi_modules["XXT"][key] = (
            module,
            kernel_name,
            compiled_kernel,
            gx,
            gy,
            gz,
        )

    ffi_args = flatten_tvm_ffi_args(
        compiled_kernel,
        kernel_kwargs,
        (gx, gy, gz),
    )
    getattr(module, kernel_name)(*ffi_args)
    return C


class GluonXXT:
    def __init__(
        self,
        BLOCK_SIZE_M: int = 128,
        BLOCK_SIZE_N: int = 128,
        BLOCK_SIZE_K: int = 64,
        GROUP_SIZE_M: int = 8,
        STAGES: int = 4,
        NUM_WARPS: int = 8,
        LOWER_UPPER: int = 1,  # NOTE (yiakwy) : 1 skip block above diaglogue; 0 skip block below diaglogue
    ):
        self.BLOCK_SIZE_M = BLOCK_SIZE_M
        self.BLOCK_SIZE_N = BLOCK_SIZE_N
        self.BLOCK_SIZE_K = BLOCK_SIZE_K

        self.a_shape = [self.BLOCK_SIZE_M, self.BLOCK_SIZE_K]
        self.c_shape = [self.BLOCK_SIZE_M, self.BLOCK_SIZE_N]

        self.GROUP_SIZE_M = GROUP_SIZE_M
        self.STAGES = STAGES
        self.NUM_WARPS = NUM_WARPS
        self.LOWER_UPPER = LOWER_UPPER

    def __call__(
        self,
        A: torch.Tensor,
        C: Optional[torch.Tensor] = None,
        use_tvm_ffi: Optional[bool] = None,
    ) -> torch.Tensor:
        M, K = A.shape[-2:]

        if C is None:
            if A.ndim == 2:
                C = torch.empty((M, M), device=A.device, dtype=A.dtype)
            else:
                C = torch.empty(A.shape[:-2] + (M, M), device=A.device, dtype=A.dtype)

        a_layout = gl.NVMMASharedLayout.get_default_for(self.a_shape, gl.bfloat16)
        c_layout = gl.NVMMASharedLayout.get_default_for(self.c_shape, gl.bfloat16)

        A_desc = TensorDescriptor.from_tensor(A, self.a_shape, a_layout)
        C_desc = TensorDescriptor.from_tensor(C, self.c_shape, c_layout)

        batch_size = A.size(0) if A.ndim == 3 else 1
        # input_batch_stride = A.stride(0) if A.ndim == 3 else 0
        # output_batch_stride = C.stride(0) if C.ndim == 3 else 0

        M, K = A.shape[-2:]

        num_pid_m = triton.cdiv(M, self.BLOCK_SIZE_M)
        grid = (batch_size * num_pid_m * num_pid_m,)

        kernel_kwargs = {
            "A_desc": A_desc,
            "C_desc": C_desc,
            "M": M,
            "K": K,
            "BLOCK_SIZE_M": self.BLOCK_SIZE_M,
            "BLOCK_SIZE_N": self.BLOCK_SIZE_N,
            "BLOCK_SIZE_K": self.BLOCK_SIZE_K,
            "GROUP_SIZE_M": self.GROUP_SIZE_M,
            "LOWER_UPPER": self.LOWER_UPPER,
            "STAGES": self.STAGES,
            "num_warps": self.NUM_WARPS,
        }

        use_ffi = USE_TVM_FFI if use_tvm_ffi is None else use_tvm_ffi
        if use_ffi:
            key = (
                A.dtype,
                C.dtype,
                tuple(A.shape),
                tuple(C.shape),
                tuple(A.stride()),
                tuple(C.stride()),
                self.BLOCK_SIZE_M,
                self.BLOCK_SIZE_N,
                self.BLOCK_SIZE_K,
                self.GROUP_SIZE_M,
                self.LOWER_UPPER,
                self.STAGES,
                self.NUM_WARPS,
            )
            return _gluon_xxt_tvm_ffi(A, C, grid, kernel_kwargs, key)

        XXT_kernel[grid](
            A_desc,
            C_desc,
            M=M,
            K=K,
            # a_stride_b=input_batch_stride, a_stride_r=A.stride(-2), a_stride_c=A.stride(-1),
            # c_stride_b=output_batch_stride, c_stride_r=C.stride(-2), c_stride_c=C.stride(-1),
            BLOCK_SIZE_M=self.BLOCK_SIZE_M,
            BLOCK_SIZE_N=self.BLOCK_SIZE_N,
            BLOCK_SIZE_K=self.BLOCK_SIZE_K,
            GROUP_SIZE_M=self.GROUP_SIZE_M,
            LOWER_UPPER=self.LOWER_UPPER,
            STAGES=self.STAGES,
            num_warps=self.NUM_WARPS,  # 8 for multi stages, 12 for 1 x producer wg, 2 x consumers wg
        )

        return C


# Unified interface for XXT and XXL
def symm_gemm(
    A: torch.Tensor, out: Optional[torch.Tensor] = None, use_gluon: bool = True
) -> torch.Tensor:
    M, K = A.shape[-2:]

    if out is None:
        if A.ndim == 2:
            out = torch.empty((M, M), device=A.device, dtype=A.dtype)
        else:
            out = torch.empty(A.shape[:-2] + (M, M), device=A.device, dtype=A.dtype)

    raise NotImplementedError(
        "symm_gemm is not implemented yet, please use XXT instead"
    )

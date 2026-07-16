import ctypes
import hashlib
import itertools
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from packaging import version

USE_TVM_FFI = False

tvm_ffi_modules = {
    "XXT": None,
    "fp8_gemm_block_scaled": None,
    "thunder_moun_gemm": None,
}


def is_hopper():
    return triton.runtime.driver.active.get_current_target().arch >= 90


if is_hopper():
    NUM_XCDS = 1
    OCP_FP8E4M3_MAX = 448.0
    FP8_MAX = OCP_FP8E4M3_MAX

    props = torch.cuda.get_device_properties(0)
    SMs = props.multi_processor_count
    NUM_CUs = SMs
else:
    raise Exception("Not supported yet!")


class LoadBalanceStrategy(Enum):
    UNDEFINE = -1
    GAUSSIAN_FOLDING = 0


# NOTE (yiakwy) : useful for multiple-die chiplet architecture :
#   - Blackwell(2 dies), Rubin(2 dies),
#   - Rubin Ultra(4 dies), MI300(4 dies),
#   - MI300X(8 dies)
#
# Make sure SPLIT-K blocks cooperatively work on the same die to maximize the cache hit re-usage
@triton.jit
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
@triton.jit
def linear_to_tril(pid):
    # row = floor((sqrt(8*pid + 1) - 1) / 2)
    row = tl.floor((tl.math.sqrt(8.0 * pid + 1.0) - 1.0) / 2.0).to(tl.int32)
    col = pid - (row * (row + 1)) // 2
    return row, col


@triton.jit
def _compute_pid(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M):
    if GROUP_SIZE_M == 1:
        pid_n = tile_id % num_pid_n
        pid_m = tile_id // num_pid_n

        num_pid_in_group = num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n

        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _symm_moun_mat_compute(
    pid_m_pair,
    pid_m,
    pid_n,
    acc,
    acc_pair,
    xq_ptr,
    wq_ptr,
    xs_lhs_ptr,
    xs_rhs_ptr,
    o_ptr,
    M,
    K,
    stride_xq_m,
    stride_xq_k,
    stride_wq_n,
    stride_wq_k,
    stride_xs_m,
    stride_xs_k,
    stride_ws_n,
    stride_ws_k,
    stride_o_m,
    stride_o_n,
    offs_k,
    start_ssteps,
    num_ssteps,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    SCALE_BLOCK_SIZE_N,
    SCALE_BLOCK_SIZE_K,
    SPLIT_K,
):
    offs_xq_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_xq_pair_m = (pid_m_pair * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M

    if BLOCK_SIZE_M != BLOCK_SIZE_N or pid_m != pid_n:
        offs_wq_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % M
    else:
        offs_wq_n = offs_xq_m

    xq_block_ptr = xq_ptr + (
        offs_xq_m[:, None] * stride_xq_m + offs_k[None, :] * stride_xq_k
    )
    xq_pair_block_ptr = xq_ptr + (
        offs_xq_pair_m[:, None] * stride_xq_m + offs_k[None, :] * stride_xq_k
    )

    wq_block_ptr = wq_ptr + (
        offs_wq_n[:, None] * stride_wq_n + offs_k[None, :] * stride_wq_k
    )

    xs_block_ptr = (
        xs_lhs_ptr
        + offs_xq_m * stride_xs_m
        + start_ssteps * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
    )
    xs_pair_block_ptr = (
        xs_lhs_ptr
        + offs_xq_pair_m * stride_xs_m
        + start_ssteps * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
    )

    offs_ws_n = offs_wq_n // SCALE_BLOCK_SIZE_N
    ws_block_ptr = (
        xs_rhs_ptr
        + offs_ws_n * stride_ws_n
        + start_ssteps * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
    )

    xq = tl.load(
        xq_block_ptr,
        mask=offs_k[None, :] < K,
        other=0.0,
    )

    xq_pair = xq
    if pid_m_pair >= pid_n:
        xq_pair = tl.load(
            xq_pair_block_ptr,
            mask=offs_k[None, :] < K,
            other=0.0,
        )

    wq = tl.load(
        wq_block_ptr,
        mask=offs_k[None, :] < K,
        other=0.0,
    )

    xs = tl.load(xs_block_ptr)

    xs_pair = xs
    if pid_n <= pid_m_pair:
        xs_pair = tl.load(xs_pair_block_ptr)

    ws = tl.load(ws_block_ptr)

    xq_next = xq
    wq_next = wq

    xs_next = xs
    ws_next = ws

    xq_pair_next = xq_pair
    xs_pair_next = xs_pair

    last_ssteps = start_ssteps * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
    end_ssteps = start_ssteps + num_ssteps
    for k in tl.range(start_ssteps, end_ssteps):
        xq_k_step = BLOCK_SIZE_K * stride_xq_k
        wq_k_step = BLOCK_SIZE_K * stride_wq_k

        cur_ssteps = (k + 1) * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
        if k + 1 < end_ssteps:
            xq_next = tl.load(
                xq_block_ptr + xq_k_step,
                mask=offs_k[None, :] < K - (k + 1) * (BLOCK_SIZE_K),
                other=0.0,
            )

            wq_next = tl.load(
                wq_block_ptr + wq_k_step,
                mask=offs_k[None, :] < K - (k + 1) * (BLOCK_SIZE_K),
                other=0.0,
            )

            if cur_ssteps > last_ssteps:
                xs_next = tl.load(xs_block_ptr + stride_xs_k)
                ws_next = tl.load(ws_block_ptr + stride_ws_k)

        acc += tl.dot(xq, wq.trans(1, 0), input_precision="ieee") * (
            xs[:, None] * ws[None, :]
        )

        if pid_m_pair >= pid_n:
            if k + 1 < end_ssteps:
                xq_pair_next = tl.load(
                    xq_pair_block_ptr + xq_k_step,
                    mask=offs_k[None, :] < K - (k + 1) * (BLOCK_SIZE_K),
                    other=0.0,
                )
                xs_pair_next = tl.load(xs_pair_block_ptr + stride_xs_k)

            acc_pair += tl.dot(xq_pair, wq.trans(1, 0), input_precision="ieee") * (
                xs_pair[:, None] * ws[None, :]
            )

        xq_block_ptr += xq_k_step
        wq_block_ptr += wq_k_step

        if cur_ssteps > last_ssteps:
            xs_block_ptr += stride_xs_k
            ws_block_ptr += stride_ws_k

        if k + 1 < end_ssteps:
            xq = xq_next
            wq = wq_next

            if cur_ssteps > last_ssteps:
                xs = xs_next
                ws = ws_next

        if pid_m_pair >= pid_n:
            xq_pair_block_ptr += xq_k_step

            if cur_ssteps > last_ssteps:
                xs_pair_block_ptr += stride_xs_k

            if k + 1 < end_ssteps:
                xq_pair = xq_pair_next

                if cur_ssteps > last_ssteps:
                    xs_pair = xs_pair_next

        last_ssteps = cur_ssteps

    if acc.dtype != o_ptr.type.element_ty:
        _acc = acc.to(o_ptr.type.element_ty)
    else:
        _acc = acc
    o_block_ptr = (
        o_ptr + offs_xq_m[:, None] * stride_o_m + offs_wq_n[None, :] * stride_o_n
    )
    o_mask = (offs_xq_m[:, None] < M) & (offs_wq_n[None, :] < M)
    if SPLIT_K == 1:
        tl.store(o_block_ptr, _acc, mask=o_mask)
    else:
        tl.atomic_add(o_block_ptr, _acc, mask=o_mask)

    if pid_n < pid_m:
        _acc = tl.trans(_acc, 1, 0)
        o_block_ptr = (
            o_ptr + offs_wq_n[:, None] * stride_o_m + offs_xq_m[None, :] * stride_o_n
        )
        o_mask = (offs_wq_n[:, None] < M) & (offs_xq_m[None, :] < M)
        if SPLIT_K == 1:
            tl.store(o_block_ptr, _acc, mask=o_mask)
        else:
            tl.atomic_add(o_block_ptr, _acc, mask=o_mask)

    if pid_n <= pid_m_pair:
        if acc_pair.dtype != o_ptr.type.element_ty:
            _acc = acc_pair.to(o_ptr.type.element_ty)
        else:
            _acc = acc_pair
        o_pair_block_ptr = (
            o_ptr
            + offs_xq_pair_m[:, None] * stride_o_m
            + offs_wq_n[None, :] * stride_o_n
        )
        o_mask_pair = (offs_xq_pair_m[:, None] < M) & (offs_wq_n[None, :] < M)
        if SPLIT_K == 1:
            tl.store(o_pair_block_ptr, _acc, mask=o_mask_pair)
        else:
            tl.atomic_add(o_pair_block_ptr, _acc, mask=o_mask_pair)

        if pid_n < pid_m_pair:
            _acc = tl.trans(_acc, 1, 0)
            o_pair_block_ptr = (
                o_ptr
                + offs_wq_n[:, None] * stride_o_m
                + offs_xq_pair_m[None, :] * stride_o_n
            )
            o_mask_pair = (offs_wq_n[:, None] < M) & (offs_xq_pair_m[None, :] < M)
            if SPLIT_K == 1:
                tl.store(o_pair_block_ptr, _acc, mask=o_mask_pair)
            else:
                tl.atomic_add(o_pair_block_ptr, _acc, mask=o_mask_pair)


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
            },
            num_stages=2,
            num_warps=8,
            maxnreg=384,
        ),
    ],
    key=("M", "K"),
    use_cuda_graph=True,
)
@triton.heuristics(
    {
        "NUM_PID_M": lambda args: triton.cdiv(args["M"] // 2, args["BLOCK_SIZE_M"]),
        "SPLIT_K": lambda args: args.get("SPLIT_K", 1),
    }
)
@triton.jit
def paired_symm_moun_mat_fp8_block_scaled_tuned_kernel(
    xq_ptr,
    wq_ptr,
    o_ptr,
    M,
    K,
    stride_xq_m,
    stride_xq_k,
    stride_o_m,
    stride_o_n,
    xs_lhs_ptr,
    xs_rhs_ptr,
    stride_xs_lhs_m,
    stride_xs_lhs_k,
    stride_xs_rhs_n,
    stride_xs_rhs_k,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    NUM_CUs: tl.constexpr,
    NUM_PID_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    M_EVEN: tl.constexpr,
):
    pid_k = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    num_pid_m = NUM_PID_M

    if M_EVEN:
        FULL_BLOCKS = num_pid_m * 2 - 1
    else:
        FULL_BLOCKS = num_pid_m * 2

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    num_ssteps = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    start_ssteps = pid_k * num_ssteps
    offs_k += start_ssteps * BLOCK_SIZE_K

    acc_dtype = tl.float32

    pid_m_pair = FULL_BLOCKS - pid_m
    for pid_n in tl.range(0, pid_m_pair + 1):
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        acc_pair = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        _symm_moun_mat_compute(
            pid_m,
            pid_m_pair,
            pid_n,
            acc,
            acc_pair,
            xq_ptr,
            wq_ptr,
            xs_lhs_ptr,
            xs_rhs_ptr,
            o_ptr,
            M,
            K,
            stride_xq_m,
            stride_xq_k,
            stride_xq_m,
            stride_xq_k,
            stride_xs_lhs_m,
            stride_xs_lhs_k,
            stride_xs_rhs_n,
            stride_xs_rhs_k,
            stride_o_m,
            stride_o_n,
            offs_k,
            start_ssteps,
            num_ssteps,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            SCALE_BLOCK_SIZE_N,
            SCALE_BLOCK_SIZE_K,
            SPLIT_K,
        )


def thunder_moun_gemm(
    xq_lhs: torch.Tensor,
    xq_rhs: torch.Tensor,
    xs_0: torch.Tensor,
    xs_1: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    online_quant: bool = False,
    SPLIT_K: int = 1,
    use_tvm_ffi: Optional[bool] = None,
):
    use_ffi = USE_TVM_FFI if use_tvm_ffi is None else use_tvm_ffi
    if use_ffi:
        raise NotImplementedError(
            "Triton 3.5 thunder_moun_gemm TVM-FFI launcher is not wired yet. "
            "Use native first, then add the launcher after native is validated."
        )

    M, K = xq_lhs.shape
    if out is None:
        out = torch.zeros((M, M), device=xq_lhs.device, dtype=torch.float16)

    grid = lambda META: (SPLIT_K, triton.cdiv(M // 2, META["BLOCK_SIZE_M"]), 1)
    kernel = paired_symm_moun_mat_fp8_block_scaled_tuned_kernel

    kernel[grid](
        xq_lhs,
        xq_rhs,
        out,
        M,
        K,
        xq_lhs.stride(0),
        xq_lhs.stride(1),
        out.stride(0),
        out.stride(1),
        xs_0,
        xs_1,
        xs_0.stride(0),
        xs_0.stride(1),
        xs_1.stride(0),
        xs_1.stride(1),
        SCALE_BLOCK_SIZE_K=128,
        SCALE_BLOCK_SIZE_N=128,
        NUM_XCDS=NUM_XCDS,
        NUM_CUs=NUM_CUs,
        SPLIT_K=SPLIT_K,
        M_EVEN=M % 2 == 0,
    )

    return out


# Ref kernels are adpated from modded-gpt
if version.parse(triton.__version__) < version.parse("3.6"):

    # adatped from triton 3.6+
    @triton.jit
    def xcd_swizzle(pid, domain_size, XCD_SWIZZLE: tl.constexpr):
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


    @triton.jit
    def swizzle2d(i, j, grid_m, grid_n, GROUP_M: tl.constexpr):
        pid = i * grid_n + j
        width = GROUP_M * grid_n

        group_id = pid // width
        first_pid_m = group_id * GROUP_M
        group_size = min(grid_m - first_pid_m, GROUP_M)
        tl.assume(group_size >= 0)
        pid_m = first_pid_m + (pid % group_size)
        pid_n = (pid % width) // group_size
        return pid_m, pid_n
    
    # NOTE (yiakwy) : FIX newer triton API
    setattr(tl, 'swizzle2d', swizzle2d)
    setattr(tl, 'xcd_swizzle', xcd_swizzle)
    

# Adpated from gemm swizzle by @byronxu99 for reference
@triton.jit
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
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx


@triton.jit
def XXT_kernel(
    A_ptr, C_ptr,
    M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Load A blocks for C[m,n] = A[m,:] @ A[n,:].T
    # Load A[m, k] -> shape (BM, BK)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    # Load A[n, k] -> shape (BN, BK). Transpose to get (BK, BN) for accumulation.
    # Loading (BN, BK) is coalesced because stride_c is 1 (contiguous dim is k).
    at_ptrs = A_ptr + (offs_n[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        at_temp = tl.load(at_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        at = tl.trans(at_temp)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def get_pack_size(
    pack_int8=False,
    pack_int16=False,
    pack_int32=False,
):
    if pack_int8:
        return 8
    if pack_int16:
        return 16
    if pack_int32:
        return 32
    return 1


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 16,
                "NUM_STAGES": 3,
            },
            num_stages=3,
            num_warps=16,
            maxnreg=384,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 16,
                "NUM_STAGES": 3,
            },
            num_stages=3,
            num_warps=8,
            maxnreg=384,
        ),
    ],
    key=("M", "N", "K"),
    use_cuda_graph=False,
)
@triton.heuristics(
    {
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
        "BLOCKS_PER_XCD": lambda args: triton.cdiv(
            triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
            * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
            NUM_XCDS,
        ),
        "NUM_PID_M": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"]),
        "NUM_PID_N": lambda args: triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
        "SPLIT_K": lambda _: 1,
        "w1a16": lambda args: args.get("pack_int8", False)
        or args.get("pack_int16", False)
        or args.get("pack_int32", False),
        "packed_size": lambda args: get_pack_size(
            pack_int8=args.get("pack_int8", False),
            pack_int16=args.get("pack_int16", False),
            pack_int32=args.get("pack_int32", False),
        ),
        "_is_hopper": lambda args: args.get("_is_hopper", is_hopper()),
    }
)
@triton.jit
def fp8_streamk_gemm_block_scaled_tuned_kernel(
    xq_ptr,
    wq_ptr,
    o_ptr,
    M,
    N,
    K,
    stride_xq_m,
    stride_xq_k,
    stride_wq_n,
    stride_wq_k,
    stride_o_m,
    stride_o_n,
    xs_ptr,
    ws_ptr,
    stride_xs_m,
    stride_xs_k,
    stride_ws_n,
    stride_ws_k,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_N: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_CUs: tl.constexpr,
    GRID_MN: tl.constexpr,
    BLOCKS_PER_XCD: tl.constexpr,
    NUM_PID_M: tl.constexpr,
    NUM_PID_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    w1a16: tl.constexpr,
    packed_size: tl.constexpr,
    use_w1a16_as_fp8: tl.constexpr,
    use_w1a8: tl.constexpr,
    _is_hopper: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = NUM_PID_M
    num_pid_n = NUM_PID_N

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if w1a16:
        offs_k_packed = tl.arange(0, BLOCK_SIZE_K // packed_size)

    num_ssteps = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    for tile_id_0 in tl.range(pid, GRID_MN * SPLIT_K, NUM_CUs):
        tile_id = tile_id_0 % GRID_MN
        pid_k = tile_id_0 // GRID_MN

        start_ssteps = pid_k * num_ssteps
        offs_k += start_ssteps * BLOCK_SIZE_K
        if w1a16:
            offs_k_packed += start_ssteps * BLOCK_SIZE_K // packed_size

        pid_m, pid_n = _compute_pid(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M)

        offs_xq_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_wq_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        xq_block_ptr = xq_ptr + (
            offs_xq_m[:, None] * stride_xq_m + offs_k[None, :] * stride_xq_k
        )
        if w1a16:
            unpack_offs = tl.arange(0, packed_size).to(wq_ptr.type.element_ty)

            wq_block_ptr = wq_ptr + (
                offs_wq_n[:, None] * stride_wq_n + offs_k_packed[None, :] * stride_wq_k
            )

        else:
            wq_block_ptr = wq_ptr + (
                offs_wq_n[:, None] * stride_wq_n + offs_k[None, :] * stride_wq_k
            )

            xs_block_ptr = (
                xs_ptr
                + offs_xq_m * stride_xs_m
                + start_ssteps * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
            )

            offs_ws_n = offs_wq_n // SCALE_BLOCK_SIZE_N
            ws_block_ptr = (
                ws_ptr
                + offs_ws_n * stride_ws_n
                + start_ssteps * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
            )

        acc_dtype = tl.float16
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        if not w1a16:
            xs_next = tl.load(xs_block_ptr)
            ws_next = tl.load(ws_block_ptr)

        last_ssteps = start_ssteps * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
        for k in tl.range(
            start_ssteps, start_ssteps + num_ssteps, num_stages=NUM_STAGES
        ):
            xq = tl.load(
                xq_block_ptr,
                mask=offs_k[None, :] < K - k * (BLOCK_SIZE_K),
                other=0.0,
            )

            if w1a16:
                wq = tl.load(
                    wq_block_ptr,
                    mask=offs_k_packed[None, :]
                    < K // packed_size - k * (BLOCK_SIZE_K // packed_size),
                    other=0.0,
                )

                wq = wq[:, :, None]

                wq = tl.inline_asm_elementwise(
                    asm="bfe.u32 $0, $1, $2, 1;",
                    constraints="=r,r,r",
                    args=[wq, unpack_offs[None, None, :]],
                    dtype=wq_ptr.type.element_ty,
                    is_pure=True,
                    pack=1,
                )

                wq = wq.reshape((BLOCK_SIZE_N, BLOCK_SIZE_K))
                wq = wq * 2 - 1
                wq = wq.to(tl.float16)

                if use_w1a16_as_fp8:
                    wq = wq.to(tl.float8e4nv)
                    xq = xq.to(tl.float8e4nv)
                elif use_w1a8:
                    wq = wq.to(tl.float8e4nv)

            else:
                wq = tl.load(
                    wq_block_ptr,
                    mask=offs_k[None, :] < K - k * (BLOCK_SIZE_K),
                    other=0.0,
                )

                if wq.dtype == tl.int8 or wq.dtype == tl.int16:
                    wq = wq.to(tl.float16)

                    if use_w1a16_as_fp8:
                        wq = wq.to(tl.float8e4nv)
                        xq = xq.to(tl.float8e4nv)
                    elif use_w1a8:
                        wq = wq.to(tl.float8e4nv)

            ws = ws_next
            xs = xs_next

            cur_ssteps = (k + 1) * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
            if cur_ssteps > last_ssteps and (k + 1) < num_ssteps:
                xs_next = tl.load(xs_block_ptr + stride_xs_k)
                ws_next = tl.load(ws_block_ptr + stride_ws_k)

            if w1a16:
                acc = tl.dot(xq, wq.trans(1, 0), acc=acc, out_dtype=acc_dtype)
            else:
                if False:
                    acc = tl.dot(
                        xq * xs[:, None],
                        (wq * ws[:, None]).trans(1, 0),
                        acc=acc,
                        input_precision="ieee",
                        out_dtype=acc_dtype,
                    )
                else:
                    acc += tl.dot(
                        xq, wq.trans(1, 0), input_precision="ieee", out_dtype=acc_dtype
                    ) * ((xs[:, None] * ws[None, :]).to(acc_dtype))

            xq_block_ptr += BLOCK_SIZE_K * stride_xq_k
            if w1a16:
                wq_block_ptr += BLOCK_SIZE_K // packed_size * stride_wq_k
            else:
                wq_block_ptr += BLOCK_SIZE_K * stride_wq_k

            if cur_ssteps > last_ssteps and (k + 1) < num_ssteps:
                xs_block_ptr += stride_xs_k
                ws_block_ptr += stride_ws_k

            last_ssteps = cur_ssteps

        o_block_ptr = (
            o_ptr + offs_xq_m[:, None] * stride_o_m + offs_wq_n[None, :] * stride_o_n
        )
        o_mask = (offs_xq_m[:, None] < M) & (offs_wq_n[None, :] < N)

        if o_ptr.type.element_ty != acc.dtype:
            _acc = acc.to(o_ptr.type.element_ty)
        else:
            _acc = acc
        if SPLIT_K == 1:
            tl.store(o_block_ptr, _acc, mask=o_mask)
        else:
            tl.atomic_add(o_block_ptr, _acc, mask=o_mask)


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 16,
                "NUM_STAGES": 3,
            },
            num_stages=3,
            num_warps=8,
            maxnreg=384,
        ),
    ],
    key=("M", "N", "K"),
    use_cuda_graph=False,
)
@triton.heuristics(
    {
        "GRID_MN": lambda args: (
            triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
            * triton.cdiv(args["N"], args["BLOCK_SIZE_N"])
            + triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        )
        // 2,
        "BLOCKS_PER_XCD": lambda args: triton.cdiv(
            triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
            * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
            NUM_XCDS,
        ),
        "NUM_PID_M": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"]),
        "NUM_PID_N": lambda args: triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
        "SPLIT_K": lambda _: 1,
        "w1a16": lambda args: args.get("pack_int8", False)
        or args.get("pack_int16", False)
        or args.get("pack_int32", False),
        "packed_size": lambda args: get_pack_size(
            pack_int8=args.get("pack_int8", False),
            pack_int16=args.get("pack_int16", False),
            pack_int32=args.get("pack_int32", False),
        ),
        "_is_hopper": lambda args: args.get("_is_hopper", is_hopper()),
        "_is_symm": lambda args: args.get("_is_symm", True),
    }
)
@triton.jit
def fp8_streamk_symm_gemm_block_scaled_tuned_kernel(
    xq_ptr,
    wq_ptr,
    o_ptr,
    M,
    N,
    K,
    stride_xq_m,
    stride_xq_k,
    stride_wq_n,
    stride_wq_k,
    stride_o_m,
    stride_o_n,
    xs_ptr,
    ws_ptr,
    stride_xs_m,
    stride_xs_k,
    stride_ws_n,
    stride_ws_k,
    SCALE_BLOCK_SIZE_K: tl.constexpr,
    SCALE_BLOCK_SIZE_N: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_CUs: tl.constexpr,
    GRID_MN: tl.constexpr,
    BLOCKS_PER_XCD: tl.constexpr,
    NUM_PID_M: tl.constexpr,
    NUM_PID_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    w1a16: tl.constexpr,
    packed_size: tl.constexpr,
    use_w1a16_as_fp8: tl.constexpr,
    use_w1a8: tl.constexpr,
    _is_hopper: tl.constexpr,
    _is_symm: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_m = NUM_PID_M
    num_pid_n = NUM_PID_N

    tl.static_assert(
        _is_symm,
        "This streamk_symm_gemm kernel is designed for symmetric matrix multiplication, please set _is_symm to True.",
    )
    tl.static_assert(
        NUM_PID_M == NUM_PID_N,
        "The streamk_symm_gemm kernel currently only supports BLOCK_SIZE_M == BLOCK_SIZE_N.",
    )

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if w1a16:
        offs_k_packed = tl.arange(0, BLOCK_SIZE_K // packed_size)

    num_ssteps = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    for tile_id_0 in tl.range(pid, GRID_MN * SPLIT_K, NUM_CUs):
        tile_id = tile_id_0 % GRID_MN
        pid_k = tile_id_0 // GRID_MN

        start_ssteps = pid_k * num_ssteps
        offs_k += start_ssteps * BLOCK_SIZE_K
        if w1a16:
            offs_k_packed += start_ssteps * BLOCK_SIZE_K // packed_size

        if _is_symm:
            pid_m, pid_n = linear_to_tril(tile_id)
        else:
            pid_m, pid_n = _compute_pid(tile_id, num_pid_m, num_pid_n, GROUP_SIZE_M)

        offs_xq_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_wq_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        xq_block_ptr = xq_ptr + (
            offs_xq_m[:, None] * stride_xq_m + offs_k[None, :] * stride_xq_k
        )
        if w1a16:
            unpack_offs = tl.arange(0, packed_size).to(wq_ptr.type.element_ty)

            wq_block_ptr = wq_ptr + (
                offs_wq_n[:, None] * stride_wq_n + offs_k_packed[None, :] * stride_wq_k
            )

        else:
            wq_block_ptr = wq_ptr + (
                offs_wq_n[:, None] * stride_wq_n + offs_k[None, :] * stride_wq_k
            )

            xs_block_ptr = (
                xs_ptr
                + offs_xq_m * stride_xs_m
                + start_ssteps * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
            )

            offs_ws_n = offs_wq_n // SCALE_BLOCK_SIZE_N
            ws_block_ptr = (
                ws_ptr
                + offs_ws_n * stride_ws_n
                + start_ssteps * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
            )

        acc_dtype = tl.float16
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        if not w1a16:
            xs_next = tl.load(xs_block_ptr)
            ws_next = tl.load(ws_block_ptr)

        last_ssteps = start_ssteps * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
        for k in tl.range(
            start_ssteps, start_ssteps + num_ssteps, num_stages=NUM_STAGES
        ):
            xq = tl.load(
                xq_block_ptr,
                mask=offs_k[None, :] < K - k * (BLOCK_SIZE_K),
                other=0.0,
            )

            if w1a16:
                wq = tl.load(
                    wq_block_ptr,
                    mask=offs_k_packed[None, :]
                    < K // packed_size - k * (BLOCK_SIZE_K // packed_size),
                    other=0.0,
                )

                wq = wq[:, :, None]

                wq = tl.inline_asm_elementwise(
                    asm="bfe.u32 $0, $1, $2, 1;",
                    constraints="=r,r,r",
                    args=[wq, unpack_offs[None, None, :]],
                    dtype=wq_ptr.type.element_ty,
                    is_pure=True,
                    pack=1,
                )

                wq = wq.reshape((BLOCK_SIZE_N, BLOCK_SIZE_K))
                wq = wq * 2 - 1
                wq = wq.to(tl.float16)

                if use_w1a16_as_fp8:
                    wq = wq.to(tl.float8e4nv)
                    xq = xq.to(tl.float8e4nv)
                elif use_w1a8:
                    wq = wq.to(tl.float8e4nv)

            else:
                wq = tl.load(
                    wq_block_ptr,
                    mask=offs_k[None, :] < K - k * (BLOCK_SIZE_K),
                    other=0.0,
                )

                if wq.dtype == tl.int8 or wq.dtype == tl.int16:
                    wq = wq.to(tl.float16)

                    if use_w1a16_as_fp8:
                        wq = wq.to(tl.float8e4nv)
                        xq = xq.to(tl.float8e4nv)
                    elif use_w1a8:
                        wq = wq.to(tl.float8e4nv)

            ws = ws_next
            xs = xs_next

            cur_ssteps = (k + 1) * BLOCK_SIZE_K // SCALE_BLOCK_SIZE_K
            if cur_ssteps > last_ssteps and (k + 1) < num_ssteps:
                xs_next = tl.load(xs_block_ptr + stride_xs_k)
                ws_next = tl.load(ws_block_ptr + stride_ws_k)

            if w1a16:
                acc = tl.dot(xq, wq.trans(1, 0), acc=acc, out_dtype=acc_dtype)
            else:
                if False:
                    acc = tl.dot(
                        xq * xs[:, None],
                        (wq * ws[:, None]).trans(1, 0),
                        acc=acc,
                        input_precision="ieee",
                        out_dtype=acc_dtype,
                    )
                else:
                    acc += tl.dot(
                        xq, wq.trans(1, 0), input_precision="ieee", out_dtype=acc_dtype
                    ) * ((xs[:, None] * ws[None, :]).to(acc_dtype))

            xq_block_ptr += BLOCK_SIZE_K * stride_xq_k
            if w1a16:
                wq_block_ptr += BLOCK_SIZE_K // packed_size * stride_wq_k
            else:
                wq_block_ptr += BLOCK_SIZE_K * stride_wq_k

            if cur_ssteps > last_ssteps and (k + 1) < num_ssteps:
                xs_block_ptr += stride_xs_k
                ws_block_ptr += stride_ws_k

            last_ssteps = cur_ssteps

        o_block_ptr = (
            o_ptr + offs_xq_m[:, None] * stride_o_m + offs_wq_n[None, :] * stride_o_n
        )
        o_mask = (offs_xq_m[:, None] < M) & (offs_wq_n[None, :] < N)

        if o_ptr.type.element_ty != acc.dtype:
            _acc = acc.to(o_ptr.type.element_ty)
        else:
            _acc = acc
        if SPLIT_K == 1:
            tl.store(o_block_ptr, _acc, mask=o_mask)
        else:
            tl.atomic_add(o_block_ptr, _acc, mask=o_mask)

        if _is_symm:
            if pid_n < pid_m:
                _acc = tl.trans(_acc, 1, 0)
                o_block_ptr = (
                    o_ptr
                    + offs_wq_n[:, None] * stride_o_m
                    + offs_xq_m[None, :] * stride_o_n
                )
                o_mask = (offs_wq_n[:, None] < M) & (offs_xq_m[None, :] < M)
                if SPLIT_K == 1:
                    tl.store(o_block_ptr, _acc, mask=o_mask)
                else:
                    tl.atomic_add(o_block_ptr, _acc, mask=o_mask)


def fp8_gemm_block_scaled(
    xq: torch.Tensor,
    wq: torch.Tensor,
    xs: torch.Tensor,
    ws: torch.Tensor,
    o: Optional[torch.Tensor] = None,
    SCALE_BLOCK_SIZE_N=128,
    SCALE_BLOCK_SIZE_K=128,
    check_input_shape=False,
    is_symm=False,
    use_tvm_ffi: Optional[bool] = None,
):
    if check_input_shape:
        assert xq.dim() >= 2, "xq must be >= 2 dims"
        assert wq.dim() >= 2, "wq must be >= 2 dims"

        if xq.dim() > 2:
            xq_flatten = xq.flatten(start_dim=1)
            wq_flatten = wq.flatten(start_dim=1)

            xq = xq_flatten
            wq = wq_flatten

        assert xq.shape[1] == wq.shape[1], "the shapes of xq and wq are incompatible!"

    use_ffi = USE_TVM_FFI if use_tvm_ffi is None else use_tvm_ffi
    if use_ffi:
        raise NotImplementedError(
            "Triton 3.5 fp8_gemm_block_scaled TVM-FFI launcher is not wired yet. "
            "Use native first, then add the launcher after native is validated."
        )

    M, K = xq.shape
    N = wq.shape[0]

    if o is None:
        o = torch.zeros((M, N), device=xq.device, dtype=torch.float16)

    if is_symm:
        grid = lambda META: (
            min(
                NUM_CUs,
                (
                    (
                        triton.cdiv(M, META["BLOCK_SIZE_M"])
                        * triton.cdiv(N, META["BLOCK_SIZE_N"])
                        + triton.cdiv(M, META["BLOCK_SIZE_M"])
                    )
                    // 2
                )
                * META["SPLIT_K"],
            ),
        )

        kernel = fp8_streamk_symm_gemm_block_scaled_tuned_kernel
    else:
        grid = lambda META: (
            min(
                NUM_CUs,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"])
                * META["SPLIT_K"],
            ),
        )

        kernel = fp8_streamk_gemm_block_scaled_tuned_kernel

    kernel[grid](
        xq,
        wq,
        o,
        M,
        N,
        K,
        xq.stride(0),
        xq.stride(1),
        wq.stride(0),
        wq.stride(1),
        o.stride(0),
        o.stride(1),
        xs,
        ws,
        xs.stride(0),
        xs.stride(1),
        ws.stride(0),
        ws.stride(1),
        SCALE_BLOCK_SIZE_N=SCALE_BLOCK_SIZE_N,
        SCALE_BLOCK_SIZE_K=SCALE_BLOCK_SIZE_K,
        NUM_XCDS=NUM_XCDS,
        NUM_CUs=NUM_CUs,
        use_w1a16_as_fp8=False,
        use_w1a8=False,
    )

    return o


def _xxt_config(K: int):
    if K == 768:
        return 128, 128, 64, 4, 8
    return 64, 128, 128, 4, 8


def _xxt_tvm_ffi(
    A: torch.Tensor,
    out: torch.Tensor,
    M: int,
    K: int,
    batch_size: int,
    input_batch_stride: int,
    output_batch_stride: int,
    BLOCK_SIZE_M: int,
    BLOCK_SIZE_N: int,
    BLOCK_SIZE_K: int,
    num_stages: int,
    num_warps: int,
):
    import tvm_ffi

    from jit_kernel.triton3_5.tvm_ffi_mod import generate_tvm_ffi_source

    global tvm_ffi_modules

    grid = (batch_size * triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(M, BLOCK_SIZE_N),)
    key = (
        str(A.dtype),
        str(out.dtype),
        M,
        K,
        batch_size,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        num_stages,
        num_warps,
    )

    cache = tvm_ffi_modules["XXT"]
    if cache is not None and key in cache:
        module, kernel_name, gx, gy, gz = cache[key]
    else:
        compiled_kernel = XXT_kernel[grid](
            A_ptr=A,
            C_ptr=out,
            M=M,
            K=K,
            a_stride_b=input_batch_stride,
            a_stride_r=A.stride(-2),
            a_stride_c=A.stride(-1),
            c_stride_b=output_batch_stride,
            c_stride_r=out.stride(-2),
            c_stride_c=out.stride(-1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=8,
            LOWER_UPPER=1,
            num_stages=num_stages,
            num_warps=num_warps,
        )

        cubin_bytes = compiled_kernel.kernel
        kernel_name = compiled_kernel.name
        sources, _constants = generate_tvm_ffi_source(compiled_kernel, kernel_name)
        module = tvm_ffi.cpp.load_inline(
            "triton3_5_xxt_loader",
            cuda_sources=sources,
            extra_ldflags=["-lcudart", "-lcuda"],
            embed_cubin={"triton_cubin": cubin_bytes},
        )

        gx, gy, gz = int(grid[0]), 1, 1
        if tvm_ffi_modules["XXT"] is None:
            tvm_ffi_modules["XXT"] = {}
        tvm_ffi_modules["XXT"][key] = (module, kernel_name, gx, gy, gz)

    getattr(module, kernel_name)(
        A,
        out,
        int(M),
        int(K),
        int(input_batch_stride),
        int(A.stride(-2)),
        int(A.stride(-1)),
        int(output_batch_stride),
        int(out.stride(-2)),
        int(out.stride(-1)),
        int(gx),
        int(gy),
        int(gz),
    )
    return out


def XXT(A: torch.Tensor, out: torch.Tensor, use_tvm_ffi: Optional[bool] = None):
    """
    Launch Triton kernel to compute C = A @ A.T
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == M, "Output matrix has incorrect shape"
    assert out.size(-1) == M, "Output matrix has incorrect shape"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps = _xxt_config(K)

    use_ffi = USE_TVM_FFI if use_tvm_ffi is None else use_tvm_ffi
    if use_ffi:
        return _xxt_tvm_ffi(
            A,
            out,
            M,
            K,
            batch_size,
            input_batch_stride,
            output_batch_stride,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            num_stages,
            num_warps,
        )

    grid = (batch_size * triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(M, BLOCK_SIZE_N),)
    XXT_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        LOWER_UPPER=1,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return out

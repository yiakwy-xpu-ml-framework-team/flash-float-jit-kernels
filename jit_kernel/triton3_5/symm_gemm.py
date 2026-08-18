import ctypes
import hashlib
import itertools
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl
from packaging import version
from triton.tools.tensor_descriptor import TensorDescriptor


USE_TVM_FFI = False

tvm_ffi_modules = {
    "XXT": None,
}


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
    def swizzle2d(pid_m, pid_n, grid_m, grid_n, GROUP_M: tl.constexpr):
        pid = pid_m * grid_n + pid_n
        width = GROUP_M * grid_n

        group_id = pid // width
        first_pid_m = group_id * GROUP_M
        group_size = min(grid_m - first_pid_m, GROUP_M)
        tl.assume(group_size >= 0)
        pid_m = first_pid_m + (pid % group_size)
        pid_n = (pid % width) // (group_size)
        return pid_m, pid_n

    # NOTE (yiakwy) : FIX newer triton API
    setattr(tl, "swizzle2d", swizzle2d)
    setattr(tl, "xcd_swizzle", xcd_swizzle)


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


# TODO (yiakwy) : remove
# adapted from modded_nanogpt as ref
@triton.jit
def XXT_kernel(
    A_ptr,
    C_ptr,
    M,
    K,
    a_stride_b,
    a_stride_r,
    a_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
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
        input_batch_stride,
        A.stride(-2),
        A.stride(-1),
        output_batch_stride,
        out.stride(-2),
        out.stride(-1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        num_stages,
        num_warps,
    )

    if tvm_ffi_modules["XXT"] is not None and key in tvm_ffi_modules["XXT"]:
        module, kernel_name, gx, gy, gz = tvm_ffi_modules["XXT"][key]
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

        gx = int(grid[0])
        gy = 1
        gz = 1

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


# TODO (yiakwy) : remove
# adapted from modded_nanogpt as ref
def XXT(
    A: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    use_tvm_ffi: Optional[bool] = None,
):
    """
    Launch Triton kernel to compute C = A @ A.T
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]

    if out is None:
        if A.ndim == 3:
            out = torch.zeros(
                A.shape[:-2] + (M, M), device=A.device, dtype=torch.float16
            )
        else:
            out = torch.zeros((M, M), device=A.device, dtype=torch.float16)

    assert out.size(-2) == M, "Output matrix has incorrect shape"
    assert out.size(-1) == M, "Output matrix has incorrect shape"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    # Hardcoded configs based on H100 autotuning
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


# TODO (yiakwy) : remove
# adapted from modded_nanogpt as ref
@triton.jit
def XTX_kernel(
    A_ptr,
    C_ptr,
    M,
    K,
    a_stride_b,
    a_stride_r,
    a_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    """
    Compute C = A.T @ A where A is (M, K) and C is (K, K).
    This is the transpose variant of XXT for tall matrices.

    The output matrix C is symmetric, so we compute upper triangle and mirror.
    We iterate over blocks of M (the reduction dimension after transpose).
    """
    pid = tl.program_id(axis=0)
    # Note: Output is (K, K), so we use K for the output grid
    batch_idx, k_idx, n_idx = _pid_to_block(
        pid, K, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed (symmetry optimization)
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= k_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (k_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # For A.T @ A:
    # - A.T has shape (K, M), so A.T[k, m] = A[m, k]
    # - We load blocks from columns k_idx and n_idx of A (which are rows of A.T)
    # - We reduce over M (the shared dimension)
    offs_k = (
        k_idx + tl.arange(0, BLOCK_SIZE_M)
    ) % K  # Output row indices (columns of A)
    offs_n = (
        n_idx + tl.arange(0, BLOCK_SIZE_N)
    ) % K  # Output col indices (columns of A)
    offs_m = tl.arange(0, BLOCK_SIZE_K)  # Reduction dimension (rows of A)

    # Pointers for loading A[:, k_idx:k_idx+BLOCK] (transposed view is A.T[k_idx:, :])
    # at_ptrs loads A.T block: A.T[offs_k, offs_m] = A[offs_m, offs_k]
    at_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    # a_ptrs loads A block for the other factor: A.T[offs_m, offs_n].T = A[offs_m, offs_n]
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_n[None, :] * a_stride_c)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of M (the reduction dimension)
    for m in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        m_remaining = M - m * BLOCK_SIZE_K
        # Load A.T[offs_k, offs_m] = A[offs_m, offs_k] -> shape (BLOCK_K, BLOCK_M)
        at = tl.load(at_ptrs, mask=offs_m[:, None] < m_remaining, other=0.0)
        # Load A[offs_m, offs_n] -> shape (BLOCK_K, BLOCK_N)
        a = tl.load(a_ptrs, mask=offs_m[:, None] < m_remaining, other=0.0)
        # C[k, n] = sum_m A.T[k, m] * A[m, n] = sum_m A[m, k] * A[m, n]
        # at.T @ a: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) = (BLOCK_M, BLOCK_N)
        accumulator = tl.dot(at.T, a, accumulator)
        at_ptrs += BLOCK_SIZE_K * a_stride_r
        a_ptrs += BLOCK_SIZE_K * a_stride_r

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_ck = k_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_ck[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_ck[:, None] < K) & (offs_cn[None, :] < K)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal (symmetry)
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_ck[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < K) & (offs_ck[None, :] < K)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


# TODO (yiakwy) : remove
# adapted from modded_nanogpt as ref
def XTX(A: torch.Tensor, out: Optional[torch.Tensor] = None):
    """
    Launch Triton kernel to compute C = A.T @ A

    For tall matrices (M > K), this is more efficient than transposing
    and using XXT because the intermediate products are smaller (K x K vs M x M).

    Args:
        A: Input tensor of shape (M, K) or (batch, M, K)
        out: Output tensor of shape (K, K) or (batch, K, K)

    Returns:
        out: The same output tensor, filled with A.T @ A
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]

    if out is None:
        if A.ndim == 3:
            out = torch.zeros(
                A.shape[:-2] + (K, K), device=A.device, dtype=torch.float16
            )
        else:
            out = torch.zeros((K, K), device=A.device, dtype=torch.float16)

    assert (
        out.size(-2) == K
    ), f"Output matrix has incorrect shape: expected ({K}, {K}), got {tuple(out.shape[-2:])}"
    assert (
        out.size(-1) == K
    ), f"Output matrix has incorrect shape: expected ({K}, {K}), got {tuple(out.shape[-2:])}"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    # Hardcoded configs based on H100 autotuning
    if K == 768:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 64
        num_stages, num_warps = 4, 8
    else:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 128, 128
        num_stages, num_warps = 4, 8

    grid = (batch_size * triton.cdiv(K, BLOCK_SIZE_M) * triton.cdiv(K, BLOCK_SIZE_N),)
    XTX_kernel[grid](
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

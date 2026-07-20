from __future__ import annotations

import functools
import os
from typing import Optional, Tuple, Union

os.environ["TVM_FFI_CUDA_ARCH_LIST"] = (
    "9.0a"  # Set the CUDA architecture to 9.0 for Hopper support
)

import warnings

import torch

USE_TORCH_JIT = False

# TODO (yaikwy) : use tvm ffi load_jit
if USE_TORCH_JIT:
    from torch.utils.cpp_extension import load as load_jit
else:
    # from jit_kernel.utils import load_jit
    from tvm_ffi.cpp import load, load_inline as load_jit

from jit_kernel.utils import KERNEL_PATH

extra_ldflags = ["-lcudart", "-lcuda"]

cuda_sources = [
    str(KERNEL_PATH / "csrc" / "thunder_moun/symm_gemm.cu"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/profiler.cuh"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/tma/tma_desc.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/tma/tma_desc_impl.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/tma/tma_copy.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/tma/tma_copy_impl.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/tma/tma_barrier.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/tma/tma_barrier.cu"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/cluster/cluster.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/cluster/cluster.cu"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/block/block.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/block/producer.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/block/nv_block_gemm_scaled_impl.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/fragment/fragment.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/fragment/nv_frag_gemm_scaled_impl.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/tensor/array_ref.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/tensor/layout.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/tensor/tuple.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/tensor/tensor_view_ref.h"),
]

_CPP_ENTRY = "symmetric_gemm_fp8_block_scaled"
_PY_SYMBOL = "symmetric_gemm_fp8_block_scaled"

common_cuda_flags = ["-O2", "-std=c++17"]


if torch.cuda.is_available():
    major, _minor = torch.cuda.get_device_capability()
else:
    major = 0

if major >= 9:
    #         "--ptxas-options=-v",
    common_cuda_flags += [
        "-O2",
        "-Xcompiler",
        "-fPIC",
        "-DENABLE_HOPPER=1",
        "-arch=compute_90a",
        "-code=sm_90a",
        "-I " + str(KERNEL_PATH / "csrc" / "thunder_moun"),
    ]


@functools.cache
def _jit_thunder_moun_module(enable_cuda_profiler: bool = False):
    if USE_TORCH_JIT:
        raise Exception(
            "Torch JIT is not supported for thunder moun kernel, please set USE_TORCH_JIT to False"
        )
    else:

        cuda_tmp_sources = [f'#include "{path}"' for path in cuda_sources]
        extra_cuda_cflags = list(common_cuda_flags)
        module_name = "thunder_moun"
        if enable_cuda_profiler:
            extra_cuda_cflags.append("-DFFJK_ENABLE_CUDA_PROFILER=1")
            module_name = "thunder_moun_profiler"

        return load_jit(
            module_name,
            cuda_sources=cuda_tmp_sources,
            # TODO (yiakwy) : add sglang style TVM_FFI warpper
            # cuda_wrappers=[(_PY_SYMBOL, _CPP_ENTRY)],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=extra_ldflags,
        )


def is_column_major(tensor: torch.Tensor):
    return tensor.stride()[0] == 1 and tensor.stride()[1] > 1


BLK_M = 128
BLK_N = 128
BLK_K = 128
DEFAULT_SM_COUNT = 132
MAX_SPLIT_K = 8
CLUSTER_SIZE_M = 2

CUDA_PROFILER_HEADER_U64_WORDS = 4
CUDA_PROFILER_RECORD_U64_WORDS = 2
CUDA_PROFILER_CTA_SLOTS = 2
CUDA_PROFILER_TASK_SLOTS = 20
CUDA_PROFILER_PER_K_SLOTS = 10


def allocate_cuda_profiler_buffer(
    max_records: int,
    device: Union[torch.device, str],
) -> torch.Tensor:
    assert max_records > 0, "max_records must be positive"
    num_words = (
        CUDA_PROFILER_HEADER_U64_WORDS
        + max_records * CUDA_PROFILER_RECORD_U64_WORDS
    )
    return torch.empty(num_words, device=device, dtype=torch.uint64)


def cuda_profiler_capacity(profiler_buffer: torch.Tensor) -> int:
    assert profiler_buffer.is_cuda, "profiler_buffer must be a CUDA tensor"
    assert profiler_buffer.dtype == torch.uint64, "profiler_buffer must be torch.uint64"
    assert profiler_buffer.is_contiguous(), "profiler_buffer must be contiguous"
    payload_words = profiler_buffer.numel() - CUDA_PROFILER_HEADER_U64_WORDS
    assert payload_words >= 0, "profiler_buffer is smaller than the profiler header"
    return payload_words // CUDA_PROFILER_RECORD_U64_WORDS


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _device_sm_count(device: Optional[Union[torch.device, str, int]] = None) -> int:
    if not torch.cuda.is_available():
        return DEFAULT_SM_COUNT

    if device is None:
        device_index = torch.cuda.current_device()
    elif isinstance(device, int):
        device_index = device
    else:
        torch_device = torch.device(device)
        device_index = (
            torch_device.index
            if torch_device.index is not None
            else torch.cuda.current_device()
        )

    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _effective_split_k(
    total_symmetric_tiles: int,
    requested_split_k: int,
    sm_count: int,
) -> Tuple[int, int]:
    grid_mn = min(sm_count, total_symmetric_tiles)
    split_k = requested_split_k

    if grid_mn < sm_count:
        if grid_mn >= 64:
            split_k = 2
        elif grid_mn >= 32:
            split_k = min(split_k, 4)
        else:
            split_k = min(split_k, 8)
    else:
        split_k = 1

    cluster_size_m = min(4, CLUSTER_SIZE_M)
    if grid_mn % cluster_size_m != 0:
        cluster_size_m = 1

    max_split_k = _ceil_div(MAX_SPLIT_K, cluster_size_m)
    split_k = min(split_k, max_split_k)
    return split_k, grid_mn


def estimate_cuda_profiler_records(
    M: int,
    N: int,
    K: int,
    requested_split_k: int = 1,
    device: Optional[Union[torch.device, str, int]] = None,
) -> int:
    num_blocks_m = _ceil_div(M, BLK_M)
    num_blocks_n = _ceil_div(N, BLK_N)
    total_symmetric_tiles = (num_blocks_m * num_blocks_n + num_blocks_m) // 2
    assert total_symmetric_tiles > 0, "profiler requires at least one output tile"

    split_k, grid_mn = _effective_split_k(
        total_symmetric_tiles,
        requested_split_k,
        _device_sm_count(device),
    )
    k_tiles_total = _ceil_div(K, BLK_K)
    max_k_tiles_per_task = _ceil_div(k_tiles_total, split_k)
    max_tasks_per_cta = _ceil_div(total_symmetric_tiles, grid_mn)
    records_per_task = (
        CUDA_PROFILER_TASK_SLOTS
        + max_k_tiles_per_task * CUDA_PROFILER_PER_K_SLOTS
    )
    records_per_cta = (
        CUDA_PROFILER_CTA_SLOTS + max_tasks_per_cta * records_per_task
    )
    num_ctas = split_k * grid_mn
    return num_ctas * records_per_cta


def symm_gemm_block_scaled(
    xq: torch.Tensor,
    wq: torch.Tensor,
    xs_lhs: torch.Tensor,
    xs_rhs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    SCALE_BLOCK_SIZE_N: int = 128,
    SCALE_BLOCK_SIZE_K: int = 128,
    check_input_shape: bool = False,
    use_mxfp8: bool = False,
    enable_cuda_profiler: bool = False,
    profiler_buffer: Optional[torch.Tensor] = None,
    max_profiler_records: Optional[int] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Thunder Moun Optimizer CUDA Kernel
    Args:
        xq: input matrix of type torch[float16 | float8_e4m3fnuz | float8_e4m3] to perform xq * wq^T, of shape (M, K), by default row major
        wq: input matrix of type torch[float16 | float8_e4m3fnuz | float8_e4m3] to perform xq * wq^T, of shape (M, K), by default row major
        out: output maxtrix placeholder of torch[bfloat16] for x * x^T, of shape (M, M), by default row major
        xs_lhs: block scale of type torch[float16] for x, of shape (M, K // SCALE_BLOCK_SIZE_K), by default row major
        xs_rhs: block scale of type torch[float16] for x, of shape (N // SCALE_BLOCK_SIZE_N, K // SCALE_BLOCK_SIZE_K), by default row major
        use_fp8: use fp8 if True
    Returns:
        The output tensor of shape (M, M)
    """
    if check_input_shape:
        assert xq.dim() >= 2, "xq must be >= 2 dims"

        assert xq.is_cuda and wq.is_cuda, "xq/wq must be CUDA tensors"
        assert xq.shape[-1] == wq.shape[-1], "xq and wq must have the same K dimension"
        assert xq.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ), "xq must be FP8 e4m3"
        assert wq.dtype == xq.dtype, "wq must have the same dtype as xq"
        assert (
            xs_lhs.dtype == torch.float32 and xs_rhs.dtype == torch.float32
        ), "xs_lhs/xs_rhs must be float32 scales"

        if is_column_major(xq):
            warnings.warn(
                "xq is preferred to be in row major but found in column major in this algorithm"
            )

        if xq.dim() > 2:
            xq = xq.flatten(start_dim=1)

    M, K = xq.shape
    N = wq.shape[0]

    split_k = max(1, K // BLK_K)

    # NOTE (yiakwy) : disable on chip split_k reduction via NoC for the moment
    split_k = 1

    if out is None:
        out = torch.zeros((M, N), device=xq.device, dtype=torch.float16)

    if enable_cuda_profiler or profiler_buffer is not None:
        if profiler_buffer is None:
            profiler_capacity = estimate_cuda_profiler_records(
                M, N, K, split_k, xq.device
            )
            if max_profiler_records is not None:
                assert max_profiler_records > 0, "max_profiler_records must be positive"
                profiler_capacity = min(profiler_capacity, max_profiler_records)
            profiler_buffer = allocate_cuda_profiler_buffer(
                profiler_capacity, xq.device
            )
        else:
            profiler_capacity = cuda_profiler_capacity(profiler_buffer)

        module = _jit_thunder_moun_module(True)

        module.symmetric_gemm_fp8_block_scaled_profiled(
            xq,
            wq,
            xs_lhs,
            xs_rhs,
            out,
            M,
            N,
            K,
            xq.stride(0),
            xq.stride(1),
            wq.stride(0),
            wq.stride(1),
            out.stride(0),
            out.stride(1),
            split_k,
            profiler_buffer,
            profiler_capacity,
        )

        return out, profiler_buffer

    module = _jit_thunder_moun_module()

    # print(f"X ptr: {xq.data_ptr()}, Aligned 16B? {xq.data_ptr() % 16 == 0}")
    # print(f"W ptr: {wq.data_ptr()}, Aligned 16B? {wq.data_ptr() % 16 == 0}")

    module.symmetric_gemm_fp8_block_scaled(
        xq,
        wq,
        xs_lhs,
        xs_rhs,
        out,
        M,
        N,
        K,
        xq.stride(0),
        xq.stride(1),
        wq.stride(0),
        wq.stride(1),
        out.stride(0),
        out.stride(1),
        split_k,
    )

    return out

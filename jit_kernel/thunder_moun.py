from __future__ import annotations

import functools
import os
from typing import Optional

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

from jit_kernel.fp8_quantize import fp8_weight_block_wise_quant

extra_ldflags = ["-lcudart", "-lcuda"]

cuda_common_sources = [
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/tma/tma_desc.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/tma/tma_desc_impl.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/tma/tma_copy.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/tma/tma_copy_impl.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/tma/tma_barrier.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/tma/tma_barrier.cu"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/cluster/cluster.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/arch/cluster/cluster.cu"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/block/block.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/block/sched.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/fragment/fragment.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/fragment/nv_frag_gemm_scaled_impl.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/tensor/array_ref.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/tensor/layout.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/tensor/tuple.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/tensor/tensor_view_ref.h"),
]

cuda_wasp_1p2c_impl_sources = [
    str(KERNEL_PATH / "csrc" / "thunder_moun/symm_1p2c_gemm.cu"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/block/wasp_producer.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/block/nv_block_1p2c_gemm_scaled_impl.h"),
]

cuda_multi_stage_impl_sources = [
    str(KERNEL_PATH / "csrc" / "thunder_moun/symm_gemm.cu"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/block/producer.h"),
    str(KERNEL_PATH / "csrc" / "thunder_moun/block/nv_block_gemm_scaled_impl.h"),
]

_CPP_ENTRY = "symmetric_gemm_fp8_block_scaled"
_PY_SYMBOL = "symmetric_gemm_fp8_block_scaled"

common_cuda_flags = ["-O2", "-std=c++17"]

# NOTE (yiakwy) : FLASH_FLOAT_BULK_SPLITK=1 switches the on-chip split-K reduction
USE_BULK_SPLITK = False
if os.environ.get("FLASH_FLOAT_BULK_SPLITK", "0") == "1":
    common_cuda_flags += ["-DUSE_BULK_SPLITK_REDUCE=1"]
    USE_BULK_SPLITK = True


if torch.cuda.is_available():
    major, _minor = torch.cuda.get_device_capability()
else:
    major = 0

if major >= 9:
    #         "--ptxas-options=-v",
    #         "-Xcudafe",
    #         "--diag_suppress=20012",
    common_cuda_flags += [
        "-O2",
        "-Xcompiler",
        "-fPIC",
        "-DENABLE_HOPPER=1",
        "-arch=compute_90a",
        "-code=sm_90a",
        "-I " + str(KERNEL_PATH / "csrc" / "thunder_moun"),
    ]

cuda_sources = cuda_common_sources + cuda_multi_stage_impl_sources


@functools.cache
def _jit_thunder_moun_module():
    if USE_TORCH_JIT:
        raise Exception(
            "Torch JIT is not supported for thunder moun kernel, please set USE_TORCH_JIT to False"
        )
    else:
        cuda_tmp_sources = [f'#include "{path}"' for path in cuda_sources]

        return load_jit(
            "thunder_moun",
            cuda_sources=cuda_tmp_sources,
            # TODO (yiakwy) : add sglang style TVM_FFI warpper
            # cuda_wrappers=[(_PY_SYMBOL, _CPP_ENTRY)],
            extra_cuda_cflags=common_cuda_flags,
            extra_ldflags=extra_ldflags,
        )


cuda_sources_v2 = cuda_common_sources + cuda_wasp_1p2c_impl_sources


@functools.cache
def _jit_thunder_moun_module_v2():
    if USE_TORCH_JIT:
        raise Exception(
            "Torch JIT is not supported for thunder moun kernel, please set USE_TORCH_JIT to False"
        )
    else:
        cuda_tmp_sources = [f'#include "{path}"' for path in cuda_sources_v2]

        return load_jit(
            "thunder_moun",
            cuda_sources=cuda_tmp_sources,
            # TODO (yiakwy) : add sglang style TVM_FFI warpper
            # cuda_wrappers=[(_PY_SYMBOL, _CPP_ENTRY)],
            extra_cuda_cflags=common_cuda_flags,
            extra_ldflags=extra_ldflags,
        )


def is_column_major(tensor: torch.Tensor):
    return tensor.stride()[0] == 1 and tensor.stride()[1] > 1


BLK_K = 128


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
    algorithm: str = "wasp_1p2c",  # "multi_stage", # "wasp_1p2c",
) -> torch.Tensor:
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

        assert algorithm in [
            "multi_stage",
            "wasp_1p2c",
        ], "algorithm must be either multi_stage or wasp_1p2c"

    is_batched = xq.dim() > 2

    # NOTE (yiakwy) : batch symmetric gemm, mimicking the triton gluon XXT_kernel:
    #   the 3D tensors are folded into 2D TMA descriptors on the device side
    #   (batch offset = batch_idx * M, i.e. block_idx_m += batch_idx * num_blocks_m),
    #   which requires a contiguous batched row-major layout (batch_stride == M * row_stride).
    if is_batched:
        assert (
            xq.is_contiguous() and wq.is_contiguous()
        ), "batched xq/wq must be contiguous (batch folding into 2D TMA descriptors requires batch_stride == M * row_stride)"

        # TODO (yiakwy) : add group symmetric gemm support where we don't need xq, wq, xs and ws to be continous along batch dimension

        B, M, K = xq.shape
        N = wq.shape[1]
    else:
        B = 1
        M, K = xq.shape
        N = wq.shape[0]

    split_k = max(1, K // BLK_K)

    if USE_BULK_SPLITK:
        split_k = min(split_k, 4) # maximum 4 for split_k, and at least 2 for cluster multicast

    if out is None:
        out = torch.empty(
            (B, M, N) if is_batched else (M, N),
            device=xq.device,
            dtype=torch.float16,
        )
    else:
        assert out.is_contiguous(), "out must be contiguous (single batch-expanded 2D output TMA descriptor)"

    # TODO (yiakwy) : unified the interface

    if algorithm == "wasp_1p2c":
        module = _jit_thunder_moun_module_v2()

        module.symmetric_gemm_fp8_block_scaled(
            xq,
            wq,
            xs_lhs,
            xs_rhs,
            out,
            M,
            N,
            K,
            B,
            xq.stride(-2), xq.stride(-1),
            wq.stride(-2), wq.stride(-1),
            out.stride(-2), out.stride(-1),
            split_k,
        )
    else:
        # TODO (yiakwy) : multi_stage variant (symm_gemm.cu) does not support batched inputs yet
        assert not is_batched, "multi_stage algorithm does not support batched inputs yet"
        module = _jit_thunder_moun_module()

        module.symmetric_gemm_fp8_block_scaled(
            xq,
            wq,
            xs_lhs,
            xs_rhs,
            out,
            M,
            N,
            K,
            xq.stride(-2), xq.stride(-1),
            wq.stride(-2), wq.stride(-1),
            out.stride(-2), out.stride(-1),
            split_k,
        )

    return out


# ---------------------------------------------------------------------------
# Harness integration (tools/kernel_agent.py OperatorSpec)
# ---------------------------------------------------------------------------
# `shape_generator` + `reference` let the generic 5-stage harness drive this
# GEMM. NOTE: the CUDA kernel is a *symmetric* GEMM (it only launches the
# lower-triangle tiles of the (M-block x N-block) output grid), so the harness
# feeds the valid symmetric regime: a single bf16 matrix ``a`` is block-wise
# FP8-quantized (jit_kernel/fp8_quantize.py) into ``a_q`` + scale ``s``; the
# same ``a_q`` is used for both xq and wq (M == N) and ``s`` is shared by both
# operands. Ground truth is the *pre-quantization* bf16 matmul ``a @ a.T``,
# so correctness includes the FP8 quantization error (not just accumulation).

ATOL = 5e-2
RTOL = 5e-2

# The CUDA kernel tiles at BLK_K granularity (TMA box of 128 x 128), so all
# dims must be BLK_K-aligned; the harness skips unaligned shapes.
ALIGNMENT = BLK_K

_HARNESS_SOURCE: Optional[torch.Tensor] = None


def _blocks(size: int) -> int:
    return (size + BLK_K - 1) // BLK_K


def shape_generator(shape, fill_val=None):
    """Build a *symmetric* FP8-quantized workload for ``symm_gemm_block_scaled``.

    The generic harness passes 1-D shapes (element-wise convention); for this
    GEMM a 1-D shape is read as a square tile (M = N = K). A 3-D shape is read
    as (M, N, K) and requires M == N (symmetric kernel contract).

    A bf16 source matrix is block-wise FP8-quantized (BLK_K x BLK_K blocks);
    the pre-quantization source is recorded for the ground-truth ``reference``.
    """
    global _HARNESS_SOURCE

    if len(shape) == 1:
        M = N = K = int(shape[0])
    elif len(shape) == 3:
        M, N, K = (int(s) for s in shape)
        assert M == N, f"symmetric GEMM requires M == N, got {M} != {N}"
    else:
        raise ValueError(f"unexpected shape {shape!r}: expected (N,) or (M, N, K)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if fill_val is None:
        a = torch.rand(M, K, device=dev, dtype=torch.float32).to(torch.bfloat16)
    else:
        a = torch.full((M, K), fill_val, device=dev, dtype=torch.bfloat16)

    _HARNESS_SOURCE = a

    # Block-wise FP8 quant: s is (m_blocks, k_blocks).
    # scale_X is per row (M, k_blocks); scale_W is per N-block (m_blocks, k_blocks).
    a_q, s = fp8_weight_block_wise_quant(a, BLK_K, BLK_K)
    xs_lhs = s.repeat_interleave(BLK_K, dim=0)[:M]
    return a_q, a_q, xs_lhs, s


def reference(xq: torch.Tensor, wq: torch.Tensor, xs_lhs: torch.Tensor, xs_rhs: torch.Tensor):
    """Ground truth: pre-quantization bf16 symmetric GEMM ``a @ a.T``.

    The source matrix recorded by ``shape_generator`` is the bf16 data the FP8
    quantization approximates, so this is the true target the kernel must match
    (within FP8 quantization tolerance), not the FP8-dequantized recompute.
    """
    a = _HARNESS_SOURCE
    if a is None:
        raise ValueError("shape_generator must run before reference (records the pre-quantization source)")
    return (a @ a.T).to(torch.float16)

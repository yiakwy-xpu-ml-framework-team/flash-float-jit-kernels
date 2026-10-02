from __future__ import annotations

import functools
import os
from typing import Optional

import torch

from jit_kernel.utils import KERNEL_PATH

# Switch with FLASH_FLOAT_TOPK_USE_TVM_FFI=1.
USE_TVM_FFI = os.environ.get("FLASH_FLOAT_TOPK_USE_TVM_FFI", "1") == "1"

_SOURCE = str(KERNEL_PATH / "csrc" / "topk_indexer/topk_indexer_radix.cu")

# NOTE (yiakwy) : pip-installed CUDA toolkits (nvidia/cu13) may ship an
# nvcc whose version is newer than the toolkit headers (CUDA_VERSION).
_COMPAT_FLAG = "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"


_CAND_FLAGS = []
_cand = os.environ.get("FLASH_FLOAT_TOPK_MAX_CANDIDATES")
if _cand:
    _CAND_FLAGS = [f"-DTOPK_MAX_CANDIDATES={int(_cand)}"]

common_cuda_flags = ["-O2", _COMPAT_FLAG] + _CAND_FLAGS


def _arch_flags():
    flags = []
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
        # NOTE (yiakwy) : this kernel is featured with cluster DSMEM via NoC
        flags += ["-DENABLE_HOPPER=1", "-arch=compute_90a", "-code=sm_90a"]
    return flags


def _tvm_ffi_flags():
    flags = ["-O2", _COMPAT_FLAG, "-std=c++17"] + _CAND_FLAGS
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
        flags += ["-DENABLE_HOPPER=1"]
    return flags


@functools.cache
def _jit_fast_topk_v3_module():
    if USE_TVM_FFI:
        os.environ.setdefault("TVM_FFI_CUDA_ARCH_LIST", "9.0a")
        from tvm_ffi.cpp import load_inline

        source = open(_SOURCE).read()
        # tvm_ffi auto-adds `-L<CUDA_HOME>/lib64`, but pip toolkits
        # (nvidia/cu13) put libcudart in `lib/`; add the real dir.
        from torch.utils.cpp_extension import CUDA_HOME

        ldflags = ["-lcuda"]
        if CUDA_HOME:
            ldflags = [f"-L{CUDA_HOME}/lib", f"-L{CUDA_HOME}/lib64"] + ldflags
        return load_inline(
            name="topk_indexer_radix_tvmffi",
            cuda_sources=[source],
            functions=[],  # entry point self-exported via TVM_FFI_DLL_EXPORT_TYPED_FUNC
            extra_cuda_cflags=_tvm_ffi_flags() + ["-DUSE_TVM_FFI"],
            extra_ldflags=ldflags,
        )

    from torch.utils.cpp_extension import load as load_jit

    return load_jit(
        name="topk_indexer_radix",
        sources=[_SOURCE],
        extra_cflags=["-O2"],
        extra_cuda_cflags=common_cuda_flags + _arch_flags(),
        verbose=True,
    )


def fast_topk_v3(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Get the topk indices of the score tensor.

    The raw ``score`` lives in the DeepSeek V3.2 normalized attention domain
    ``[0, 1]``; the kernel maps it onto an 8-bit monotonic radix key.

    Args:
        score: (B, L) score tensor (float32). ``stride(1)`` must be 1.
        lengths: (B) int32 row lengths.
        topk: number of indices to select; one of {512, 1024, 2048}.
        row_starts: (B) int32 per-row window start. When provided, topk applies
            to ``score[i, row_starts[i] : row_starts[i] + lengths[i]]`` and the
            returned indices are relative to that window.

    Returns:
        (B, topk) int32 topk indices.
    """
    assert topk in (512, 1024, 2048), (
        "fast_topk_v3 supports topk in {512, 1024, 2048}, " f"got {topk}"
    )
    assert score.dim() == 2

    topk_indices = torch.full(
        (score.size(0), topk), -1, dtype=torch.int32, device=score.device
    )

    module = _jit_fast_topk_v3_module()

    if USE_TVM_FFI:
        rs = (
            row_starts
            if row_starts is not None
            else torch.empty(0, dtype=torch.int32, device=score.device)
        )
        module.fast_topk(score, topk_indices, lengths, rs, int(row_starts is not None))
    else:
        module.fast_topk(score, topk_indices, lengths, row_starts)
    return topk_indices

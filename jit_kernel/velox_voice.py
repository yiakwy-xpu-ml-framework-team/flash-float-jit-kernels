from __future__ import annotations

import functools
import hashlib
import os
from typing import Optional
from pathlib import Path

from jit_kernel import helper_cuda
from jit_kernel.helper_cuda import cuda_build_flags, cuda_arch_str

os.environ["TVM_FFI_CUDA_ARCH_LIST"] = (
    "12.1a"  # Set the CUDA architecture to 12.1 for DGX Spark NVFP4 support
)

import warnings

import torch

USE_TORCH_JIT = False

# TODO (yaikwy) : use tvm ffi load_jit
if USE_TORCH_JIT:
    from torch.utils.cpp_extension import load as load_jit
else:
    # from jit_kernel.utils import load_jit
    import tvm_ffi
    from tvm_ffi.cpp import load, load_inline as load_jit

from jit_kernel.utils import KERNEL_PATH

CSRC = KERNEL_PATH / "csrc" / "velox_voice"
BUILD_CACHE = Path.home() / ".cache" / "veloxvoice" / "dgx" / "tvmffi"

if os.environ.get("KERNEL_CACHE"):
    BUILD_CACHE = Path(os.environ["KERNEL_CACHE"]) / ".cache" / "veloxvoice" / "dgx" / "tvmffi"


def read_source(rel: str) -> str:
    return (CSRC / rel).read_text(encoding="utf-8")


def source_key(name: str, sources: list[str], flags: list[str], arch: str) -> str:
    h = hashlib.sha256()
    h.update(f"veloxvoice:{name}\n".encode())
    for s in sources:
        h.update(s.encode())
    h.update(" | ".join(flags).encode())
    h.update(arch.encode())
    return h.hexdigest()[:16]


@functools.cache
def _build_cuda_module(
    name: str,
    csrc_files: tuple[str, ...],
    functions: tuple[str, ...],
    extra_cuda_cflags: tuple[str, ...] = (),
    arch_override: str | None = None,
):
    """Compile csrc/*.cu files into one TVM-FFI module (disk-cached, process-cached).

    Returns `tvm_ffi.Module`; entry points are `mod.<function>` callables that take
    torch tensors zero-copy (DLPack).
    """

    sources = tuple(read_source(f"{f}") for f in csrc_files)
    if arch_override:
        flags = list(extra_cuda_cflags)  # replace arch flags entirely
        arch = arch_override
    else:
        flags = cuda_build_flags() + list(extra_cuda_cflags)
        arch = cuda_arch_str()

    key = source_key(name, list(sources), flags, arch)
    build_dir = BUILD_CACHE / f"{name}-{key}"
    build_dir.mkdir(parents=True, exist_ok=True)

    prev = os.environ.get("TVM_FFI_CUDA_ARCH_LIST")
    if arch_override:
        os.environ["TVM_FFI_CUDA_ARCH_LIST"] = arch
    try:
        return load_jit(
            name=f"veloxvoice_{name}",
            cuda_sources=list(sources),
            functions=[],  # entry points are self-exported via TVM_FFI_DLL_EXPORT_TYPED_FUNC
            extra_cuda_cflags=flags,
            extra_ldflags=[f"-L/usr/lib/aarch64-linux-gnu", "-lcuda"],
            build_directory=str(build_dir),
        )
    finally:
        os.environ.pop("TVM_FFI_CUDA_ARCH_LIST", None)
        if prev is not None:
            os.environ["TVM_FFI_CUDA_ARCH_LIST"] = prev


@functools.cache
def _jit_vx_dgx_nvfp4_gemm_module(num_producer_warps=1, num_consumer_warps=8, group_size_m=16, cluster_size_m=1):
    """Build dgx_mxfp4_gemm with P producer warps and C consumer warps."""

    return _build_cuda_module(
        f"mxfp4_gemm_p{num_producer_warps}_c{num_consumer_warps}",
        ("dgx/mxfp4_gemm.cu",),
        (),
        extra_cuda_cflags=(
            "-O2",
            "--use_fast_math",
            "-std=c++17",
            f"-I{CSRC}/dgx",
            f"-DNUM_PRODUCER_WARPS={num_producer_warps}",
            f"-DNUM_CONSUMER_WARPS={num_consumer_warps}",
            f"-DK_GROUP_SIZE_M={group_size_m}",
            f"-DK_CLUSTER_SIZE_M={cluster_size_m}",
        ),
        arch_override="12.1a",
    )

def dgx_mxfp4_gemm(pack_a, pack_b, scale_a, scale_b,
                    num_producer_warps=1, num_consumer_warps=8,
                    group_size_m=16, cluster_size_m=1):
    """pack_a [M, K/2] u8; pack_b [N, K/2] u8; row/col scale tensor u8; -> out [M, N] f32."""

    M, K2 = pack_a.shape
    out = torch.empty(M, pack_b.shape[0], device=pack_a.device, dtype=torch.float32)

    module = _jit_vx_dgx_nvfp4_gemm_module(num_producer_warps, num_consumer_warps, group_size_m, cluster_size_m)

    with tvm_ffi.use_torch_stream():
        module.dgx_mxfp4_gemm(
            pack_a, pack_b, scale_a, scale_b, out
        )
    return out
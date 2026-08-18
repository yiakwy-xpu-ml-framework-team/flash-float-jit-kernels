#
# Adapted from sglang.jit_kernel.utils to avoid circular import
#
import torch

import functools
import pathlib

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    TypeVar,
)

F = TypeVar("F", bound=Callable[..., Any])

from jit_kernel.helper_cuda import (
    CUDA_HOME,
    _is_cuda,
    get_cuda_libraries,
    get_nvcc_cuda_version,
    get_nvcc_flags,
)
from jit_kernel.helper_rocm import (
    ROCM_HOME,
    _is_hip,
    get_hip_libraries,
    get_hipcc_flags,
    get_hipcc_rocm_version,
)

## Constant
SUPPORTED_DEVICES = ["CUDA", "ROCm"]

operator_namespace = "flash_float-jit-kernel"

common_libs = ["c10", "torch", "torch_python"]

common_flags = [
    "-DNDEBUG",
    # f"-DOPERATOR_NAMESPACE={operator_namespace}",
    "-O0",
    "-Xcompiler",
    "-fPIC",
    "-std=c++17",
    "-DENABLE_BF16",
    "-DENABLE_BF16x2",
    "-DENABLE_HALF2",
    "-DENABLE_OCP_MXFP8",
    "-DENABLE_ML_DTYPE_FP8_E4M3_FNUZ",
    "-DENABLE_OCP_FP4",
]

cxx_flags = ["-O0", "-g"]
extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", "-L/usr/lib/x86_64-linux-gnu"]


def get_device_flags():
    if _is_hip:
        return get_hipcc_flags()
    elif _is_cuda:
        return get_nvcc_flags()
    else:
        raise RuntimeError(
            f"Unknown runtime environment, only these {SUPPORTED_DEVICES} supported for the moment."
        )


def get_device_libs():
    if _is_hip:
        return get_hip_libraries()
    elif _is_cuda:
        return get_cuda_libraries()
    else:
        raise RuntimeError(
            f"Unknown runtime environment, only these {SUPPORTED_DEVICES} supported for the moment."
        )

# Enforce CXX14_API with C++17 for internal libraries
torch._C._GLIBCXX_USE_CXX14_ABI = True


def cache_once(fn: F) -> F:
    """
    NOTE: `functools.lru_cache` is not compatible with `torch.compile`
    So we manually implement a simple cache_once decorator to replace it.
    """
    result_map = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in result_map:
            result_map[key] = fn(*args, **kwargs)
        return result_map[key]

    return wrapper  # type: ignore

@cache_once
def _resolve_kernel_path() -> pathlib.Path:
    cur_dir = pathlib.Path(__file__).parent.resolve()

    # first, try this directory structure
    def _environment_install():
        candidate = cur_dir.resolve()
        if (candidate / "csrc").exists():
            return candidate
        return None

    def _package_install():
        # TODO: support find path by package
        return None

    path = _environment_install() or _package_install()
    if path is None:
        raise RuntimeError("Cannot find flash-float-jit-kernel.jit_kernel path")
    return path


KERNEL_PATH = _resolve_kernel_path()

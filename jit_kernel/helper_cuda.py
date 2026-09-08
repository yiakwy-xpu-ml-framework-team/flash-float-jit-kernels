# Copyright 2025 FlashFloat authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import re
import subprocess

import torch
from packaging.version import Version, parse

try:
    from torch.utils.cpp_extension import CUDA_HOME
except:
    raise RuntimeError(
        "Base env does not provide Torch with support of CUDA SDK. Exit."
    )

CUDA_VERSION_PAT = r"CUDA version: (\S+)"
CUDA_SDK_ROOT = CUDA_HOME  # "/usr/local/cuda"
# currently only support MI30X (MI308X, MI300XA) datacenter intelligent computing accelerator
ALLOWED_NVGPU_ARCHS = ["sm75", "sm_80", "sm_90", "sm_100"]


def is_cuda(cuda_sdk_root=None) -> bool:
    SDK_ROOT = f"{cuda_sdk_root or CUDA_SDK_ROOT}"

    def _check_sdk_installed() -> bool:
        # return True if this dir points to a directory or symbolic link
        return os.path.isdir(SDK_ROOT)

    if not _check_sdk_installed():
        return False, None

    # we provide torch for the base env, check whether it is valid installation
    def get_sdk_ver(dir):
        result = subprocess.run(
            [
                f"{dir}/bin/nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -o -m1 '.*'"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=True,
        )

        if result.returncode != 0:
            print(f"Use CUDA pytorch ({CUDA_HOME}), but no devices found!")
            return False, result

        return True, result

    status, result = get_sdk_ver("/usr")
    if not status:
        status, result = get_sdk_ver(SDK_ROOT)
        if not status:
            return False, None

    sm_ver = result.stdout.strip()
    print(f"target NV gpu arch {sm_ver}")
    return True, sm_ver


_is_cuda, sm_ver = is_cuda()

if _is_cuda:
    assert CUDA_HOME is not None, "CUDA_HOME is not set"

    CUDA_HOME = os.environ.get("CUDA_HOME", CUDA_HOME)
    pass


def get_nvcc_cuda_version(cuda_sdk_root=None):
    assert _is_cuda

    SDK_ROOT = f"{cuda_sdk_root or CUDA_SDK_ROOT}"

    nvcc_output = subprocess.check_output(
        [f"{SDK_ROOT}/bin/nvcc", "-V"], universal_newlines=True
    )

    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


nvcc_flags = [
    "-gencode=arch=compute_75,code=sm_75",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
    "-gencode=arch=compute_90,code=sm_100",
    "-std=c++17",
    "-use_fast_math",
    "-Xcompiler=-Wconversion",
    "-Xcompiler=-fno-strict-aliasing",
]

cuda_libraries = ["cuda", "cublas"]

for flag in [
    "-D__CUDA_NO_HALF_OPERATORS__=1",
    "-D__CUDA_NO_HALF_CONVERSIONS__=1",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__=1",
    "-D__CUDA_NO_HALF2_OPERATORS__=1",
]:
    try:
        from torch.utils.cpp_extension import COMMON_NVCC_FLAGS

        COMMON_NVCC_FLAGS.remove(flag)
    except ValueError:
        pass


def get_cuda_libraries():
    return cuda_libraries


def get_nvcc_flags():
    return nvcc_flags


def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


# TODO (yiakwy) : remove
def compute_capability(device: int = 0) -> tuple[int, int]:
    import torch

    return torch.cuda.get_device_capability(device)


# TODO (yiakwy) : remove
def cuda_arch_str(device: int = 0) -> str:
    """e.g. '12.1a' for GB10, '9.0a' for Hopper."""
    return ".".join(str(x) for x in compute_capability(device))


def cuda_gencode_flag(device: int = 0) -> str:
    arch = cuda_arch_str(device).replace(".", "")
    return f"-gencode=arch=compute_{arch},code=sm_{arch}"


def cuda_build_flags(device: int = 0) -> list[str]:
    return ["-O3", "--use_fast_math", "-std=c++17", cuda_gencode_flag(device)]


def is_hopper_or_newer(device: int = 0) -> bool:
    return compute_capability(device)[0] >= 9
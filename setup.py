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

import multiprocessing
import os
import sys
from pathlib import Path
from typing import List

import torch
from setuptools import find_packages, setup
from setuptools_scm import get_version as py_get_base_version

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except:
    raise Exception("Base env does not provide torch distribution. Exit.")


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

from jit_kernel.utils import SUPPORTED_DEVICES

PROJECT_ROOT = Path(__file__).parent.resolve()

def get_path(*filepath) -> str:
    return os.path.join(PROJECT_ROOT / "requirements", *filepath)


def get_requirements() -> List[str]:
    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif (
                not line.startswith("--")
                and not line.startswith("#")
                and line.strip() != ""
            ):
                resolved_requirements.append(line)
        return resolved_requirements

    if _is_hip:
        requirements = _read_requirements("rocm-build.txt")
    elif _is_cuda:
        requirements = _read_requirements("cuda-build.txt")
    else:
        raise ValueError(f"Unsupported platform, please use {SUPPORTED_DEVICES}")
    return requirements


if "bdist_wheel" in sys.argv and "--plat-name" not in sys.argv:
    sys.argv.extend(["--plat-name", "manylinux2014_x86_64"])


def get_version():
    version = py_get_base_version(write_to="jit_kernel/__version__.py")
    sep = "+" if "+" not in version else "."

    if _is_hip:
        rocm_version = torch.version.hip or get_hipcc_rocm_version(ROCM_HOME)
        version += f"{sep}rocm{rocm_version.replace('.', '')[:3]}"
    elif _is_cuda:
        cuda_version = torch.version.cuda or get_nvcc_cuda_version(CUDA_HOME)
        version += f"{sep}cu{cuda_version.replace('.', '')[:3]}"
    else:
        raise RuntimeError(
            f"Unknown runtime environment, only these {SUPPORTED_DEVICES} supported for the moment."
        )

    return version


# NOTE (yiakwy) : this is a jit-kernel lib, so we don't have pre-compiled sources. The sources will be provided at runtime by the jit wrapper.
ext_modules = []

setup(
    name="flash-float-jit-kernel",
    author="yiakwy, and other flash-float authors",
    author_email="yiak.wy@gmail.com",
    description="A high-throughput and memory-efficient JIT kernel library for low bit floats of LLM inference",
    version=get_version(),
    packages=find_packages(where=".", exclude=("benchmark*",)),
    install_requires=get_requirements(),
    package_dir={"": "."},
    package_data={
        "jit_kernel": ["csrc/**/*.cu", "csrc/**/*.h", "csrc/**/*.cpp", "csrc/**/*.cuh"],
    },
    include_package_data=True,
    zip_safe=False,
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=True, max_jobs=multiprocessing.cpu_count()
        )
    },
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)

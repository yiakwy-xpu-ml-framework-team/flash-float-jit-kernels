from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


THIS_DIR = Path(__file__).resolve().parent
SOURCE_PATH = THIS_DIR / "cupti_activity_collector.cpp"
BUILD_DIR = THIS_DIR / "build" / "cupti_activity"


def _cuda_home() -> Path:
    candidates = []
    env_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if env_home:
        candidates.append(Path(env_home))
    candidates.extend([Path("/usr/local/cuda"), Path("/usr/local/cuda-12.8")])

    for candidate in candidates:
        if (candidate / "include" / "cuda.h").exists():
            return candidate
    raise RuntimeError(
        "Could not find CUDA_HOME. Set CUDA_HOME, for example "
        "export CUDA_HOME=/usr/local/cuda-12.8"
    )


def _cupti_paths(cuda_home: Path) -> tuple[Path, Path]:
    include_candidates = [
        cuda_home / "extras" / "CUPTI" / "include",
        cuda_home / "include",
    ]
    lib_candidates = [
        cuda_home / "extras" / "CUPTI" / "lib64",
        cuda_home / "lib64",
    ]

    include_dir = next((path for path in include_candidates if (path / "cupti.h").exists()), None)
    lib_dir = next((path for path in lib_candidates if (path / "libcupti.so").exists()), None)
    if include_dir is None or lib_dir is None:
        raise RuntimeError(
            "Could not find CUPTI headers/libs under CUDA_HOME. Expected "
            "$CUDA_HOME/extras/CUPTI/include/cupti.h and "
            "$CUDA_HOME/extras/CUPTI/lib64/libcupti.so"
        )
    return include_dir, lib_dir


def _shared_lib_name() -> str:
    if platform.system() == "Windows":
        return "ffjk_cupti_activity.dll"
    if platform.system() == "Darwin":
        return "libffjk_cupti_activity.dylib"
    return "libffjk_cupti_activity.so"


def _compile_library(force_rebuild: bool = False) -> Path:
    cuda_home = _cuda_home()
    cupti_include, cupti_lib = _cupti_paths(cuda_home)
    output_path = BUILD_DIR / _shared_lib_name()

    if (
        not force_rebuild
        and output_path.exists()
        and output_path.stat().st_mtime >= SOURCE_PATH.stat().st_mtime
    ):
        return output_path

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "g++",
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        str(SOURCE_PATH),
        "-o",
        str(output_path),
        "-I",
        str(cuda_home / "include"),
        "-I",
        str(cupti_include),
        "-L",
        str(cuda_home / "lib64"),
        "-L",
        str(cupti_lib),
        f"-Wl,-rpath,{cuda_home / 'lib64'}",
        f"-Wl,-rpath,{cupti_lib}",
        "-lcupti",
        "-lcuda",
        "-lcudart",
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("g++ is required to build the CUPTI activity collector") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to build CUPTI activity collector. Command was:\n"
            + " ".join(cmd)
        ) from exc
    return output_path


class CuptiActivityCollector:
    def __init__(self, force_rebuild: bool = False) -> None:
        lib_path = _compile_library(force_rebuild=force_rebuild)
        self._lib = ctypes.CDLL(str(lib_path))

        self._lib.ffjk_cupti_start.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.ffjk_cupti_start.restype = ctypes.c_int
        self._lib.ffjk_cupti_stop.argtypes = []
        self._lib.ffjk_cupti_stop.restype = ctypes.c_int
        self._lib.ffjk_cupti_last_error.argtypes = []
        self._lib.ffjk_cupti_last_error.restype = ctypes.c_char_p

    def _last_error(self) -> str:
        value = self._lib.ffjk_cupti_last_error()
        if value is None:
            return "unknown CUPTI collector error"
        return value.decode("utf-8", errors="replace")

    def start(
        self,
        output_path: Path,
        *,
        runtime: bool = True,
        driver: bool = True,
        memcpy: bool = True,
        memset: bool = True,
        latency_timestamps: bool = False,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = self._lib.ffjk_cupti_start(
            str(output_path).encode("utf-8"),
            int(runtime),
            int(driver),
            int(memcpy),
            int(memset),
            int(latency_timestamps),
        )
        if result != 0:
            raise RuntimeError(self._last_error())

    def stop(self) -> None:
        result = self._lib.ffjk_cupti_stop()
        if result != 0:
            raise RuntimeError(self._last_error())


@contextmanager
def collect_activity(
    output_path: Path,
    *,
    collector: Optional[CuptiActivityCollector] = None,
    runtime: bool = True,
    driver: bool = True,
    memcpy: bool = True,
    memset: bool = True,
    latency_timestamps: bool = False,
) -> Iterator[None]:
    owns_collector = collector is None
    if collector is None:
        collector = CuptiActivityCollector()
    collector.start(
        output_path,
        runtime=runtime,
        driver=driver,
        memcpy=memcpy,
        memset=memset,
        latency_timestamps=latency_timestamps,
    )
    try:
        yield
    finally:
        collector.stop()
        if owns_collector:
            del collector

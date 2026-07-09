from __future__ import annotations

import ctypes
import csv
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Callable, Iterable, List, Sequence


THIS_DIR = Path(__file__).resolve().parent
SOURCE_PATH = THIS_DIR / "cupti_range_profiler_collector.cpp"
BUILD_DIR = THIS_DIR / "build" / "cupti_range_profiler"

DEFAULT_RANGE_METRICS = (
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_bytes.sum",
    "sm__ctas_launched.sum",
    "sm__warps_launched.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct",
)


def parse_metric_csv(value: str | None) -> List[str]:
    if not value:
        return list(DEFAULT_RANGE_METRICS)
    return [item.strip() for item in value.split(",") if item.strip()]


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

    include_dir = next(
        (
            path
            for path in include_candidates
            if (path / "cupti_range_profiler.h").exists()
            and (path / "cupti_profiler_host.h").exists()
        ),
        None,
    )
    lib_dir = next((path for path in lib_candidates if (path / "libcupti.so").exists()), None)
    if include_dir is None or lib_dir is None:
        raise RuntimeError(
            "Could not find CUPTI Range Profiler headers/libs under CUDA_HOME. "
            "Expected cupti_range_profiler.h, cupti_profiler_host.h, and libcupti.so."
        )
    return include_dir, lib_dir


def _shared_lib_name() -> str:
    if platform.system() == "Windows":
        return "ffjk_cupti_range_profiler.dll"
    if platform.system() == "Darwin":
        return "libffjk_cupti_range_profiler.dylib"
    return "libffjk_cupti_range_profiler.so"


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

    nvperf_host = cupti_lib / "libnvperf_host.so"
    if nvperf_host.exists():
        cmd.append("-lnvperf_host")

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("g++ is required to build the CUPTI range profiler") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to build CUPTI range profiler. Command was:\n"
            + " ".join(cmd)
        ) from exc
    return output_path


def safe_metric_filename(metric: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", metric)


class CuptiRangeProfiler:
    def __init__(self, force_rebuild: bool = False) -> None:
        lib_path = _compile_library(force_rebuild=force_rebuild)
        self._lib = ctypes.CDLL(str(lib_path))

        self._lib.ffjk_cupti_range_prepare.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self._lib.ffjk_cupti_range_prepare.restype = ctypes.c_int
        self._lib.ffjk_cupti_range_start_pass.argtypes = []
        self._lib.ffjk_cupti_range_start_pass.restype = ctypes.c_int
        self._lib.ffjk_cupti_range_stop_pass.argtypes = []
        self._lib.ffjk_cupti_range_stop_pass.restype = ctypes.c_int
        self._lib.ffjk_cupti_range_finish.argtypes = [ctypes.c_char_p]
        self._lib.ffjk_cupti_range_finish.restype = ctypes.c_int
        self._lib.ffjk_cupti_range_abort.argtypes = []
        self._lib.ffjk_cupti_range_abort.restype = None
        self._lib.ffjk_cupti_range_num_passes.argtypes = []
        self._lib.ffjk_cupti_range_num_passes.restype = ctypes.c_size_t
        self._lib.ffjk_cupti_range_last_error.argtypes = []
        self._lib.ffjk_cupti_range_last_error.restype = ctypes.c_char_p

    def _last_error(self) -> str:
        value = self._lib.ffjk_cupti_range_last_error()
        if value is None:
            return "unknown CUPTI range profiler error"
        return value.decode("utf-8", errors="replace")

    def profile(
        self,
        output_path: Path,
        range_name: str,
        metrics: Sequence[str],
        fn: Callable[[], object],
    ) -> None:
        if not metrics:
            raise ValueError("metrics must not be empty")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        metric_csv = ",".join(metrics).encode("utf-8")
        prepared = False
        result = self._lib.ffjk_cupti_range_prepare(
            metric_csv,
            range_name.encode("utf-8"),
        )
        if result != 0:
            raise RuntimeError(self._last_error())
        prepared = True

        num_passes = max(1, int(self._lib.ffjk_cupti_range_num_passes()))
        max_replays = max(16, num_passes * 4)
        all_passes_submitted = False
        try:
            for _ in range(max_replays):
                start_result = self._lib.ffjk_cupti_range_start_pass()
                if start_result != 0:
                    raise RuntimeError(self._last_error())
                try:
                    fn()
                finally:
                    stop_result = self._lib.ffjk_cupti_range_stop_pass()
                if stop_result < 0:
                    raise RuntimeError(self._last_error())
                if stop_result == 1:
                    all_passes_submitted = True
                    break

            if not all_passes_submitted:
                raise RuntimeError(
                    "CUPTI Range Profiler did not submit all passes after "
                    f"{max_replays} user replays; expected about {num_passes} pass(es)."
                )

            finish_result = self._lib.ffjk_cupti_range_finish(
                str(output_path).encode("utf-8")
            )
            if finish_result != 0:
                raise RuntimeError(self._last_error())
            prepared = False
        except Exception:
            if prepared:
                self._lib.ffjk_cupti_range_abort()
            raise


def write_metric_error(path: Path, range_name: str, metrics: Iterable[str], message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "range_index",
        "range_name",
        "metric",
        "value",
        "chip_name",
        "num_passes",
        "all_passes_submitted",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "range_index": "0",
                    "range_name": range_name,
                    "metric": metric,
                    "value": "",
                    "chip_name": "",
                    "num_passes": "",
                    "all_passes_submitted": "0",
                    "error": message,
                }
            )

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jit_kernel.thunder_moun import (  # noqa: E402
    allocate_cuda_profiler_buffer,
    estimate_cuda_profiler_records,
    symm_gemm_block_scaled,
)


BLOCK_SIZE = 128
DEFAULT_SEED = 42


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def make_inputs(
    m: int,
    device: torch.device,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)

    x = torch.randn((m, m), dtype=torch.bfloat16, device=device)
    xs_lhs = torch.ones(
        (m, ceil_div(m, BLOCK_SIZE)), dtype=torch.float32, device=device
    )
    xs_rhs = torch.ones(
        (ceil_div(m, BLOCK_SIZE), ceil_div(m, BLOCK_SIZE)),
        dtype=torch.float32,
        device=device,
    )

    x_fp8 = x.to(torch.float8_e4m3fn)
    w_fp8 = x.to(torch.float8_e4m3fn)
    return x_fp8, w_fp8, xs_lhs, xs_rhs


def maybe_start_cuda_profiler(enabled: bool) -> bool:
    if not enabled:
        return False
    try:
        torch.cuda.cudart().cudaProfilerStart()
    except Exception as exc:
        raise RuntimeError("failed to call cudaProfilerStart") from exc
    return True


def maybe_stop_cuda_profiler(started: bool) -> None:
    if not started:
        return
    try:
        torch.cuda.cudart().cudaProfilerStop()
    except Exception as exc:
        raise RuntimeError("failed to call cudaProfilerStop") from exc


def maybe_push_nvtx(enabled: bool, message: str) -> bool:
    if not enabled:
        return False
    torch.cuda.nvtx.range_push(message)
    return True


def maybe_pop_nvtx(pushed: bool) -> None:
    if pushed:
        torch.cuda.nvtx.range_pop()


def unwrap_result(result: Any) -> torch.Tensor:
    if isinstance(result, tuple):
        return result[0]
    return result


def run_target(
    *,
    m: int,
    warmup: int,
    iters: int,
    device: torch.device,
    seed: int,
    profiled: bool,
    max_profiler_records: Optional[int],
    cuda_profiler_api: bool,
    nvtx: bool,
    sync_each_iter: bool,
) -> Tuple[float, float]:
    if iters <= 0:
        raise ValueError("iters must be positive")

    x_fp8, w_fp8, xs_lhs, xs_rhs = make_inputs(m, device, seed)
    out = torch.empty((m, m), dtype=torch.float16, device=device)

    profiler_buffer = None
    if profiled:
        profiler_capacity = estimate_cuda_profiler_records(m, m, m)
        if max_profiler_records is not None:
            profiler_capacity = min(profiler_capacity, max_profiler_records)
        profiler_buffer = allocate_cuda_profiler_buffer(profiler_capacity, device)

    def run_once() -> torch.Tensor:
        return unwrap_result(
            symm_gemm_block_scaled(
                x_fp8,
                w_fp8,
                xs_lhs,
                xs_rhs,
                out=out,
                enable_cuda_profiler=profiled,
                profiler_buffer=profiler_buffer,
                max_profiler_records=max_profiler_records,
            )
        )

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    profiler_started = False
    nvtx_pushed = False
    start_time = time.time()
    try:
        profiler_started = maybe_start_cuda_profiler(cuda_profiler_api)
        nvtx_pushed = maybe_push_nvtx(nvtx, "ffjk_thunder_moun_cuda_path")

        start_event.record()
        for _ in range(iters):
            run_once()
            if sync_each_iter:
                torch.cuda.synchronize(device)
        end_event.record()
        torch.cuda.synchronize(device)
    finally:
        maybe_pop_nvtx(nvtx_pushed)
        maybe_stop_cuda_profiler(profiler_started)

    host_ms = (time.time() - start_time) / iters * 1000.0
    device_ms = start_event.elapsed_time(end_event) / iters
    return host_ms, device_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="External-profiler target for the ThunderMoun CUDA path."
    )
    parser.add_argument("--m", type=int, default=4096, help="Use M=N=K=m.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--profiled",
        action="store_true",
        help="Run the embedded-profiler CUDA symbol instead of the normal path.",
    )
    parser.add_argument("--max-profiler-records", type=int, default=None)
    parser.add_argument(
        "--cuda-profiler-api",
        action="store_true",
        help="Call cudaProfilerStart/Stop around the measured loop.",
    )
    parser.add_argument(
        "--no-nvtx",
        action="store_true",
        help="Do not add an NVTX range around the measured loop.",
    )
    parser.add_argument(
        "--sync-each-iter",
        action="store_true",
        help="Synchronize after every measured kernel launch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this target")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must be a CUDA device")
    if device.index is not None:
        torch.cuda.set_device(device)

    stream = torch.cuda.Stream(device=device)
    torch.cuda.set_stream(stream)

    host_ms, device_ms = run_target(
        m=args.m,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
        seed=args.seed,
        profiled=args.profiled,
        max_profiler_records=args.max_profiler_records,
        cuda_profiler_api=args.cuda_profiler_api,
        nvtx=not args.no_nvtx,
        sync_each_iter=args.sync_each_iter,
    )

    print("ThunderMoun CUDA path target")
    print(f"target_kernel=hopper_symm_gemm_kernel_entry")
    print(
        f"M=N=K={args.m} warmup={args.warmup} iters={args.iters} "
        f"profiled={args.profiled}"
    )
    print(f"device={torch.cuda.get_device_name(device)}")
    print(f"host_ms_per_iter={host_ms:.6f}")
    print(f"device_ms_per_iter={device_ms:.6f}")


if __name__ == "__main__":
    main()

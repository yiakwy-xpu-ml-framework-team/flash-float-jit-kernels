from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jit_kernel.thunder_moun import (  # noqa: E402
    allocate_cuda_profiler_buffer,
    estimate_cuda_profiler_records,
    symm_gemm_block_scaled,
)


SEED = 42
BLOCK_SIZE = 128


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def make_inputs(m: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(SEED)
    x = torch.randn(m, m, dtype=torch.bfloat16, device="cuda")
    xs_lhs = torch.ones(
        (m, ceil_div(m, BLOCK_SIZE)), dtype=torch.float32, device="cuda"
    )
    xs_rhs = torch.ones(
        (ceil_div(m, BLOCK_SIZE), ceil_div(m, BLOCK_SIZE)),
        dtype=torch.float32,
        device="cuda",
    )
    x_fp8 = x.to(torch.float8_e4m3fn)
    w_fp8 = x.to(torch.float8_e4m3fn)
    return x_fp8, w_fp8, xs_lhs, xs_rhs


def clone_result(result):
    if isinstance(result, tuple):
        result = result[0]
    torch.cuda.synchronize()
    return result.detach().clone()


def run_baseline(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    xs_lhs: torch.Tensor,
    xs_rhs: torch.Tensor,
) -> torch.Tensor:
    return clone_result(symm_gemm_block_scaled(x_fp8, w_fp8, xs_lhs, xs_rhs))


def run_profiled(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    xs_lhs: torch.Tensor,
    xs_rhs: torch.Tensor,
    profiler_buffer: torch.Tensor,
) -> torch.Tensor:
    return clone_result(
        symm_gemm_block_scaled(
            x_fp8,
            w_fp8,
            xs_lhs,
            xs_rhs,
            enable_cuda_profiler=True,
            profiler_buffer=profiler_buffer,
        )
    )


def compare_tensors(name: str, actual: torch.Tensor, expected: torch.Tensor) -> Dict[str, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    abs_diff = (actual_f - expected_f).abs()
    mismatch = actual != expected
    mismatch_count = int(mismatch.sum().item())
    total = actual.numel()

    if mismatch_count == 0:
        stats = {
            "mismatch_count": 0,
            "mismatch_pct": 0.0,
            "max_abs": 0.0,
            "max_rel": 0.0,
            "max_index_flat": -1,
        }
    else:
        flat_abs = abs_diff.flatten()
        max_abs, max_idx = flat_abs.max(dim=0)
        rel = abs_diff / expected_f.abs().clamp_min(1e-12)
        max_rel = rel.flatten().max()
        stats = {
            "mismatch_count": mismatch_count,
            "mismatch_pct": mismatch_count / total * 100.0,
            "max_abs": float(max_abs.item()),
            "max_rel": float(max_rel.item()),
            "max_index_flat": int(max_idx.item()),
        }

    print(
        f"{name:<32} mismatches={stats['mismatch_count']:>8}/{total} "
        f"({stats['mismatch_pct']:.4f}%) "
        f"max_abs={stats['max_abs']:.6g} max_rel={stats['max_rel']:.6g} "
        f"flat_idx={stats['max_index_flat']}"
    )
    return stats


def run_correctness_probe(m: int, runs: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this correctness probe")

    x_fp8, w_fp8, xs_lhs, xs_rhs = make_inputs(m)
    profiler_capacity = estimate_cuda_profiler_records(m, m, m)
    profiler_buffer = allocate_cuda_profiler_buffer(profiler_capacity, x_fp8.device)

    baseline_outputs: List[torch.Tensor] = []
    profiled_outputs: List[torch.Tensor] = []

    print(f"CUDA ThunderMoun correctness probe: M=N=K={m}, runs={runs}")
    print("Running baseline kernels...")
    for i in range(runs):
        baseline_outputs.append(run_baseline(x_fp8, w_fp8, xs_lhs, xs_rhs))
        print(f"  baseline run {i} done")

    print("Running profiled kernels...")
    for i in range(runs):
        profiled_outputs.append(run_profiled(x_fp8, w_fp8, xs_lhs, xs_rhs, profiler_buffer))
        print(f"  profiled run {i} done")

    print("\nWithin-mode consistency:")
    for i in range(1, runs):
        compare_tensors(f"baseline[0] vs baseline[{i}]", baseline_outputs[i], baseline_outputs[0])
    for i in range(1, runs):
        compare_tensors(f"profiled[0] vs profiled[{i}]", profiled_outputs[i], profiled_outputs[0])

    print("\nBaseline vs profiled:")
    for i in range(runs):
        compare_tensors(f"baseline[{i}] vs profiled[{i}]", profiled_outputs[i], baseline_outputs[i])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check ThunderMoun CUDA baseline/profiled output consistency."
    )
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--runs", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_correctness_probe(m=args.m, runs=args.runs)

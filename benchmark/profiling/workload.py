from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import triton

from jit_kernel.thunder_moun import symm_gemm_block_scaled
import jit_kernel.triton3_4.symm_gemm as triton_symm_gemm


SEED = 42
DEFAULT_SCENARIOS = (
    "cuda_cold",
    "cuda_warm",
    "triton_moun_native_cold",
    "triton_moun_native_warm",
    "triton_symm_native_cold",
    "triton_symm_native_warm",
)


def parse_csv_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def prepare_inputs(m: int, device: str) -> Dict[str, torch.Tensor]:
    torch.manual_seed(SEED)

    x = torch.randn(m, m, dtype=torch.bfloat16, device=device)
    x_fp8 = x.to(torch.float8_e4m3fn)
    xs_lhs = torch.ones((m, triton.cdiv(m, 128)), dtype=torch.float32, device=device)
    xs_rhs = torch.ones(
        (triton.cdiv(m, 128), triton.cdiv(m, 128)),
        dtype=torch.float32,
        device=device,
    )

    return {
        "x_fp8": x_fp8,
        "w_fp8": x_fp8,
        "xs_lhs": xs_lhs,
        "xs_rhs": xs_rhs,
    }


@contextmanager
def triton_ffi_mode(use_tvm_ffi: Optional[bool]):
    old = triton_symm_gemm.USE_TVM_FFI
    if use_tvm_ffi is not None:
        triton_symm_gemm.USE_TVM_FFI = use_tvm_ffi
    try:
        yield
    finally:
        triton_symm_gemm.USE_TVM_FFI = old


def make_provider(
    provider: str,
    tensors: Dict[str, torch.Tensor],
    out: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    x_fp8 = tensors["x_fp8"]
    w_fp8 = tensors["w_fp8"]
    xs_lhs = tensors["xs_lhs"]
    xs_rhs = tensors["xs_rhs"]

    if provider == "cuda":
        return lambda: symm_gemm_block_scaled(
            x_fp8,
            w_fp8,
            xs_lhs,
            xs_rhs,
            out=out,
        )
    if provider == "triton_moun":
        return lambda: triton_symm_gemm.thunder_moun_gemm(
            x_fp8,
            w_fp8,
            xs_lhs,
            xs_rhs,
            out=out,
        )
    if provider == "triton_symm":
        return lambda: triton_symm_gemm.fp8_gemm_block_scaled(
            x_fp8,
            w_fp8,
            xs_lhs,
            xs_rhs,
            o=out,
            is_symm=True,
        )
    raise ValueError(
        f"Unknown provider {provider!r}. Expected one of: "
        "cuda, triton_moun, triton_symm"
    )


def scenario_config(name: str, args: argparse.Namespace) -> Dict[str, object]:
    if name == "cuda_cold":
        return {
            "scenario": name,
            "provider": "cuda",
            "launcher": "cuda_tvm_ffi_extension",
            "warmup": 0,
            "iters": args.cold_iters,
            "triton_use_tvm_ffi": None,
        }
    if name == "cuda_warm":
        return {
            "scenario": name,
            "provider": "cuda",
            "launcher": "cuda_tvm_ffi_extension",
            "warmup": args.warmup,
            "iters": args.iters,
            "triton_use_tvm_ffi": None,
        }
    if name == "triton_moun_native_cold":
        return {
            "scenario": name,
            "provider": "triton_moun",
            "launcher": "triton_native_kernel_grid_call",
            "warmup": 0,
            "iters": args.cold_iters,
            "triton_use_tvm_ffi": False,
        }
    if name == "triton_moun_native_warm":
        return {
            "scenario": name,
            "provider": "triton_moun",
            "launcher": "triton_native_kernel_grid_call",
            "warmup": args.warmup,
            "iters": args.iters,
            "triton_use_tvm_ffi": False,
        }
    if name == "triton_symm_native_cold":
        return {
            "scenario": name,
            "provider": "triton_symm",
            "launcher": "triton_native_kernel_grid_call",
            "warmup": 0,
            "iters": args.cold_iters,
            "triton_use_tvm_ffi": False,
        }
    if name == "triton_symm_native_warm":
        return {
            "scenario": name,
            "provider": "triton_symm",
            "launcher": "triton_native_kernel_grid_call",
            "warmup": args.warmup,
            "iters": args.iters,
            "triton_use_tvm_ffi": False,
        }
    raise ValueError(f"Unknown scenario {name!r}")


def warmup(fn: Callable[[], torch.Tensor], iters: int) -> None:
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()


def timed_calls(fn: Callable[[], torch.Tensor], iters: int) -> Dict[str, float]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0 / iters
    event_ms = start_event.elapsed_time(end_event) / iters

    return {
        "wall_latency_ms": wall_ms,
        "event_latency_ms": event_ms,
    }


def summarize_output_checks(
    m: int,
    outputs: Dict[str, torch.Tensor],
    rtol: float = 5e-1,
    atol: float = 1e-3,
) -> List[Dict[str, str]]:
    names = list(outputs)
    if len(names) < 2:
        return []

    ref_name = names[0]
    ref = outputs[ref_name]
    ref_f32 = ref.float()
    rows: List[Dict[str, str]] = []
    for name in names[1:]:
        candidate = outputs[name]
        candidate_f32 = candidate.float()
        close = torch.isclose(candidate, ref, rtol=rtol, atol=atol, equal_nan=True)
        mismatched = (~close).sum().item()
        total = ref.numel()
        diff = (candidate_f32 - ref_f32).abs()
        max_abs = diff.max().item()
        rel = diff / ref_f32.abs().clamp_min(1e-30)
        max_rel = rel.max().item()
        rows.append(
            {
                "m": str(m),
                "reference": ref_name,
                "candidate": name,
                "matched": str(mismatched == 0),
                "mismatched_elements": str(mismatched),
                "total_elements": str(total),
                "mismatch_pct": f"{(mismatched / total) * 100.0:.6f}",
                "max_abs_diff": f"{max_abs:.6f}",
                "max_rel_diff": f"{max_rel:.6f}",
                "rtol": str(rtol),
                "atol": str(atol),
            }
        )
    return rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_check_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "m",
        "reference",
        "candidate",
        "matched",
        "mismatched_elements",
        "total_elements",
        "mismatch_pct",
        "max_abs_diff",
        "max_rel_diff",
        "rtol",
        "atol",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

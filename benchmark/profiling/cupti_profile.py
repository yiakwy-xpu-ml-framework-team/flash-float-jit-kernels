from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from workload import (
    DEFAULT_SCENARIOS,
    SEED,
    make_provider,
    parse_csv_ints,
    parse_csv_strings,
    prepare_inputs,
    scenario_config,
    summarize_output_checks,
    timed_calls,
    triton_ffi_mode,
    warmup,
    write_check_csv,
    write_json,
)
from cupti_activity import CuptiActivityCollector
from cupti_range_profiler import (
    CuptiRangeProfiler,
    parse_metric_csv,
    safe_metric_filename,
    write_metric_error,
)
from visualize_cupti_profile import write_report


def summarize_activity(activity_path: Path) -> List[Dict[str, str]]:
    if not activity_path.exists():
        return []

    groups: Dict[tuple[str, str], Dict[str, float]] = defaultdict(
        lambda: {
            "count": 0,
            "total_us": 0.0,
            "max_us": 0.0,
            "min_us": float("inf"),
        }
    )
    with activity_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record_type = row["record_type"]
            name = row["name"] or f"cbid_{row['cbid']}"
            key = (record_type, name)
            duration_us = float(row["duration_us"])
            group = groups[key]
            group["count"] += 1
            group["total_us"] += duration_us
            group["max_us"] = max(group["max_us"], duration_us)
            group["min_us"] = min(group["min_us"], duration_us)

    rows: List[Dict[str, str]] = []
    for (record_type, name), values in sorted(
        groups.items(),
        key=lambda item: item[1]["total_us"],
        reverse=True,
    ):
        count = int(values["count"])
        total_us = values["total_us"]
        rows.append(
            {
                "record_type": record_type,
                "name": name,
                "count": str(count),
                "total_us": f"{total_us:.3f}",
                "avg_us": f"{(total_us / count) if count else 0.0:.3f}",
                "min_us": f"{values['min_us'] if count else 0.0:.3f}",
                "max_us": f"{values['max_us']:.3f}",
            }
        )
    return rows


def write_activity_summary(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "record_type",
        "name",
        "count",
        "total_us",
        "avg_us",
        "min_us",
        "max_us",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
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
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def range_metrics_complete(path: Path) -> bool:
    rows = read_rows(path)
    if not rows:
        return False
    return all(row.get("all_passes_submitted", "1") == "1" for row in rows)


def run_scenario_with_cupti(
    m: int,
    config: Dict[str, object],
    tensors: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    config_dir: Path,
    activity_dir: Path,
    collector: CuptiActivityCollector,
) -> tuple[Dict[str, str], torch.Tensor, Path, Path]:
    scenario = str(config["scenario"])
    provider = str(config["provider"])
    warmup_iters = int(config["warmup"])
    timed_iters = int(config["iters"])
    use_tvm_ffi = config["triton_use_tvm_ffi"]

    out = torch.empty((m, m), device=args.device, dtype=torch.float16)
    fn = make_provider(provider, tensors, out)

    config_with_shape = dict(config)
    config_with_shape["m"] = m
    activity_path = activity_dir / f"{scenario}_m{m}_activity.csv"
    summary_path = activity_dir / f"{scenario}_m{m}_activity_summary.csv"
    config_with_shape["cupti_activity_path"] = str(activity_path)
    config_with_shape["cupti_summary_path"] = str(summary_path)
    write_json(config_dir / f"{scenario}_m{m}.json", config_with_shape)

    with triton_ffi_mode(use_tvm_ffi if isinstance(use_tvm_ffi, bool) else None):
        if warmup_iters > 0:
            warmup(fn, warmup_iters)

        if args.poison_output:
            out.fill_(args.poison_value)
            torch.cuda.synchronize()

        collector.start(
            activity_path,
            runtime=args.runtime,
            driver=args.driver,
            memcpy=args.memcpy,
            memset=args.memset,
            latency_timestamps=args.latency_timestamps,
        )
        try:
            timings = timed_calls(fn, timed_iters)
        finally:
            collector.stop()

    sentinel_remaining = ""
    if args.poison_output:
        sentinel_remaining = str((out == args.poison_value).sum().item())

    row = {
        "scenario": scenario,
        "provider": provider,
        "launcher": str(config["launcher"]),
        "m": str(m),
        "warmup": str(warmup_iters),
        "iters": str(timed_iters),
        "triton_use_tvm_ffi": str(use_tvm_ffi),
        "wall_latency_ms": f"{timings['wall_latency_ms']:.6f}",
        "event_latency_ms": f"{timings['event_latency_ms']:.6f}",
        "poison_value": str(args.poison_value) if args.poison_output else "",
        "sentinel_remaining": sentinel_remaining,
    }
    row["cupti_activity_path"] = str(activity_path)
    row["cupti_summary_path"] = str(summary_path)
    row["cupti_range_path"] = ""

    summary_rows = summarize_activity(activity_path)
    write_activity_summary(summary_path, summary_rows)
    return row, out.detach().clone(), activity_path, summary_path


def collect_range_counters_once(
    profiler: CuptiRangeProfiler,
    output_path: Path,
    range_name: str,
    metrics: List[str],
    fn,
) -> None:
    profiler.profile(output_path, range_name, metrics, fn)


def collect_range_counters_with_fallback(
    profiler: CuptiRangeProfiler,
    output_path: Path,
    range_name: str,
    metrics: List[str],
    fn,
    *,
    fallback_individual: bool,
) -> None:
    try:
        collect_range_counters_once(profiler, output_path, range_name, metrics, fn)
        if range_metrics_complete(output_path):
            return
        if not fallback_individual:
            return
        print(
            "CUPTI range grouped metrics did not submit all passes; "
            "retrying metrics individually."
        )
    except RuntimeError as exc:
        if not fallback_individual:
            write_metric_error(output_path, range_name, metrics, str(exc))
            raise
        print(
            "CUPTI range grouped metrics failed; retrying metrics individually: "
            f"{exc}"
        )

    rows: List[Dict[str, str]] = []
    parts_dir = output_path.parent / f"{output_path.stem}_metric_parts"
    for metric in metrics:
        metric_path = parts_dir / f"{safe_metric_filename(metric)}.csv"
        try:
            collect_range_counters_once(profiler, metric_path, range_name, [metric], fn)
            metric_rows = read_rows(metric_path)
            if metric_rows:
                for row in metric_rows:
                    row.setdefault("error", "")
                rows.extend(metric_rows)
            else:
                rows.append(
                    {
                        "range_index": "0",
                        "range_name": range_name,
                        "metric": metric,
                        "value": "",
                        "chip_name": "",
                        "num_passes": "",
                        "all_passes_submitted": "0",
                        "error": "no rows returned",
                    }
                )
        except RuntimeError as exc:
            rows.append(
                {
                    "range_index": "0",
                    "range_name": range_name,
                    "metric": metric,
                    "value": "",
                    "chip_name": "",
                    "num_passes": "",
                    "all_passes_submitted": "0",
                    "error": str(exc),
                }
            )
    write_rows(output_path, rows)


def run_scenario_range_counters(
    m: int,
    config: Dict[str, object],
    tensors: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    counter_dir: Path,
    profiler: CuptiRangeProfiler,
) -> Path:
    scenario = str(config["scenario"])
    provider = str(config["provider"])
    warmup_iters = int(config["warmup"])
    use_tvm_ffi = config["triton_use_tvm_ffi"]

    out = torch.empty((m, m), device=args.device, dtype=torch.float16)
    fn = make_provider(provider, tensors, out)
    metrics = parse_metric_csv(args.range_metrics)
    output_path = counter_dir / f"{scenario}_m{m}_range_metrics.csv"
    range_name = f"{scenario}_m{m}"

    with triton_ffi_mode(use_tvm_ffi if isinstance(use_tvm_ffi, bool) else None):
        if warmup_iters > 0:
            warmup(fn, warmup_iters)
        collect_range_counters_with_fallback(
            profiler,
            output_path,
            range_name,
            metrics,
            fn,
            fallback_individual=args.range_fallback_individual,
        )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the native Triton/CUDA comparison under a direct CUPTI Activity "
            "collector. Each scenario gets a separate activity CSV."
        )
    )
    parser.add_argument("--m", default="4096", help="Comma-separated M values.")
    parser.add_argument(
        "--scenarios",
        default="cuda_warm,triton_symm_native_warm,triton_moun_native_warm",
        help="Comma-separated scenarios defined in workload.py.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--cold-iters", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark/profiling/traces/cupti_native_triton_cuda"),
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--strict-check", action="store_true")
    parser.add_argument(
        "--no-poison-output",
        action="store_false",
        dest="poison_output",
        help="Do not fill output with a sentinel before timed calls.",
    )
    parser.add_argument("--poison-value", type=float, default=12345.0)
    parser.add_argument("--no-runtime", action="store_false", dest="runtime")
    parser.add_argument("--no-driver", action="store_false", dest="driver")
    parser.add_argument("--no-memcpy", action="store_false", dest="memcpy")
    parser.add_argument("--no-memset", action="store_false", dest="memset")
    parser.add_argument(
        "--latency-timestamps",
        action="store_true",
        help=(
            "Reserved for CUPTI latency timestamp support. The current collector "
            "records queued/submitted fields when the CUPTI activity record provides them."
        ),
    )
    parser.add_argument(
        "--force-rebuild-cupti",
        action="store_true",
        help="Rebuild the local CUPTI collector shared library.",
    )
    parser.add_argument(
        "--range-counters",
        action="store_true",
        help=(
            "Run an additional CUPTI Range Profiling pass to collect hardware "
            "counters. The counter pass is separate from benchmark timing."
        ),
    )
    parser.add_argument(
        "--range-metrics",
        default=None,
        help=(
            "Comma-separated CUPTI/Nsight Compute metric names. Defaults to a "
            "small roofline-oriented metric set."
        ),
    )
    parser.add_argument(
        "--force-rebuild-cupti-range",
        action="store_true",
        help="Rebuild the local CUPTI Range Profiler shared library.",
    )
    parser.add_argument(
        "--no-range-fallback-individual",
        action="store_false",
        dest="range_fallback_individual",
        help=(
            "Do not retry metrics one-by-one if the grouped Range Profiler "
            "configuration fails."
        ),
    )
    parser.add_argument(
        "--no-visualize",
        action="store_false",
        dest="visualize",
        help="Do not generate report.html after profiling.",
    )
    parser.set_defaults(
        poison_output=True,
        runtime=True,
        driver=True,
        memcpy=True,
        memset=True,
        visualize=True,
        range_fallback_individual=True,
    )
    return parser


def write_cupti_latency_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "provider",
        "launcher",
        "m",
        "warmup",
        "iters",
        "triton_use_tvm_ffi",
        "wall_latency_ms",
        "event_latency_ms",
        "poison_value",
        "sentinel_remaining",
        "cupti_activity_path",
        "cupti_summary_path",
        "cupti_range_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / run_name
    config_dir = run_dir / "configs"
    activity_dir = run_dir / "activities"
    counter_dir = run_dir / "range_counters"

    shapes = parse_csv_ints(args.m)
    scenario_names = parse_csv_strings(args.scenarios)
    known = set(DEFAULT_SCENARIOS)
    unknown = [name for name in scenario_names if name not in known]
    if unknown:
        raise ValueError(f"Unknown scenario(s): {unknown}")
    configs = [scenario_config(name, args) for name in scenario_names]

    write_json(
        run_dir / "run_config.json",
        {
            "m": shapes,
            "scenarios": configs,
            "seed": SEED,
            "device": args.device,
            "poison_output": args.poison_output,
            "poison_value": args.poison_value,
            "cupti_activity_kinds": {
                "runtime": args.runtime,
                "driver": args.driver,
                "kernel": True,
                "memcpy": args.memcpy,
                "memset": args.memset,
                "latency_timestamps": args.latency_timestamps,
            },
            "cupti_range_counters": {
                "enabled": args.range_counters,
                "metrics": parse_metric_csv(args.range_metrics),
                "fallback_individual": args.range_fallback_individual,
            },
        },
    )

    collector = CuptiActivityCollector(force_rebuild=args.force_rebuild_cupti)
    range_profiler = (
        CuptiRangeProfiler(force_rebuild=args.force_rebuild_cupti_range)
        if args.range_counters
        else None
    )
    rows: List[Dict[str, str]] = []
    outputs_by_shape: Dict[int, Dict[str, torch.Tensor]] = {}

    for m in shapes:
        tensors = prepare_inputs(m, args.device)
        for config in configs:
            row, output, activity_path, summary_path = run_scenario_with_cupti(
                m,
                config,
                tensors,
                args,
                config_dir,
                activity_dir,
                collector,
            )
            rows.append(row)
            if args.check:
                outputs_by_shape.setdefault(m, {})[row["scenario"]] = output
            print(
                f"{row['scenario']:>25} m={m} "
                f"event={row['event_latency_ms']} ms "
                f"activity={activity_path} "
                f"summary={summary_path}"
            )
            if range_profiler is not None:
                range_path = run_scenario_range_counters(
                    m,
                    config,
                    tensors,
                    args,
                    counter_dir,
                    range_profiler,
                )
                row["cupti_range_path"] = str(range_path)
                print(
                    f"{row['scenario']:>25} m={m} "
                    f"range_counters={range_path}"
                )

    latency_path = run_dir / "latency.csv"
    write_cupti_latency_csv(latency_path, rows)

    check_rows: List[Dict[str, str]] = []
    if args.check:
        for m, outputs in outputs_by_shape.items():
            check_rows.extend(summarize_output_checks(m, outputs))
        write_check_csv(run_dir / "check_summary.csv", check_rows)

    print(f"Wrote run config: {run_dir / 'run_config.json'}")
    print(f"Wrote latency CSV: {latency_path}")
    if args.check:
        print(f"Wrote check summary: {run_dir / 'check_summary.csv'}")
    if args.visualize:
        report_path = write_report(run_dir)
        print(f"Wrote HTML report: {report_path}")

    failed_checks = [row for row in check_rows if row["matched"] != "True"]
    for row in failed_checks:
        print(
            "CHECK WARNING: "
            f"{row['candidate']} differs from {row['reference']} "
            f"for m={row['m']} with {row['mismatched_elements']}/"
            f"{row['total_elements']} mismatches, "
            f"max_abs={row['max_abs_diff']}, max_rel={row['max_rel_diff']}"
        )
    if args.strict_check and failed_checks:
        raise AssertionError(
            f"{len(failed_checks)} output check(s) failed. "
            f"See {run_dir / 'check_summary.csv'}"
        )


if __name__ == "__main__":
    main()

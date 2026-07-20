from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jit_kernel.thunder_moun import (  # noqa: E402
    CUDA_PROFILER_HEADER_U64_WORDS,
    CUDA_PROFILER_RECORD_U64_WORDS,
    allocate_cuda_profiler_buffer,
    estimate_cuda_profiler_records,
    symm_gemm_block_scaled,
)


SEED = 42
BLOCK_SIZE = 128

EVENT_KIND_INSTANT = 0
EVENT_KIND_BEGIN = 1
EVENT_KIND_END = 2

EVENT_NAMES = {
    1: "kernel_enter",
    2: "pipeline_enter",
    10: "task",
    11: "task_map",
    20: "prefetch_tma",
    21: "scale_x_load",
    22: "scale_w_load",
    30: "tma_wait",
    31: "mma_issue",
    32: "producer_load_once",
    33: "wgmma_wait",
    34: "scale_apply_accum",
    40: "epilogue_smem_store",
    41: "splitk_reduce",
    42: "store_lower",
    43: "mirror_transpose",
    44: "mirror_store",
    50: "task_done",
}


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

    xq = x
    wq = x
    x_fp8 = xq.to(torch.float8_e4m3fn)
    w_fp8 = wq.to(torch.float8_e4m3fn)
    return x_fp8, w_fp8, xs_lhs, xs_rhs


def time_cuda_path(
    fn: Callable[[], Any],
    warmup: int,
    iters: int,
) -> Tuple[float, float, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start = time.time()
    start_event.record()
    last_result = None
    for _ in range(iters):
        last_result = fn()
    end_event.record()
    torch.cuda.synchronize()

    host_ms = (time.time() - start) / iters * 1000
    device_ms = start_event.elapsed_time(end_event) / iters
    return host_ms, device_ms, last_result


def _u32_lo(value: int) -> int:
    return value & 0xFFFFFFFF


def _u32_hi(value: int) -> int:
    return (value >> 32) & 0xFFFFFFFF


def _u16_lo(value: int) -> int:
    return value & 0xFFFF


def _u16_hi(value: int) -> int:
    return (value >> 16) & 0xFFFF


def decode_profiler_records(
    profiler_buffer: torch.Tensor,
) -> Tuple[Dict[str, Any], List[Dict[str, int]]]:
    words = [int(v) for v in profiler_buffer.detach().cpu().tolist()]
    if len(words) < CUDA_PROFILER_HEADER_U64_WORDS:
        raise ValueError("profiler buffer is smaller than the profiler header")

    header_word0 = words[0]
    header_word1 = words[1]
    header_word2 = words[2]
    header_word3 = words[3]
    grid_xy = _u32_hi(header_word0)
    grid_x = _u16_lo(grid_xy)
    grid_y = _u16_hi(grid_xy)
    header = {
        "capacity": _u32_lo(header_word0),
        "grid_x": grid_x,
        "grid_y": grid_y,
        "num_ctas": grid_x * grid_y,
        "records_per_cta": _u32_lo(header_word1),
        "records_per_task": _u32_hi(header_word1),
        "max_tasks_per_cta": _u32_lo(header_word2),
        "max_k_tiles_per_task": _u32_hi(header_word2),
        "cta_slots": _u32_lo(header_word3),
        "per_k_slots": _u32_hi(header_word3),
    }
    header["required_capacity"] = header["num_ctas"] * header["records_per_cta"]
    header["static_slots"] = (
        header["records_per_task"]
        - header["max_k_tiles_per_task"] * header["per_k_slots"]
    )

    record_count = min(header["capacity"], header["required_capacity"])
    record_offset = CUDA_PROFILER_HEADER_U64_WORDS
    record_words = words[
        record_offset : record_offset + record_count * CUDA_PROFILER_RECORD_U64_WORDS
    ]

    records = []
    empty_slots = 0
    for slot_idx in range(record_count):
        i = slot_idx * CUDA_PROFILER_RECORD_U64_WORDS
        word0, word1 = record_words[i : i + CUDA_PROFILER_RECORD_U64_WORDS]
        if word0 == 0 and word1 == 0:
            empty_slots += 1
            continue

        payload = _u32_lo(word1)
        tag = (word1 >> 32) & 0xFFFF
        smid = (word1 >> 48) & 0xFFFF
        event_id = tag & 0xFF
        if event_id == 0:
            empty_slots += 1
            continue

        kind = (tag >> 8) & 0x3
        flags = (tag >> 10) & 0x3F
        cta_id = slot_idx // header["records_per_cta"]
        slot_in_cta = slot_idx % header["records_per_cta"]
        block_x = cta_id % header["grid_x"] if header["grid_x"] else 0
        block_y = (cta_id // header["grid_x"]) if header["grid_x"] else 0
        task_iter = -1
        k_iter = -1
        slot_scope = "cta"
        event_slot = slot_in_cta

        if slot_in_cta >= header["cta_slots"]:
            slot_scope = "task"
            task_region = slot_in_cta - header["cta_slots"]
            task_iter = task_region // header["records_per_task"]
            slot_in_task = task_region % header["records_per_task"]
            event_slot = slot_in_task
            if slot_in_task >= header["static_slots"]:
                slot_scope = "k"
                k_region = slot_in_task - header["static_slots"]
                k_iter = k_region // header["per_k_slots"]
                event_slot = k_region % header["per_k_slots"]

        records.append(
            {
                "timestamp": word0,
                "event_id": event_id,
                "kind": kind,
                "flags": flags,
                "cta_id": cta_id,
                "block_x": block_x,
                "block_y": block_y,
                "task_iter": task_iter,
                "task_id": (
                    block_y + task_iter * header["grid_y"]
                    if task_iter >= 0
                    else -1
                ),
                "k_iter": k_iter,
                "slot_scope": slot_scope,
                "event_slot": event_slot,
                "thread_id": 0,
                "sm_id": smid,
                "payload": payload,
            }
        )

    header["slots_scanned"] = record_count
    header["empty_slots"] = empty_slots
    header["unused_capacity"] = max(0, header["capacity"] - record_count)
    return header, records


def _duration_stats(durations: List[int]) -> Dict[str, float]:
    if not durations:
        return {
            "count": 0,
            "total_us": 0.0,
            "mean_us": 0.0,
            "p50_us": 0.0,
            "p90_us": 0.0,
            "p99_us": 0.0,
            "max_us": 0.0,
        }

    sorted_values = sorted(durations)

    def percentile(q: float) -> float:
        idx = min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * q)))
        return sorted_values[idx] / 1000.0

    total = sum(sorted_values)
    return {
        "count": len(sorted_values),
        "total_us": total / 1000.0,
        "mean_us": (total / len(sorted_values)) / 1000.0,
        "p50_us": percentile(0.50),
        "p90_us": percentile(0.90),
        "p99_us": percentile(0.99),
        "max_us": sorted_values[-1] / 1000.0,
    }


def build_profiler_report(profiler_buffer: torch.Tensor) -> Dict[str, Any]:
    header, records = decode_profiler_records(profiler_buffer)

    open_events = defaultdict(list)
    durations_by_event = defaultdict(list)
    instant_counts = defaultdict(int)
    unmatched_end = defaultdict(int)

    for record in records:
        event_id = record["event_id"]
        key = (
            event_id,
            record["cta_id"],
            record["task_iter"],
            record["k_iter"],
            record["sm_id"],
        )

        if record["kind"] == EVENT_KIND_BEGIN:
            open_events[key].append(record)
        elif record["kind"] == EVENT_KIND_END:
            if open_events[key]:
                begin = open_events[key].pop()
                duration = max(0, record["timestamp"] - begin["timestamp"])
                durations_by_event[event_id].append(duration)
            else:
                unmatched_end[event_id] += 1
        elif record["kind"] == EVENT_KIND_INSTANT:
            instant_counts[event_id] += 1

    unmatched_begin = defaultdict(int)
    for key, stack in open_events.items():
        if stack:
            unmatched_begin[key[0]] += len(stack)

    event_summaries = {}
    for event_id, durations in sorted(durations_by_event.items()):
        event_summaries[EVENT_NAMES.get(event_id, f"event_{event_id}")] = (
            _duration_stats(durations)
        )

    return {
        "header": header,
        "records": {
            "parsed": len(records),
            "slots_scanned": header["slots_scanned"],
            "capacity": header["capacity"],
            "required": header["required_capacity"],
            "empty_slots": header["empty_slots"],
            "unused_capacity": header["unused_capacity"],
            "truncated": header["capacity"] < header["required_capacity"],
        },
        "events": event_summaries,
        "instant_counts": {
            EVENT_NAMES.get(event_id, f"event_{event_id}"): count
            for event_id, count in sorted(instant_counts.items())
        },
        "unmatched_begin": {
            EVENT_NAMES.get(event_id, f"event_{event_id}"): count
            for event_id, count in sorted(unmatched_begin.items())
        },
        "unmatched_end": {
            EVENT_NAMES.get(event_id, f"event_{event_id}"): count
            for event_id, count in sorted(unmatched_end.items())
        },
    }


def _record_label(record: Dict[str, int]) -> str:
    if record["slot_scope"] == "k":
        return (
            f"cta={record['cta_id']} task={record['task_iter']} "
            f"k={record['k_iter']} sm={record['sm_id']}"
        )
    if record["slot_scope"] == "task":
        return f"cta={record['cta_id']} task={record['task_iter']} sm={record['sm_id']}"
    return f"cta={record['cta_id']} sm={record['sm_id']}"


def _trace_thread_id(record: Dict[str, int], lane_mode: str) -> str:
    if lane_mode == "cta":
        return f"cta{record['cta_id']}"
    if record["task_iter"] >= 0:
        return f"cta{record['cta_id']}:task{record['task_iter']}"
    return f"cta{record['cta_id']}:meta"


def _cta_trace_path(trace_path: Path) -> Path:
    return trace_path.with_name(f"{trace_path.stem}_by_cta{trace_path.suffix}")


def export_chrome_trace(
    profiler_buffer: torch.Tensor,
    trace_path: Path,
    lane_mode: str = "task",
) -> Dict[str, int]:
    if lane_mode not in {"task", "cta"}:
        raise ValueError(f"Unsupported trace lane mode: {lane_mode}")

    _header, records = decode_profiler_records(profiler_buffer)
    metadata = {
        "source": "ThunderMoun CUDA embedded profiler",
        "lane_mode": lane_mode,
    }
    if not records:
        trace = {
            "traceEvents": [],
            "displayTimeUnit": "ns",
            "metadata": metadata,
        }
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
        return {"events": 0, "duration_events": 0, "instant_events": 0}

    base_ts = min(record["timestamp"] for record in records)
    pid = f"ThunderMoun CUDA ({lane_mode} lanes)"
    trace_events = []
    open_events = defaultdict(list)
    instant_events = 0
    duration_events = 0

    for record in sorted(records, key=lambda item: item["timestamp"]):
        event_id = record["event_id"]
        name = EVENT_NAMES.get(event_id, f"event_{event_id}")
        tid = _trace_thread_id(record, lane_mode)
        ts_us = (record["timestamp"] - base_ts) / 1000.0
        args = {
            "cta_id": record["cta_id"],
            "block_x": record["block_x"],
            "block_y": record["block_y"],
            "task_iter": record["task_iter"],
            "task_id": record["task_id"],
            "k_iter": record["k_iter"],
            "sm_id": record["sm_id"],
            "payload": record["payload"],
            "scope": record["slot_scope"],
            "slot": record["event_slot"],
        }

        key = (
            event_id,
            record["cta_id"],
            record["task_iter"],
            record["k_iter"],
            record["sm_id"],
        )
        if record["kind"] == EVENT_KIND_BEGIN:
            open_events[key].append(record)
        elif record["kind"] == EVENT_KIND_END:
            if not open_events[key]:
                continue
            begin = open_events[key].pop()
            begin_ts_us = (begin["timestamp"] - base_ts) / 1000.0
            dur_us = max(0.0, (record["timestamp"] - begin["timestamp"]) / 1000.0)
            args["begin_payload"] = begin["payload"]
            args["end_payload"] = record["payload"]
            trace_events.append(
                {
                    "name": name,
                    "cat": record["slot_scope"],
                    "ph": "X",
                    "ts": begin_ts_us,
                    "dur": dur_us,
                    "pid": pid,
                    "tid": tid,
                    "args": args,
                }
            )
            duration_events += 1
        elif record["kind"] == EVENT_KIND_INSTANT:
            trace_events.append(
                {
                    "name": name,
                    "cat": record["slot_scope"],
                    "ph": "i",
                    "s": "t",
                    "ts": ts_us,
                    "pid": pid,
                    "tid": tid,
                    "args": args,
                }
            )
            instant_events += 1

    trace_events.sort(key=lambda item: (item["ts"], str(item["tid"]), item["name"]))
    trace = {
        "traceEvents": trace_events,
        "displayTimeUnit": "ns",
        "metadata": metadata,
    }
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return {
        "events": len(trace_events),
        "duration_events": duration_events,
        "instant_events": instant_events,
    }


def print_profiler_report(report: Dict[str, Any], top_k: int = 20) -> None:
    records = report["records"]
    print("\nProfiler records:")
    print(
        f"  parsed={records['parsed']} slots_scanned={records['slots_scanned']} "
        f"capacity={records['capacity']} required={records['required']} "
        f"empty={records['empty_slots']} truncated={records['truncated']}"
    )

    if report["instant_counts"]:
        print("  instant events:")
        for name, count in report["instant_counts"].items():
            print(f"    {name:<24} {count}")

    rows = sorted(
        report["events"].items(),
        key=lambda item: item[1]["total_us"],
        reverse=True,
    )
    print("\nProfiler summary (globaltimer ticks treated as ns):")
    print(
        "  "
        + f"{'event':<24} {'count':>8} {'total_us':>12} {'mean_us':>10} "
        + f"{'p50_us':>10} {'p90_us':>10} {'p99_us':>10} {'max_us':>10}"
    )
    for name, stats in rows[:top_k]:
        print(
            "  "
            + f"{name:<24} {stats['count']:>8} {stats['total_us']:>12.3f} "
            + f"{stats['mean_us']:>10.3f} {stats['p50_us']:>10.3f} "
            + f"{stats['p90_us']:>10.3f} {stats['p99_us']:>10.3f} "
            + f"{stats['max_us']:>10.3f}"
        )

    if report["unmatched_begin"] or report["unmatched_end"]:
        print("\nProfiler pairing warnings:")
        for name, count in report["unmatched_begin"].items():
            print(f"  unmatched begin {name}: {count}")
        for name, count in report["unmatched_end"].items():
            print(f"  unmatched end   {name}: {count}")


def bench_cuda_symm_gemm(
    m: int = 4096,
    warmup: int = 25,
    iters: int = 100,
    max_profiler_records: Optional[int] = None,
    trace_json: Optional[Path] = None,
) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    x_fp8, w_fp8, xs_lhs, xs_rhs = make_inputs(m)
    profiler_capacity = estimate_cuda_profiler_records(m, m, m)
    if max_profiler_records is not None:
        profiler_capacity = min(profiler_capacity, max_profiler_records)
    profiler_buffer = allocate_cuda_profiler_buffer(profiler_capacity, x_fp8.device)

    baseline_fn = lambda: symm_gemm_block_scaled(x_fp8, w_fp8, xs_lhs, xs_rhs)
    profile_fn = lambda: symm_gemm_block_scaled(
        x_fp8,
        w_fp8,
        xs_lhs,
        xs_rhs,
        enable_cuda_profiler=True,
        profiler_buffer=profiler_buffer,
    )

    baseline_host_ms, baseline_device_ms, _ = time_cuda_path(
        baseline_fn, warmup, iters
    )
    profile_host_ms, profile_device_ms, _ = time_cuda_path(profile_fn, warmup, iters)

    report = build_profiler_report(profiler_buffer)
    report["timing"] = {
        "baseline_host_ms": baseline_host_ms,
        "baseline_device_ms": baseline_device_ms,
        "profile_host_ms": profile_host_ms,
        "profile_device_ms": profile_device_ms,
        "profile_device_overhead_ms": profile_device_ms - baseline_device_ms,
        "profile_device_overhead_pct": (
            (profile_device_ms / baseline_device_ms - 1.0) * 100.0
            if baseline_device_ms > 0
            else 0.0
        ),
    }

    print(f"\nCUDA ThunderMoun Symm GEMM: M=N=K={m}")
    print(f"warmup={warmup}, iters={iters}")
    print(
        f"Cuda Muon Symm Gemm          : {baseline_host_ms:.3f} ms, "
        f"device {baseline_device_ms:.3f} ms"
    )
    print(
        f"Cuda Muon Symm Gemm Profiled : {profile_host_ms:.3f} ms, "
        f"device {profile_device_ms:.3f} ms"
    )
    print(
        "Profiler overhead            : "
        f"{report['timing']['profile_device_overhead_ms']:.3f} ms, "
        f"{report['timing']['profile_device_overhead_pct']:.2f}% device"
    )
    print_profiler_report(report)
    if trace_json is not None:
        trace_stats = export_chrome_trace(profiler_buffer, trace_json, lane_mode="task")
        cta_trace_json = _cta_trace_path(trace_json)
        cta_trace_stats = export_chrome_trace(
            profiler_buffer, cta_trace_json, lane_mode="cta"
        )
        report["trace"] = {
            "path": str(trace_json),
            **trace_stats,
        }
        report["trace_by_cta"] = {
            "path": str(cta_trace_json),
            **cta_trace_stats,
        }
        print(
            "\nChrome trace: "
            f"{trace_json} "
            f"({trace_stats['duration_events']} durations, "
            f"{trace_stats['instant_events']} instants)"
        )
        print(
            "Chrome trace by CTA: "
            f"{cta_trace_json} "
            f"({cta_trace_stats['duration_events']} durations, "
            f"{cta_trace_stats['instant_events']} instants)"
        )

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ThunderMoun CUDA path.")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--max-profiler-records", type=int, default=None)
    parser.add_argument(
        "--trace-json",
        type=Path,
        default=None,
        help="Write Chrome Trace JSON for chrome://tracing or Perfetto.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bench_cuda_symm_gemm(
        m=args.m,
        warmup=args.warmup,
        iters=args.iters,
        max_profiler_records=args.max_profiler_records,
        trace_json=args.trace_json,
    )

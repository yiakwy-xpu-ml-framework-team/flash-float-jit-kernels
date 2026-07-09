from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_ROOT = Path("benchmark/profiling/traces/cupti_native_triton_cuda")
STALL_METRICS = (
    ("long_scoreboard", "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"),
    ("short_scoreboard", "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct"),
    ("mio_throttle", "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct"),
    ("math_pipe_throttle", "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"),
    ("barrier", "smsp__warp_issue_stalled_barrier_per_warp_active.pct"),
    ("not_selected", "smsp__warp_issue_stalled_not_selected_per_warp_active.pct"),
    ("no_instruction", "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct"),
    ("wait", "smsp__warp_issue_stalled_wait_per_warp_active.pct"),
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def latest_run(root: Path) -> Path:
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No CUPTI profiling runs found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def numeric(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def resolve_profile_path(run_dir: Path, path_text: str, fallback: Path) -> Path:
    if path_text:
        path = Path(path_text)
        if path.is_absolute():
            candidates = [path]
        else:
            candidates = [Path.cwd() / path, run_dir / path, run_dir.parent / path]
    else:
        candidates = []

    candidates.append(fallback)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def activity_path_for(run_dir: Path, row: Dict[str, str]) -> Path:
    fallback = run_dir / "activities" / f"{row['scenario']}_m{row['m']}_activity.csv"
    return resolve_profile_path(run_dir, row.get("cupti_activity_path", ""), fallback)


def summary_path_for(run_dir: Path, row: Dict[str, str]) -> Path:
    fallback = run_dir / "activities" / f"{row['scenario']}_m{row['m']}_activity_summary.csv"
    return resolve_profile_path(run_dir, row.get("cupti_summary_path", ""), fallback)


def range_path_for(run_dir: Path, row: Dict[str, str]) -> Path:
    fallback = run_dir / "range_counters" / f"{row['scenario']}_m{row['m']}_range_metrics.csv"
    return resolve_profile_path(run_dir, row.get("cupti_range_path", ""), fallback)


def table(headers: Iterable[str], rows: Iterable[Dict[str, str]]) -> str:
    header_cells = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{esc(row.get(header, ''))}</td>" for header in headers)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header_cells}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def latency_cards(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "<p>No latency.csv found.</p>"

    max_event = max(numeric(row.get("event_latency_ms", "0")) for row in rows) or 1.0
    cards = []
    for row in rows:
        scenario = row["scenario"]
        event_ms = numeric(row["event_latency_ms"])
        wall_ms = numeric(row["wall_latency_ms"])
        width = max(2.0, event_ms / max_event * 100.0)
        sentinel = row.get("sentinel_remaining", "")
        sentinel_class = "ok" if sentinel in ("", "0") else "bad"
        cards.append(
            f"""
            <section class="card">
              <div class="card-title">{esc(scenario)}</div>
              <div class="metric"><span>event</span><strong>{event_ms:.4f} ms</strong></div>
              <div class="bar"><div style="width:{width:.2f}%"></div></div>
              <div class="metric"><span>wall</span><strong>{wall_ms:.4f} ms</strong></div>
              <div class="metric"><span>sentinel</span><strong class="{sentinel_class}">{esc(sentinel)}</strong></div>
            </section>
            """
        )
    return f"<div class=\"cards\">{''.join(cards)}</div>"


def summarize_kernel_configs(run_dir: Path, latency_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    rows = []
    for latency_row in latency_rows:
        activity_path = activity_path_for(run_dir, latency_row)
        kernel_rows = [
            row for row in read_csv(activity_path)
            if row.get("record_type") == "kernel"
        ]
        groups: Dict[Tuple[str, str, str, str, str, str, str], Dict[str, object]] = {}
        for kernel_row in kernel_rows:
            key = (
                kernel_row.get("name", ""),
                kernel_row.get("grid_x", ""),
                kernel_row.get("grid_y", ""),
                kernel_row.get("grid_z", ""),
                kernel_row.get("block_x", ""),
                kernel_row.get("block_y", ""),
                kernel_row.get("block_z", ""),
            )
            group = groups.setdefault(
                key,
                {
                    "first": kernel_row,
                    "count": 0,
                    "total_us": 0.0,
                    "min_us": math.inf,
                    "max_us": 0.0,
                },
            )
            duration_us = numeric(kernel_row.get("duration_us", "0"))
            group["count"] = int(group["count"]) + 1
            group["total_us"] = float(group["total_us"]) + duration_us
            group["min_us"] = min(float(group["min_us"]), duration_us)
            group["max_us"] = max(float(group["max_us"]), duration_us)

        for group in groups.values():
            first = group["first"]
            assert isinstance(first, dict)
            count = int(group["count"])
            total_us = float(group["total_us"])
            rows.append(
                {
                    "scenario": latency_row["scenario"],
                    "kernel": first.get("name", ""),
                    "count": str(count),
                    "avg_us": f"{(total_us / count) if count else 0.0:.3f}",
                    "min_us": f"{float(group['min_us']) if count else 0.0:.3f}",
                    "max_us": f"{float(group['max_us']):.3f}",
                    "grid": (
                        f"{first.get('grid_x', '')}x"
                        f"{first.get('grid_y', '')}x"
                        f"{first.get('grid_z', '')}"
                    ),
                    "block": (
                        f"{first.get('block_x', '')}x"
                        f"{first.get('block_y', '')}x"
                        f"{first.get('block_z', '')}"
                    ),
                    "regs/thread": first.get("registers_per_thread", ""),
                    "static_smem": first.get("static_smem", ""),
                    "dynamic_smem": first.get("dynamic_smem", ""),
                    "stream": first.get("stream_id", ""),
                }
            )
    return rows


def kernel_config_section(run_dir: Path, latency_rows: List[Dict[str, str]]) -> str:
    rows = summarize_kernel_configs(run_dir, latency_rows)
    if not rows:
        return "<p>No kernel activity records found.</p>"
    return table(
        [
            "scenario",
            "kernel",
            "count",
            "avg_us",
            "min_us",
            "max_us",
            "grid",
            "block",
            "regs/thread",
            "static_smem",
            "dynamic_smem",
            "stream",
        ],
        rows,
    )


def kernel_avg_us(run_dir: Path, latency_row: Dict[str, str]) -> float:
    summary_rows = read_csv(summary_path_for(run_dir, latency_row))
    for row in summary_rows:
        if row.get("record_type") == "kernel":
            return numeric(row.get("avg_us", "0"))

    kernel_rows = [
        row for row in read_csv(activity_path_for(run_dir, latency_row))
        if row.get("record_type") == "kernel"
    ]
    if kernel_rows:
        total_us = sum(numeric(row.get("duration_us", "0")) for row in kernel_rows)
        return total_us / len(kernel_rows)

    return numeric(latency_row.get("event_latency_ms", "0")) * 1000.0


def flops_for_m(m: int) -> Tuple[float, float]:
    k = m
    symmetric_flops = float(m) * float(m + 1) * float(k)
    full_equiv_flops = 2.0 * float(m) * float(m) * float(k)
    return symmetric_flops, full_equiv_flops


def metric_map(run_dir: Path, latency_row: Dict[str, str]) -> Dict[str, str]:
    rows = read_csv(range_path_for(run_dir, latency_row))
    values: Dict[str, str] = {}
    for row in rows:
        metric = row.get("metric", "")
        if not metric:
            continue
        if row.get("error"):
            values[f"{metric}::error"] = row["error"]
        values[metric] = row.get("value", "")
    return values


def metric_float(metrics: Dict[str, str], *names: str) -> Optional[float]:
    for name in names:
        value = metrics.get(name, "")
        if value == "":
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def diagnose_counter_row(
    sm_pct: Optional[float],
    dram_pct: Optional[float],
    dram_bytes: float,
    l2_bytes: Optional[float],
    top_stall: str,
) -> str:
    notes = []
    if sm_pct is not None and dram_pct is not None:
        if dram_pct > sm_pct + 15.0:
            notes.append("DRAM pressure likely")
        elif sm_pct > dram_pct + 15.0:
            notes.append("SM/compute-side pressure likely")
        else:
            notes.append("SM and DRAM pressure are both relevant")
    elif dram_bytes > 0:
        notes.append("DRAM traffic measured")

    if l2_bytes is not None and dram_bytes > 0 and l2_bytes > dram_bytes * 4.0:
        notes.append("high L2 traffic vs DRAM")
    if top_stall:
        notes.append(f"top stall: {top_stall}")

    return "; ".join(notes) if notes else "needs more counters"


def top_stall_reason(metrics: Dict[str, str]) -> str:
    best_name = ""
    best_value = -1.0
    for display_name, metric_name in STALL_METRICS:
        value = metric_float(metrics, metric_name)
        if value is not None and value > best_value:
            best_name = display_name
            best_value = value
    if best_value < 0:
        return ""
    return f"{best_name} {best_value:.1f}%"


def measured_roofline_rows(run_dir: Path, latency_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    rows = []
    for latency_row in latency_rows:
        m = int(numeric(latency_row.get("m", "0")))
        if m <= 0:
            continue

        metrics = metric_map(run_dir, latency_row)
        if not metrics:
            continue

        read_bytes = metric_float(metrics, "dram__bytes_read.sum")
        write_bytes = metric_float(metrics, "dram__bytes_write.sum")
        total_dram_bytes = metric_float(metrics, "dram__bytes.sum")
        if total_dram_bytes is None and read_bytes is not None and write_bytes is not None:
            total_dram_bytes = read_bytes + write_bytes
        if total_dram_bytes is None or total_dram_bytes <= 0.0:
            continue

        activity_us = kernel_avg_us(run_dir, latency_row)
        gpu_duration_raw = metric_float(metrics, "gpu__time_duration.sum")
        gpu_duration_us = (gpu_duration_raw / 1000.0) if gpu_duration_raw else None
        time_us = gpu_duration_us if gpu_duration_us and gpu_duration_us > 0.0 else activity_us

        symmetric_flops, full_equiv_flops = flops_for_m(m)
        seconds = time_us / 1_000_000.0
        l2_bytes = metric_float(metrics, "lts__t_bytes.sum")
        sm_pct = metric_float(metrics, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
        dram_pct = metric_float(metrics, "dram__throughput.avg.pct_of_peak_sustained_elapsed")
        top_stall = top_stall_reason(metrics)

        rows.append(
            {
                "scenario": latency_row["scenario"],
                "m": str(m),
                "time_us": f"{time_us:.3f}",
                "time_source": "gpu__time_duration.sum" if gpu_duration_us else "activity_kernel_avg",
                "symm_TFLOP/s": f"{symmetric_flops / seconds / 1e12:.2f}",
                "full_equiv_TFLOP/s": f"{full_equiv_flops / seconds / 1e12:.2f}",
                "dram_bytes_MB": f"{total_dram_bytes / 1e6:.2f}",
                "dram_TB/s": f"{total_dram_bytes / seconds / 1e12:.2f}",
                "AI_F/B": f"{symmetric_flops / total_dram_bytes:.2f}",
                "l2_bytes_MB": "" if l2_bytes is None else f"{l2_bytes / 1e6:.2f}",
                "sm_pct": "" if sm_pct is None else f"{sm_pct:.1f}",
                "dram_pct": "" if dram_pct is None else f"{dram_pct:.1f}",
                "top_stall": top_stall,
                "diagnosis": diagnose_counter_row(
                    sm_pct,
                    dram_pct,
                    total_dram_bytes,
                    l2_bytes,
                    top_stall,
                ),
            }
        )
    return rows


def roofline_section(run_dir: Path, latency_rows: List[Dict[str, str]]) -> str:
    measured_rows = measured_roofline_rows(run_dir, latency_rows)
    if not measured_rows:
        return ""

    note = (
        "This roofline uses CUPTI Range Profiler hardware counters for DRAM "
        "traffic and GPU time when available."
    )
    return (
        f"<p class=\"note\">{esc(note)}</p>"
        + table(
            [
                "scenario",
                "m",
                "time_us",
                "time_source",
                "symm_TFLOP/s",
                "full_equiv_TFLOP/s",
                "dram_bytes_MB",
                "dram_TB/s",
                "AI_F/B",
                "l2_bytes_MB",
                "sm_pct",
                "dram_pct",
                "top_stall",
                "diagnosis",
            ],
            measured_rows,
        )
    )


def counter_metric_section(run_dir: Path, latency_rows: List[Dict[str, str]]) -> str:
    rows = []
    for latency_row in latency_rows:
        range_rows = read_csv(range_path_for(run_dir, latency_row))
        for row in range_rows:
            rows.append(
                {
                    "scenario": latency_row["scenario"],
                    "range_name": row.get("range_name", ""),
                    "metric": row.get("metric", ""),
                    "value": row.get("value", ""),
                    "chip_name": row.get("chip_name", ""),
                    "num_passes": row.get("num_passes", ""),
                    "all_passes_submitted": row.get("all_passes_submitted", ""),
                    "error": row.get("error", ""),
                }
            )
    if not rows:
        return "<p>No CUPTI Range Profiler counter CSV found.</p>"
    return table(
        [
            "scenario",
            "range_name",
            "metric",
            "value",
            "chip_name",
            "num_passes",
            "all_passes_submitted",
            "error",
        ],
        rows,
    )


def roofline_block(run_dir: Path, latency_rows: List[Dict[str, str]]) -> str:
    section = roofline_section(run_dir, latency_rows)
    if not section:
        return ""
    return f"""
      <h2>Roofline</h2>
      <section class="panel">{section}</section>
    """


def counter_metric_block(run_dir: Path, latency_rows: List[Dict[str, str]]) -> str:
    section = counter_metric_section(run_dir, latency_rows)
    if "No CUPTI Range Profiler counter CSV found." in section:
        return ""
    return f"""
      <h2>CUPTI Range Counters</h2>
      <section class="panel">{section}</section>
    """


def activity_sections(run_dir: Path, latency_rows: List[Dict[str, str]], top: int) -> str:
    sections = []
    for row in latency_rows:
        scenario = row["scenario"]
        summary_path = summary_path_for(run_dir, row)
        summary_rows = read_csv(summary_path)[:top]
        if not summary_rows:
            sections.append(
                f"<section class=\"panel\"><h3>{esc(scenario)}</h3><p>No activity summary found.</p></section>"
            )
            continue

        max_total = max(numeric(item["total_us"]) for item in summary_rows) or 1.0
        bars = []
        for item in summary_rows:
            width = max(2.0, numeric(item["total_us"]) / max_total * 100.0)
            label = f"{item['record_type']}: {item['name']}"
            bars.append(
                f"""
                <div class="activity-row">
                  <div class="activity-name">{esc(label)}</div>
                  <div class="activity-bar"><div style="width:{width:.2f}%"></div></div>
                  <div class="activity-value">{esc(item['total_us'])} us / {esc(item['count'])}x</div>
                </div>
                """
            )

        sections.append(
            f"""
            <section class="panel">
              <h3>{esc(scenario)}</h3>
              {''.join(bars)}
              {table(['record_type', 'name', 'count', 'total_us', 'avg_us', 'min_us', 'max_us'], summary_rows)}
            </section>
            """
        )
    return "\n".join(sections)


def check_section(run_dir: Path) -> str:
    rows = read_csv(run_dir / "check_summary.csv")
    if not rows:
        return "<p>No check_summary.csv found.</p>"
    return table(
        [
            "reference",
            "candidate",
            "matched",
            "mismatched_elements",
            "mismatch_pct",
            "max_abs_diff",
            "max_rel_diff",
        ],
        rows,
    )


def write_report(run_dir: Path, output_path: Optional[Path] = None, top: int = 12) -> Path:
    run_dir = run_dir.resolve()
    output_path = output_path or run_dir / "report.html"
    latency_rows = read_csv(run_dir / "latency.csv")
    body = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>CUPTI Symmetric GEMM Profile</title>
      <style>
        body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 32px; color: #202124; background: #f7f8fa; }}
        h1, h2, h3 {{ margin: 0 0 14px; }}
        h1 {{ font-size: 28px; }}
        h2 {{ margin-top: 34px; font-size: 20px; }}
        .subtle {{ color: #5f6368; margin-bottom: 24px; }}
        .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }}
        .card, .panel {{ background: white; border: 1px solid #dfe3e8; border-radius: 8px; padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
        .card-title {{ font-weight: 700; margin-bottom: 12px; }}
        .metric {{ display: flex; justify-content: space-between; gap: 18px; margin: 7px 0; }}
        .metric span {{ color: #5f6368; }}
        .ok {{ color: #137333; }}
        .bad {{ color: #b3261e; }}
        .bar, .activity-bar {{ height: 10px; background: #edf0f3; border-radius: 999px; overflow: hidden; }}
        .bar div {{ height: 100%; background: #1a73e8; }}
        .activity-row {{ display: grid; grid-template-columns: minmax(220px, 1.5fr) minmax(160px, 2fr) 160px; gap: 12px; align-items: center; margin: 8px 0; }}
        .activity-name {{ overflow-wrap: anywhere; font-size: 13px; }}
        .activity-value {{ text-align: right; font-variant-numeric: tabular-nums; color: #5f6368; }}
        .activity-bar div {{ height: 100%; background: #34a853; }}
        .note {{ color: #5f6368; line-height: 1.45; margin: 0 0 12px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 14px; font-size: 13px; background: white; }}
        th, td {{ border-bottom: 1px solid #e8eaed; padding: 8px 10px; text-align: left; vertical-align: top; }}
        th {{ background: #f1f3f4; position: sticky; top: 0; }}
        td {{ overflow-wrap: anywhere; }}
        .panel {{ margin-top: 16px; overflow-x: auto; }}
      </style>
    </head>
    <body>
      <h1>CUPTI Symmetric GEMM Profile</h1>
      <div class="subtle">{esc(run_dir)}</div>

      <h2>Latency</h2>
      {latency_cards(latency_rows)}

      <h2>Kernel Launch Config</h2>
      <section class="panel">{kernel_config_section(run_dir, latency_rows)}</section>

      {roofline_block(run_dir, latency_rows)}

      {counter_metric_block(run_dir, latency_rows)}

      <h2>Correctness Check</h2>
      <section class="panel">{check_section(run_dir)}</section>

      <h2>CUPTI Activity Summary</h2>
      {activity_sections(run_dir, latency_rows, top)}
    </body>
    </html>
    """
    output_path.write_text(body, encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an HTML report for a CUPTI profiling run.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory. Defaults to the newest run under benchmark/profiling/traces/cupti_native_triton_cuda.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir or latest_run(DEFAULT_ROOT)
    report_path = write_report(run_dir, args.output, args.top)
    print(f"Wrote HTML report: {report_path}")


if __name__ == "__main__":
    main()

# CUPTI Symmetric GEMM Profiling

This folder contains the CUPTI-based profiling harness for comparing the CUDA
Thunder Moun symmetric GEMM implementation with native Triton symmetric GEMM
providers.

Quick run from the repository root:

```bash
python benchmark/profiling/init.py
```

Equivalent explicit command:

```bash
python benchmark/profiling/cupti_profile.py \
  --m 4096 \
  --scenarios cuda_warm,triton_symm_native_warm,triton_moun_native_warm \
  --warmup 10 \
  --iters 5 \
  --check \
  --range-counters
```

Force rebuild the Range Profiler collector after C++ changes:

```bash
python benchmark/profiling/cupti_profile.py \
  --m 4096 \
  --scenarios cuda_warm,triton_symm_native_warm,triton_moun_native_warm \
  --warmup 10 \
  --iters 5 \
  --check \
  --range-counters \
  --force-rebuild-cupti-range
```

Outputs are written under:

```text
benchmark/profiling/traces/cupti_native_triton_cuda/<timestamp>/
```

Important files in each run:

- `report.html`: human-readable HTML summary.
- `latency.csv`: wall-clock and CUDA-event latency by scenario.
- `check_summary.csv`: output mismatch diagnostics when `--check` is used.
- `activities/*_activity.csv`: raw CUPTI activity records.
- `activities/*_activity_summary.csv`: grouped CUPTI activity summary.
- `range_counters/*_range_metrics.csv`: CUPTI Range Profiler hardware counters
  when `--range-counters` is enabled.
- `configs/*.json`: per-scenario config snapshots.
- `run_config.json`: full run config.

The HTML report includes:

- latency cards from CUDA events and wall-clock timing;
- correctness mismatch summaries when `--check` is enabled;
- CUPTI Activity summaries for runtime, driver, kernel, memcpy, and memset records;
- kernel launch configuration from raw CUPTI records, including grid, block,
  registers per thread, and shared-memory usage;
- a hardware-counter roofline when Range Profiler counters are available.

If Range Profiler counters are unavailable, the report omits the roofline and
counter sections.

The Range Profiler pass is separate from benchmark timing. It uses CUPTI
UserRange/UserReplay, so the Python wrapper relaunches the same workload for
each counter pass. Do not use Range Profiler wall time as latency.

Hardware counters identify whether a kernel is leaning toward DRAM pressure,
L2 traffic, SM/compute-side pressure, or a dominant warp stall reason such as
long scoreboard, barrier, math pipe throttle, MIO throttle, or no instruction.
They still do not pinpoint a specific source line inside one kernel. For
source/SASS-level hotspots, use CUPTI PC sampling, CUPTI SASS metrics, Nsight
Compute source counters, or explicit in-kernel instrumentation after the
counter pass narrows down the bottleneck.

If counter collection fails with a permission error, enable NVIDIA performance
counters on the VM or run with sufficient privileges. The usual symptom is an
`ERR_NVGPUCTRPERM`-style failure.

If you already have a run and only want to regenerate the HTML report:

```bash
python benchmark/profiling/visualize_cupti_profile.py \
  --run-dir benchmark/profiling/traces/cupti_native_triton_cuda/<timestamp>
```

Source files:

- `init.py`: tiny default runner for Activity trace plus Range Profiler counters.
- `cupti_profile.py`: main CUPTI profiling CLI.
- `workload.py`: provider/scenario/input/timing/check helpers.
- `cupti_activity.py`: Python loader for the CUPTI shared library.
- `cupti_activity_collector.cpp`: direct CUPTI Activity API collector.
- `cupti_range_profiler.py`: Python loader for the CUPTI Range Profiler library.
- `cupti_range_profiler_collector.cpp`: CUPTI Range Profiler hardware-counter
  collector.
- `visualize_cupti_profile.py`: HTML report generator.

After changes to `cupti_activity_collector.cpp` or
`cupti_range_profiler_collector.cpp`, run with rebuild flags once so the shared
libraries are rebuilt:

```bash
python benchmark/profiling/init.py --force-rebuild-cupti --force-rebuild-cupti-range
```

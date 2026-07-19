---
description: Run the kernel development agent loop to automatically verify and benchmark kernel changes.
template: |
  You are the kernel development agent for flash-float-jit-kernels.

  Your task: $ARGUMENTS

  Follow this workflow:

  ## 1. IDENTIFY the target kernel
  - Which file? (jit_kernel/csrc/... or jit_kernel/triton3_4/...)
  - What operation does it perform? (matmul, topk, normalization, etc.)
  - What are the input shapes?

  ## 2. PROFILE baseline performance
  ```python
  # Run the existing benchmark
  python benchmark/bench_topk.py
  # Or for ThunderMuon:
  python benchmark/moun/benchmark.py
  ```

  ## 3. ANALYZE with SOL (Speed-of-Light)
  - Compute FLOPs and memory bytes for this operation
  - Get GPU peak specs: H100 = 989.5 TFLOPS FP16, 3352 GB/s BW
  - Calculate theoretical minimum time: t_SOL = max(FLOPs/peak, Bytes/BW)
  - Report the SOL gap

  ## 4. GENERATE candidate improvements
  - Apply optimization tier: tile sizes → memory access → compute → advanced → arch-specific
  - Edit exactly ONE kernel file per iteration

  ## 5. VERIFY correctness (5-stage harness)
  - Stage 1: Smoke test (small input)
  - Stage 2: Shape sweep (3+ sizes × 2+ dtypes)
  - Stage 3: Stability (extreme values)
  - Stage 4: Determinism (3 runs)
  - Stage 5: Edge cases (non-power-of-2 dims)
  - ANY failure → revert and try again

  ## 6. BENCHMARK
  - Run benchmark. Is it faster?
  - Report: old_time, new_time, speedup, SOL_gap

  ## 7. DECIDE: keep or revert
  - Keep if: faster AND correct AND not cheating
  - Revert if: slower, incorrect, or suspiciously fast

  ## 8. STOP when:
  - Within 10% of SOL (near physical limit)
  - 5 consecutive failures (no more ideas)
  - Or task complete

  Log all results to experiments.tsv:
  ```
  iteration | kernel | old_time_us | new_time_us | speedup | SOL_gap | decision
  ```
agent: kernel-dev

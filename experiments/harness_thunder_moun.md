# Harness Report — `thunder_moun`

- **File**: `jit_kernel/thunder_moun.py`
- **Timestamp**: 2026-08-22T17:06:27
- **Correctness**: PASS (stage: n/a)
- **Reference compared**: yes

## Correctness (5-stage)

| Stage | Status |
|-------|--------|
| smoke | pass |
| shape_sweep | pass |
| stability | pass |
| determinism | pass |
| edge_cases | pass |

## Performance (triton.testing.do_bench, quantiles [0.5, 0.2, 0.8])

- **Input size**: N = 4096
- **Time (median)**: 144.4 us (min 143.1 / max 145.5)
- **Reference time**: 215.7 us (speedup 1.49x)
- **SOL time**: 138.9 us
- **SOL gap**: 1.04x
- **Classification**: compute-bound (compute 138.9 us / mem 20.2 us)

| Metric | Value |
|--------|-------|
| median (us) | 144.4 |
| min (us) | 143.1 |
| max (us) | 145.5 |
| reference (us) | 215.7 |
| speedup vs ref | 1.49x |

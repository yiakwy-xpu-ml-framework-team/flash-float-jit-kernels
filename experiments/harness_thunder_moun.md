# Harness Report — `thunder_moun`

- **File**: `jit_kernel/thunder_moun.py`
- **Timestamp**: 2026-08-19T08:39:49
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

### Skipped (kernel alignment constraint)

- shape (63,) skipped (kernel requires 128-aligned dims)
- shape (4097,) skipped (kernel requires 128-aligned dims)
- shape (1023,) skipped (kernel requires 128-aligned dims)
- shape (1023,) skipped (kernel requires 128-aligned dims)
- shape (4097,) skipped (kernel requires 128-aligned dims)
- shape (1537,) skipped (kernel requires 128-aligned dims)

## Performance (triton.testing.do_bench, quantiles [0.5, 0.2, 0.8])

- **Input size**: N = 1024
- **Time (median)**: 24.4 us (min 24.1 / max 25.0)
- **Reference time**: 15.4 us (speedup 0.63x)
- **SOL time**: 2.2 us
- **SOL gap**: 11.22x
- **Classification**: compute-bound (compute 2.2 us / mem 1.3 us)

| Metric | Value |
|--------|-------|
| median (us) | 24.4 |
| min (us) | 24.1 |
| max (us) | 25.0 |
| reference (us) | 15.4 |
| speedup vs ref | 0.63x |

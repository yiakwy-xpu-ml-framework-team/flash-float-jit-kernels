# Harness Report — `thunder_moun`

- **File**: `jit_kernel/thunder_moun.py`
- **Timestamp**: 2026-08-19T14:50:36
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

- **Input size**: N = 4096
- **Time (median)**: 143.3 us (min 142.2 / max 144.2)
- **Reference time**: 216.0 us (speedup 1.51x)
- **SOL time**: 138.9 us
- **SOL gap**: 1.03x
- **Classification**: compute-bound (compute 138.9 us / mem 20.2 us)

| Metric | Value |
|--------|-------|
| median (us) | 143.3 |
| min (us) | 142.2 |
| max (us) | 144.2 |
| reference (us) | 216.0 |
| speedup vs ref | 1.51x |

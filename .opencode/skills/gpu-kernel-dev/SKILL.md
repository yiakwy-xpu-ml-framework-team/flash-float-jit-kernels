---
name: gpu-kernel-dev
description: General GPU kernel development skill covering profiling, benchmarking, optimization strategies, and correctness verification for CUDA, Triton, and Metal kernels. Use when writing, optimizing, or debugging any GPU kernel code.
---

# GPU Kernel Development

## Core Principles

### Memory Hierarchy (fastest to slowest)
- **Registers**: Private per-thread (~256 per thread). Extremely fast, tiny.
- **Shared Memory (SMEM)**: Shared by thread block (~48-228 KB). Use for data reuse.
- **L2 Cache**: Shared across SMs (~50 MB on H100). Automatic but unpredictable.
- **Global Memory (HBM)**: Main GPU RAM (80 GB H100). ~100x slower than shared memory.

### Performance Metrics
- **Arithmetic Intensity** = FLOPs / Bytes moved. High = compute-bound, low = memory-bound.
- **Occupancy** = active warps / max warps. Higher is better for hiding latency.
- **Throughput** = FLOPs/s or GB/s. Compare against GPU peak specs.
- **Speed-of-Light (SOL)** = theoretical minimum time = max(FLOPs/peak_FLOPs, Bytes/peak_BW).

### Optimization Tiers (Apply in Order)
1. **Tile/Block sizes**: Powers of 2, try rectangular tiles. Registers per block must fit.
2. **Memory access**: Coalesced loads/stores, vectorized (float4, half2), shared memory padding.
3. **Compute**: Fuse operations, hoist invariants, use tensor cores (WMMA/WGMMA on NVIDIA).
4. **Advanced**: Persistent kernels, split-K, software pipelining, warp specialization.
5. **Architecture-specific**: TMA on Hopper, cp.async on Ampere, SIMD groups on Metal.

## Correctness Verification (Mandatory)
Before measuring speed, verify correctness through:
1. **Smoke test**: Single small input, tight tolerances
2. **Shape sweep**: Test 8-10 different input sizes × 3 dtypes (FP16, BF16, FP32)
3. **Stability**: Test adversarial inputs (extreme values, zeros, near-zero variance)
4. **Determinism**: Run 3 times, outputs must match bit-for-bit
5. **Edge cases**: Non-power-of-2 dimensions (e.g., 1023, 4097, 1537)

Tolerances: FP16 atol=1e-2, BF16 atol=2e-2, FP32 atol=1e-4.

## Common Pitfalls
- Bank conflicts in shared memory (pad arrays to avoid same-bank access)
- Warp divergence (all threads in a warp must take same branch)
- Register spilling (too many local variables → spills to slow L1/local memory)
- Uncoalesced global memory access (threads accessing non-sequential addresses)
- Race conditions in atomic operations or parallel reductions
- Forgetting threadgroup_barrier / __syncthreads() after shared memory writes

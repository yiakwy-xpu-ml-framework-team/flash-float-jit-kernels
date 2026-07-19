---
name: metal-kernel
description: Apple Metal Shading Language kernel development for Apple Silicon GPUs (M1-M4, M1 Ultra-M3 Ultra). Covers mlx.fast.metal_kernel, SIMD group operations, threadgroup memory, 1-bit quantized GEMM, and StreamK scheduling. Use when working with Metal GPU kernels or Apple Silicon optimization.
---

# Metal Kernel Development (Apple Silicon)

## Metal via MLX

The primary interface for Metal kernels in this project is `mx.fast.metal_kernel`:

```python
def make_my_kernel(**config):
    header = """
    #include <metal_atomic>
    #include <metal_simdgroup>
    """

    source = """
    // Metal Shading Language (C++14-based)
    kernel void my_kernel(
        device const half* x [[buffer(0)]],
        device half* out [[buffer(1)]],
        constant uint& N [[buffer(2)]],
        uint tid [[thread_position_in_grid]]
    ) {
        if (tid < N) {
            out[tid] = x[tid] * 2.0h;
        }
    }
    """

    return mx.fast.metal_kernel(
        name="my_kernel",
        input_names=["x"],
        output_names=["out"],
        source=source,
        header=header,
        ensure_row_contiguous=True,
    )
```

## Metal Terminology vs CUDA

| Metal | CUDA | Notes |
|-------|------|-------|
| `threadgroup` | `block` | Group of threads sharing memory |
| `simdgroup` | `warp` | 32 threads executing in lockstep |
| `thread_position_in_grid` | `blockIdx * blockDim + threadIdx` | Global thread ID |
| `thread_position_in_threadgroup` | `threadIdx` | Local thread ID within block |
| `threadgroup_position_in_grid` | `blockIdx` | Block ID |
| `threads_per_threadgroup` | `blockDim` | Block size |
| `threadgroup_barrier(mem_flags::mem_threadgroup)` | `__syncthreads()` | Barrier |
| `device` | `__global__` | Global memory pointer |
| `threadgroup` | `__shared__` | Shared memory |
| `constant` | kernel argument / constant memory | Read-only kernel param |
| `atomic_fetch_add_explicit` | `atomicAdd` | Atomic addition |

## Metal SIMD Group Operations

```metal
#include <metal_simdgroup>

// SIMD group matrix multiply-accumulate (like tensor cores)
simdgroup_float8x8 c;
simdgroup_multiply_accumulate(c, a, b, c);

// SIMD shuffle (like __shfl_xor_sync)
float val = simd_shuffle(val, lane);
```

## 1-Bit Quantized GEMM Pattern (from PR #609)

### Weight Packing (Python side)
```python
def pack_1_bit_weights(weights):
    N, K = weights.shape
    pack_size = 16  # uint16 = 16 bits
    binary = ((weights + 1) // 2).to(torch.uint16)
    packed = torch.zeros((N, K // pack_size), dtype=torch.uint16)
    for i in range(pack_size):
        packed |= binary.reshape(N, -1, pack_size)[:, :, i] << i
    return packed
```

### Bit Unpacking in Metal (XOR trick)
```metal
// NOT_USE_XOR=0 (fast path): extract sign bit with XOR
// Input: packed uint16, output: sign-matched half
uint16_t wq_nk = tile_wq[n][k / packed_size];
int16_t wq_depacked = ((wq_nk << (15 - k % packed_size)) & 0x8000) ^ 0x8000;

// This produces: bit=0 → 0x8000 (negative half), bit=1 → 0x0000 (positive half)
// Then XOR with float representation to flip sign
datum.f_val = xq_mk;
datum.u_val ^= wq_depacked;
acc[i][j] += datum.f_val;
```

### StreamK Scheduling
```metal
// Split K dimension across Metal cores for load balancing
for (uint tile_id = pid; tile_id < kGRID_MNK; tile_id += NUM_METAL_CORES) {
    uint pid_k = tile_id / kGRID_MN;
    // Each core handles a K-slice, accumulates atomically
    atomic_fetch_add_explicit(
        (device atomic<float> *)(out + ...),
        acc[i][j] * scale,
        memory_order_relaxed
    );
}
```

## Metal Tuning Parameters

```python
# Thread organization
simd_group_size = 32  # Always 32 on Apple GPUs
warps = 4             # Concurrent warps per threadgroup
threadgroup = (simd_group_size, warps, 1)

# Tile sizes (tune these)
BLOCK_SIZE_M = tuning_config.get("BLOCK_SIZE_M", 32)
BLOCK_SIZE_N = tuning_config.get("BLOCK_SIZE_N", 128)
BLOCK_SIZE_K = tuning_config.get("BLOCK_SIZE_K", 64)

# GPU core count
NUM_METAL_CORES = get_mlx_gpu_cores()[1][0]
```

## Common Metal Errors
- "type mismatch" → Check half vs float, ensure explicit casts
- "buffer binding" → Check input_names/output_names match Metal `[[buffer(N)]]`
- "undefined identifier" → Check template substitutions match source code
- Non-contiguous tensors → Use `ensure_row_contiguous=True` or manually call `.contiguous()`
- `atomic<half>` not supported on some devices → Use float32 for atomic outputs

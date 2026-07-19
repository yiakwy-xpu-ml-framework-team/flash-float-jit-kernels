---
name: triton-kernel
description: Triton kernel development for NVIDIA and AMD GPUs. Covers block-level tiled programming, autotuning, epilogue fusion, and performance optimization patterns from the DRTriton and AutoKernel papers. Use when working with Triton .py kernel files or converting PyTorch ops to Triton.
---

# Triton Kernel Development

## Triton Conventions

### Basic Kernel Structure
```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, y_ptr, out_ptr,
              N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y  # Fused compute
    tl.store(out_ptr + offsets, output, mask=mask)

def fused_operator(x, y):
    N = x.numel()
    output = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    my_kernel[grid](x, y, output, N, BLOCK_SIZE=BLOCK_SIZE)
    return output
```

### Key Triton Features
- `tl.constexpr`: Compile-time constants (block sizes, dimensions)
- `tl.load/tl.store`: With mask and `other=` for bounds
- `tl.arange`: Vectorized thread indices
- `tl.program_id`: Grid position (0, 1, 2 for 3D grid)
- `tl.where`: Conditional (avoids warp divergence)
- `triton.cdiv`: Ceiling division for grid size

### Optimization Patterns from DRTriton

**Coalesced Memory Access**
```python
# Good: contiguous load along last dimension
x = tl.load(x_ptr + row * stride_row + tl.arange(0, BLOCK), mask=mask)
```

**Epilogue Fusion** (from SOLAR paper)
```python
# Fuse activation into kernel instead of separate pass
output = gelu(x @ w + bias)  # All in one kernel
```

**Autotuning** (from AutoKernel)
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256}, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def matmul_kernel(...): ...
```

## Common Triton Patterns for This Repo

### Matrix Multiplication (GEMM)
```python
@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)
```

### Normalization (RMSNorm / LayerNorm)
```python
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, N, eps: tl.constexpr,
                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    # Welford single-pass for numerical stability
    x2 = x * x
    rms = tl.sqrt(tl.sum(x2) / N + eps)
    out = x / rms * w
    tl.store(out_ptr + offsets, out)
```

## References
- Triton docs: https://triton-lang.org/
- DRTriton paper: Synthetic data + curriculum RL for Triton generation
- AutoKernel paper: 5-stage correctness harness for Triton/CUDA kernels

---
name: cuda-kernel
description: CUDA C++ kernel development for NVIDIA GPUs (Ampere, Hopper, Blackwell). Covers JIT compilation via torch.utils.cpp_extension, tensor core programming, warp-level primitives, TMA, and shared memory optimization. Use when working with .cu files in jit_kernel/csrc/.
---

# CUDA Kernel Development

## CUDA Conventions in This Repo

### JIT Compilation Pattern
```python
@functools.cache
def _jit_my_module():
    return load_jit(
        name="my_kernel",
        sources=[str(KERNEL_PATH / "csrc" / "my_kernel.cu")],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2", "-arch=sm_90", "-DENABLE_HOPPER=1"],
        verbose=True,
    )
```

### Platform Flags
- Hopper (SM 90): `-DENABLE_HOPPER=1 -arch=sm_90`
- Ampere (SM 80): `-arch=sm_80`
- ROCm: Use `helper_rocm.py` for HIP flags

## Tensor Core Programming (Hopper WGMMA)

### Warp Matrix Multiply-Accumulate (WMMA)
```cpp
// 16x16x16 half-precision
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
wmma::load_matrix_sync(a_frag, a_ptr, lda);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

### TMA (Tensor Memory Accelerator) on Hopper
```cpp
// Create TMA descriptor for 2D tensor
CUtensorMap tensor_map;
// Use tma_desc.h and tma_copy_impl.h from this repo
// See jit_kernel/csrc/thunder_moun/arch/tma/
```

## Bank Conflict Avoidance
- Pad shared memory arrays: `__shared__ float smem[BLOCK_SIZE][BLOCK_SIZE + 1]`
- Use `float4` / `half2` for vectorized access
- Swizzle patterns for multi-dimensional access

## Warp-Level Reduction
```cpp
// Butterfly shuffle reduction
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
}
```

## Common CUDA Errors to Avoid
- `cudaErrorIllegalAddress`: Out-of-bounds memory access
- `cudaErrorMisalignedAddress`: Unaligned vector access
- `cudaErrorLaunchOutOfResources`: Too many registers/too much shared memory
- Undefined behavior from missing `__syncthreads()` after shared memory writes

## Performance Profiling
```bash
ncu --set full -o profile.ncu-rep python benchmark/bench_topk.py
nsys profile -o profile python benchmark/bench_topk.py
```

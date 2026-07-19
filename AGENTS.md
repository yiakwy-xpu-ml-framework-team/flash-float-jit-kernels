# Instructions for AI Agents Working on flash-float-jit-kernels

## Project Overview
This repository is a high-performance JIT (Just-In-Time) GPU kernel library for low-latency LLM inference. It contains:
- **CUDA kernels**: Ultra-low-latency TopK indexer (Distributed Radix Sort via NoC) and ThunderMuon symmetric GEMM
- **Triton kernels**: Symmetric GEMM via Triton 3.4
- **Metal kernels**: Sub-1-bit StreamK GEMM for Apple Silicon (via mlx-lm PR #609)
- **Build system**: `torch.utils.cpp_extension.load` for JIT compilation, supporting CUDA (Hopper sm_90) and ROCm

## Repository Layout
```
jit_kernel/
├── csrc/
│   ├── topk_indexer/topk_indexer_radix.cu    # Distributed Radix Sort TopK
│   └── thunder_moun/
│       ├── symm_gemm.cu                        # Symmetric GEMM kernel
│       ├── arch/tma/                           # Hopper TMA abstractions
│       ├── block/                              # Block-level GEMM implementations
│       ├── fragment/                           # Tensor core fragment types
│       └── tensor/                             # Tensor views and tuples
├── triton3_4/
│   ├── symm_gemm.py                            # Triton symmetric GEMM
│   └── tvm_ffi_mod.py                          # TVM FFI integration
├── topk_indexer.py                             # JIT wrapper for TopK CUDA kernel
├── thunder_moun.py                             # JIT wrapper for ThunderMuon
├── helper_cuda.py / helper_rocm.py             # Platform detection and flags
└── utils.py                                    # Kernel path resolution, flags
benchmark/
├── bench_topk.py                               # TopK benchmarking
└── moun/                                       # ThunderMuon benchmarking
```

## Key Conventions
1. **File naming**: CUDA kernels in `csrc/` use `.cu` extension. Triton kernels use `.py`.
2. **JIT compilation**: Kernels are compiled at runtime via `torch.utils.cpp_extension.load()` wrapped in `functools.cache`
3. **Platform detection**: Use `_is_cuda` and `_is_hip` from helper modules
4. **Error handling**: Use `torch.jit.isinstance` style assertions for shape/dtype validation
5. **Versioning**: Uses `setuptools_scm`, version is in `jit_kernel/__version__.py`
6. **Architecture flags**: Hopper uses `-DENABLE_HOPPER=1 -arch=sm_90`

## When Writing GPU Kernels
1. Always verify correctness against a PyTorch reference implementation
2. Test multiple input shapes (small, medium, large, non-power-of-2)
3. For CUDA: use vectorized loads (float4, half2), coalesced memory access
4. For Triton: use tl.constexpr for block sizes, masks for bounds checking
5. For Metal: use threadgroup memory, SIMD operations, ensure_row_contiguous
6. Profile with benchmark suite before claiming speedup
7. Never assume a kernel is correct just because it compiles

# Instructions for AI Agents Working on flash-float-jit-kernels

## Project Overview
This repository is a high-performance JIT (Just-In-Time) GPU kernel library for low-latency LLM inference. It contains:
- **CUDA kernels**: 
  - Hopper (sm90a) : Ultra-low-latency TopK indexer (Distributed Radix Sort via NoC) / ThunderMuon symmetric GEMM (on chip Split-K Bulk Reduce) / Mega MoE / Mega NS5
  - DGX-Spark (sm121a) : VeloxVoice NVFP4 GEMM / VeloxVoice Fused Mel Log 
- **Triton kernels**: Symmetric GEMM via Triton 3.4
- **Metal kernels**: Sub-1-bit StreamK JIT XOR GEMM kernel for Apple Silicon, via our first attempt in mlx-lm [PR#609](https://github.com/ml-explore/mlx-lm/pull/609)
- **Build system**: `torch.utils.cpp_extension.load` for JIT compilation, supporting CUDA (Hopper sm_90) and ROCm
- **Agent system**: Headless opencode server + kernel_agent.py orchestrator for automated kernel development

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
tools/
├── kernel_agent.py                             # Kernel agent orchestrator (Python CLI)
├── opencode_client.py                          # REST client for opencode headless server
├── example_kernel.py                           # Example kernel for testing
└── KERNEL_AGENT.md                             # Architecture documentation
```

## Key Conventions
1. **File naming**: CUDA kernels in `csrc/` use `.cu` extension. Triton kernels use `.py`.
2. **JIT compilation**: Kernels are compiled at runtime via `torch.utils.cpp_extension.load()` wrapped in `functools.cache`
3. **TVM-FFI**: Production kernels (thunder_moun) use `tvm_ffi.cpp.load_inline` for lower overhead
4. **Platform detection**: Use `_is_cuda` and `_is_hip` from helper modules
5. **Error handling**: Use `torch.jit.isinstance` style assertions for shape/dtype validation
6. **Versioning**: Uses `setuptools_scm`, version is in `jit_kernel/__version__.py`
7. **Architecture flags**: Hopper uses `-DENABLE_HOPPER=1 -arch=compute_90a -code=sm_90a`

## Target Platform
Production kernels run on **H800 SuperPod with IB9700** (NDR 400G InfiniBand) interconnect.
- H800 SXM compute peaks match H100: 989.5 TFLOPS (FP16 dense), 3,352 GB/s (HBM3)
- **NVLink = 400 GB/s on H800 (half of H100's 900 GB/s)** — do not design multi-GPU kernels that depend on high NVLink throughput; prefer IB9700 for cross-node traffic.

## When Writing GPU Kernels
1. Always verify correctness against a PyTorch reference implementation
2. Test multiple input shapes (small, medium, large, non-power-of-2)
3. For CUDA: use vectorized loads (float4, half2), coalesced memory access
4. For Triton: use tl.constexpr for block sizes, masks for bounds checking
5. For Metal: use threadgroup memory, SIMD operations, ensure_row_contiguous
6. Profile with benchmark suite before claiming speedup
7. Never assume a kernel is correct just because it compiles

## Hopper Coding Style (SM 90+)
This repo targets Hopper exclusively for complex kernels. Key patterns:
- **TVM-FFI entry**: `TVM_FFI_DLL_EXPORT_TYPED_FUNC` wrapping `extern "C"` CUDA entry
- **WGMMA**: Warp Group MMA via inline PTX (`wgmma.mma_async.sync.aligned.m64n128k32`)
- **TMA**: Tensor Memory Accelerator (`cuTensorMapEncodeTiled`, `cp.async.bulk.tensor.2d`)
- **mbarrier**: Pipeline sync with parity-based wait (`mbarrier.try_wait.parity`)
- **NoC clusters**: Cross-SM reduction via `cluster.map_shared_rank` (not atomicAdd)
- **Zero cutlass**: Header-only abstractions in `fragment/`, `block/`, `tensor/`

See `.opencode/skills/hopper-kernel/SKILL.md` and `RESEARCH_NOTES.md` for details.

## Agent Workflow (Headless Mode)
1. Start opencode server: `python tools/kernel_agent.py serve --port 8096`
2. Create session: `python tools/kernel_agent.py session create`
3. Run optimization: `python tools/kernel_agent.py run --kernel <path> --max-iter 5`
4. Or standalone: `python tools/kernel_agent.py harness --kernel <path>`

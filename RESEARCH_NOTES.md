# Research Notes: flash-float-jit-kernels

## thunder_moun GEMM 优化方向偏好（人工指定的优先级）

仅在以下三个方向上尝试优化，按优先级排序：
1. **on-chip split-k all-reduce**：降低跨 block 归约（k-dim 归并）的开销。
2. **inplace transpose**：优化 inplace transpose，并完善 output transpose 路径。
3. **Fragment shape**：尝试不同的 Fragment shape 组合，用足更大算力（MFU）。

硬约束：
- **不要** 把原来定义在 `arch/`, `block/`, `fragment/`, `tensor/`, 里的东西挪出去。
- **不要** 重写与现有功能一致的实现, 没有可量化加速的改写直接省略。
- 每次 edit 都必须带有明确的性能动机（approx SOL / bottleneck 分析），并保持
  `jit_kernel/thunder_moun.py` 与 `csrc/` 的现有接口不变。

优化记录：
- 优化的前形成 plan 记录为 md 文档，放在 `sandbox/probes` 下面，和计划修改文件，修改方向
- 修改不要直接 edit 源文件，而先放在 `sandbox` 下形成方案，进行替换；将实验结果，按日期修改内容记录
- 最后形成 ablation study，将有效的修改整合规约，进行集成测试
- 只提交成功的优化，并同步到文静，并在 summary 文档记录最终的修改方案 和 内容

## Paper References

### AutoKernel [^1]
- **Core idea**: Automatic kernel optimization via CPU-first safety harness + keep/revert loop
- **What we took**:
  - 5-stage correctness harness: smoke → shape sweep → stability → determinism → edge cases
  - Keep/revert optimization loop with consecutive-revert stopping
  - Amdahl's Law prioritization for optimization candidates
- **Usage in this repo**: `tools/kernel_agent.py` implements the full 5-stage harness

### DRTriton [^2]
- **Core idea**: Curriculum RL for Triton kernel generation with CSP-DAG test synthesis
- **What we took**:
  - Synthetic test generation via constraint satisfaction
  - Curriculum difficulty escalation (small shapes → large shapes → edge cases)
  - Test-time kernel search with multiple candidates
- **Usage in this repo**: Test generation patterns in benchmark scripts, shape sweep in safety harness

### SOLAR [^3]
- **Core idea**: Speed-of-Light analysis for systematic kernel optimization
- **What we took**:
  - SOL gap metric: `g = t_best / max(T_compute, T_mem)`
  - MANTIS optimization loop: Measure → Analyze → Nominate → Triage → Implement → Summarize
  - Compute-bound vs memory-bound classification
- **Usage in this repo**: `.opencode/agent/kernel-dev.md` implements MANTIS loop

### DeepSeek Harness
- **Core idea**: Self-Envolv Agent
- **Waht we took**:
  - Defining kerenl-dev loop contract for runtime monkey patch
  - 

---

## Hopper Platform Coding Patterns

### 1. Kernel Entry via TVM-FFI

We use TVM FFI as the kernel entry point, NOT `torch.utils.cpp_extension` for complex kernels.
This gives us lower launch overhead and better integration with the SGLang/TVM ecosystem.

```cpp
// Raw C entry point
extern "C" int my_kernel_entry(void* X_ptr, void* Out_ptr, int M, int N, ...) {
    // CUDA kernel setup and launch
}

// TVM-FFI wrapper
#ifdef USE_TVM_FFI
namespace tvm_c_loader {
using namespace tvm::ffi;
void tvm_jit_my_kernel_launch(TensorView X, TensorView Out, int M, int N, ...) {
    DLDevice device = X.device();
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(device.device_type, device.device_id));
    my_kernel_entry(X.data_ptr(), Out.data_ptr(), M, N, ..., stream);
}
} // namespace tvm_c_loader
TVM_FFI_DLL_EXPORT_TYPED_FUNC(my_kernel, (tvm_c_loader::tvm_jit_my_kernel_launch));
#endif
```

Python side:
```python
from tvm_ffi.cpp import load_inline
module = load_inline("my_kernel", cuda_sources=[...], extra_cuda_cflags=[...])
module.my_kernel(x_tensor, out_tensor, M, N)
```

### 2. Hopper-Only Features (SM 90+)

Our kernels are designed for Hopper (H100/H800) and above. We do NOT fall back to Ampere.
Production target: H800 SuperPod with IB9700 (NDR 400G InfiniBand) interconnect. H800 compute peaks match H100 (989.5 TFLOPS FP16 dense, 3,352 GB/s HBM3) but NVLink is reduced to 400 GB/s — so cluster-scale kernels should lean on IB9700 for cross-node traffic rather than NVLink.

| Feature | Usage | PTX |
|---------|-------|-----|
| **WGMMA** | Tensor core GEMM | `wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3` |
| **TMA** | Async bulk copy | `cp.async.bulk.tensor.2d.shared::cluster.global` |
| **mbarrier** | Pipeline sync | `mbarrier.init`, `mbarrier.arrive.expect_tx`, `mbarrier.try_wait.parity` |
| **Cooperative groups** | Cluster-level ops | `this_cluster()`, `cluster.sync()`, `cluster.map_shared_rank` |
| **NoC multicast** | Cross-SM reduction | `cp.async.bulk.tensor.2d...multicast::cluster` |

### 3. Software Pipelining Pattern

4-stage pipeline with mbarrier synchronization:

```cpp
// Stage layout: [stage0, stage1, stage2, stage3]
// Ramp-up: fill stages 0..2, then enter main loop
// Main loop: wait on read_stage, compute, prefetch into write_stage
// Epilogue: reduce split-K results, TMA store

__shared__ __align__(128) uint64_t barriers[STAGES];

// Init (thread 0 only)
for (int s = 0; s < STAGES; ++s) {
    uint32_t bar_ptr = __cvta_generic_to_shared(&barriers[s]);
    asm volatile("mbarrier.init.shared.b64 [%0], 1;\n" :: "r"(bar_ptr));
}
asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
__syncthreads();

// Prefetch: arrive + TMA copy
asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
    :: "r"(bar_ptr), "r"(total_bytes));
asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.L2::cache_hint"
    " [%0], [%1, {%2, %3}], [%4], %5;\n" ...);

// Wait: parity-based wait with nanosleep
asm volatile(
    "{\n"
    ".reg .pred P;\n"
    "WAIT_LOOP:\n"
    "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
    "@P bra DONE;\n"
    "nanosleep.u32 64;\n"
    "bra WAIT_LOOP;\n"
    "DONE:\n"
    "}\n" :: "r"(bar_ptr), "r"(phase) : "memory");
```

### 4. TMA Descriptor Setup

```cpp
CUtensorMap desc;
uint64_t shape[2] = {K, M};       // inner dim first for row-major
uint64_t stride[1] = {M * sizeof(T)};
uint32_t box[2] = {BLOCK_K, BLOCK_M};
uint32_t step[2] = {1, 1};

cuTensorMapEncodeTiled(
    &desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, ptr,
    shape, stride, box, step,
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    nvgpu::arch::get_tma_swizzle_mode<BLOCK_K>(),
    CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
```

### 5. NoC-Based Split-K Reduction

Instead of atomicAdd, we use cluster-level shared memory reduction:

```cpp
auto cluster = cooperative_groups::this_cluster();

if (split_k_id == 0) {
    cluster.sync(); // cluster_consumers_sync();

    for (int r = 1; r < split_k; ++r) {
        OutDtype* remote = cluster.map_shared_rank<OutDtype>(&shmem_epilogue[0], r);
        for (int idx = tid; idx < BM * BN; idx += threads_per_block) {
            shmem_epilogue[idx] += remote[idx];
        }
    }
    // ...
}

cluster.sync(); // cluster_consumers_sync();
```

### 6. Shared Memory Abstraction

We use template-based header-only abstractions (zero cutlass dependency):

```cpp
// SharedBlock: 128-byte aligned, typed shared memory tile
template <typename T, int M, int N>
struct alignas(128) SharedBlock {
    T data[M * N];
};

// FragmentView: in-place transpose view over shared memory
template <typename T, int M, int N, MemoryDomain domain>
struct FragmentView {
    T* smem_ptr;
    void _transpose();  // cooperative in-place transpose
};

// HopperWGMMAAccumulator: register-backed accumulator fragment
template <int BM, int BN, int BK>
struct HopperWGMMAAccumulator {
    float regs[64];  // 64 registers per thread for m64n128k32
    void clear();
    void store(OutDtype* smem_ptr);
    void mul_(const fp8_t* smem_x, const fp8_t* smem_w, int k_step);
    void add_(const HopperWGMMAAccumulator& other);
};
```

### 7. Compile Flags

```python
# Hopper-specific flags
cuda_flags = [
    "-O2", "-std=c++17",
    "-Xcompiler", "-fPIC",
    "-DENABLE_HOPPER=1",
    "-arch=compute_90a",
    "-code=sm_90a",
    "--ptxas-options=-v",
]
```

### 8. Zero Cutlass/Cutlass Dependency

Our codebase deliberately avoids cutlass and cutlass. All abstractions are:
- Header-only templates in `jit_kernel/csrc/thunder_moun/`
- Named with `xpu::` or `nvgpu::` namespace
- Algorithm-driven: the pipeline structure IS the algorithm, not a library call
- File structure: `fragment/`, `block/`, `tensor/`, `arch/tma/`

---

## Design Principles

1. **Neat and symbolic**: Code reads like the algorithm, not like a CUDA manual
2. **Algorithm-driven**: Template parameters encode algorithmic choices (tile sizes, stages, cluster size)
3. **Zero heavy dependencies**: No cutlass, no cuBLAS, no cuDNN for kernel internals
4. **TVM-FFI for entry**: Lower overhead than PyTorch JIT, integrates with SGLang
5. **Hopper-first**: Design for SM 90+, do not compromise for older architectures
6. **Persistent kernels**: One block per SM, task-based scheduling via atomic counters

## References

[1] J. Jaber and O. Jaber, "AutoKernel: Autonomous GPU Kernel Optimization via Iterative
Agent-Driven Search," arXiv:2603.21331, 2026.

[2] S. Guo, M. Lin, and T. Yang, "DRTriton: Large-Scale Synthetic Data Driven Reinforcement
Learning for Triton Kernel Generation," arXiv:2603.21465, 2026.

[3] S. K. S. Hari et al., "Improving EHiciency of GPU Kernel Optimization Agents using a DSL
and Speed-of-Light Guidance," arXiv:2603.29010, 2026.

[4] DeepSeek Harness : https://github.com/deepseek-ai/deepseek-harness. Accessed online on Sep 1st, 2026.

[5] Yifan Shi, Wei Zhang, Tianyi Cui, "A Programming Paradigm for Spatiotemporal Composabilit", https://arxiv.org/pdf/2608.25512. Accessed online on Sep 1st, 2026.
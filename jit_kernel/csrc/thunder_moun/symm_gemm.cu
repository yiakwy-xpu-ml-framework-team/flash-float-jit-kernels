/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#include <cuda.h>
#include <cuda_runtime.h>

// TODO (yiakwy) : add fast dtype impl for Hopper and MI300X/MI355X/MI400X platform
// TODO (yiakwy) : use "include/flash_float/dtype/fp8_e4m3_fnuz.h",
#include <cuda_fp8.h>

#include <cuda/ptx>

// enable cooperative blocks launch
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include <type_traits>

#define USE_TVM_FFI 1

#ifdef USE_TVM_FFI
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#endif


#define ENABLE_HOPPER 1

// NOTE (yiakwy) : TVM FFI does not recognize __CUDA_ARCH__ macro for *.h files
#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 900
#endif

#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif

#ifndef MAX
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif

#ifndef CEILDIV
#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#endif

#ifndef SMs
#define SMs 132
#endif

// TODO (yiakwy) : use "include/flash_float/dtype/mxfp4.h"
// TODO (yiakwy) : use "include/flash_float/dtype/fp8_e8m0_fnu.h"

#include "tensor/tuple.h"
#include "tensor/tensor_view_ref.h"
#include "profiler.cuh"
#include "fragment/fragment.h"

// TODO (yiakwy) : move to xpu general interface
#include "fragment/nv_frag_gemm_scaled_impl.h"

#include "block/block.h"

// TODO (yiakwy) : move to xpu general interface
#include "block/nv_block_gemm_scaled_impl.h"

#include "arch/tma/tma_desc.h"

constexpr int K_BLOCK_M = 128;
constexpr int K_BLOCK_N = 128;
constexpr int K_BLOCK_K = 128;

constexpr int K_BLOCK_N_SWIZZLE = 64;

constexpr int K_STAGES = 4;

constexpr int MAX_SPLIT_K = 8;

constexpr int GROUP_SIZE_M = 16;
// NOTE (yiakwy) : see our paper for details
constexpr int CLUSTER_SIZE_M = 2; // GROUP_SIZE_M / 2;

// 8 warps per block
#ifndef WARP_SIZE
constexpr int WARP_SIZE = 32;
#endif

constexpr int NUM_WARPS = 8;

constexpr int TOTAL_WARP_THREADS = NUM_WARPS * WARP_SIZE;

extern "C" __global__
void hopper_symm_gemm_kernel_entry(
    const __nv_fp8_e4m3* __restrict__ X,
    const __nv_fp8_e4m3* __restrict__ W,
    const float* __restrict__ scale_X,
    const float* __restrict__ scale_W,
    half* __restrict__ Out,
    const int M, const int N, const int K,
    const int total_symmetric_tiles,
    const int num_blocks_m, const int num_blocks_n, const int cluster_size_m,
    __grid_constant__ const CUtensorMap tma_desc_X,
    __grid_constant__ const CUtensorMap tma_desc_W,
    __grid_constant__ const CUtensorMap tma_desc_O,
    __grid_constant__ const CUtensorMap tma_desc_O_swizzle
    FFJK_PROFILER_KERNEL_PARAMS)
{
    // extern __shared__ uint8_t total_smem_space[];
    extern __shared__ __align__(128) uint8_t smem_buffer[];

    /*
    uintptr_t raw_smem = reinterpret_cast<uintptr_t>(total_smem_space);
    uintptr_t aligned_smem = (raw_smem + 127) & ~127;
    uint8_t* smem_buffer = reinterpret_cast<uint8_t*>(aligned_smem);
    */

#ifdef FFJK_ENABLE_CUDA_PROFILER
    const int profiler_k_tiles_total = CEILDIV(K, K_BLOCK_K);
    const int profiler_k_tiles_per_slice = CEILDIV(profiler_k_tiles_total, gridDim.x);
    FFJK_PROFILER_DEFINE_LAYOUT(total_symmetric_tiles, profiler_k_tiles_per_slice);
    FFJK_PROF_CTA_EVENT_PAYLOAD(
        ffjk::kProfilerEventKernelLaunch,
        M,
        N);
#endif

    xpu::HopperWGMMAAccumulator<K_BLOCK_M, K_BLOCK_N, K_BLOCK_K> accumulator_fragment;
    accumulator_fragment.clear();

    // if (threadIdx.x == 0 && blockIdx.x == 1) {
    //     printf("[hopper_symm_gemm_kernel_entry] [split_k#%d] [SM#%d] enter into HopperPersistentSplitKPipeline ...\n", blockIdx.x, blockIdx.y);
    // }
    // __syncthreads();

    if (cluster_size_m < 8) {
        xpu::HopperPersistentSplitKPipeline<K_BLOCK_M, K_BLOCK_N, K_BLOCK_K, K_STAGES, GROUP_SIZE_M, 4>::run_persistent(
            accumulator_fragment,
            &tma_desc_X,
            &tma_desc_W,
            &tma_desc_O,
            &tma_desc_O_swizzle,
            scale_X,
            scale_W,
            Out, M, N, K,
            total_symmetric_tiles,
            num_blocks_m,
            num_blocks_n,
            smem_buffer
            FFJK_PROFILER_KERNEL_ARGS
        );
    } else
    if (cluster_size_m < 4) {
        xpu::HopperPersistentSplitKPipeline<K_BLOCK_M, K_BLOCK_N, K_BLOCK_K, K_STAGES, GROUP_SIZE_M, 2>::run_persistent(
            accumulator_fragment,
            &tma_desc_X,
            &tma_desc_W,
            &tma_desc_O,
            &tma_desc_O_swizzle,
            scale_X,
            scale_W,
            Out, M, N, K,
            total_symmetric_tiles,
            num_blocks_m,
            num_blocks_n,
            smem_buffer
            FFJK_PROFILER_KERNEL_ARGS
        );
    } else
    if (cluster_size_m < 2) {
        xpu::HopperPersistentSplitKPipeline<K_BLOCK_M, K_BLOCK_N, K_BLOCK_K, K_STAGES, GROUP_SIZE_M, 1>::run_persistent(
            accumulator_fragment,
            &tma_desc_X,
            &tma_desc_W,
            &tma_desc_O,
            &tma_desc_O_swizzle,
            scale_X,
            scale_W,
            Out, M, N, K,
            total_symmetric_tiles,
            num_blocks_m,
            num_blocks_n,
            smem_buffer
            FFJK_PROFILER_KERNEL_ARGS
        );
    }

}

// TVM-FFI raw C interface for JIT kernel invocation
extern "C" int symm_gemm_fp8_block_scaled(
    void* X_ptr, void* W_ptr, void* scale_X, void* scale_W, void* Out_ptr,
    int M, int N, int K,
    int x_stride_m, int x_stride_k,
    int w_stride_n, int w_stride_k,
    int o_stride_m, int o_stride_n,
    unsigned int split_k,
    cudaStream_t stream = 0
#ifdef FFJK_ENABLE_CUDA_PROFILER
    , void* profiler_buffer = nullptr,
    unsigned int profiler_capacity = 0
#endif
    )
{
    using fp8_t = __nv_fp8_e4m3;
    using fp16_t = half;
    using bf16_t = __nv_bfloat16;
    using fp32_t = float;

#ifdef FFJK_ENABLE_CUDA_PROFILER
    ffjk::CudaProfilerBuffer* profiler = nullptr;
    if (profiler_buffer != nullptr && profiler_capacity > 0) {
        profiler = reinterpret_cast<ffjk::CudaProfilerBuffer*>(profiler_buffer);
    }
#endif

// TODO(yiakwy) : move to op constructor
/*
#ifdef USE_TVM_FFI
    DLDevice device;
    device.device_type = kDLCUDA;

    cudaPointerAttributes attrs;

    if (cudaPointerGetAttributes(&attrs, X_ptr) == cudaSuccess) {
        device.device_id = attrs.device;
    } else {
        cudaGetDevice(&device.device_id);
    }

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));
#endif
*/

#if __CUDA_ARCH__ >= 900 && ENABLE_HOPPER // Hopper 900+ GPU with TMA support
    CUtensorMap desc_X, desc_W;
    CUtensorMap desc_O, desc_O_swizzle;
#else

#error "Compilation Failed: This Flash-Float JIT Kernel relies heavily on Hopper and above hardware features specifically, " \
       "such as TMA descriptors to reduce registers overhead in multi-stage execution, NoC communication (mapa) for Split-K reudction and new TC core with WGMMA instructions." \
       "Please ensure you are compiling with CUDA 12.8+ targeting sm90+ and have set -DENABLE_HOPPER=1." \
       "For older non-TMA architectures (Ampere/Ada), please use the triton alternate as fallbacks."

#endif

    // default to row major layout, but CUDA TMA requires the inner most dim to be contiguous, so we will swap the dim and stride for K and M/N
    // see references (12.8) :
    //   - https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
    //   - https://zhuanlan.zhihu.com/p/1985678344352731952
    uint64_t shape_x[2] = {static_cast<uint64_t>(K), static_cast<uint64_t>(M)};
    uint64_t stride_x[1] = {static_cast<uint64_t>(x_stride_m)};
    uint32_t box_x[2] = {K_BLOCK_K, K_BLOCK_M};
    uint32_t step_x[2] = {1, 1};

#if __CUDA_ARCH__ >= 900 && ENABLE_HOPPER // Hopper 900+ GPU with TMA support
{
    // CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B
    CUresult res = cuTensorMapEncodeTiled(
        &desc_X, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (fp8_t*)X_ptr,
        shape_x, stride_x, box_x, step_x,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        nvgpu::arch::get_tma_swizzle_mode<K_BLOCK_K>(), // CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (res != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(res, &errStr);
        std::cerr << "TMA Encode Failed! Error: " << errStr << " | Check tensorDim, boxSize, and strides!" << std::endl;
        return -1;
    }
}
#else

#error "Not supported. We heavily rely on Hopper's TMA Multicast via NoC communication for efficient Split-K reduction and better IO/L2 cache reuse."

#endif

    uint64_t shape_w[2] = {static_cast<uint64_t>(K), static_cast<uint64_t>(N)};
    uint64_t stride_w[1] = {static_cast<uint64_t>(w_stride_n)};
    uint32_t box_w[2] = {K_BLOCK_K, K_BLOCK_N};
    uint32_t step_w[2] = {1, 1};

#if __CUDA_ARCH__ >= 900 && ENABLE_HOPPER // Hopper 900+ GPU with TMA support
{
     // CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B
    CUresult res = cuTensorMapEncodeTiled(
        &desc_W, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (fp8_t*)W_ptr,
        shape_w, stride_w, box_w, step_w,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        nvgpu::arch::get_tma_swizzle_mode<K_BLOCK_K>(),// CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE); // CU_TENSOR_MAP_L2_PROMOTION_L2_256B

    if (res != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(res, &errStr);
        std::cerr << "TMA Encode Failed! Error: " << errStr << " | Check tensorDim, boxSize, and strides!" << std::endl;
        return -1;
    }
}
#else

#error "Not supported. We heavily rely on Hopper's TMA Multicast via NoC communication for efficient Split-K reduction and better IO/L2 cache reuse."

#endif

    // === TMapOut ===
    uint64_t shape_o[2] = {static_cast<uint64_t>(N), static_cast<uint64_t>(M)};
    uint64_t stride_o[1] = {static_cast<uint64_t>(o_stride_m) * sizeof(fp16_t)};
    uint32_t box_o[2] = {K_BLOCK_N, K_BLOCK_M};
    uint32_t step_o[2] = {1, 1};

#if __CUDA_ARCH__ >= 900 && ENABLE_HOPPER // Hopper 900+ GPU with TMA support
    {
        CUresult res = cuTensorMapEncodeTiled(
            &desc_O, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, (fp16_t*)Out_ptr,
            shape_o, stride_o, box_o, step_o,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE); // CU_TENSOR_MAP_L2_PROMOTION_L2_256B

        if (res != CUDA_SUCCESS) {
            const char* errStr;
            cuGetErrorString(res, &errStr);
            std::cerr << "TMA Encode Failed! Error: " << errStr << " | Check desc_O's tensorDim, boxSize, and strides!" << std::endl;
            return -1;
        }
    }

    // Optional TMA descriptor for transpose store with swizzle.
    uint32_t box_o_swizzle[2] = {K_BLOCK_N_SWIZZLE, K_BLOCK_M};
    {
        CUresult res = cuTensorMapEncodeTiled(
            &desc_O_swizzle, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, (fp16_t*)Out_ptr,
            shape_o, stride_o, box_o_swizzle, step_o,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B, // CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE); // CU_TENSOR_MAP_L2_PROMOTION_L2_256B

        if (res != CUDA_SUCCESS) {
            const char* errStr;
            cuGetErrorString(res, &errStr);
            std::cerr << "TMA Encode Failed! Error: " << errStr << " | Check desc_O_swizzle's tensorDim, boxSize, and strides!" << std::endl;
            return -1;
        }
    }
#else

#error "Not supported. We heavily rely on Hopper's TMA Multicast via NoC communication for efficient Split-K reduction and better IO/L2 cache reuse."

#endif

    auto& kernel = hopper_symm_gemm_kernel_entry;

    uint32_t required_smem_bytes = sizeof(xpu::SharedBlock<fp8_t, K_BLOCK_M, K_BLOCK_K>) * K_STAGES +
                                   sizeof(xpu::SharedBlock<fp8_t, K_BLOCK_N, K_BLOCK_K>) * K_STAGES +
                                   sizeof(xpu::SharedBlock<fp16_t, K_BLOCK_M, K_BLOCK_N>);

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, required_smem_bytes);

    int num_blocks_m = CEILDIV(M, K_BLOCK_M);
    int num_blocks_n = CEILDIV(N, K_BLOCK_N);

    // static_assert( num_blocks_m == num_blocks_n , "Unexpected value. For the momnet we only support K_BLOCK_M == K_BLOCK_N" );

    int total_symmetric_tiles = (num_blocks_m * num_blocks_n + num_blocks_m) / 2;

    int grid_mn = MIN(SMs, total_symmetric_tiles);

    if (grid_mn < SMs) {
#if ENABLE_HOPPER
        if (grid_mn >= 64) {
            split_k = 2;
        } else if (grid_mn >= 32) {
            split_k = MIN(split_k, 4);
        } else {
            split_k = MIN(split_k, 8);
        }
#endif
    } else {
        split_k = 1;
    }

    int cluster_size_m = MIN(4, CLUSTER_SIZE_M);
    if (grid_mn % cluster_size_m != 0) {
        cluster_size_m = 1;
    }

    int max_split_k = CEILDIV(MAX_SPLIT_K, cluster_size_m);
    split_k = MIN(split_k, max_split_k);

    dim3 grid(split_k, grid_mn, 1);
    dim3 block(TOTAL_WARP_THREADS, 1, 1);

#ifdef FFJK_ENABLE_CUDA_PROFILER
    if (profiler != nullptr) {
        const uint32_t k_tiles_total = CEILDIV(K, K_BLOCK_K);
        const uint32_t k_tiles_per_slice = CEILDIV(k_tiles_total, split_k);
        const uint32_t max_tasks_per_cta = CEILDIV(total_symmetric_tiles, grid_mn);

        cudaError_t init_state = ffjk::cuda_profiler_init(
            profiler_buffer,
            profiler_capacity,
            grid.x,
            grid.y,
            max_tasks_per_cta,
            k_tiles_per_slice,
            stream);
        if (init_state != cudaSuccess) {
            printf("[ERROR] cuda profiler init failed: %s\n", cudaGetErrorString(init_state));
            return -1;
        }
    }
#endif

#if __CUDA_ARCH__ >= 900 && ENABLE_HOPPER // Hopper 900+ GPU with TMA support
    cudaLaunchConfig_t launch_config = {0};
    launch_config.gridDim = grid;
    launch_config.blockDim = block;
    launch_config.dynamicSmemBytes = required_smem_bytes;

// #ifdef USE_TVM_FFI
    launch_config.stream = stream;
// #endif

    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {split_k, (unsigned int)cluster_size_m, 1};
    launch_config.attrs = &attr;
    launch_config.numAttrs = 1;

    // printf("[symm_gemm_fp8_block_scaled] launching stream-split-k (%d) grid_mn#%d with groups %d (%d x %d), %d (%dx1) multcast kernel via NoC...\n", split_k, grid_mn, GROUP_SIZE_M, GROUP_SIZE_M, GROUP_SIZE_M, cluster_size_m, cluster_size_m);

    cudaError_t state = cudaLaunchKernelEx(&launch_config, kernel,
        (const fp8_t*)X_ptr, (const fp8_t*)W_ptr, (const fp32_t*)scale_X, (fp32_t*)scale_W, (fp16_t*)Out_ptr,
        M, N, K, total_symmetric_tiles, num_blocks_m, num_blocks_n, cluster_size_m, desc_X, desc_W, desc_O, desc_O_swizzle
        FFJK_PROFILER_LAUNCH_ARG(profiler)
    );

    if (state != cudaSuccess) {
        printf("[ERROR] cudaLaunchKernelEx failed to launch: %s\n", cudaGetErrorString(state));
        return -1;
    }
#else

    // printf("[symm_gemm_fp8_block_scaled] launching stream-split-k kernel with L2 cache...\n");
    // launch with L2 cache
#error "Not supported. We heavily rely on Hopper's TMA Multicast via NoC communication for efficient Split-K reduction and better IO/L2 cache reuse."

#endif
    return 0;
}

#ifdef USE_TVM_FFI

namespace tvm_c_loader {
using namespace tvm::ffi;

void tvm_jit_symm_gemm_fp8_block_scaled_launch(
    TensorView X, TensorView W, TensorView scale_X, TensorView scale_W, TensorView Out,
    int M, int N, int K,
    int x_stride_m, int x_stride_k,
    int w_stride_n, int w_stride_k,
    int o_stride_m, int o_stride_n,
    unsigned int split_k)
{
    DLDevice device = X.device();
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));

    symm_gemm_fp8_block_scaled(
        X.data_ptr(), W.data_ptr(), scale_X.data_ptr(), scale_W.data_ptr(), Out.data_ptr(),
        M, N, K,
        x_stride_m, x_stride_k,
        w_stride_n, w_stride_k,
        o_stride_m, o_stride_n,
        split_k, stream
    );
}

#ifdef FFJK_ENABLE_CUDA_PROFILER
void tvm_jit_symm_gemm_fp8_block_scaled_profiled_launch(
    TensorView X, TensorView W, TensorView scale_X, TensorView scale_W, TensorView Out,
    int M, int N, int K,
    int x_stride_m, int x_stride_k,
    int w_stride_n, int w_stride_k,
    int o_stride_m, int o_stride_n,
    unsigned int split_k,
    TensorView profiler_buffer,
    int profiler_capacity)
{
    DLDevice device = X.device();
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));

    symm_gemm_fp8_block_scaled(
        X.data_ptr(), W.data_ptr(), scale_X.data_ptr(), scale_W.data_ptr(), Out.data_ptr(),
        M, N, K,
        x_stride_m, x_stride_k,
        w_stride_n, w_stride_k,
        o_stride_m, o_stride_n,
        split_k, stream,
        profiler_buffer.data_ptr(), static_cast<unsigned int>(profiler_capacity)
    );
}
#endif

} // namespace tvm_c_loader

TVM_FFI_DLL_EXPORT_TYPED_FUNC(symmetric_gemm_fp8_block_scaled, (tvm_c_loader::tvm_jit_symm_gemm_fp8_block_scaled_launch));
#ifdef FFJK_ENABLE_CUDA_PROFILER
TVM_FFI_DLL_EXPORT_TYPED_FUNC(symmetric_gemm_fp8_block_scaled_profiled, (tvm_c_loader::tvm_jit_symm_gemm_fp8_block_scaled_profiled_launch));
#endif
#endif

/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#include <cuda.h>

/* Packed nvfp4/mxfp4 GEMM for DGX-Spark (sm_121a) */
#if defined(CUDA_VERSION) && CUDA_VERSION >= 1200

#include "block/nv_block_wasp_mxfp4.h"

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

namespace {

constexpr int K_BM = 128;
constexpr int K_BN = 128;
constexpr int K_BK = 64;

constexpr int K_STAGES = 4;

#ifndef K_GROUP_SIZE_M
#define K_GROUP_SIZE_M 16
#endif

#ifndef K_CLUSTER_SIZE_M
#define K_CLUSTER_SIZE_M 2
#endif

#ifndef NUM_PRODUCER_WARPS
#define NUM_PRODUCER_WARPS 4
#endif

#ifndef NUM_CONSUMER_WARPS
#define NUM_CONSUMER_WARPS 4
#endif

constexpr int P = NUM_PRODUCER_WARPS;
constexpr int C = NUM_CONSUMER_WARPS;

constexpr int THREADS = (P + C) * 32;  // 160 = (128 + 32) threads, producer warp + 8 consumer warps

// TODO (yiakwy) : cluster multicast on sm121a is not supported for the moment
#if K_CLUSTER_SIZE_M > 1

extern "C" __global__ void __launch_bounds__(THREADS) __cluster_dims__(K_CLUSTER_SIZE_M, 1, 1)
mxfp4_gemm_kernel_entry(const __grid_constant__ CUtensorMap tma_A,
                        const __grid_constant__ CUtensorMap tma_B,
                        const uint8_t* __restrict__ scale_A,
                        const uint8_t* __restrict__ scale_B,
                        float* __restrict__ Out,
                        int M, int N, int K,
                        int num_blocks_m, int num_blocks_n) {
  extern __shared__ __align__(128) uint8_t smem_buffer[];
  xpu::BlackwellPersistentMxfp4Pipeline<K_BM, K_BN, K_BK, K_STAGES, K_GROUP_SIZE_M, K_CLUSTER_SIZE_M>
      ::run_persistent(&tma_A, &tma_B, scale_A, scale_B, Out, M, N, K,
                       num_blocks_m, num_blocks_n, smem_buffer);
}

#else

extern "C" __global__ void __launch_bounds__(THREADS)
mxfp4_gemm_kernel_entry(const __grid_constant__ CUtensorMap tma_A,
                        const __grid_constant__ CUtensorMap tma_B,
                        const uint8_t* __restrict__ scale_A,
                        const uint8_t* __restrict__ scale_B,
                        float* __restrict__ Out,
                        int M, int N, int K,
                        int num_blocks_m, int num_blocks_n) {
  extern __shared__ __align__(128) uint8_t smem_buffer[];
  xpu::BlackwellPersistentMxfp4Pipeline<K_BM, K_BN, K_BK, K_STAGES, K_GROUP_SIZE_M, K_CLUSTER_SIZE_M>
      ::run_persistent(&tma_A, &tma_B, scale_A, scale_B, Out, M, N, K,
                       num_blocks_m, num_blocks_n, smem_buffer);
}

#endif

} // anonymous namespace

void velox_dgx_mxfp4_gemm(tvm::ffi::TensorView packA, tvm::ffi::TensorView packB,
                          tvm::ffi::TensorView scaleA, tvm::ffi::TensorView scaleB,
                          tvm::ffi::TensorView out) {
  int64_t M = packA.size(0), K2 = packA.size(1), K = K2 * 2;
  int64_t N = packB.size(0);

  TVM_FFI_ICHECK_EQ(K % K_BK, 0) << "K must be divisible by BK=" << K_BK;
  TVM_FFI_ICHECK_EQ(M % K_BM, 0);
  TVM_FFI_ICHECK_EQ(N % K_BN, 0);

  static CUtensorMap desc_A, desc_B;

  {
    auto result = nvgpu::arch::make_2d_u8_desc(&desc_A, packA.data_ptr(), K2, M, K_BK / 2, K_BM, K2);
    TVM_FFI_ICHECK(result == CUDA_SUCCESS);       
  } // desc_A

  {
    auto result = nvgpu::arch::make_2d_u8_desc(&desc_B, packB.data_ptr(), K2, N, K_BK / 2, K_BN, K2);
    TVM_FFI_ICHECK(result == CUDA_SUCCESS);
  } //  desc_B

  // TODO (yiakwy) : use CEIL_DIV
  int nb_m = (M + K_BM - 1) / K_BM;
  int nb_n = (N + K_BN - 1) / K_BN;

  // TODO (yiakwy) : use CEIL_DIV
  int total_tiles = nb_m * nb_n;

  int grid_mn;
  if constexpr (K_CLUSTER_SIZE_M > 1) {
    grid_mn = (total_tiles + K_CLUSTER_SIZE_M - 1) / K_CLUSTER_SIZE_M;
  } else {
    grid_mn = total_tiles;
  }

  // TODO (yiakwy) : add split-k support
  dim3 grid(grid_mn, 1, 1);
  dim3 block(THREADS, 1, 1);

  size_t smem = sizeof(xpu::BlackwellPersistentMxfp4Pipeline<
    K_BM, K_BN, K_BK, K_STAGES, K_GROUP_SIZE_M, K_CLUSTER_SIZE_M>::SmemLayout);

  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(packA.device().device_type, packA.device().device_id));

  cudaFuncSetAttribute(mxfp4_gemm_kernel_entry,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  
  mxfp4_gemm_kernel_entry<<<grid, block, smem, stream>>>(
      desc_A, desc_B, 
      (const uint8_t*)scaleA.data_ptr(), (const uint8_t*)scaleB.data_ptr(),
      (float*)out.data_ptr(), (int)M, (int)N, (int)K, nb_m, nb_n);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dgx_mxfp4_gemm, velox_dgx_mxfp4_gemm);

#endif // defined(CUDA_VERSION) && CUDA_VERSION >= 1200

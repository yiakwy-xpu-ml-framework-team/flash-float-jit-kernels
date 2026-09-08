/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#include <cuda.h>

#if defined(CUDA_VERSION) && CUDA_VERSION >= 1200

#include "block/nv_block_power_mel_log.h"

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

namespace {

constexpr int K_BM = 64;
constexpr int K_BN = 80;
constexpr int K_BK = 64;

constexpr int NUM_SMs = 48;

using Pipeline = xpu::PowerMelLogPipeline<K_BM, K_BN, K_BK>;

extern "C" __global__ void __launch_bounds__(Pipeline::TOTAL_THREADS)
power_log_bf16_kernel_entry(
    const float2* __restrict__ spec,
    const float* __restrict__ mel,
    const float* __restrict__ cmvn_mean,
    const float* __restrict__ cmvn_istd,
    float* __restrict__ out,
    int T, int F_pad, int M, int F) {
  Pipeline pipeline;

  pipeline.run(nullptr, mel, cmvn_mean, cmvn_istd, out,
               T, F_pad, M, spec, F);
}

}  // namespace

void velox_power_log_bf16(tvm::ffi::TensorView spec,
                           tvm::ffi::TensorView mel,
                           tvm::ffi::TensorView cmvn_mean,
                           tvm::ffi::TensorView cmvn_istd,
                           tvm::ffi::TensorView out,
                           int64_t has_cmvn) {
  int64_t Tdim = spec.size(0);
  int64_t Fdim = spec.size(1);
  int64_t Mdim = mel.size(1);
  int64_t Fpad = mel.size(0);

  TVM_FFI_ICHECK_EQ(Fpad % K_BK, 0)
      << "F_padded must be divisible by BK=" << K_BK;

  TVM_FFI_ICHECK_LE(Mdim, K_BN);

  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(spec.device().device_type, spec.device().device_id));

  size_t smem = Pipeline::TOTAL_SMEM;

  cudaFuncSetAttribute(power_log_bf16_kernel_entry,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

  int nb_m = (Tdim + K_BM - 1) / K_BM;
  nb_m = (nb_m < NUM_SMs) ? nb_m : NUM_SMs;

  dim3 grid(nb_m, 1, 1);
  dim3 block(Pipeline::TOTAL_THREADS, 1, 1);

  const float* cm = (has_cmvn && cmvn_mean.data_ptr())
      ? static_cast<const float*>(cmvn_mean.data_ptr()) : nullptr;

  const float* ci = (has_cmvn && cmvn_istd.data_ptr())
      ? static_cast<const float*>(cmvn_istd.data_ptr()) : nullptr;

  power_log_bf16_kernel_entry<<<grid, block, smem, stream>>>(
      static_cast<const float2*>(spec.data_ptr()),
      static_cast<const float*>(mel.data_ptr()),
      cm, ci,
      static_cast<float*>(out.data_ptr()),
      (int)Tdim, (int)Fpad, (int)Mdim, (int)Fdim);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(power_mel_log, velox_power_log_bf16);

#endif

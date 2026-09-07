/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>

// TODO (yiakwy) : move to common files

namespace nvgpu { 
namespace arch {

inline CUresult make_2d_u8_desc(CUtensorMap* desc, const void* base,
                                uint64_t inner, uint64_t outer,
                                uint32_t box_inner, uint32_t box_outer,
                                uint64_t global_stride_inner_bytes) {
  uint64_t shape[2] = {inner, outer};
  uint64_t stride[1] = {global_stride_inner_bytes};
  uint32_t box[2] = {box_inner, box_outer};
  uint32_t step[2] = {1, 1};
  return cuTensorMapEncodeTiled(
      desc, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
      const_cast<void*>(base), shape, stride, box, step,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

__device__ inline void tma_load_2d_bytes(const CUtensorMap* desc, void* dst,
                                         int32_t off_i, int32_t off_o,
                                         uint64_t* bar) {
  const uint32_t dst_smem = (uint32_t)__cvta_generic_to_shared(dst);
  const uint64_t bar_smem = (uint64_t)__cvta_generic_to_shared(bar);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];\n"
      :: "r"(dst_smem), "l"(desc), "r"(off_i), "r"(off_o),
         "r"((uint32_t)bar_smem)
      : "memory");
}

__device__ inline void tma_load_2d_bytes_multicast(const CUtensorMap* desc,
                                                    void* dst,
                                                    int32_t off_i, int32_t off_o,
                                                    uint64_t* bar,
                                                    uint16_t cluster_mask) {
  const uint32_t dst_smem = (uint32_t)__cvta_generic_to_shared(dst);
  const uint64_t bar_smem = (uint64_t)__cvta_generic_to_shared(bar);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global"
      ".mbarrier::complete_tx::bytes"
      ".multicast::cluster"
      " [%0], [%1, {%2, %3}], [%4], %5;\n"
      :: "r"(dst_smem), "l"(desc), "r"(off_i), "r"(off_o),
         "r"((uint32_t)bar_smem), "h"(cluster_mask)
      : "memory");
}

__device__ inline void mbar_init(uint64_t* bar, uint32_t count) {
  uint32_t bar_addr = __cvta_generic_to_shared(bar);
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
               :: "r"(bar_addr), "r"(count) : "memory");
}

__device__ inline void mbar_expect_tx(uint64_t* bar, uint32_t tx) {
  uint32_t bar_addr = __cvta_generic_to_shared(bar);
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
      :: "r"(bar_addr), "r"(tx) : "memory");
}

__device__ inline void mbar_arrive(uint64_t* bar) {
  uint32_t bar_addr = __cvta_generic_to_shared(bar);
  asm volatile(
      "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], 1;\n"
      :: "r"(bar_addr) : "memory");
}

__device__ inline void mbar_wait(uint64_t* bar, uint32_t parity) {
  uint32_t bar_addr = __cvta_generic_to_shared(bar);
  asm volatile(
      "{.reg .pred p;\n"
      " WAIT_LOOP:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra.uni WAIT_LOOP;\n"
      "}\n"
      :: "r"(bar_addr), "r"(parity) : "memory");
}

__device__ inline void tma_expect_bytes(uint64_t* bar, uint32_t bytes) {
  uint32_t bar_addr = __cvta_generic_to_shared(bar);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               :: "r"(bar_addr), "r"(bytes) : "memory");
}

__device__ inline void tma_expect_bytes_u32(uint32_t bar_addr, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               :: "r"(bar_addr), "r"(bytes) : "memory");
}

__device__ inline uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(r) :);
  return r;
}

__device__ inline uint32_t cluster_map_shared_rank(uint32_t local_addr,
                                                   uint32_t cta_rank) {
  uint32_t mapped;
  asm("mapa.shared::cluster.u32 %0, %1, %2;"
      : "=r"(mapped) : "r"(local_addr), "r"(cta_rank));
  return mapped;
}

__device__ inline uint32_t cluster_map_shared_rank(void* local_ptr,
                                                   int target_rank) {
  uint32_t local_smem = __cvta_generic_to_shared(local_ptr);
  return cluster_map_shared_rank(local_smem, static_cast<uint32_t>(target_rank));
}

__device__ inline void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.aligned;\n" :: :);
}

__device__ inline void cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;\n" :: :);
}

__device__ inline void cluster_sync() {
  cluster_arrive();
  cluster_wait();
}

__device__ inline void mbar_arrive_cluster_release(uint64_t* bar,
                                                   uint32_t cta_rank) {
  uint32_t mapped = cluster_map_shared_rank(__cvta_generic_to_shared(bar),
                                            cta_rank);
  asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0], 1;\n"
               :: "r"(mapped) : "memory");
}

__device__ inline void cluster_cp_async_bulk(void* dst_local_smem,
                                             const void* src_remote_smem,
                                             uint32_t bytes,
                                             uint64_t* s_mbar) {
  uint32_t dst_addr = __cvta_generic_to_shared(dst_local_smem);
  uint32_t src_addr = __cvta_generic_to_shared(src_remote_smem);
  uint32_t mbar_addr = __cvta_generic_to_shared(s_mbar);
  asm volatile(
      "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];\n"
      :: "r"(dst_addr), "r"(src_addr), "r"(bytes), "r"(mbar_addr)
      : "memory");
}

__device__ inline void tma_store_fence() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

template<int Count = 0>
__device__ inline void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group.read %0;\n"
               :: "n"(Count) : "memory");
}

template<int num_threads = 288>
__device__ __forceinline__ void warpgroup_sync(int barrier_id = 7) {
  asm volatile("barrier.sync %0, %1;\n" ::
               "r"(barrier_id), "n"(num_threads) : "memory");
}

template<uint32_t RegCount>
__device__ inline void reg_alloc_increase_registers() {
  static_assert(RegCount % 8 == 0, "n_reg must be a multiple of 8");
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(RegCount));
}

template<uint32_t RegCount>
__device__ inline void reg_dealloc_decrease_registers() {
  static_assert(RegCount % 8 == 0, "n_reg must be a multiple of 8");
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(RegCount));
}

} // namespace arch
} // namespace nvgpu

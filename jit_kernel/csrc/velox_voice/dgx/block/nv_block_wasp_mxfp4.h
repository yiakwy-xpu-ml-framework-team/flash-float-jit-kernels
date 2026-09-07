/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include <cstdint>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include "fragment/nv_frag_mxfp4.h"
#include "fragment/nv_frag_mxfp4_accumulator.h"
#include "arch/tma/tma_lite.h"

#ifndef NUM_PRODUCER_WARPS
#define NUM_PRODUCER_WARPS 4
#endif
#ifndef NUM_CONSUMER_WARPS
#define NUM_CONSUMER_WARPS 4
#endif

#define CONSUMER_THREADS NUM_CONSUMER_WARPS * 32

namespace xpu {

// TODO (yiakwy) : remove
#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif

__device__ inline float ue8m0_to_fp32(uint8_t u) {
  return ldexpf(1.0f, static_cast<int>(u) - 127);
}

  // TODO (yiakwy) : move standard swizzle2d to common
  template<int GROUP_SIZE_M>
  static __device__ inline void swizzle2d(
      int tile, int num_blocks_m, int num_blocks_n, int& bm, int& bn) {
    if constexpr (GROUP_SIZE_M <= 1) {
      bm = tile / num_blocks_n;
      bn = tile % num_blocks_n;
    } else {
      const int num_groups = (num_blocks_m + GROUP_SIZE_M - 1) / GROUP_SIZE_M;

      const int tiles_per_group = GROUP_SIZE_M * num_blocks_n;

      const int group_id = tile / tiles_per_group;
      const int group_off_row = group_id * GROUP_SIZE_M;

      const int group_size_m = MIN(num_blocks_m - group_off_row, GROUP_SIZE_M);

      const int tile_in_group = tile % tiles_per_group;

      bm = group_off_row + tile_in_group % group_size_m;
      bn = tile_in_group / group_size_m;
    }
  }

template <int BM, int BN, int BK, int STAGES, int GROUP_SIZE_M, int CLUSTER_SIZE_M>
struct BlackwellPersistentMxfp4Pipeline {
  static constexpr int WARP = 32;

  static constexpr int P = NUM_PRODUCER_WARPS;
  static constexpr int C = NUM_CONSUMER_WARPS;

  static constexpr int TOTAL_WARPS = P + C;

  static constexpr int TOTAL_THREADS = TOTAL_WARPS * WARP;

  static constexpr int WN = 2;
  static constexpr int WM = C / WN;

  // NOTE (yiakwy) : warps layout
  static constexpr int TM_ROWS = BM / WM;
  static constexpr int TN_COLS = BN / WN;

  static constexpr int TM = TM_ROWS / 16;
  static constexpr int TN = TN_COLS / 8;
  static constexpr int KWP = BK / 8;

  static_assert(C % WN == 0, "C must be divisible by WN=2");
  static_assert(TM_ROWS % 16 == 0, "BM/WM must be divisible by 16 (MMA m)");
  static_assert(TN_COLS % 8 == 0, "BN/WN must be divisible by 8 (MMA n)");

  // NOTE (yiakwy) : shared memory descriptor
  struct SmemLayout {
    // NOTE (yiakwy) : multi stages buffer for producers
    uint32_t shmem_X[STAGES][BM * KWP];
    uint32_t shmem_W[STAGES][BN * KWP];

    // NOTE (yiakwy) : mbarriers for coordinating producers and consumers
    uint64_t full[STAGES];
    uint64_t empty[STAGES];

    // NOTE (yiakwy) : epilogue memory buffer
    float acc_smem[BM * BN];

    // NOTE (yiakwy) : row/col scales
    float xs_fp32[BM];
    float ws_fp32[BN];
  };

  static __device__ inline void run_persistent(
      const CUtensorMap* tma_desc_A, const CUtensorMap* tma_desc_B,
      const uint8_t* scale_A, const uint8_t* scale_B,
      float* Out, int M, int N, int K,
      int num_blocks_m, int num_blocks_n, uint8_t* smem_buffer) {

    SmemLayout* smem = reinterpret_cast<SmemLayout*>(smem_buffer);

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP;
    const int lane_id = tid % WARP;

    const bool is_producer = (warp_id < P);

    // TODO (yiakwy) : prefetch

    // TODO (yiakwy) : init NoC multicast cluster for sm121a

    // prepare
    if (tid < STAGES) {
      nvgpu::arch::mbar_init(&smem->full[tid], 1);
      nvgpu::arch::mbar_init(&smem->empty[tid], C);
    }
    nvgpu::arch::warpgroup_sync<TOTAL_THREADS>();

    const int total_tiles = num_blocks_m * num_blocks_n;
    int local_task_id = blockIdx.x;
     
    nvgpu::arch::warpgroup_sync<TOTAL_THREADS>();

    for (int tile = local_task_id; tile < total_tiles; tile += gridDim.x) {
      int bm, bn;
      swizzle2d<GROUP_SIZE_M>(tile, num_blocks_m, num_blocks_n, bm/*dst*/, bn/*dst*/);

      const int baseM = bm * BM;
      const int baseN = bn * BN;

      const int ns = K / BK;

      // Start TMA pipeline
      if (is_producer) {
        const int producer_id = warp_id;

        if (lane_id == 0) {
          for (int s = producer_id; s < ns; s += P) {
            const int buf = s % STAGES;

            if (s >= STAGES) {
              nvgpu::arch::mbar_wait(&smem->empty[buf], ((s / STAGES) & 1) ^ 1);
            }

            nvgpu::arch::mbar_expect_tx(&smem->full[buf], BM * KWP * 4 + BN * KWP * 4);

            // TODO (yiakwy) : rename shmem_X, shmem_W
            nvgpu::arch::tma_load_2d_bytes(tma_desc_A, smem->shmem_X[buf],
                                           s * (BK / 2), baseM,
                                           &smem->full[buf]);
            nvgpu::arch::tma_load_2d_bytes(tma_desc_B, smem->shmem_W[buf],
                                           s * (BK / 2), baseN,
                                           &smem->full[buf]);
          }
        }
      }

      if (!is_producer) {
        const int cw = warp_id - P;
        const int _c_tid = cw * WARP + lane_id;

        // NOTE (yiakwy) : warp offset
        const int cm = cw / WN;
        const int cn = cw % WN;

        // NOTE (yiakwy) : fragment offset
        const int warpM = cm * TM_ROWS;
        const int warpN = cn * TN_COLS;

        Mxfp4Accumulator<TM, TN> accum;
        accum.clear();

        // TODO (yiakwy) : support mxfp4 / nvfp4 with mma_scaled

        // NOTE (yiakwy) : prefetch all per-row, per-col e8m0 scales
    #pragma unroll 4
        for (int i = _c_tid; i < BM; i += CONSUMER_THREADS) {
            int g_row = baseM + i;
            smem->xs_fp32[i] = (g_row < M) ? ue8m0_to_fp32(scale_A[g_row]) : 0.f;
        }
    #pragma unroll 4
        for (int i = _c_tid; i < BN; i += CONSUMER_THREADS) {
            int g_col = baseN + i;
            smem->ws_fp32[i] = (g_col < N) ? ue8m0_to_fp32(scale_B[g_col]) : 0.f;
        }
        nvgpu::arch::warpgroup_sync<CONSUMER_THREADS>(1);

        for (int s = 0; s < ns; ++s) {
          const int buf = s % STAGES;
          nvgpu::arch::mbar_wait(&smem->full[buf], (s / STAGES) & 1);

          // TODO (yiakwy) : rename shmem_X, shmem_W
          accum.mma_scaled(smem->shmem_X[buf], smem->shmem_W[buf],
                          warpM, warpN, KWP, lane_id);

          if (lane_id == 0) {
            nvgpu::arch::mbar_arrive(&smem->empty[buf]);
          }
        }

        accum.store<BM, BN>(Out, smem->xs_fp32, smem->ws_fp32,
                            baseM, baseN, M, N, warpM, warpN, lane_id);
      }

      // TODO (yiakwy) : using TMA store and remove syncs cross producer and consumer
      nvgpu::arch::warpgroup_sync<TOTAL_THREADS>();
    }
  }
};

}  // namespace xpu

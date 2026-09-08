/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once

#include <cstdint>
#include <cuda.h>

#include "fragment/nv_frag_mxfp4.h"

// USE_LDMATRIX=1 triggers cooperative warp-level fragment
#ifndef USE_LDMATRIX
#define USE_LDMATRIX 0
#endif

namespace xpu {

template <int _TM/*2*/, int _TN/*8*/>
struct Mxfp4Accumulator {
  static constexpr int TM = _TM;
  static constexpr int TN = _TN;
  static constexpr int REGS = TM * TN * 4;

  float acc[REGS]; // 4x 16 (repeats) floats per thread, to form 32x64 distributed layout (check triton for the distributed layout) 

  __device__ inline void clear() {
#pragma unroll
    // TODO (yiakwy) : memset
    for (int i = 0; i < REGS; ++i) acc[i] = 0.f;
  }

#if USE_LDMATRIX

  __device__ inline void load_af(const uint32_t* shmem_X, int warpM,
                                 int KWP, int lane_id, uint32_t af[TM][4]) {
    // A: row = warpM + mi*16 + 2*g (+0 or +1 for rE/rO)
    //    k  = t*2 (+0 or +1 for the two halves)  
    const int g = lane_id / 4, t = lane_id % 4;
#pragma unroll
    for (int mi = 0; mi < TM; ++mi) {
      const int base = warpM + mi * 16;
      /* 4 ldmatrix.x4: rows [0..3], [4..7], [8..11], [12..15] */
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        int row = base + k * 4 + (lane_id % 4);
        uint32_t ptr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shmem_X[row * KWP + t * 2]));
        asm volatile(
            "ldmatrix.sync.aligned.m8n4.x4.shared.b128 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(af[mi][k*4+0]), "=r"(af[mi][k*4+1]),
              "=r"(af[mi][k*4+2]), "=r"(af[mi][k*4+3])
            : "r"(ptr));
      }
    }
  }

  __device__ inline void load_bf(const uint32_t* shmem_W, int warpN,
                                 int KWP, int lane_id, uint32_t bf[TN][2]) {
    // B : col = warpN + ni*8 + g
    //     k  = t*2 (+0 or +1) 
    const int g = lane_id / 4, t = lane_id % 4;
#pragma unroll
    for (int ni = 0; ni < TN; ++ni) {
      int row = warpN + ni * 8 + g;
      uint32_t ptr = static_cast<uint32_t>(
          __cvta_generic_to_shared(&shmem_W[row * KWP + t * 2]));
      asm volatile(
          "ldmatrix.sync.aligned.m8n4.x2.shared.b128 {%0,%1}, [%2];\n"
          : "=r"(bf[ni][0]), "=r"(bf[ni][1])
          : "r"(ptr));
    }
  }

#else
  /* Scalar path: 6 lds.u32 per (mi,ni) pair. */
  __device__ inline void load_af(const uint32_t* shmem_X, int warpM,
                                 int KWP, int lane_id, uint32_t af[TM][4]) {
    const int g = lane_id / 4, t = lane_id % 4;
#pragma unroll
    for (int mi = 0; mi < TM; ++mi) {
      int rE = warpM + mi * 16 + 2 * g, rO = rE + 1;
      af[mi][0] = shmem_X[rE * KWP + t * 2];
      af[mi][1] = shmem_X[rO * KWP + t * 2];
      af[mi][2] = shmem_X[rE * KWP + t * 2 + 1];
      af[mi][3] = shmem_X[rO * KWP + t * 2 + 1];
    }
  }

  __device__ inline void load_bf(const uint32_t* shmem_W, int warpN,
                                 int KWP, int lane_id, uint32_t bf[TN][2]) {
    const int g = lane_id / 4, t = lane_id % 4;
#pragma unroll
    for (int ni = 0; ni < TN; ++ni) {
      int col = warpN + ni * 8 + g;
      bf[ni][0] = shmem_W[col * KWP + t * 2];
      bf[ni][1] = shmem_W[col * KWP + t * 2 + 1];
    }
  }
#endif

  __device__ inline void mma_scaled(const uint32_t* shmem_X, const uint32_t* shmem_W,
                                    int warpM, int warpN,
                                    int KWP, int lane_id) {
    uint32_t af[TM][4], bf[TN][2];
    load_af(shmem_X, warpM, KWP, lane_id, af);
    load_bf(shmem_W, warpN, KWP, lane_id, bf);

#pragma unroll
    for (int mi = 0; mi < TM; ++mi)
#pragma unroll
      for (int ni = 0; ni < TN; ++ni)
        mxfp4_mma(acc + (mi * TN + ni) * 4, af[mi], bf[ni]);
  }

  template <int BM, int BN>
  __device__ inline void store(float* smem, int warpM, int warpN,
                               int lane_id, int M, int N) {
    const int g = lane_id / 4, t = lane_id % 4;
#pragma unroll
    for (int mi = 0; mi < TM; ++mi) {
#pragma unroll
      for (int ni = 0; ni < TN; ++ni) {
        const int r0 = warpM + mi * 16 + 2 * g;
        const int c0 = warpN + ni * 8 + 2 * t;
        const int base = (mi * TN + ni) * 4;
        if (r0   < M && c0   < N) smem[r0 * BN + c0]         = acc[base];
        if (r0   < M && c0+1 < N) smem[r0 * BN + c0 + 1]     = acc[base+1];
        if (r0+1 < M && c0   < N) smem[(r0+1) * BN + c0]     = acc[base+2];
        if (r0+1 < M && c0+1 < N) smem[(r0+1) * BN + c0 + 1] = acc[base+3];
      }
    }
  }

  // TODO (yiakwy) : override epilogue store without scale
  template <int BM, int BN>
  __device__ inline void store(float* Out, const float* xs_fp32,
                               const float* ws_fp32,
                               int baseM, int baseN, int M, int N,
                               int warpM, int warpN, int lane_id) {
    const int g = lane_id / 4, t = lane_id % 4;
#pragma unroll
    for (int mi = 0; mi < TM; ++mi) {
#pragma unroll
      for (int ni = 0; ni < TN; ++ni) {
        const int row0 = baseM + warpM + mi * 16 + 2 * g;
        const int col0 = baseN + warpN + ni * 8 + 2 * t;
        const int base = (mi * TN + ni) * 4;
        const float s_r0 = (row0   < M) ? xs_fp32[row0 - baseM] : 0.f;
        const float s_r1 = (row0+1 < M) ? xs_fp32[row0+1 - baseM] : 0.f;
        const float s_c0 = (col0   < N) ? ws_fp32[col0 - baseN] : 0.f;
        const float s_c1 = (col0+1 < N) ? ws_fp32[col0+1 - baseN] : 0.f;

        // TODO (yiakwy) : using TMA store
        if (row0   < M && col0   < N) Out[(row0+0)*N+col0+0] = acc[base+0] * s_r0 * s_c0;
        if (row0   < M && col0+1 < N) Out[(row0+0)*N+col0+1] = acc[base+1] * s_r0 * s_c1;
        if (row0+1 < M && col0   < N) Out[(row0+1)*N+col0+0] = acc[base+2] * s_r1 * s_c0;
        if (row0+1 < M && col0+1 < N) Out[(row0+1)*N+col0+1] = acc[base+3] * s_r1 * s_c1;

      } // TN
    } // TM
  }
};

template <typename _T, int BM, int BN>
struct FragmentView {
  using T = _T;
  T* shared_ptr;

  __device__ inline FragmentView(T* smem) : shared_ptr(smem) {}

  __device__ inline T& operator()(int m, int n) {
    return shared_ptr[m * BN + n];
  }

  __device__ inline const T& operator()(int m, int n) const {
    return shared_ptr[m * BN + n];
  }
};

}  // namespace xpu

/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once
#include <cuda_runtime.h>

#ifndef CONSUMER_THREADS
#define CONSUMER_THREADS 256
#endif

#ifndef WARP_SIZE

#define WARP_SIZE 32

#endif

#ifndef SWIZZLE_64B_STORE

#define SWIZZLE_64B_STORE 0

#endif



static __device__ __forceinline__ void _warpgroup_sync(int barrier_id=7) {
    asm volatile("barrier.cta.sync %0, %1;\n" ::"r"(barrier_id), "n"(128) : "memory");
}


namespace xpu {

enum class MemoryDomain { kShared, kRegister };

template <typename _T, int BM, int BN, MemoryDomain Domain>
struct FragmentView {

    using T = _T;

    static constexpr int VEC_SIZE = sizeof(T);

    T* shared_ptr;

    __device__ inline FragmentView(T* smem) : shared_ptr(smem) {}

    __device__ inline T& operator()(int m, int n) {
        return shared_ptr[m * BN + n];
    }

    // NOTE (yiakwy) : implement inplace transpose for 128x128 fp32 fragment
    // TODO (yaikwy) : add inplace transpose for 128x128 fp16/bf16 fragment
    __device__ inline void _transpose() {
        static_assert(BM == BN, "inplace transpose can be only applied to square fragment.");

        const int tid = threadIdx.x;
        constexpr int threads_per_block = CONSUMER_THREADS;

        const int lane_id = threadIdx.x % WARP_SIZE;
        const int warp_id = threadIdx.x / WARP_SIZE;

        const int wg_id = warp_id / 4;

        constexpr int FRAG_M = 16;
        constexpr int TOTAL_ELEMENTS = BM * BN;

        constexpr int M_STEPS = BM / FRAG_M;

        constexpr int total_diagonal_pairs = (FRAG_M * FRAG_M - FRAG_M) / 2;

        constexpr int SWIZZLE_SHIFT = (sizeof(T) == 2) ? 3 : 2;

        // NOTE (yiakwy) : process lower left sub fragment
        #pragma unroll
        for (int sub_frag_idx_m = 1; sub_frag_idx_m < M_STEPS; ++sub_frag_idx_m) {

            int sub_frag_idx_m_off = sub_frag_idx_m * FRAG_M;

            #pragma unroll
            for (int sub_frag_idx_n = 0; sub_frag_idx_n < sub_frag_idx_m; ++sub_frag_idx_n) {

                int sub_frag_idx_n_off = sub_frag_idx_n * FRAG_M;

                #pragma unroll
                for (int e_idx = tid; e_idx < FRAG_M * FRAG_M; e_idx += threads_per_block) {
                    int thr_col = e_idx % FRAG_M;
                    int thr_row = e_idx / FRAG_M;

                    int row_src = sub_frag_idx_m_off + thr_row;
                    int col_src = sub_frag_idx_n_off + thr_col;

                    int row_dst = sub_frag_idx_n_off + thr_col;
                    int col_dst = sub_frag_idx_m_off + thr_row;

                    int src_idx, dst_idx;

                    // NOTE(yiakwy) : only valid for 16x16 fragment
#if SWIZZLE_64B_STORE
                    int swizzle_col = thr_col ^ (thr_row % 8);
                    int swizzle_row = thr_row ^ (thr_col % 8);

                    int swizzle_col_src = col_src ^ ((row_src & 7) << SWIZZLE_SHIFT);
                    int swizzle_col_dst = col_dst ^ ((row_dst & 7) << SWIZZLE_SHIFT);

                    src_idx = row_src * BM + swizzle_col_src;
                    dst_idx = row_dst * BM + swizzle_col_dst;
#else
                    src_idx = row_src * BM + col_src;
                    dst_idx = row_dst * BM + col_dst;

                    int swizzle_col = thr_col;
                    int swizzle_row = thr_row;
#endif

                    T src_val = shared_ptr[src_idx];
                    T dst_val = shared_ptr[dst_idx];

                    shared_ptr[src_idx] = dst_val;
                    shared_ptr[dst_idx] = src_val;
                } // end of e_idx

            } // end of sub_frag_idx_n

        } // end of sub_frag_idx_m

        _warpgroup_sync(wg_id);

        #pragma unroll
        for (int task_idx = 0; task_idx < M_STEPS; task_idx++) {

            int sub_frag_idx_m_off = task_idx * FRAG_M;
            int sub_frag_idx_n_off = task_idx * FRAG_M;

            for (int pair_idx = tid; pair_idx < total_diagonal_pairs; pair_idx += threads_per_block) {
                int thr_row = static_cast<int>((1 + __fsqrt_rn(1 + 8 * pair_idx)) / 2);
                int thr_col = pair_idx - (thr_row * (thr_row - 1)) / 2;

                int row_src = sub_frag_idx_m_off + thr_row;
                int col_src = sub_frag_idx_n_off + thr_col;

                int row_dst = sub_frag_idx_n_off + thr_col;
                int col_dst = sub_frag_idx_m_off + thr_row;

                int src_idx, dst_idx;

#if SWIZZLE_64B_STORE
                int swizzle_col_src = col_src ^ ((row_src & 7) << SWIZZLE_SHIFT);
                int swizzle_col_dst = col_dst ^ ((row_dst & 7) << SWIZZLE_SHIFT);

                src_idx = row_src * BM + swizzle_col_src;
                dst_idx = row_dst * BM + swizzle_col_dst;
#else
                src_idx = row_src * BM + col_src;
                dst_idx = row_dst * BM + col_dst;
#endif

                T src_val = shared_ptr[src_idx];
                T dst_val = shared_ptr[dst_idx];

                shared_ptr[src_idx] = dst_val;
                shared_ptr[dst_idx] = src_val;
            }

        } // end of task_idx

        _warpgroup_sync(wg_id);

    }
};

} // namespace xpu

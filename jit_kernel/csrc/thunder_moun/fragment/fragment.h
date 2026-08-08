/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

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

    __device__ __forceinline__ static uint32_t warp_transpose_8x8_half2(
        uint32_t packed,
        int lane_id) {
        const int out_row = lane_id / 4;
        const int out_pair = lane_id % 4;

        const int source_pair = out_row / 2;
        const int half_select = out_row % 2;
        const int bit_shift = half_select * 16;

        const int source_lane_0 = (2 * out_pair) * 4 + source_pair;
        const int source_lane_1 = (2 * out_pair + 1) * 4 + source_pair;

        const uint32_t word_0 = __shfl_sync(0xffffffffu, packed, source_lane_0);
        const uint32_t word_1 = __shfl_sync(0xffffffffu, packed, source_lane_1);

        const uint32_t value_0 = (word_0 >> bit_shift) & 0xffffu;
        const uint32_t value_1 = (word_1 >> bit_shift) & 0xffffu;
        return value_0 | (value_1 << 16);
    }

    // Experimental row-major-only transpose using a direct 8x8 tiling.
    __device__ inline void _transpose_8x8() {
        constexpr int TILE = 8;
        constexpr int NUM_TILES = BM / TILE;
        constexpr int TOTAL_OFFDIAGONAL_TILE_PAIRS =
            NUM_TILES * (NUM_TILES - 1) / 2;
        constexpr int WARPS_PER_CTA = 8;

        static_assert(sizeof(T) == 2, "optimized transpose only supports 16-bit elements.");
        static_assert(BM == BN, "inplace transpose can be only applied to square fragment.");
        static_assert(BM % TILE == 0, "optimized transpose requires BM to be divisible by 8.");

        const int lane_id = threadIdx.x % WARP_SIZE;
        const int warp_id = threadIdx.x / WARP_SIZE;

        const int wg_id = warp_id / 4;

        const int lane_row = lane_id / 4;
        const int lane_pair = lane_id % 4;
        const int local_col = 2 * lane_pair;

        // Each of the eight warps handles one off-diagonal 8x8 tile pair.
        #pragma unroll
        for (int pair_base = 0;
             pair_base < TOTAL_OFFDIAGONAL_TILE_PAIRS;
             pair_base += WARPS_PER_CTA) {
            const int tile_pair_idx = pair_base + warp_id;

            if (tile_pair_idx < TOTAL_OFFDIAGONAL_TILE_PAIRS) {
                int tile_m = 1;

                #pragma unroll
                for (int candidate_m = 2; candidate_m < NUM_TILES; ++candidate_m) {
                    if (tile_pair_idx >= candidate_m * (candidate_m - 1) / 2) {
                        tile_m = candidate_m;
                    }
                }

                const int tile_n =
                    tile_pair_idx - tile_m * (tile_m - 1) / 2;

                const int src_row_base = tile_m * TILE;
                const int src_col_base = tile_n * TILE;
                const int dst_row_base = tile_n * TILE;
                const int dst_col_base = tile_m * TILE;

                const int src_idx =
                    (src_row_base + lane_row) * BM + src_col_base + local_col;
                const int dst_idx =
                    (dst_row_base + lane_row) * BM + dst_col_base + local_col;

                const uint32_t src_old =
                    *reinterpret_cast<const uint32_t*>(shared_ptr + src_idx);
                const uint32_t dst_old =
                    *reinterpret_cast<const uint32_t*>(shared_ptr + dst_idx);

                const uint32_t src_t = warp_transpose_8x8_half2(src_old, lane_id);
                const uint32_t dst_t = warp_transpose_8x8_half2(dst_old, lane_id);

                *reinterpret_cast<uint32_t*>(shared_ptr + src_idx) = dst_t;
                *reinterpret_cast<uint32_t*>(shared_ptr + dst_idx) = src_t;
            }
        }

        // The same eight warps transpose the sixteen diagonal 8x8 tiles in two rounds.
        #pragma unroll
        for (int tile_base = 0; tile_base < NUM_TILES; tile_base += WARPS_PER_CTA) {
            const int tile_idx = tile_base + warp_id;

            if (tile_idx < NUM_TILES) {
                const int row_base = tile_idx * TILE;
                const int col_base = tile_idx * TILE;
                const int packed_idx =
                    (row_base + lane_row) * BM + col_base + local_col;

                const uint32_t old_value =
                    *reinterpret_cast<const uint32_t*>(shared_ptr + packed_idx);
                const uint32_t transposed =
                    warp_transpose_8x8_half2(old_value, lane_id);

                *reinterpret_cast<uint32_t*>(shared_ptr + packed_idx) = transposed;
            }
        }

        // __syncthreads();
        _warpgroup_sync(wg_id);
    }

    // Experimental row-major-only out-of-place transpose using direct 8x8 tiles.
    __device__ inline void _transpose_outplace_8x8(T* dst_shared_ptr) const {
        constexpr int TILE = 8;
        constexpr int NUM_TILES = BM / TILE;
        constexpr int TOTAL_TILES = NUM_TILES * NUM_TILES;
        constexpr int WARPS_PER_CTA = 8;

        static_assert(sizeof(T) == 2, "optimized transpose only supports 16-bit elements.");
        static_assert(BM == BN, "out-of-place transpose can be only applied to square fragment.");
        static_assert(BM % TILE == 0, "optimized transpose requires BM to be divisible by 8.");

        const int lane_id = threadIdx.x % WARP_SIZE;
        const int warp_id = threadIdx.x / WARP_SIZE;

        const int wg_id = warp_id / 4;

        const int lane_row = lane_id / 4;
        const int lane_pair = lane_id % 4;
        const int local_col = 2 * lane_pair;

        // Source and destination do not overlap. Each warp therefore handles one
        // complete 8x8 tile rather than exchanging an off-diagonal tile pair.
        #pragma unroll
        for (int tile_base = 0;
             tile_base < TOTAL_TILES;
             tile_base += WARPS_PER_CTA) {
            const int tile_idx = tile_base + warp_id;

            if (tile_idx < TOTAL_TILES) {
                const int src_tile_row = tile_idx / NUM_TILES;
                const int src_tile_col = tile_idx % NUM_TILES;

                const int src_row_base = src_tile_row * TILE;
                const int src_col_base = src_tile_col * TILE;
                const int dst_row_base = src_tile_col * TILE;
                const int dst_col_base = src_tile_row * TILE;

                const int src_idx =
                    (src_row_base + lane_row) * BM + src_col_base + local_col;
                const int dst_idx =
                    (dst_row_base + lane_row) * BM + dst_col_base + local_col;

                const uint32_t old_value =
                    *reinterpret_cast<const uint32_t*>(shared_ptr + src_idx);
                const uint32_t transposed =
                    warp_transpose_8x8_half2(old_value, lane_id);

                *reinterpret_cast<uint32_t*>(dst_shared_ptr + dst_idx) = transposed;
            }
        }

        _warpgroup_sync(wg_id);
    }
};

} // namespace xpu

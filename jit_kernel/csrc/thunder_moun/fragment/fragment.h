/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <cuda_fp16.h>

#ifndef CONSUMER_THREADS
#define CONSUMER_THREADS 256
#endif

#ifndef WARP_SIZE

#define WARP_SIZE 32

#endif

#ifndef WARPS_PER_CTA

#define WARPS_PER_CTA 8

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

    // NOTE (yiakwy) : implement inplace transpose for 128x128 fp32 fragment
    // TODO (yaikwy) : add inplace transpose for 128x128 fp16/bf16 fragment
    __device__ inline void _transpose_opt() {
        static_assert(BM == BN, "inplace transpose can be only applied to square fragment.");

        const int tid = threadIdx.x;
        constexpr int threads_per_block = CONSUMER_THREADS;

        const int lane_id = threadIdx.x % WARP_SIZE;
        const int warp_id = threadIdx.x / WARP_SIZE;

        const int wg_id = warp_id / 4;

        constexpr int FRAG_M = 16;
        constexpr int TOTAL_ELEMENTS = BM * BN;

        constexpr int M_STEPS = BM / FRAG_M;

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
                    int thr_row = e_idx % FRAG_M;
                    int thr_offset = lane_id / FRAG_M;

                    int thr_col = (thr_row + thr_offset + warp_id * 2) % FRAG_M;

                    int row_src = sub_frag_idx_m_off + thr_row;
                    int col_src = sub_frag_idx_n_off + thr_col;

                    int row_dst = sub_frag_idx_n_off + thr_col;
                    int col_dst = sub_frag_idx_m_off + thr_row;

                    int src_idx, dst_idx;

                    // NOTE(erke) : Inplace Swizzle not available yet
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

        const int warp_rank_in_group = warp_id % 4;
        const int half_warp_id = lane_id / FRAG_M;
        const int cyclic_distance = 1 + half_warp_id + 2 * warp_rank_in_group;
        const int thr_col = lane_id % FRAG_M;
        const int thr_row = (thr_col + cyclic_distance) % FRAG_M;
        const bool is_unique_pair =
            cyclic_distance != FRAG_M / 2 || thr_col < FRAG_M / 2;

        #pragma unroll
        for (int task_idx = 0; task_idx < M_STEPS; task_idx+=2) {
            int sub_frag_idx = task_idx + wg_id;
            int sub_frag_idx_m_off = sub_frag_idx * FRAG_M;
            int sub_frag_idx_n_off = sub_frag_idx * FRAG_M;

            if (is_unique_pair) {
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

        } // end of tile_pair_base
        _warpgroup_sync(wg_id);

    }

    template <int FRAG_M>
    __device__ __forceinline__ static void warp_transpose_16x16(half2* values, int lane_id) {
        constexpr int PACKS_PER_ROW = FRAG_M / 2;
        constexpr int ROWS_PER_ACCESS = WARP_SIZE / PACKS_PER_ROW;
        constexpr int VALUES_PER_LANE = FRAG_M / ROWS_PER_ACCESS;

        static_assert(FRAG_M == 16, "warp transpose only supports 16x16 fragments.");

        // NOTE (erke) : swap row bit 0 with column bit 0
        #pragma unroll
        for (int e_idx = 0; e_idx < VALUES_PER_LANE; ++e_idx) {
            const half2 peer = __shfl_xor_sync(0xffffffffu, values[e_idx], 8);

            if ((lane_id & 8) == 0) {
                values[e_idx] = __lows2half2(values[e_idx], peer);
            } else {
                values[e_idx] = __highs2half2(peer, values[e_idx]);
            }
        }

        // NOTE (erke) : swap row bit 1 with column bit 1
        const int source_lane = (lane_id & ~(16 | 1)) | ((lane_id & 1) << 4) | ((lane_id & 16) >> 4);

        #pragma unroll
        for (int e_idx = 0; e_idx < VALUES_PER_LANE; ++e_idx) {
            values[e_idx] = __shfl_sync(0xffffffffu, values[e_idx], source_lane);
        }

        // NOTE (erke) : swap row bit 3 with column bit 3
        const bool is_high_col_group = (lane_id & 4) != 0;

        #pragma unroll
        for (int e_idx = 0; e_idx < VALUES_PER_LANE / 2; ++e_idx) {
            const half2 cross_value = is_high_col_group ? values[e_idx] : values[VALUES_PER_LANE / 2 + e_idx];
            const half2 peer = __shfl_xor_sync(0xffffffffu, cross_value, 4);

            if (is_high_col_group) {
                values[e_idx] = peer;
            } else {
                values[VALUES_PER_LANE / 2 + e_idx] = peer;
            }
        }

        // NOTE (erke) : swap row bit 2 with column bit 2
        const bool is_odd_col_group = (lane_id & 2) != 0;

        #pragma unroll
        for (int e_idx = 0; e_idx < VALUES_PER_LANE; e_idx += 2) {
            const half2 cross_value = is_odd_col_group ? values[e_idx] : values[e_idx + 1];
            const half2 peer = __shfl_xor_sync(0xffffffffu, cross_value, 2);

            if (is_odd_col_group) {
                values[e_idx] = peer;
            } else {
                values[e_idx + 1] = peer;
            }
        }
    }

    // NOTE (erke) : implement inplace transpose for 128x128 fp16 fragment
    __device__ inline void _transpose_fp16() {
        static_assert(BM == BN, "inplace transpose can be only applied to square fragment.");
        static_assert(sizeof(T) == 2, "optimized fp16 transpose only supports 16-bit elements.");

        const int lane_id = threadIdx.x % WARP_SIZE;
        const int warp_id = threadIdx.x / WARP_SIZE;

        const int wg_id = warp_id / 4;

        constexpr int FRAG_M = 16;
        constexpr int M_STEPS = BM / FRAG_M;
        constexpr int PACKS_PER_ROW = FRAG_M / 2;
        constexpr int ROWS_PER_ACCESS = WARP_SIZE / PACKS_PER_ROW;
        constexpr int VALUES_PER_LANE = FRAG_M / ROWS_PER_ACCESS;

        static_assert(BM % FRAG_M == 0, "optimized fp16 transpose requires BM to be divisible by 16.");

        constexpr int total_off_diagonal_pairs = (M_STEPS * M_STEPS - M_STEPS) / 2;
        constexpr int total_tasks = total_off_diagonal_pairs + M_STEPS / 2;

        const int thr_row = lane_id / PACKS_PER_ROW;
        const int thr_col = 2 * (lane_id % PACKS_PER_ROW);

        // NOTE (erke) : one off-diagonal pair has the same work as two diagonal tiles
        #pragma unroll
        for (int task_idx = warp_id; task_idx < total_tasks; task_idx += WARPS_PER_CTA) {
            if (task_idx < total_off_diagonal_pairs) {
                const int sub_frag_idx_m = static_cast<int>((1 + __fsqrt_rn(1 + 8 * task_idx)) / 2);
                const int sub_frag_idx_n = task_idx - (sub_frag_idx_m * (sub_frag_idx_m - 1)) / 2;

                const int sub_frag_idx_m_off = sub_frag_idx_m * FRAG_M;
                const int sub_frag_idx_n_off = sub_frag_idx_n * FRAG_M;

                half2 src_values[VALUES_PER_LANE];
                half2 dst_values[VALUES_PER_LANE];

                #pragma unroll
                for (int e_idx = 0; e_idx < VALUES_PER_LANE; ++e_idx) {
                    const int row_offset = ROWS_PER_ACCESS * e_idx + thr_row;
                    const int src_idx = (sub_frag_idx_m_off + row_offset) * BM + sub_frag_idx_n_off + thr_col;
                    const int dst_idx = (sub_frag_idx_n_off + row_offset) * BM + sub_frag_idx_m_off + thr_col;

                    src_values[e_idx] = *reinterpret_cast<const half2*>(shared_ptr + src_idx);
                    dst_values[e_idx] = *reinterpret_cast<const half2*>(shared_ptr + dst_idx);
                }

                warp_transpose_16x16<FRAG_M>(src_values, lane_id);
                warp_transpose_16x16<FRAG_M>(dst_values, lane_id);

                #pragma unroll
                for (int e_idx = 0; e_idx < VALUES_PER_LANE; ++e_idx) {
                    const int row_offset = ROWS_PER_ACCESS * e_idx + thr_row;
                    const int src_idx = (sub_frag_idx_m_off + row_offset) * BM + sub_frag_idx_n_off + thr_col;
                    const int dst_idx = (sub_frag_idx_n_off + row_offset) * BM + sub_frag_idx_m_off + thr_col;

                    *reinterpret_cast<half2*>(shared_ptr + src_idx) = dst_values[e_idx];
                    *reinterpret_cast<half2*>(shared_ptr + dst_idx) = src_values[e_idx];
                }
            } else {
                const int diagonal_pair_idx = task_idx - total_off_diagonal_pairs;

                #pragma unroll
                for (int diagonal_idx = 0; diagonal_idx < 2; ++diagonal_idx) {
                    const int sub_frag_idx = 2 * diagonal_pair_idx + diagonal_idx;
                    const int sub_frag_idx_off = sub_frag_idx * FRAG_M;

                    half2 values[VALUES_PER_LANE];

                    #pragma unroll
                    for (int e_idx = 0; e_idx < VALUES_PER_LANE; ++e_idx) {
                        const int row_offset = ROWS_PER_ACCESS * e_idx + thr_row;
                        const int shared_idx = (sub_frag_idx_off + row_offset) * BM + sub_frag_idx_off + thr_col;

                        values[e_idx] = *reinterpret_cast<const half2*>(shared_ptr + shared_idx);
                    }

                    warp_transpose_16x16<FRAG_M>(values, lane_id);

                    #pragma unroll
                    for (int e_idx = 0; e_idx < VALUES_PER_LANE; ++e_idx) {
                        const int row_offset = ROWS_PER_ACCESS * e_idx + thr_row;
                        const int shared_idx = (sub_frag_idx_off + row_offset) * BM + sub_frag_idx_off + thr_col;

                        *reinterpret_cast<half2*>(shared_ptr + shared_idx) = values[e_idx];
                    }
                }
            }
        }

        _warpgroup_sync(wg_id);
    }

    __device__ __forceinline__ static half2 warp_transpose_8x8_half2(half2 packed, int lane_id) {
        const int out_row = lane_id / 4;
        const int out_pair = lane_id % 4;

        const int source_pair = out_row / 2;
        const int half_select = out_row % 2;

        const int source_lane_0 = (2 * out_pair) * 4 + source_pair;
        const int source_lane_1 = (2 * out_pair + 1) * 4 + source_pair;

        const half2 word_0 = __shfl_sync(0xffffffffu, packed, source_lane_0);
        const half2 word_1 = __shfl_sync(0xffffffffu, packed, source_lane_1);

        return half_select == 0 ? __lows2half2(word_0, word_1) : __highs2half2(word_0, word_1);
    }

    // Experimental row-major-only transpose using a direct 8x8 tiling.
    __device__ inline void _transpose_8x8() {
        static_assert(BM == BN, "inplace transpose can be only applied to square fragment.");
        static_assert(sizeof(T) == 2, "optimized transpose only supports 16-bit elements.");

        const int tid = threadIdx.x;
        constexpr int threads_per_block = CONSUMER_THREADS;

        const int lane_id = threadIdx.x % WARP_SIZE;
        const int warp_id = threadIdx.x / WARP_SIZE;

        const int wg_id = warp_id / 4;

        constexpr int FRAG_M = 8;
        constexpr int TOTAL_ELEMENTS = BM * BN;

        constexpr int M_STEPS = BM / FRAG_M;

        constexpr int total_off_diagonal_pairs = (M_STEPS * M_STEPS - M_STEPS) / 2;

        constexpr int SWIZZLE_SHIFT = (sizeof(T) == 2) ? 3 : 2;

        const int lane_row = lane_id / 4;
        const int lane_pair = lane_id % 4;
        const int local_col = 2 * lane_pair;

        // Each of the eight warps handles one off-diagonal 8x8 tile pair.
        #pragma unroll
        for (int pair_idx = 0; pair_idx < total_off_diagonal_pairs; pair_idx += WARPS_PER_CTA) {
            const int tile_pair_idx = pair_idx + warp_id;
            if (tile_pair_idx < total_off_diagonal_pairs) {

                const int sub_frag_idx_m = static_cast<int>((1 + __fsqrt_rn(1 + 8 * tile_pair_idx)) / 2);
                const int sub_frag_idx_n = tile_pair_idx - (sub_frag_idx_m * (sub_frag_idx_m - 1)) / 2;

                const int sub_frag_idx_m_off = sub_frag_idx_m * FRAG_M;
                const int sub_frag_idx_n_off = sub_frag_idx_n * FRAG_M;

                const int src_idx = (sub_frag_idx_m_off + lane_row) * BM + sub_frag_idx_n_off + local_col;
                const int dst_idx = (sub_frag_idx_n_off + lane_row) * BM + sub_frag_idx_m_off + local_col;

                const half2 src_old = *reinterpret_cast<const half2*>(shared_ptr + src_idx);
                const half2 dst_old = *reinterpret_cast<const half2*>(shared_ptr + dst_idx);

                const half2 src_t = warp_transpose_8x8_half2(src_old, lane_id);
                const half2 dst_t = warp_transpose_8x8_half2(dst_old, lane_id);

                *reinterpret_cast<half2*>(shared_ptr + src_idx) = dst_t;
                *reinterpret_cast<half2*>(shared_ptr + dst_idx) = src_t;
            }
        }

        // The same eight warps transpose the sixteen diagonal 8x8 tiles in two rounds.
        #pragma unroll
        for (int task_idx = 0; task_idx < M_STEPS; task_idx += WARPS_PER_CTA) {
            const int sub_frag_idx = task_idx + warp_id;

            if (sub_frag_idx < M_STEPS) {
                const int sub_frag_idx_m_off = sub_frag_idx * FRAG_M;
                const int sub_frag_idx_n_off = sub_frag_idx * FRAG_M;
                const int packed_idx =
                    (sub_frag_idx_m_off + lane_row) * BM + sub_frag_idx_n_off + local_col;

                const half2 old_value =
                    *reinterpret_cast<const half2*>(shared_ptr + packed_idx);
                const half2 transposed =
                    warp_transpose_8x8_half2(old_value, lane_id);

                *reinterpret_cast<half2*>(shared_ptr + packed_idx) = transposed;
            }
        }

        // __syncthreads();
        _warpgroup_sync(wg_id);
    }

    // Experimental row-major-only out-of-place transpose using direct 8x8 tiles.
    __device__ inline void transpose_8x8(T* dst_shared_ptr) const {
        constexpr int FRAG_M = 8;
        constexpr int M_STEPS = BM / FRAG_M;
        constexpr int total_sub_fragments = M_STEPS * M_STEPS;

        static_assert(sizeof(T) == 2, "optimized transpose only supports 16-bit elements.");
        static_assert(BM == BN, "out-of-place transpose can be only applied to square fragment.");
        static_assert(BM % FRAG_M == 0, "optimized transpose requires BM to be divisible by 8.");

        const int lane_id = threadIdx.x % WARP_SIZE;
        const int warp_id = threadIdx.x / WARP_SIZE;

        const int wg_id = warp_id / 4;

        const int thr_row = lane_id / 4;
        const int thr_col = 2 * (lane_id % 4);

        constexpr int ELEMENTS_PER_PANEL = 128 / sizeof(T);
        constexpr int SWIZZLE_SHIFT = (sizeof(T) == 2) ? 3 : 2;

        // Source and destination do not overlap. Each warp therefore handles one
        // complete 8x8 tile rather than exchanging an off-diagonal tile pair.
        #pragma unroll
        for (int task_idx = 0; task_idx < total_sub_fragments; task_idx += WARPS_PER_CTA) {
            const int sub_frag_idx = task_idx + warp_id;

            if (sub_frag_idx < total_sub_fragments) {
                const int sub_frag_idx_m = sub_frag_idx / M_STEPS;
                const int sub_frag_idx_n = sub_frag_idx % M_STEPS;

                const int sub_frag_idx_m_off = sub_frag_idx_m * FRAG_M;
                const int sub_frag_idx_n_off = sub_frag_idx_n * FRAG_M;

                int row_src = sub_frag_idx_m_off + thr_row;
                int col_src = sub_frag_idx_n_off + thr_col;

                int row_dst = sub_frag_idx_n_off + thr_row;
                int col_dst = sub_frag_idx_m_off + thr_col;

                const int src_idx = row_src * BM + col_src;
                const int dst_idx = row_dst * BM + col_dst;

                int dst_store_idx;
#if SWIZZLE_64B_STORE
                int panel_idx_dst = col_dst / ELEMENTS_PER_PANEL;

                int swizzle_col_dst = (col_dst % ELEMENTS_PER_PANEL) ^ ((row_dst & 7) << SWIZZLE_SHIFT);
                    
                dst_store_idx = panel_idx_dst * BM * ELEMENTS_PER_PANEL + row_dst * ELEMENTS_PER_PANEL + swizzle_col_dst;
#else
                dst_store_idx = dst_idx;
#endif

                const half2 old_value =
                    *reinterpret_cast<const half2*>(shared_ptr + src_idx);
                const half2 transposed =
                    warp_transpose_8x8_half2(old_value, lane_id);

                *reinterpret_cast<half2*>(dst_shared_ptr + dst_store_idx) = transposed;
            }
        }

        _warpgroup_sync(wg_id);
    }


    // NOTE : implement outplace transpose for 128x128 fp32 fragment
    // TODO  : add outplace transpose for 128x128 fp16/bf16 fragment
    __device__ inline void transpose(T* dst_shared_ptr) {
        static_assert(BM == BN, "transpose can be only applied to square fragment.");

        const int tid = threadIdx.x;
        constexpr int threads_per_block = CONSUMER_THREADS;

        const int lane_id = threadIdx.x % WARP_SIZE;
        const int warp_id = threadIdx.x / WARP_SIZE;

        const int wg_id = warp_id / 4;

        constexpr int FRAG_M = 16;
        constexpr int TOTAL_ELEMENTS = BM * BN;

        constexpr int M_STEPS = BM / FRAG_M;

        constexpr int total_diagonal_pairs = (FRAG_M * FRAG_M + FRAG_M) / 2;

        constexpr int ELEMENTS_PER_PANEL = 128 / sizeof(T);
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

                    int src_idx = row_src * BM + col_src;
                    int dst_idx = row_dst * BM + col_dst;
                    
                    int src_store_idx, dst_store_idx;
                    // NOTE(yiakwy) : only valid for 16x16 fragment
#if SWIZZLE_64B_STORE
                    int panel_idx_src = col_src / ELEMENTS_PER_PANEL;
                    int panel_idx_dst = col_dst / ELEMENTS_PER_PANEL;

                    int swizzle_col_src = (col_src % ELEMENTS_PER_PANEL) ^ ((row_src & 7) << SWIZZLE_SHIFT);
                    int swizzle_col_dst = (col_dst % ELEMENTS_PER_PANEL) ^ ((row_dst & 7) << SWIZZLE_SHIFT);
                    
                    src_store_idx = panel_idx_src * BM * ELEMENTS_PER_PANEL + row_src * ELEMENTS_PER_PANEL + swizzle_col_src;
                    dst_store_idx = panel_idx_dst * BM * ELEMENTS_PER_PANEL + row_dst * ELEMENTS_PER_PANEL + swizzle_col_dst;
#else
                    src_store_idx = src_idx;
                    dst_store_idx = dst_idx;
#endif

                    T src_val = shared_ptr[src_idx];
                    T dst_val = shared_ptr[dst_idx];

                    dst_shared_ptr[src_store_idx] = dst_val;
                    dst_shared_ptr[dst_store_idx] = src_val;
                } // end of e_idx

            } // end of sub_frag_idx_n

        } // end of sub_frag_idx_m

        _warpgroup_sync(wg_id);

        #pragma unroll
        for (int task_idx = 0; task_idx < M_STEPS; task_idx++) {

            int sub_frag_idx_m_off = task_idx * FRAG_M;
            int sub_frag_idx_n_off = task_idx * FRAG_M;

            #pragma unroll
            for (int e_idx = tid; e_idx < FRAG_M * FRAG_M; e_idx += threads_per_block) {
                int thr_col = e_idx % FRAG_M;
                int thr_row = e_idx / FRAG_M;

                int row_src = sub_frag_idx_m_off + thr_row;
                int col_src = sub_frag_idx_n_off + thr_col;

                int row_dst = sub_frag_idx_n_off + thr_col;
                int col_dst = sub_frag_idx_m_off + thr_row;

                int src_idx = row_src * BM + col_src;
                int dst_store_idx;

#if SWIZZLE_64B_STORE
                int panel_idx_dst = col_dst / ELEMENTS_PER_PANEL;
                int swizzle_col_dst = (col_dst % ELEMENTS_PER_PANEL) ^ ((row_dst & 7) << SWIZZLE_SHIFT);

                dst_store_idx = panel_idx_dst * BM * ELEMENTS_PER_PANEL + row_dst * ELEMENTS_PER_PANEL + swizzle_col_dst;
#else
                dst_store_idx = row_dst * BM + col_dst;
#endif

                dst_shared_ptr[dst_store_idx] = shared_ptr[src_idx];
            }

        } // end of task_idx

        _warpgroup_sync(wg_id);

    }
};

} // namespace xpu

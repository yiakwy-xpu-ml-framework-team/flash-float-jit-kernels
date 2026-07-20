/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once
#include <cooperative_groups.h>

#include <cuda/barrier>

namespace cg = cooperative_groups;

#include "../arch/tma/tma_copy.h"
#include "../arch/tma/tma_barrier.h"
#include "../arch/cluster/cluster.h"

#include "../arch/warpgroup/reg_allocator.h"
#include "../profiler.cuh"

// TODO (yiakwy) : refactor flashFloat FragView with Shape and Layout component with support of device side Flash Float datatype

#include "../tensor/tensor_view_ref.h"
#include "../fragment/nv_frag_gemm_scaled_impl.h"

#include "block.h"
#include "producer.h"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef SWIZZLE_64B_STORE
#define SWIZZLE_64B_STORE 1
#endif

namespace cg = cooperative_groups;

#define USE_LINEAR_TO_TRIL_LAYOUT 1

#define USE_CLUSTER_MULTICAST 1

#define USE_INPALCE_TRI_TRANSPOSE 1

namespace xpu {

template <int BM, int BN, int BK, int STAGES, int GROUP_SIZE_M, int CLUSTER_SIZE_M>
struct HopperPersistentSplitKPipeline {

    // sm90+
    static __device__ inline void run_persistent(
        HopperWGMMAAccumulator<BM, BN, BK>& accum,
        const CUtensorMap* tma_desc_X,
        const CUtensorMap* tma_desc_W,
        const CUtensorMap* tma_desc_O,
        const CUtensorMap* tma_desc_O_swizzle,
        const float* scale_X,
        const float* scale_W,
        half* Out,
        int M, int N, int K,
        int total_symmetric_tiles,
        int num_blocks_m,
        int num_blocks_n,
        uint8_t* smem_buffer
        FFJK_PROFILER_KERNEL_PARAMS
    ) {
        using fp8_t = __nv_fp8_e4m3;
        using fp32_t = float;

        using AccDtype = fp32_t;
        using OutDtype = half;

        const int tid = threadIdx.x;
        // const int lane_id = threadIdx.x % WARP_SIZE;
        // const int warp_id = threadIdx.x / WARP_SIZE;

#if USE_CLUSTER_MULTICAST
        uint32_t cluster_rank;
        cluster_rank = nvgpu::arch::cluster_ctarank();

        auto cluster = cooperative_groups::this_cluster();
        auto clusterDim = cluster.dim_blocks();

        const int cluster_group_m_rank = cluster_rank / clusterDim.x;
#else
        uint32_t cluster_rank = 0;
#endif

        // prepare
        auto* shmem_X = reinterpret_cast<SharedBlock<fp8_t, BM, BK>*>(smem_buffer);
        auto* shmem_W = reinterpret_cast<SharedBlock<fp8_t, BN, BK>*>(smem_buffer + STAGES * sizeof(*shmem_X));

        int threads_per_block = blockDim.x;

        OutDtype* shmem_epilogue = reinterpret_cast<OutDtype *>(smem_buffer + STAGES * (BM * BK + BN * BK));
        FragmentView<OutDtype, BM, BN, MemoryDomain::kShared> frag_view(shmem_epilogue);

        constexpr uint32_t tma_bytes_X = BM * BK;
        constexpr uint32_t tma_bytes_W = BN * BK;

        constexpr uint32_t total_stage_bytes = tma_bytes_X + tma_bytes_W;

        constexpr int SCLAE_BLOCK_SIZE_K = 128;
        constexpr int K_TILES_TOTAL = (8192 + SCLAE_BLOCK_SIZE_K - 1) / SCLAE_BLOCK_SIZE_K;

        __shared__ __align__(128) float shmem_XS[BM * K_TILES_TOTAL];
        __shared__ __align__(128) float shmem_WS[K_TILES_TOTAL];

        // TODO (yiakwy) : init full barriers for TMA, in mult-stages pipeline, we combine writer and reader barrieres in the same stage into one
        __shared__ __align__(128) uint64_t barriers[STAGES];

        __shared__ __align__(128) OutDtype *dst[8];

        if (threadIdx.x == 0) {
            #pragma unroll
            for (int s = 0; s < STAGES; ++s) {
                uint32_t s_bar_ptr = __cvta_generic_to_shared(&barriers[s]);
                nvgpu::arch::tma_init_barrier<USE_CLUSTER_MULTICAST>(&barriers[s], 1);
            }
        }
        nvgpu::arch::tma_store_fence();
        __syncthreads();

        uint64_t cache_hint_lhs = static_cast<uint64_t>(nvgpu::arch::CacheHintSm90::EVICT_NORMAL);
        uint64_t cache_hint_rhs = static_cast<uint64_t>(nvgpu::arch::CacheHintSm90::EVICT_NORMAL);

        int split_k_id = blockIdx.x;
        int split_k = gridDim.x;

#if USE_CLUSTER_MULTICAST
        nvgpu::arch::cluster_sync();

        uint16_t cluster_mask = 0;
        const int num_splits = clusterDim.x; // split_k
        const int _assumed_group_size_m = clusterDim.y; // GROUP_SIZE_M
        for (int r =0; r < _assumed_group_size_m ; r++) {
            int target_rank = split_k_id + r * num_splits;
            cluster_mask |= (1 << target_rank);
        }
#endif
        const int k_tiles_total = (K + BK - 1) / BK;

        int k_tiles_per_slice = (k_tiles_total + split_k - 1) / split_k;

#ifdef FFJK_ENABLE_CUDA_PROFILER
        FFJK_PROFILER_DEFINE_LAYOUT(total_symmetric_tiles, k_tiles_per_slice);
        int ffjk_prof_task_iter = 0;
        int ffjk_prof_k_iter = -1;
        FFJK_PROF_CTA_EVENT_PAYLOAD(
            ffjk::kProfilerEventPipelineEnter,
            K,
            total_symmetric_tiles);
#endif

        int k_start = split_k_id * k_tiles_per_slice;
        int k_end = min(k_tiles_total, (split_k_id + 1) * k_tiles_per_slice);

        alignas(128) __shared__ int local_task_id;
        if (threadIdx.x == 0) {
            local_task_id = blockIdx.y;
        }
        __syncthreads();

        while (local_task_id < total_symmetric_tiles) {
            const int task_id = local_task_id;
#ifdef FFJK_ENABLE_CUDA_PROFILER
            ffjk_prof_k_iter = -1;
#endif
            FFJK_PROF_BEGIN(
                ffjk::kProfilerEventTask,
                task_id,
                ffjk::cuda_profiler_pack_u16(blockIdx.x, blockIdx.y));

            accum.clear();

            // if (threadIdx.x == 0 && blockIdx.x == 0) {
            //     printf("[Split#%d] [SM#%d] tile#%d (cluster_rank#%d) enter into persistent loop ...\n", blockIdx.x, blockIdx.y, local_task_id, cluster_rank);
            // }

            // NOTE (yiakwy) : we only support symmetric gemm, hence force to use linear to triangular mapping for block-level tile assignment.
            // TODO (yiakwy) : precompute the block_idx_m and block_idx_n for each local_task_id and store in shared memory to avoid redundant computation on the fly and reduce the latency.
#ifdef USE_LINEAR_TO_TRIL_LAYOUT
            int block_idx_m = int((sqrt(8.0 * local_task_id + 1.0) - 1.0) / 2.0);
            int block_idx_n = local_task_id - (block_idx_m * (block_idx_m + 1)) / 2;
#else
            int block_idx_m = local_task_id / num_blocks_n;
            int block_idx_n = local_task_id % num_blocks_n;
#endif

            // if (threadIdx.x == 0 && blockIdx.x == 0) {
            //     printf("[pre] [Split#%d] [SM#%d] block#(%d, %d) initiate TAM loading ...\n", blockIdx.x, blockIdx.y, block_idx_m, block_idx_n);
            // }

            // Grouping for better L2 cache locality in TMA load
            const uint32_t group_id = block_idx_m / GROUP_SIZE_M;

            // NOTE (yaikwy) : group swizzle after Down-Left Triangular Mapping for better L2 cache locality in TMA load
            if constexpr (GROUP_SIZE_M > 1) {
                const uint32_t group_off_row = group_id * GROUP_SIZE_M;
                const uint32_t group_size_m = min(num_blocks_m - group_off_row, static_cast<uint32_t>(GROUP_SIZE_M));

                auto sum_tri = [](int h) {
                    return h * (h + 1) / 2;
                };

                const uint32_t og_off = sum_tri(group_off_row);
                const uint32_t res = sum_tri(group_size_m);

                const uint32_t num_blocks_in_group = sum_tri(group_off_row + group_size_m) - og_off;
                const uint32_t in_group_idx = local_task_id - og_off;

                const uint32_t test_col = in_group_idx / group_size_m;

                if (test_col < group_off_row) {
                    block_idx_m = group_off_row + (in_group_idx % group_size_m);
                    block_idx_n = test_col;
                } else {
                    const uint32_t ig_col_off = group_off_row;
                    const uint32_t sub_in_group_id = in_group_idx - ig_col_off * group_size_m;

                    // NOTE (yiakwy) : since int ( sqrt( gm^2 + gm - 2ig_id - 1/4) - 1/2 ) - 1 < c0 <= sqrt( gm^2 + gm - 2ig_id - 1/4) - 1/2 )
                    const uint32_t c0 = int ( sqrt( group_size_m*group_size_m + group_size_m - 2*sub_in_group_id - 1.0/4) - 1.0/2 );
                    const uint32_t c0_plus_1 = c0 + 1;
                    const uint32_t c1 = group_size_m - c0_plus_1;

                    block_idx_n = ig_col_off + c1;
                    block_idx_m = group_off_row + (sub_in_group_id - (group_size_m + c0_plus_1 + 1) * (group_size_m - c0_plus_1) / 2) + c1;
                }
            }

            FFJK_PROF_EVENT_PAYLOAD(
                ffjk::kProfilerEventTaskMap,
                task_id,
                ffjk::cuda_profiler_pack_u16(block_idx_m, block_idx_n));

            // if (threadIdx.x == 0 && blockIdx.x == 0) {
            //     printf("[after] [Split#%d] [SM#%d] block#(%d, %d) initiate TAM loading ...\n", blockIdx.x, blockIdx.y, block_idx_m, block_idx_n);
            // }

            int write_stage = 0;
            int read_stage = 0;

            // 1. Ramp Up Fill : to initiate the pipeline, we will fill STAGES-1 stages of data before entering the main loop, and then maintain 1 stage ahead of the main loop to keep the pipeline full.

            FFJK_PROF_BEGIN(
                ffjk::kProfilerEventPrefetchTma,
                task_id,
                ffjk::cuda_profiler_pack_u16(k_start, k_end));
#if  USE_CLUSTER_MULTICAST
            producer<STAGES, GROUP_SIZE_M, BM, BN, BK, USE_CLUSTER_MULTICAST, USE_LINEAR_TO_TRIL_LAYOUT>::load(
                tid, group_id, block_idx_m, block_idx_n,
                k_start, k_end, total_stage_bytes,
                tma_desc_X, tma_desc_W,
                shmem_X, shmem_W, barriers,
                cluster_mask, cluster_group_m_rank, cache_hint_lhs, cache_hint_rhs,
                write_stage/*src & dst*/);
#else
            producer<STAGES, GROUP_SIZE_M, BM, BN, BK, USE_CLUSTER_MULTICAST, USE_LINEAR_TO_TRIL_LAYOUT>::load(
                tid, group_id, block_idx_m, block_idx_n,
                k_start, k_end, total_stage_bytes,
                tma_desc_X, tma_desc_W,
                shmem_X, shmem_W, barriers,
                cache_hint_lhs, cache_hint_rhs,
                write_stage/*src & dst*/);
#endif
            FFJK_PROF_END(
                ffjk::kProfilerEventPrefetchTma,
                task_id,
                ffjk::cuda_profiler_pack_u16(k_start, k_end));

            // if (threadIdx.x == 0 && blockIdx.x == 0) {
            //     printf("[Prefetch] [Split#%d] [SM#%d] block#(%d, %d) enter into main loop ...\n", blockIdx.x, blockIdx.y, block_idx_m, block_idx_n);
            // }
            // __syncthreads();

            // NOTE (yiakwy) : rows of 1 or more BLOCKS share a scale
            const int stride_xs_m = K / SCLAE_BLOCK_SIZE_K;
            const int stride_ws_n = K / SCLAE_BLOCK_SIZE_K;

            constexpr int shares_per_scale = SCLAE_BLOCK_SIZE_K / BK;
            // static_assert(shares_per_scale == 2, "with BK=%d, each scale will be shared by 2 tiles, please adjust SCLAE_BLOCK_SIZE_K or BK to ensure that.");

            const int k_tiles = k_end - k_start;
            const int total_xs_elements = BM * k_tiles;
            const int total_ws_elements = k_tiles;

            // NOTE (yiakwy) : prefetch all scale without TMA

            // TODO (yiakwy) : remap shmem_XS to per-thread registers to reduce the latency, since the scale load is on the critical path of the main loop.
            FFJK_PROF_BEGIN(
                ffjk::kProfilerEventScaleXLoad,
                task_id,
                total_xs_elements);
            #pragma unroll 4
            for (int i = tid; i < total_xs_elements; i += threads_per_block) {
                int s_row = i / k_tiles;
                int s_col = i % k_tiles;

                int g_row = block_idx_m * BM + s_row;
                int g_col = (k_start + s_col) / shares_per_scale;

                shmem_XS[s_col * BM + s_row] = scale_X[g_row * stride_xs_m + g_col];
            }
            __syncthreads();
            FFJK_PROF_END(
                ffjk::kProfilerEventScaleXLoad,
                task_id,
                total_xs_elements);

           // TODO (yiakwy) : remap shmem_WS to per-thread registers to reduce the latency, since the scale load is on the critical path of the main loop.
            FFJK_PROF_BEGIN(
                ffjk::kProfilerEventScaleWLoad,
                task_id,
                total_ws_elements);
            #pragma unroll 4
            for (int i = tid; i < total_ws_elements; i += threads_per_block) {
                int s_col = i;

                int g_row = block_idx_n;
                int g_col = (k_start + s_col) / shares_per_scale;

                shmem_WS[s_col] = scale_W[g_row * stride_ws_n + g_col];
            }
            __syncthreads();
            FFJK_PROF_END(
                ffjk::kProfilerEventScaleWLoad,
                task_id,
                total_ws_elements);

            int tma_phase = 0;

            // 2. main loop
            for (int k_tile = k_start; k_tile < k_end; ++k_tile) {
#ifdef FFJK_ENABLE_CUDA_PROFILER
                ffjk_prof_k_iter = k_tile - k_start;
#endif
                uint32_t current_barrier = __cvta_generic_to_shared(&barriers[read_stage]);

                // if (threadIdx.x == 0 && blockIdx.x == 0) {
                //     printf("[MainLoop] [Split#%d] [SM#%d] block#(%d, %d) k_tile=%d, k_start=%d, k_end=%d\n", blockIdx.x, blockIdx.y, block_idx_m, block_idx_n, k_tile, k_start, k_end);
                // }

                if (threadIdx.x == 0) {
                    // NOTE (yiakwy) : wait parity switch from phase (1 at prfetch stage 0) to ^phase (0 when TMA finish stage 0 transactions)
                    FFJK_PROF_BEGIN(
                        ffjk::kProfilerEventTmaWait,
                        task_id,
                        ffjk::cuda_profiler_pack_u16(k_tile, read_stage));
                    nvgpu::arch::tma_wait(current_barrier, tma_phase);
                    FFJK_PROF_END(
                        ffjk::kProfilerEventTmaWait,
                        task_id,
                        ffjk::cuda_profiler_pack_u16(k_tile, read_stage));
                }
                __syncthreads();

                // if (threadIdx.x == 0 && blockIdx.x == 0 && k_tile == k_end - 1) {
                //     printf("[Split#%d/%d] [SM#%d] block#(%d, %d) k_tile=%d, inputs are ready.\n", blockIdx.x, gridDim.x, blockIdx.y, block_idx_m, block_idx_n, k_tile);
                // }
                // __syncthreads();

                HopperWGMMAAccumulator<BM, BN, BK> local_step_accum;
                local_step_accum.clear();

                uint32_t active_smem_x = __cvta_generic_to_shared(&shmem_X[read_stage]);
                uint32_t active_smem_w = __cvta_generic_to_shared(&shmem_W[read_stage]);

                // NOTE (yiakwy) : hopper (SM90a) does not support mma_scaled instruction, sx, sw will be ignored in the current implementation, and the scaling will be applied in the epilogue.
                FFJK_PROF_BEGIN(
                    ffjk::kProfilerEventMmaIssue,
                    task_id,
                    ffjk::cuda_profiler_pack_u16(k_tile, read_stage));
                HopperWGMMAExecutor::mma_scaled(local_step_accum, active_smem_x, active_smem_w);
                FFJK_PROF_END(
                    ffjk::kProfilerEventMmaIssue,
                    task_id,
                    ffjk::cuda_profiler_pack_u16(k_tile, read_stage));

                int next_k = k_tile + (STAGES - 1);
                if (next_k < k_end) {

                    FFJK_PROF_BEGIN(
                        ffjk::kProfilerEventProducerLoadOnce,
                        task_id,
                        ffjk::cuda_profiler_pack_u16(next_k, write_stage));
#if  USE_CLUSTER_MULTICAST
                    producer<STAGES, GROUP_SIZE_M, BM, BN, BK, USE_CLUSTER_MULTICAST, USE_LINEAR_TO_TRIL_LAYOUT>::load_once(
                        tid, group_id,
                        next_k, block_idx_m, block_idx_n, total_stage_bytes,
                        tma_desc_X, tma_desc_W,
                        shmem_X, shmem_W, barriers,
                        cluster_mask, cluster_group_m_rank, cache_hint_lhs, cache_hint_rhs, write_stage, tma_phase
                    );
#else
                    producer<STAGES, GROUP_SIZE_M, BM, BN, BK, USE_CLUSTER_MULTICAST, USE_LINEAR_TO_TRIL_LAYOUT>::load_once(
                        tid, group_id,
                        next_k, block_idx_m, block_idx_n, total_stage_bytes,
                        tma_desc_X, tma_desc_W,
                        shmem_X, shmem_W, barriers,
                        cache_hint_lhs, cache_hint_rhs, write_stage, tma_phase
                    );
#endif
                    FFJK_PROF_END(
                        ffjk::kProfilerEventProducerLoadOnce,
                        task_id,
                        ffjk::cuda_profiler_pack_u16(next_k, write_stage));

                } // next_k < k_end

                FFJK_PROF_BEGIN(
                    ffjk::kProfilerEventWgmmaWait,
                    task_id,
                    ffjk::cuda_profiler_pack_u16(k_tile, read_stage));
                HopperWGMMAExecutor::commit_and_wait();
                FFJK_PROF_END(
                    ffjk::kProfilerEventWgmmaWait,
                    task_id,
                    ffjk::cuda_profiler_pack_u16(k_tile, read_stage));

                FFJK_PROF_BEGIN(
                    ffjk::kProfilerEventScaleApplyAccum,
                    task_id,
                    k_tile);
                local_step_accum.mul_(&shmem_XS[0], &shmem_WS[0], k_tile - k_start);
                accum.add_(local_step_accum);
                FFJK_PROF_END(
                    ffjk::kProfilerEventScaleApplyAccum,
                    task_id,
                    k_tile);
                __syncwarp();

                read_stage = (read_stage + 1) % STAGES;
                if (read_stage == 0) {
                    tma_phase ^= 1;
                }
            }

            // 3. Epilogue
            //   - first write data back to share memory for SPLIT-K reduction via NoC
            //   - applying successive operations upon tile results in the epilogue, such as bias add, activation, etc, can be fused in this step to save memory bandwidth.
            FFJK_PROF_BEGIN(
                ffjk::kProfilerEventEpilogueSmemStore,
                task_id,
                ffjk::cuda_profiler_pack_u16(block_idx_m, block_idx_n));
            accum.store(shmem_epilogue);
            __syncthreads();
            FFJK_PROF_END(
                ffjk::kProfilerEventEpilogueSmemStore,
                task_id,
                ffjk::cuda_profiler_pack_u16(block_idx_m, block_idx_n));

            // if (threadIdx.x == 0 && blockIdx.x == 1) {
            //     printf("[Epilogue] [Split#%d] [SM#%d] write block <%d, %d> back to shared memory...\n", blockIdx.x, blockIdx.y, block_idx_m, block_idx_n);
            // }
            // __syncthreads();

#if __CUDA_ARCH__ >= 900 && ENABLE_HOPPER // Hopper 900+ GPU with TMA support
            // if (threadIdx.x == 0 && blockIdx.x == 1) {
            //     printf("[Epilogue] [Split#%d] [SM#%d] write split_k#%d block <%d, %d> on-chip reduce via NoC...\n", blockIdx.x, blockIdx.y, split_k, block_idx_m, block_idx_n);
            // }
            // __syncthreads();

            auto cluster = cooperative_groups::this_cluster();
            cluster.sync();

            FFJK_PROF_BEGIN(
                ffjk::kProfilerEventSplitKReduce,
                task_id,
                split_k);
            if (split_k > 1) {
                if (split_k_id == 0) {
                    if (threadIdx.x == 0) {
                        for (int r = 1; r < split_k; ++r) {
                            OutDtype* dst_shmem_epilogue = cluster.map_shared_rank<OutDtype>(&shmem_epilogue[0], r);
                            dst[r] = dst_shmem_epilogue;
                        }
                    }
                    __syncthreads();

                    for (int r = 1; r < split_k; ++r) {
                        OutDtype* dst_shmem_epilogue = dst[r];
                        for (int idx = tid; idx < BM * BN; idx += threads_per_block) {
                            shmem_epilogue[idx] += dst_shmem_epilogue[idx];
                        }
                    }
                }
            } //  split_k > 1

            __syncthreads();
            FFJK_PROF_END(
                ffjk::kProfilerEventSplitKReduce,
                task_id,
                split_k);

            if (split_k_id == 0) {
                if (threadIdx.x == 0) {
                    FFJK_PROF_BEGIN(
                        ffjk::kProfilerEventStoreLower,
                        task_id,
                        ffjk::cuda_profiler_pack_u16(block_idx_m, block_idx_n));
                    uint64_t tma_o_addr = reinterpret_cast<uint64_t>(tma_desc_O);
                    uint32_t smem_epilogue_addr  = static_cast<uint32_t>(__cvta_generic_to_shared(&shmem_epilogue[0]));

                    asm volatile (
                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
                        " [%0, {%2, %3}], [%1];"
                        :
                        : "l"(tma_o_addr), "r"(smem_epilogue_addr),
                        "r"(block_idx_n * BN), "r"(block_idx_m * BM)
                        : "memory"
                    );
                    FFJK_PROF_END(
                        ffjk::kProfilerEventStoreLower,
                        task_id,
                        ffjk::cuda_profiler_pack_u16(block_idx_m, block_idx_n));
                }

#if (defined(USE_LINEAR_TO_TRIL_LAYOUT)) && USE_LINEAR_TO_TRIL_LAYOUT
                // NOTE (yiakwy) :  transpose copy to upper right
                if (block_idx_m > block_idx_n) {

#if (defined(USE_INPALCE_TRI_TRANSPOSE)) && USE_INPALCE_TRI_TRANSPOSE
                    if (threadIdx.x == 0) {
                        asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
                    }
                    __syncthreads();

                    // NOTE (yiakwy) : inplace transpose
                    FFJK_PROF_BEGIN(
                        ffjk::kProfilerEventMirrorTranspose,
                        task_id,
                        ffjk::cuda_profiler_pack_u16(block_idx_m, block_idx_n));
                    frag_view._transpose();
                    FFJK_PROF_END(
                        ffjk::kProfilerEventMirrorTranspose,
                        task_id,
                        ffjk::cuda_profiler_pack_u16(block_idx_m, block_idx_n));
#else
                    // NOTE (yiakwy) : outplace transpose
                    // TODO (yiakwy) : outplace transpose
#error "Outplace transpose is not implemented yet, please enable USE_INPALCE_TRI_TRANSPOSE to use inplace transpose."

#endif // USE_INPALCE_TRI_TRANSPOSE

                    if (threadIdx.x == 0) {
                        FFJK_PROF_BEGIN(
                            ffjk::kProfilerEventMirrorStore,
                            task_id,
                            ffjk::cuda_profiler_pack_u16(block_idx_m, block_idx_n));
#if SWIZZLE_64B_STORE
                        uint64_t tma_o_addr = reinterpret_cast<uint64_t>(tma_desc_O_swizzle);
#else
                        uint64_t tma_o_addr = reinterpret_cast<uint64_t>(tma_desc_O);
#endif // SWIZZLE_64B_STORE
                        uint32_t smem_epilogue_addr  = static_cast<uint32_t>(__cvta_generic_to_shared(&shmem_epilogue[0]));

#if SWIZZLE_64B_STORE
                        asm volatile (
                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
                            " [%0, {%2, %3}], [%1];"
                            :
                            : "l"(tma_o_addr), "r"(smem_epilogue_addr),
                            "r"(block_idx_n * BN), "r"(block_idx_m * BM)
                            : "memory"
                        );

                        const uint32_t smem_epilogue_addr_next = smem_epilogue_addr + 128;

                        asm volatile (
                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
                            " [%0, {%2, %3}], [%1];"
                            :
                            : "l"(tma_o_addr), "r"(smem_epilogue_addr_next),
                                "r"(block_idx_n * BN + 64), "r"(block_idx_m * BM)
                            : "memory"
                        );
#else
                        asm volatile (
                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
                            " [%0, {%2, %3}], [%1];"
                            :
                            : "l"(tma_o_addr), "r"(smem_epilogue_addr),
                            "r"(block_idx_m * BN), "r"(block_idx_n * BM)
                            : "memory"
                        );
#endif // SWIZZLE_64B_STORE
                        FFJK_PROF_END(
                            ffjk::kProfilerEventMirrorStore,
                            task_id,
                            ffjk::cuda_profiler_pack_u16(block_idx_m, block_idx_n));
                    }

                    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

                } // block_idx_m > block_idx_n

#endif // USE_LINEAR_TO_TRIL_LAYOUT

            } // split_id == 0
            cluster.sync();
#else
            for (int idx = threadIdx.x; idx < BM * BN; idx += threads_per_block) {
                int local_m = idx / BN;
                int local_n = idx % BN;

                // float val = frag_view(local_m, local_n);
                half val = static_cast<half>(shmem_epilogue[idx]);

                // write to lower left
                int global_m = block_idx_m * BM + local_m;
                int global_n = block_idx_n * BN + local_n;
                if (global_m < M && global_n < N) {
                    if (split_k > 1) {
                        atomicAdd(&Out[global_m * N + global_n], val);
                    } else {
                        Out[global_m * N + global_n] = val;
                    }
                } // end of write to lower left

                // tranpose copy to upper right
                if (block_idx_m > block_idx_n) {
                    int sym_global_m = block_idx_n * BN + local_n;
                    int sym_global_n = block_idx_m * BM + local_m;
                    if (sym_global_m < M && sym_global_n < N) {
                        if (split_k > 1) {
                            atomicAdd(&Out[sym_global_m * N + sym_global_n], val);
                        } else {
                            Out[sym_global_m * N + sym_global_n] = val;
                        }
                    }
                } // end of tranpose copy to upper right
            }
            __syncthreads();

#endif // __CUDA_ARCH__ >= 900 && ENABLE_HOPPER

            // fetch next task
            if (threadIdx.x == 0) {
                local_task_id += gridDim.y;
            }
            __syncthreads();

            FFJK_PROF_END(
                ffjk::kProfilerEventTask,
                task_id,
                ffjk::cuda_profiler_pack_u16(block_idx_m, block_idx_n));
            FFJK_PROF_EVENT_PAYLOAD(
                ffjk::kProfilerEventTaskDone,
                task_id,
                local_task_id);
#ifdef FFJK_ENABLE_CUDA_PROFILER
            ++ffjk_prof_task_iter;
#endif

        } // while
    }
};

} // namespace xpu

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

// TODO (yiakwy) : refactor flashFloat FragView with Shape and Layout component with support of device side Flash Float datatype

#include "../tensor/tensor_view_ref.h"
#include "../fragment/nv_frag_gemm_scaled_impl.h"

#include "block.h"
#include "wasp_producer.h"
#include "sched.h"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef SWIZZLE_64B_STORE
#define SWIZZLE_64B_STORE 1
#endif

#define USE_LINEAR_TO_TRIL_LAYOUT 1

#define USE_CLUSTER_MULTICAST 1

// NOTE (yiakwy) : set to 1 to enable bulk-DMA reduction (cp.async.bulk.shared::cluster) via vectorized half2 (simd_vadd); toggle with FLASH_FLOAT_BULK_SPLITK=1
#ifndef USE_BULK_SPLITK_REDUCE
#define USE_BULK_SPLITK_REDUCE 1
#endif

#define USE_INPALCE_TRI_TRANSPOSE 1

#ifndef CONSUMER_THREADS
#define CONSUMER_THREADS 256
#endif

#ifndef PRODUCER_THREADS
#define PRODUCER_THREADS 128
#endif

#ifndef WARP_GROUP
#define WARP_GROUP 4
#endif

#ifndef WARP_GROUP_SIZE
#define WARP_GROUP_SIZE 128
#endif

#define CONSUMER_WARPS (CONSUMER_THREADS / WARP_SIZE) // 8

#define CONSUMER_WARPGROUPS (CONSUMER_WARPS / WARP_GROUP) // 2

// NOTE (yiakwy) : deepgeem uses legacy "bar.sync" which does not support compile time constant.
template<int num_threads=CONSUMER_THREADS>
static __device__ __forceinline__ void warpgroup_sync(int barrier_id=7) {
    asm volatile("barrier.sync %0, %1;\n" :: "r"(barrier_id), "n"(num_threads) : "memory");
}

// TODO (yiakwy) : remove
static __device__ __forceinline__ uint32_t mapa_shared_cluster(uint32_t local_addr, uint32_t cta_rank) {
    uint32_t mapped;
    asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(mapped) : "r"(local_addr), "r"(cta_rank));
    return mapped;
}

// TODO (yiakwy) : remove
static __device__ __forceinline__ void mbar_arrive_cluster_release(uint64_t* bar, uint32_t cta_rank) {
    const uint32_t mapped = mapa_shared_cluster(__cvta_generic_to_shared(bar), cta_rank);
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0], 1;" ::"r"(mapped));
}

namespace nvgpu {
    namespace arch {

// TODO (yiakwy) : remove
static __device__ __forceinline__ void cluster_cp_async_bulk(
    void* dst_local_smem,
    const void* src_remote_smem,
    uint32_t bytes,
    uint64_t* s_mbar) {

    uint32_t dst_addr = __cvta_generic_to_shared(dst_local_smem);
    const uint32_t src_addr = __cvta_generic_to_shared(src_remote_smem);

    uint32_t mbar_addr = __cvta_generic_to_shared(s_mbar);

    // NOTE (yiakwy) : push only smem to smem bulk copy
    asm volatile(
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :: "r"(dst_addr), "r"(src_addr), "r"(bytes), "r"(mbar_addr)
        : "memory"
    );
}

// TODO (yiakwy) : remove
static __device__ __forceinline__ void simd_vadd(half2* dst, const half2* src) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) { // 128-bit = 8x half = 4x half2
        dst[i] = __hadd2(dst[i], src[i]);
    }
}

    } // namespace arch
} //  namespace nvgpu

namespace xpu {

template <int BM, int BN, int BK, int STAGES, int GROUP_SIZE_M, int CLUSTER_SIZE_M>
struct HopperPersistentSplitKPipeline {

    // sm90+
    static __device__ inline void run_persistent(
        const CUtensorMap* tma_desc_X,
        const CUtensorMap* tma_desc_W,
        const CUtensorMap* tma_desc_O,
        const CUtensorMap* tma_desc_O_swizzle,
        const CUtensorMap* tma_desc_O_trans,
        const float* scale_X,
        const float* scale_W,
        half* Out,
        int M, int N, int K, int B,
        int total_symmetric_tiles,
        int num_blocks_m,
        int num_blocks_n,
        uint8_t* smem_buffer
    ) {
        using fp8_t = __nv_fp8_e4m3;
        using fp32_t = float;

        using AccDtype = fp32_t;
        using OutDtype = half;

        const int tid = threadIdx.x;
        // const int lane_id = threadIdx.x % WARP_SIZE;

        const int wg_lane_id = tid % 128;
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int wg_id = warp_id / WARP_GROUP;

        bool is_consumer = wg_id < CONSUMER_WARPGROUPS;
        bool is_producer = !is_consumer; // select one warp group

        bool is_leader_thr_in_wgs = is_consumer ? (tid == 0) : (tid == CONSUMER_THREADS);

        if (is_producer && is_leader_thr_in_wgs) {
            // NOTE (yiakwy) : batch symmetric gemm : a single batch-expanded 2D TMA descriptor
            // serves all batches (see symm_gemm_fp8_block_scaled), so prefetch only those two.
            {
                uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(tma_desc_X);
                asm volatile (
                    "prefetch.tensormap [%0];"
                    :
                    : "l"(gmem_int_desc)
                    : "memory");
            } // prefetch tma_desc_X

            {
                uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(tma_desc_W);
                asm volatile (
                    "prefetch.tensormap [%0];"
                    :
                    : "l"(gmem_int_desc)
                    : "memory");
            } // prefetch tma_desc_W
        } // prefetch tma_desc_X and tma_desc_W
        __syncwarp();

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

        constexpr uint32_t tma_bytes_X = BM * BK;
        constexpr uint32_t tma_bytes_W = BN * BK;

        constexpr uint32_t total_stage_bytes = tma_bytes_X + tma_bytes_W;

        // NOTE (yiakwy) : init full barriers for TMA, in wasp, we have empty barriers to indicate the smem to be writable and full barriers readable
        __shared__ __align__(128) uint64_t barriers[STAGES]; // full_barriers
        __shared__ __align__(128) uint64_t empty_barriers[STAGES];

        // NOTE (yiakwy) : epilogue barrier for on chip split-k reduction
        __shared__ __align__(128) uint64_t epilogue_barriers[1];
        __shared__ __align__(128) uint64_t epilogue_readable_barriers[1];
#if USE_BULK_SPLITK_REDUCE
        // NOTE (yiakwy) : completion barrier of the bulk copies in split-k reduction
        __shared__ __align__(128) uint64_t bulk_reduce_barriers[1];
#endif

        if (threadIdx.x == 0) {
            #pragma unroll
            for (int s = 0; s < STAGES; ++s) {
                nvgpu::arch::tma_init_barrier<USE_CLUSTER_MULTICAST>(&barriers[s], 1);
                nvgpu::arch::tma_init_barrier<USE_CLUSTER_MULTICAST>(&empty_barriers[s], CLUSTER_SIZE_M * CONSUMER_WARPGROUPS);
            }
        }
        nvgpu::arch::tma_store_fence();
        __syncthreads();

        uint64_t cache_hint_lhs = static_cast<uint64_t>(nvgpu::arch::CacheHintSm90::EVICT_NORMAL);
        uint64_t cache_hint_rhs = static_cast<uint64_t>(nvgpu::arch::CacheHintSm90::EVICT_NORMAL);

        int split_k_id = blockIdx.x;
        int split_k = gridDim.x;

        if (split_k > 1) {
            if (threadIdx.x == 0) {
                nvgpu::arch::tma_init_barrier<false>(&epilogue_barriers[0], split_k - 1);
                nvgpu::arch::tma_init_barrier<false>(&epilogue_readable_barriers[0], split_k - 1);
#if USE_BULK_SPLITK_REDUCE
                nvgpu::arch::tma_init_barrier<false>(&bulk_reduce_barriers[0], 1);
#endif
            }
        }
        nvgpu::arch::tma_store_fence();
        __syncthreads();

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

        int k_start = split_k_id * k_tiles_per_slice;
        int k_end = min(k_tiles_total, (split_k_id + 1) * k_tiles_per_slice);

        int local_task_id = blockIdx.y;
        __syncthreads();

        // NOTE (yiakwy) : batch symmetric gemm, similar to the triton gluon XXT_kernel:
        //   batch_idx = pid // (num_pid_m * num_pid_n); tile_id = pid % (num_pid_m * num_pid_n)
        //   i.e. the batch is the outter-most scheduling dimension of the linear tile id.
        const int tiles_per_batch = total_symmetric_tiles / B;

        // Start TMA pipeline
        if (is_producer) {
            nvgpu::arch::reg_dealloc_decrease_registers<40>();

            int write_stage = 0;
            // TODO (yiakwy) : rename
            int tma_phase = 1;

            if ( warp_id == CONSUMER_WARPS && is_leader_thr_in_wgs) {

                while (local_task_id < total_symmetric_tiles) {

                    const int batch_idx = local_task_id / tiles_per_batch;
                    const int local_tile_id = local_task_id % tiles_per_batch;

                    auto idx = get_block_indices_tri_linear(local_tile_id);
                    int block_idx_m = xpu::get<0>(idx);
                    int block_idx_n = xpu::get<1>(idx);

                    // Grouping for better L2 cache locality in TMA load
                    const uint32_t group_id = block_idx_m / GROUP_SIZE_M;

                    get_block_indices_tri_linear_swizzled<GROUP_SIZE_M>(local_tile_id, block_idx_m/*dest*/, block_idx_n/*dest*/, num_blocks_m, group_id);

                    // NOTE (yiakwy) : add batch symmetric gemm support
                    const int batch_offset_m = batch_idx * M;
                    const int batch_offset_n = batch_idx * N;
                    // For B > 1 cluster multicast is disabled by the caller (use_multicast = (B == 1)).

                    // 1. Ramp Up Fill : to initiate the pipeline, we will fill STAGES-1 stages of data before entering the main loop, and then maintain 1 stage ahead of the main loop to keep the pipeline full.
#if defined(DEBUG_BLOCK) && DEBUG_BLOCK
                    printf("  [Producer] [Split#%d] [SM#%d] [local_task_id#%d] [tid#%d] : group_id#%d, start to filling buffer ... \n", blockIdx.x, blockIdx.y, local_task_id, tid, group_id);
#endif

        #if  USE_CLUSTER_MULTICAST
                    producer<STAGES, GROUP_SIZE_M, BM, BN, BK, USE_CLUSTER_MULTICAST, USE_LINEAR_TO_TRIL_LAYOUT>::load(
                        tid, group_id, batch_offset_m, batch_offset_n, block_idx_m, block_idx_n,
                        k_start, k_end, total_stage_bytes,
                        tma_desc_X, tma_desc_W,
                        shmem_X, shmem_W, barriers, empty_barriers,
                        cluster_mask, cluster_group_m_rank, cache_hint_lhs, cache_hint_rhs,
                        /*use_multicast=*/(B == 1),
                        write_stage/*src & dst*/, tma_phase/*src & dst*/);
        #else
                    producer<STAGES, GROUP_SIZE_M, BM, BN, BK, USE_CLUSTER_MULTICAST, USE_LINEAR_TO_TRIL_LAYOUT>::load(
                        tid, group_id, batch_offset_m, batch_offset_n, block_idx_m, block_idx_n,
                        k_start, k_end, total_stage_bytes,
                        tma_desc_X, tma_desc_W,
                        shmem_X, shmem_W, barriers, empty_barriers,
                        cache_hint_lhs, cache_hint_rhs,
                        write_stage/*src & dst*/, tma_phase/*src & dst*/);
        #endif

#if defined(DEBUG_BLOCK) && DEBUG_BLOCK
                    printf("  [Producer] [Split#%d] [SM#%d] [local_task_id#%d] [tid#%d] : buffer filled. \n", blockIdx.x, blockIdx.y, local_task_id, tid);
#endif

                    local_task_id += gridDim.y;

                } // while

#if defined(DEBUG_BLOCK) && DEBUG_BLOCK
                printf("  [Producer] [Split#%d] [SM#%d] [local_task_id#%d] [tid#%d] :  ===== The Block Finished producing ===== \n", blockIdx.x, blockIdx.y, local_task_id, tid);
#endif

            } // if ( warp_id == CONSUMER_WARPS && is_leader_thr_in_wgs)

        } else { // compute groups
            nvgpu::arch::reg_alloc_increase_registers<232>();

            int read_stage = 0;
            // TODO (yiakwy) : rename
            int tma_phase = 0;

            // TODO (yiakwy) : rename
            uint32_t epilogue_phase = 0;
            uint32_t epilogue_readable_phase = 0;
#if USE_BULK_SPLITK_REDUCE
            uint32_t bulk_reduce_phase = 0;
#endif

            int offset = STAGES * (BM * BK + BN * BK);
            OutDtype* shmem_epilogue = reinterpret_cast<OutDtype *>(smem_buffer + offset);

            OutDtype* shmem_epilogue_trans = shmem_epilogue + BM * BN;
            FragmentView<OutDtype, BM, BN, MemoryDomain::kShared> frag_view(shmem_epilogue);

            constexpr int SCLAE_BLOCK_SIZE_K = 128;
            constexpr int K_TILES_TOTAL = (8192 + SCLAE_BLOCK_SIZE_K - 1) / SCLAE_BLOCK_SIZE_K;

            offset += sizeof(OutDtype) * BM * BN;
            auto* shmem_XS = reinterpret_cast<float*>(smem_buffer + offset);
            auto* shmem_WS = reinterpret_cast<float*>(smem_buffer + offset + sizeof(fp32_t) * BM * K_TILES_TOTAL);

#if USE_BULK_SPLITK_REDUCE
            // NOTE (yiakwy) : bulk-DMA reduce staging buffer. This region can be reused in epilogue.
            OutDtype* splitk_staging = reinterpret_cast<OutDtype*>(shmem_XS);
#endif

            xpu::HopperWGMMAAccumulator<BM, BN, BK> accum;

            // NOTE (yiakwy) : keep WGMMA of k-tile k+1 on the tensor core while scale_add (FMA) of k-tile k on CUDA cores
            xpu::HopperWGMMAAccumulator<BM, BN, BK> step_even;
            xpu::HopperWGMMAAccumulator<BM, BN, BK> step_odd;

            __shared__ __align__(128) OutDtype *dst[8];

            while (local_task_id < total_symmetric_tiles) {
                accum.clear();

#if USE_BULK_SPLITK_REDUCE

                if (split_k >= 2 && split_k_id == 0 && threadIdx.x == 0) {
                    nvgpu::arch::tma_expect_bytes(&bulk_reduce_barriers[0],
                        static_cast<uint32_t>(BM * BN * sizeof(OutDtype)));
                }
#endif

                const int batch_idx = local_task_id / tiles_per_batch;
                const int local_tile_id = local_task_id % tiles_per_batch;

                auto idx = get_block_indices_tri_linear(local_tile_id);
                int block_idx_m = xpu::get<0>(idx);
                int block_idx_n = xpu::get<1>(idx);

                // Grouping for better L2 cache locality in TMA load
                const uint32_t group_id = block_idx_m / GROUP_SIZE_M;

                get_block_indices_tri_linear_swizzled<GROUP_SIZE_M>(local_tile_id, block_idx_m/*dest*/, block_idx_n/*dest*/, num_blocks_m, group_id);

                // NOTE (yiakwy) : add batch symmetric gemm support
                const int _block_idx_m = block_idx_m;
                const int _block_idx_n = block_idx_n;

                const int batch_offset_m = batch_idx * M;

                block_idx_m += batch_idx * num_blocks_m;
                block_idx_n += batch_idx * num_blocks_n;

#if defined(DEBUG_BLOCK) && DEBUG_BLOCK
                if (threadIdx.x == 0 && blockIdx.x == 0) {
                    printf("[Consumer] [Prefetch] [Split#%d] [SM#%d] : batch#%d block#(%d, %d), group_id#%d prefetching scales ...\n", blockIdx.x, blockIdx.y, batch_idx, _block_idx_m, _block_idx_n, group_id);
                }
                warpgroup_sync();
#endif

                // NOTE (yiakwy) : rows of 1 or more BLOCKS share a scale
                const int stride_xs_m = K / SCLAE_BLOCK_SIZE_K;
                const int stride_ws_n = K / SCLAE_BLOCK_SIZE_K;

                constexpr int shares_per_scale = SCLAE_BLOCK_SIZE_K / BK;
                // static_assert(shares_per_scale == 2, "with BK=%d, each scale will be shared by 2 tiles, please adjust SCLAE_BLOCK_SIZE_K or BK to ensure that.");

                const int k_tiles = k_end - k_start;
                const int total_xs_elements = BM * k_tiles;
                const int total_ws_elements = k_tiles;

                // NOTE (yiakwy) : prefetch all scale without TMA

#if !(defined(ABL_NO_SCALE_FFMA) && ABL_NO_SCALE_FFMA)
                // TODO (yiakwy) : remap shmem_XS to per-thread registers to reduce the latency, since the scale load is on the critical path of the main loop.
                #pragma unroll 4
                for (int i = tid; i < total_xs_elements; i += CONSUMER_THREADS) {
                    int s_row = i / k_tiles;
                    int s_col = i % k_tiles;

                    int g_row = block_idx_m * BM + s_row;
                    int g_col = (k_start + s_col) / shares_per_scale;

                    shmem_XS[s_col * BM + s_row] = scale_X[g_row * stride_xs_m + g_col];
                }

                // TODO (yiakwy) : remap shmem_WS to per-thread registers to reduce the latency, since the scale load is on the critical path of the main loop.
                #pragma unroll 4
                for (int i = tid; i < total_ws_elements; i += CONSUMER_THREADS) {
                    int s_col = i;

                    int g_row = block_idx_n;
                    int g_col = (k_start + s_col) / shares_per_scale;

                    shmem_WS[s_col] = scale_W[g_row * stride_ws_n + g_col];
                }
#endif
                // one 256-thread barrier publishes both scale arrays to both
                // consumer warp groups (was: one barrier per array)
                warpgroup_sync();

                // 2. main loop

                if (k_start < k_end) {
                    // consumer_full_barrier_wait(
                    nvgpu::arch::tma_wait(
                        __cvta_generic_to_shared(&barriers[read_stage]), tma_phase);

                    // wgmma_batch_commit(step_even,
                    HopperWGMMAExecutor::mma_scaled_interleaved_commit(step_even,
                        __cvta_generic_to_shared(&shmem_X[read_stage]),
                        __cvta_generic_to_shared(&shmem_W[read_stage]));
                }

                for (int k_tile = k_start; k_tile < k_end; ++k_tile) {
                    const bool has_next = (k_tile + 1 < k_end);
                    const int k_off = k_tile - k_start;

                    // 1. issue NEXT k-tile's batch: the first k-step uses
                    //    scale-d = 0, so no explicit accumulator clear and no
                    //    stale-buffer hazard across tile boundaries.
                    if (has_next) {
                        int next_stage = read_stage + 1;
                        if (next_stage == STAGES) next_stage = 0;
                        const int next_phase = (next_stage == 0) ? (tma_phase ^ 1) : tma_phase;

                        // consumer_full_barrier_wait(
                        nvgpu::arch::tma_wait(
                            __cvta_generic_to_shared(&barriers[next_stage]), next_phase);

                        // compile-time ping-pong buffer selection
                        if ((k_off & 1) == 0) {
                            // wgmma_batch_commit(step_odd,
                            HopperWGMMAExecutor::mma_scaled_interleaved_commit(step_odd,
                                __cvta_generic_to_shared(&shmem_X[next_stage]),
                                __cvta_generic_to_shared(&shmem_W[next_stage]));
                        } else {
                            // wgmma_batch_commit(step_even,
                            HopperWGMMAExecutor::mma_scaled_interleaved_commit(step_even,
                                __cvta_generic_to_shared(&shmem_X[next_stage]),
                                __cvta_generic_to_shared(&shmem_W[next_stage]));
                        }
                    }

                    // 2. wait for THIS k-tile's batch (leave the next batch in flight)
                    if (has_next) {
                        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
                    } else {
                        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
                    }

                    // 3. release the consumed stage: batch k's WGMMA smem reads
                    //    are complete after the wait above.
#if USE_CLUSTER_MULTICAST
                    if (wg_lane_id < CLUSTER_SIZE_M) {
                        mbar_arrive_cluster_release(&empty_barriers[read_stage], wg_lane_id);
                    }
#else
                    if (threadIdx.x == 0) {
                        uint32_t bar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&empty_barriers[read_stage]));
                        asm volatile(
                            "{\n" ".reg .b64 state; \n"
                            "mbarrier.arrive.shared::cta.b64 state, [%0];\n" "}\n"
                            :: "r"(bar_addr) : "memory"
                        );
                    }
#endif

                    // 4. fused scale + accumulate (FFMA) for this k-tile;
                    //    overlaps the next batch's WGMMA on the tensor core.
#if defined(ABL_NO_SCALE_FFMA) && ABL_NO_SCALE_FFMA
                    if ((k_off & 1) == 0) {
                        accum.add_(step_even);
                    } else {
                        accum.add_(step_odd);
                    }
#else
                    if ((k_off & 1) == 0) {
                        xpu::HopperWGMMAAccumulator<BM, BN, BK>::scaled_add(accum, step_even,
                            &shmem_XS[0], &shmem_WS[0], k_off);
                    } else {
                        xpu::HopperWGMMAAccumulator<BM, BN, BK>::scaled_add(accum, step_odd,
                            &shmem_XS[0], &shmem_WS[0], k_off);
                    }
#endif

                    read_stage += 1;
                    if (read_stage == STAGES) {
                        read_stage = 0;
                        tma_phase ^= 1;
                    }
                } // main loop

                // 3. Epilogue
                //   - first write data back to share memory for SPLIT-K reduction via NoC
                //   - applying successive operations upon tile results in the epilogue, such as bias add, activation, etc, can be fused in this step to save memory bandwidth.
#if defined(ABL_NO_EPILOGUE) && ABL_NO_EPILOGUE
                if (false) {
#endif
#if defined(DEBUG_BLOCK) && DEBUG_BLOCK
                if (threadIdx.x == 0 && blockIdx.x == 0) {
                    printf("[Epilogue] [Split#%d] [SM#%d] copying acc to shared memory...\n", blockIdx.x, blockIdx.y);
                }
#endif

                if (threadIdx.x == 0) {
                    nvgpu::arch::tma_store_wait();
                }
                warpgroup_sync<128>(wg_id);

                accum.store(shmem_epilogue);
                asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
                warpgroup_sync();

#if defined(DEBUG_BLOCK) && DEBUG_BLOCK
                if (threadIdx.x == 0 && blockIdx.x == 0) {
                    printf("[Epilogue] [Split#%d] [SM#%d] Completes copying acc to shmem. Writing local_task_id %d (<%d, %d>) to global ...\n", blockIdx.x, blockIdx.y, local_task_id, block_idx_m, block_idx_n);
                }
#endif

    #if __CUDA_ARCH__ >= 900 && ENABLE_HOPPER // Hopper 900+ GPU with TMA support
                // NOTE (yiakwy) : debug trace, keep silent by default (printf serializes the SM)
                if constexpr (DEBUG_BLOCK) {
                    if (threadIdx.x == 0 && blockIdx.x == 1) {
                        printf("[Epilogue] [Split#%d] [SM#%d] write split_k#%d block <%d, %d> on-chip reduce via NoC...\n", blockIdx.x, blockIdx.y, split_k, block_idx_m, block_idx_n);
                    }
                }

                auto cluster = cooperative_groups::this_cluster();

                if (split_k > 1) {
                    if (split_k_id > 0) {

#if USE_BULK_SPLITK_REDUCE
                        // NOTE (yiakwy) : we tried split_k >=2 with on chip reduction, and it does not work; split-2 itself can bring 22% improvement
                        if (split_k >= 2 && threadIdx.x == 0) {
                            OutDtype* rank0_staging = cluster.map_shared_rank<OutDtype>(
                                reinterpret_cast<OutDtype*>(shmem_XS), 0);
                            uint64_t* rank0_bulk_bar =
                                cluster.map_shared_rank<uint64_t>(&bulk_reduce_barriers[0], 0);
                            nvgpu::arch::cluster_cp_async_bulk(
                                rank0_staging,
                                shmem_epilogue,
                                static_cast<uint32_t>(BM * BN * sizeof(OutDtype)),
                                rank0_bulk_bar);
                        }
#endif

                        if (threadIdx.x == 0) {
                            uint32_t target_cta_rank = 0;
                            mbar_arrive_cluster_release(&epilogue_barriers[0], target_cta_rank);
                        }

                        nvgpu::arch::tma_wait(__cvta_generic_to_shared(&epilogue_readable_barriers[0]), epilogue_readable_phase);
                        epilogue_readable_phase ^= 1;
                        warpgroup_sync();

                    } else if (split_k_id == 0) {

                        if (threadIdx.x == 0) {
                            nvgpu::arch::tma_wait(__cvta_generic_to_shared(&epilogue_barriers[0]), epilogue_phase);
                        }
                        warpgroup_sync();

                        // flip once
                        epilogue_phase ^= 1;

                        if (threadIdx.x == 0) {
                            for (int r = 1; r < split_k; ++r) {
                                OutDtype* dst_shmem_epilogue = cluster.map_shared_rank<OutDtype>(&shmem_epilogue[0], r);
                                dst[r] = dst_shmem_epilogue;
                            }
                        }
                        warpgroup_sync();

#if USE_BULK_SPLITK_REDUCE

                        // NOTE (yiakwy) : we tried split_k >=2 with on chip reduction, and it does not work; split-2 itself can bring 22% improvement
                        if (split_k >= 2) {
                            
                            nvgpu::arch::tma_wait(__cvta_generic_to_shared(&bulk_reduce_barriers[0]), bulk_reduce_phase);
                            warpgroup_sync();
                            bulk_reduce_phase ^= 1;

                            constexpr uint32_t reduce_halves = BM * BN;
                            float4* reduce_dst4 = reinterpret_cast<float4*>(&shmem_epilogue[0]);
                            const float4* staging_src4 = reinterpret_cast<const float4*>(splitk_staging);

                            #pragma unroll
                            for (int idx = tid; idx < reduce_halves / 8; idx += CONSUMER_THREADS) {
                                float4 vd = reduce_dst4[idx];
                                float4 vs = staging_src4[idx];
                                nvgpu::arch::simd_vadd(
                                    reinterpret_cast<half2*>(&vd),
                                    reinterpret_cast<const half2*>(&vs));
                                reduce_dst4[idx] = vd;
                            }
                            warpgroup_sync();
                        } else
#endif
                        {
                            // NOTE (yiakwy) : naive solution
                            for (int r = 1; r < split_k; ++r) {
                                OutDtype* dst_shmem_epilogue = dst[r];
                                for (int idx = tid; idx < BM * BN; idx += CONSUMER_THREADS) {
                                    shmem_epilogue[idx] += dst_shmem_epilogue[idx];
                                }
                            }
                            warpgroup_sync();
                        }

                        if (threadIdx.x == 0) {
                            for (int r = 1; r < split_k; ++r) {
                                mbar_arrive_cluster_release(&epilogue_readable_barriers[0], static_cast<uint32_t>(r));
                            }
                        }
                        warpgroup_sync();

                    } // split_k_id == 0
                } //  split_k > 1

#if defined(DEBUG_BLOCK) && DEBUG_BLOCK
                if (threadIdx.x == 0 && blockIdx.x == 0) {
                    printf("[Epilogue] [Split#%d] [SM#%d] [local_task_id%d] issuing [Copy 1] ... \n", blockIdx.x, blockIdx.y, local_task_id);
                }
#endif

                if (split_k_id == 0) {

                    if (threadIdx.x == 0) {
                        uint64_t tma_o_addr = reinterpret_cast<uint64_t>(tma_desc_O);
                        uint32_t smem_epilogue_addr  = static_cast<uint32_t>(__cvta_generic_to_shared(&shmem_epilogue[0]));

                        asm volatile (
                            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
                            " [%0, {%2, %3}], [%1];"
                            :
                            : "l"(tma_o_addr), "r"(smem_epilogue_addr),
                            "r"(_block_idx_n * BN), "r"(_block_idx_m * BM + batch_offset_m)
                            : "memory"
                        );
    #if (defined(USE_INPALCE_TRI_TRANSPOSE)) && USE_INPALCE_TRI_TRANSPOSE
                        asm volatile("cp.async.bulk.commit_group;");
    #endif
                    }

#if defined(DEBUG_BLOCK) && DEBUG_BLOCK
                    if (threadIdx.x == 0) {
                        printf("[Epilogue] [Split#%d] [SM#%d] [local_task_id#%d] [Copy 1] issued.\n", blockIdx.x, blockIdx.y, local_task_id);
                    }
#endif

                    // NOTE (yiakwy) :  transpose copy to upper right
                    if (_block_idx_m > _block_idx_n) {

#if defined(DEBUG_BLOCK) && DEBUG_BLOCK
                        if (threadIdx.x == 0) {
                            printf("[Epilogue] [Split#%d] [SM#%d] [local_task_id#%d] issuing [Inplace Transpose] ...\n", blockIdx.x, blockIdx.y, local_task_id);
                        }
#endif

    #if (defined(USE_INPALCE_TRI_TRANSPOSE)) && USE_INPALCE_TRI_TRANSPOSE
                        if (threadIdx.x == 0) {
                            // asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
                            nvgpu::arch::tma_store_wait();
                        }
                        warpgroup_sync();

                        // NOTE (yiakwy) : inplace transpose
                        frag_view._transpose();

                        if (threadIdx.x == 0) {

    #if SWIZZLE_64B_STORE
                            uint64_t tma_o_addr = reinterpret_cast<uint64_t>(tma_desc_O_swizzle);
    #else
                            uint64_t tma_o_addr = reinterpret_cast<uint64_t>(tma_desc_O);
    #endif // SWIZZLE_64B_STORE

                            uint32_t smem_epilogue_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&shmem_epilogue[0]));

    #if SWIZZLE_64B_STORE
                            asm volatile (
                                "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
                                " [%0, {%2, %3}], [%1];"
                                :
                                : "l"(tma_o_addr), "r"(smem_epilogue_addr),
                                "r"(_block_idx_n * BN), "r"(_block_idx_m * BM + batch_offset_m)
                            );

                            const uint32_t smem_epilogue_addr_next = smem_epilogue_addr + 128;

                            asm volatile (
                                "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
                                " [%0, {%2, %3}], [%1];"
                                :
                                : "l"(tma_o_addr), "r"(smem_epilogue_addr_next),
                                    "r"(_block_idx_n * BN + 64), "r"(_block_idx_m * BM + batch_offset_m)
                            );
    #else
                            asm volatile (
                                "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
                                " [%0, {%2, %3}], [%1];"
                                :
                                : "l"(tma_o_addr), "r"(smem_epilogue_addr),
                                "r"(_block_idx_m * BN), "r"(_block_idx_n * BM + batch_offset_m)
                            );
    #endif // SWIZZLE_64B_STORE
                            asm volatile("cp.async.bulk.commit_group;");
                        } // inplace copy

    #else
                        // NOTE (yiakwy) : outplace transpose
                        if (threadIdx.x == 0) {
                            // outplace transpose copy

                            uint64_t tma_o_addr = reinterpret_cast<uint64_t>(tma_desc_O_trans);

                            uint32_t smem_epilogue_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&shmem_epilogue[0]));

                            asm volatile (
                                "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
                                " [%0, {%2, %3}], [%1];"
                                :
                                : "l"(tma_o_addr), "r"(smem_epilogue_addr),
                                "r"(_block_idx_m * BN), "r"(_block_idx_n * BM + batch_offset_m)
                            );

                        } // outplace copy

                        asm volatile("cp.async.bulk.commit_group;");
    #endif // USE_INPALCE_TRI_TRANSPOSE

#if defined(DEBUG_BLOCK) && DEBUG_BLOCK
                        if (threadIdx.x == 0) {
                            printf("[Epilogue] [Split#%d] [SM#%d] [local_task_id#%d] [Copy2] issued\n", blockIdx.x, blockIdx.y, local_task_id);
                        }
#endif

                    } // block_idx_m > block_idx_n

                } // split_id == 0

                warpgroup_sync();

    #else

                for (int idx = threadIdx.x; idx < BM * BN; idx += CONSUMER_THREADS) {
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
                __syncwarp();

    #endif // __CUDA_ARCH__ >= 900 && ENABLE_HOPPER

#if defined(ABL_NO_EPILOGUE) && ABL_NO_EPILOGUE
                } // ABL_NO_EPILOGUE
#endif

                // fetch next task
                local_task_id += gridDim.y;
                __syncwarp();

            } // while

#if defined(DEBUG_BLOCK) && DEBUG_BLOCK
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                printf("***** [Epilogue] [Split#%d] [SM#%d] Compelte Writing local_task_id#%d back to global. *****\n", blockIdx.x, blockIdx.y, local_task_id);
            }
#endif

        } // compute groups

    } // run_persistent

};

} // namespace xpu

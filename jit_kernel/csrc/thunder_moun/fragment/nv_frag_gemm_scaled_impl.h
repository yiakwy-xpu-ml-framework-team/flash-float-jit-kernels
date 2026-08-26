/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

#pragma once
#include <type_traits>

#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include <cuda_bf16.h>

#include "fragment.h"

#ifndef CONSUMER_THREADS
#define CONSUMER_THREADS 256
#endif

#ifndef WARP_GROUP
#define WARP_GROUP 4
#endif

#ifndef WARP_GROUP_SIZE
#define WARP_GROUP_SIZE 128
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif


// NOTE (yiakwy) : for debug purpose
// TODO (yiakwy) : remove
static __device__ __forceinline__ void _warpgroup_sync_256(int barrier_id=7) {
    asm volatile("barrier.sync %0, %1;\n" :: "r"(barrier_id), "n"(256) : "memory");
}


namespace xpu {

// WGMMA register based fragment accumulator
template <int _BM, int _BN, int _BK>
struct HopperWGMMAAccumulator {
    using AccDtype = float;

    static constexpr int BM = _BM;

    static constexpr int BN = _BN;

    static constexpr int BK = _BK;

    static constexpr int FRAG_M = 64;

    static constexpr int FRAG_N = 128;

    static constexpr int FRAG_K = 32;

    // NOTE (yiakwy) : Hopper m64n128k32 requires BM * BN / 128 (4 warps consists wgmma warp group) = 64 registers per thread, this is a fixed mapping determined by the hardware and cannot be changed.
    static constexpr int kRegistersPerThread = 64;

    // NOTE (yiakwy) : for BM = 128, threads_per_block = 128, MAX_M_STEPS == 2
    static constexpr int MAX_M_STEPS = 1;

    AccDtype regs[MAX_M_STEPS][kRegistersPerThread] = {0.0f};

    __device__ inline void clear() {
        #pragma unroll
        for (int j = 0; j < MAX_M_STEPS; ++j) {
            #pragma unroll
            for (int i = 0; i < kRegistersPerThread; ++i) { regs[j][i] = 0.0f; }
        }
    }

    __device__ inline float* get_reg_ptr() { return reinterpret_cast<float *>(&regs[0][0]); }

    __device__ inline uint32_t get_reg_num_per_frag() { return kRegistersPerThread; }

    __device__ inline int getTargetWgmmaSmemOffset(int wg_id, int wg_lane_id, int reg_idx, int m_step, int M_STEPS, int* dest_row=nullptr, int* dest_col=nullptr) {
        // NOTE (yiakwy) : 16x8 layout per warp, each register holds 2x(2 consecutive elements) (float2) with row stride 8
#define ROWS_PER_WARP (FRAG_M / WARP_GROUP)

#define THREADS_PER_ROW 4

#define ROW_STRIDE 8
#define COL_STRIDE 8

#define ROW_REPEATS 2
#define VEC_SIZE 2

#define ELE_PER_THREAD ((ROW_REPEATS) * (VEC_SIZE))

        static_assert(ROWS_PER_WARP == 16, "must be 16 in this mma instruction configuration.");

        const int warp_id = wg_lane_id / WARP_SIZE;
        const int lane_id = wg_lane_id % WARP_SIZE;

        const int wg_off_row = wg_id * FRAG_M * M_STEPS;
        const int frag_off_row = m_step * FRAG_M;

        const int thr_off_row = lane_id / THREADS_PER_ROW;
        const int thr_off_col = lane_id % THREADS_PER_ROW;

        const int i = reg_idx / ELE_PER_THREAD;
        const int sub_idx = reg_idx % ELE_PER_THREAD;

        const int local_row = warp_id * ROWS_PER_WARP + thr_off_row + ((sub_idx / VEC_SIZE) * ROW_STRIDE);
        const int row = local_row + m_step * FRAG_M + wg_id * FRAG_M * M_STEPS; // check

        const int col = thr_off_col * VEC_SIZE + i * COL_STRIDE + (sub_idx % VEC_SIZE);

        if (dest_row != nullptr) {
            *dest_row = row;
        }

        if (dest_col != nullptr) {
            *dest_col = col;
        }

        return row * BN + col;
    }

    template<typename Dtype>
    __device__ inline void store(Dtype* smem) {
        const int warp_id = threadIdx.x / WARP_SIZE;
        // const int lane_id = threadIdx.x % WARP_SIZE;

        const int wg_id = warp_id / WARP_GROUP;
        const int wg_lane_id = threadIdx.x % WARP_GROUP_SIZE;

        constexpr int wgs = CONSUMER_THREADS / WARP_GROUP_SIZE;

        const int M_STEPS = BM / FRAG_M / wgs;

        #pragma unroll
        for (int m_step = 0; m_step < M_STEPS; ++m_step) {
            #pragma unroll
            for (int i = 0; i < kRegistersPerThread; ++i) {
                int smem_offset = getTargetWgmmaSmemOffset(wg_id, wg_lane_id, i, m_step, M_STEPS);

                if constexpr (!std::is_same_v<Dtype, AccDtype>) {
                    smem[smem_offset] = static_cast<Dtype>(regs[m_step][i]);
                } else {
                    smem[smem_offset] = regs[m_step][i];
                }
            }
        }
        __syncwarp();
    }

    __device__ inline void mul_(float scale) {
        constexpr int wgs = CONSUMER_THREADS / WARP_GROUP_SIZE;

        const int M_STEPS = BM / FRAG_M / wgs;

        #pragma unroll
        for (int m_step = 0; m_step < M_STEPS; ++m_step) {
            #pragma unroll
            for (int i = 0; i < kRegistersPerThread; ++i) {
                regs[m_step][i] *= scale;
            }
        }
    }

    __device__ inline void mul_(float* xs, float* ws, int k_offset=0) {
        const int warp_id = threadIdx.x / WARP_SIZE;
        // const int lane_id = threadIdx.x % WARP_SIZE;

        const int wg_id = warp_id / WARP_GROUP;
        const int wg_lane_id = threadIdx.x % WARP_GROUP_SIZE;

        constexpr int wgs = CONSUMER_THREADS / WARP_GROUP_SIZE;

        const int M_STEPS = BM / FRAG_M / wgs;

        float _ws = ws[k_offset];

        #pragma unroll
        for (int m_step = 0; m_step < M_STEPS; ++m_step) {
            #pragma unroll
            for (int i = 0; i < kRegistersPerThread; ++i) {
                int row, col;
                int _ = getTargetWgmmaSmemOffset(wg_id, wg_lane_id, i, m_step, M_STEPS, &row, &col);

                float scale = xs[k_offset * (BM + 1) + row] * _ws;

                regs[m_step][i] *= scale;
            }
        }
    }

    __device__ inline void add_(const HopperWGMMAAccumulator& b) {
        constexpr int wgs = CONSUMER_THREADS / WARP_GROUP_SIZE;

        const int M_STEPS = BM / FRAG_M / wgs;

        #pragma unroll
        for (int m_step = 0; m_step < M_STEPS; ++m_step) {
            #pragma unroll
            for (int i = 0; i < kRegistersPerThread; ++i) {
                this->regs[m_step][i] += b.regs[m_step][i];
            }
        }
    }

    __device__ inline void fma_scaled_(const HopperWGMMAAccumulator& local, float* xs, float* ws, int k_offset = 0) {
        const int warp_id = threadIdx.x / WARP_SIZE;
        // const int lane_id = threadIdx.x % WARP_SIZE;

        const int wg_id = warp_id / WARP_GROUP;
        const int wg_lane_id = threadIdx.x % WARP_GROUP_SIZE;

        const int wgs = blockDim.x / WARP_GROUP_SIZE;
        const int M_STEPS = BM / FRAG_M / wgs;

        float _ws = ws[k_offset];

        const int warp_id_in_wg = wg_lane_id / WARP_SIZE;
        const int lane_id = wg_lane_id % WARP_SIZE;

        #pragma unroll
        for (int m_step = 0; m_step < M_STEPS; ++m_step) {
            const int row0 =
                wg_id * FRAG_M * M_STEPS
                + m_step * FRAG_M
                + warp_id_in_wg * 16
                + lane_id / 4;

            const int row1 = row0 + 8;

            const float scale0 = xs[k_offset * (BM + 1) + row0] * _ws;
            const float scale1 = xs[k_offset * (BM + 1) + row1] * _ws;

            #pragma unroll
            for (int i = 0; i < (kRegistersPerThread / 4); ++i) {
                const int base = 4 * i;
                this->regs[m_step][base + 0] =
                    __fmaf_rn(local.regs[m_step][base + 0], scale0, this->regs[m_step][base + 0]);

                this->regs[m_step][base + 1] =
                    __fmaf_rn(local.regs[m_step][base + 1], scale0, this->regs[m_step][base + 1]);

                this->regs[m_step][base + 2] =
                    __fmaf_rn(local.regs[m_step][base + 2], scale1, this->regs[m_step][base + 2]);

                this->regs[m_step][base + 3] =
                    __fmaf_rn(local.regs[m_step][base + 3], scale1, this->regs[m_step][base + 3]);
            }
        }
    }

};

union TmaDesc {
    uint64_t desc_;
    uint32_t reg32_[2];
    uint16_t reg16_[4];

    struct {
        // start_address, bit [0,14), 4LSB not included
        uint16_t start_address_ : 14, : 2;        // 14 bits [0,14), 2 bits unused

        // leading dimension byte offset, bit [16,30), 4LSB not included
        // For N: This is the stride from the first col to the second col of the 8x2 brick in INTERLEAVED
        //   Unused for all SWIZZLE_* layouts (and assumed to be 1)
        // For T: This is the stride from the first 8 rows to the next 8 rows.
        uint16_t leading_byte_offset_ : 14, : 2;  // 14 bits [0,14), 2 bits unused

        // stride dimension byte offset, bit [32,46), 4LSB not included
        // For N: This is the stride from the first 8 rows to the next 8 rows.
        // For T: This is the stride fro mthe first 8 cols to the next 8 cols.
        uint16_t stride_byte_offset_ : 14, : 2;   // 14 bits [0,14), 2 bits unused

        // base_offset, bit [48,52)
        // Valid only for SWIZZLE_128B and SWIZZLE_64B
        uint8_t : 1, base_offset_ : 3, : 4;       // 1 bit unused, 3 bits [1,4), 4 bits unused

        // 56 + 6 + 2 + 2

        // layout type, bit [62,64)
        // SWIZZLE_NONE = 0, SWIZZLE_32B = 3, SWIZZLE_64B = 2, SWIZZLE_128B = 1
        uint8_t : 6, layout_type_ : 2;            // 6 bits unused, 2 bits [6,8)
    } bitfield;

};

__device__ inline uint64_t
make_smem_desc(uint32_t s_addr, const int layout_type,
    const uint32_t&  leading_byte_offset = 16,
    const uint32_t& stride_byte_offset = 512) {
    TmaDesc desc;

    desc.bitfield.start_address_ = s_addr >> 4;
    desc.bitfield.leading_byte_offset_ = leading_byte_offset >> 4;
    desc.bitfield.stride_byte_offset_ = stride_byte_offset >> 4;

    desc.bitfield.base_offset_ = 0;
    desc.bitfield.layout_type_ = layout_type;

    return desc.desc_;
}

struct HopperWGMMAExecutor {

    static constexpr int FRAG_M = 64;

    static constexpr int FRAG_N = 128;

    static constexpr int FRAG_K = 32;

    // NOTE (yiakwy) : WGMMA uses 4 warps to collaboratively compute 64x128x32 fragment
    static constexpr int WARP_LAYOUT_COL = 4;

    // 注入底层 Hopper 的 wgmma.mma_async 指令，直接使用共享内存地址做 128-bit 加载
    template <typename AccType>
    static __device__ inline void mma_scaled(
        AccType& accum,
        uint32_t smem_x_ptr,
        uint32_t smem_w_ptr)
    {
        const int warp_id = threadIdx.x / WARP_SIZE;
        // const int lane_id = threadIdx.x % WARP_SIZE;

        const int wg_id = warp_id / WARP_GROUP;
        // const int wg_lane_id = threadIdx.x % WARP_GROUP_SIZE;

        constexpr int wgs = CONSUMER_THREADS / WARP_GROUP_SIZE;

        const int M_STEPS = AccType::BM / FRAG_M / wgs;

        constexpr int N_STEPS = AccType::BN / FRAG_N;
        constexpr int K_STEPS = AccType::BK / FRAG_K;

        float* reg_ptr = accum.get_reg_ptr();

        uint32_t reg_num_per_frag = AccType::kRegistersPerThread; // accum.get_reg_num_per_frag();

        constexpr uint32_t ld_x = 0;
        constexpr uint32_t ld_w = 0;

        constexpr uint32_t swizzle_stride_x = 8 * AccType::BK; // 8 * 64 = 512
        constexpr uint32_t swizzle_stride_w = 8 * AccType::BK;

        // NOTE (yiakwy) :
        //   - ref : https://github.com/NVIDIA/cutlass/blob/5f06f5fc1a072bbe4815fae7ae8470b876ed603a/include/cute/arch/mma_sm90_desc.hpp#L108

        auto shift_sign = (1ULL << 62);
        // auto shift_sign = (3ULL << 60);

        // TODO (yiakwy) : nvgpu::arch::wgmma_fence_sync_aligned()
        asm volatile(
            "wgmma.fence.sync.aligned;\n" ::: "memory"
        );

        // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("[Consumer] [mma_scaled] M_STEPS=%d, N_STEPS=%d, K_STEPS=%d, wgs=%d, blockDim.x=%d, FRAG_M=%d\n", M_STEPS, N_STEPS, K_STEPS, wgs, blockDim.x, FRAG_M);
        // }
        // _warpgroup_sync_256();

        #pragma unroll
        for (int m_step = 0; m_step < M_STEPS; ++m_step) {
            uint32_t m_wg_stride_bytes_off_X =
                wg_id * M_STEPS * FRAG_M * AccType::BK;

            uint32_t m_frag_stride_bytes_off_X =
                m_step * FRAG_M * AccType::BK;

            uint32_t current_smem_x = smem_x_ptr +
                m_frag_stride_bytes_off_X +
                m_wg_stride_bytes_off_X;

            #pragma unroll
            for (int n_step = 0; n_step < N_STEPS; ++n_step) {
                int reg_offset = (m_step * N_STEPS + n_step) * reg_num_per_frag;

                uint32_t n_frag_stride_bytes_off_W =
                    n_step * FRAG_N * AccType::BK;
                uint32_t current_smem_w = smem_w_ptr + n_frag_stride_bytes_off_W;

                constexpr uint32_t desc_off = (uint32_t)FRAG_K;

                for (int k_step = 0; k_step < K_STEPS; ++k_step) {
                    uint32_t addr_x = current_smem_x + k_step * desc_off;
                    uint32_t addr_w = current_smem_w + k_step * desc_off;

                    /*
                    uint64_t desc_x = (((uint64_t)(addr_x >> 4)) & 0x3FFFU) |
                                      (((uint64_t)(ld_x >> 4)) << 16) |
                                      (((uint64_t)(swizzle_stride_x >> 4)) << 32) |
                                      shift_sign;

                    uint64_t desc_w = (((uint64_t)(addr_w >> 4)) & 0x3FFFU) |
                                      ((uint64_t)(ld_w >> 4) << 16) |
                                      ((uint64_t)(swizzle_stride_w >> 4) << 32) |
                                      shift_sign;
                    */

                    uint64_t desc_x = make_smem_desc(addr_x, 1, 0, swizzle_stride_x);
                    uint64_t desc_w = make_smem_desc(addr_w, 1, 0, swizzle_stride_w);

                    // NOTE (yiakwy) : ref https://github.com/NVIDIA/cutlass/blob/5c54bee12b1efc83fda51bbfb116b2f0a66b4591/include/cute/arch/mma_sm90_gmma.hpp#L14090-L14165
                    asm volatile (
                        "{\n"
                        "  wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3\n"
                        "  {\n"
                        "   %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, \n"
                        "   %8,  %9,  %10, %11, %12, %13, %14, %15,\n"
                        "   %16, %17, %18, %19, %20, %21, %22, %23,\n"
                        "   %24, %25, %26, %27, %28, %29, %30, %31,\n"
                        "   %32, %33, %34, %35, %36, %37, %38, %39,\n"
                        "   %40, %41, %42, %43, %44, %45, %46, %47,\n"
                        "   %48, %49, %50, %51, %52, %53, %54, %55,\n"
                        "   %56, %57, %58, %59, %60, %61, %62, %63\n"
                        "  },\n"
                        "  %64, %65, "
                        "  1, 1, 1;\n"
                        "}\n"
                        :
                        "+f"(reg_ptr[reg_offset + 0]),  "+f"(reg_ptr[reg_offset + 1]),  "+f"(reg_ptr[reg_offset + 2]),  "+f"(reg_ptr[reg_offset + 3]),
                        "+f"(reg_ptr[reg_offset + 4]),  "+f"(reg_ptr[reg_offset + 5]),  "+f"(reg_ptr[reg_offset + 6]),  "+f"(reg_ptr[reg_offset + 7]),
                        "+f"(reg_ptr[reg_offset + 8]),  "+f"(reg_ptr[reg_offset + 9]),  "+f"(reg_ptr[reg_offset + 10]), "+f"(reg_ptr[reg_offset + 11]),
                        "+f"(reg_ptr[reg_offset + 12]), "+f"(reg_ptr[reg_offset + 13]), "+f"(reg_ptr[reg_offset + 14]), "+f"(reg_ptr[reg_offset + 15]),
                        "+f"(reg_ptr[reg_offset + 16]), "+f"(reg_ptr[reg_offset + 17]), "+f"(reg_ptr[reg_offset + 18]), "+f"(reg_ptr[reg_offset + 19]),
                        "+f"(reg_ptr[reg_offset + 20]), "+f"(reg_ptr[reg_offset + 21]), "+f"(reg_ptr[reg_offset + 22]), "+f"(reg_ptr[reg_offset + 23]),
                        "+f"(reg_ptr[reg_offset + 24]), "+f"(reg_ptr[reg_offset + 25]), "+f"(reg_ptr[reg_offset + 26]), "+f"(reg_ptr[reg_offset + 27]),
                        "+f"(reg_ptr[reg_offset + 28]), "+f"(reg_ptr[reg_offset + 29]), "+f"(reg_ptr[reg_offset + 30]), "+f"(reg_ptr[reg_offset + 31]),
                        "+f"(reg_ptr[reg_offset + 32]), "+f"(reg_ptr[reg_offset + 33]), "+f"(reg_ptr[reg_offset + 34]), "+f"(reg_ptr[reg_offset + 35]),
                        "+f"(reg_ptr[reg_offset + 36]), "+f"(reg_ptr[reg_offset + 37]), "+f"(reg_ptr[reg_offset + 38]), "+f"(reg_ptr[reg_offset + 39]),
                        "+f"(reg_ptr[reg_offset + 40]), "+f"(reg_ptr[reg_offset + 41]), "+f"(reg_ptr[reg_offset + 42]), "+f"(reg_ptr[reg_offset + 43]),
                        "+f"(reg_ptr[reg_offset + 44]), "+f"(reg_ptr[reg_offset + 45]), "+f"(reg_ptr[reg_offset + 46]), "+f"(reg_ptr[reg_offset + 47]),
                        "+f"(reg_ptr[reg_offset + 48]), "+f"(reg_ptr[reg_offset + 49]), "+f"(reg_ptr[reg_offset + 50]), "+f"(reg_ptr[reg_offset + 51]),
                        "+f"(reg_ptr[reg_offset + 52]), "+f"(reg_ptr[reg_offset + 53]), "+f"(reg_ptr[reg_offset + 54]), "+f"(reg_ptr[reg_offset + 55]),
                        "+f"(reg_ptr[reg_offset + 56]), "+f"(reg_ptr[reg_offset + 57]), "+f"(reg_ptr[reg_offset + 58]), "+f"(reg_ptr[reg_offset + 59]),
                        "+f"(reg_ptr[reg_offset + 60]), "+f"(reg_ptr[reg_offset + 61]), "+f"(reg_ptr[reg_offset + 62]), "+f"(reg_ptr[reg_offset + 63])
                        : "l"(desc_x), "l"(desc_w)
                    );

                    // asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
                    // asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

                    // // if (threadIdx.x == 255 && blockIdx.x == 1 && blockIdx.y == 2) {
                    // if (threadIdx.x == 255 && blockIdx.x == 0 && blockIdx.y == 0) {
                    //     printf("[HopperWGMMAExecutor::mma_scaled] [wg_id#%d] [m_step#%d] [k_step#%d] reg[%d ~ %d] = \n", wg_id, m_step, k_step, reg_offset, reg_offset + reg_num_per_frag - 1);
                    //     for (int i = 0; i < 64; ++i) {
                    //         printf("[HopperWGMMAExecutor::mma_scaled] [wg_id#%d] [m_step#%d] [k_step#%d] RegAcc[%d] = %f \n", wg_id, m_step, k_step, i, reg_ptr[reg_offset + i]);
                    //     }
                    //     printf("\n");
                    // }
                    // _warpgroup_sync_256();

                } // K_STEPS

                    // asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
                    // asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

                    // // if (threadIdx.x == 255 && blockIdx.x == 1 && blockIdx.y == 2) {
                    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
                    //     printf("[HopperWGMMAExecutor::mma_scaled] [wg_id#%d] [m_step#%d] reg[%d ~ %d] = \n", wg_id, m_step, reg_offset, reg_offset + reg_num_per_frag - 1);
                    //     for (int i = 0; i < 64; ++i) {
                    //         printf("[HopperWGMMAExecutor::mma_scaled] [wg_id#%d] [m_step#%d] RegAcc[%d] = %f \n", wg_id, m_step, i, reg_ptr[reg_offset + i]);
                    //     }
                    //     printf("\n");
                    // }
                    // _warpgroup_sync_256();

            } // N_STEPS

        } // M_STEPS
    }

    static __device__ inline void commit_and_wait() {
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    }

    // NOTE (yiakwy) : for compilation safety
    // TODO (yiakwy) : the data dependancy problem (acc and "wgmma.wait_group.sync.aligned 0") has been fixed in compiler, hence the operation should be removed
    template <typename AccType>
    static __device__ inline void commit_and_wait(AccType& accum) {
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");

        float* reg_ptr = accum.get_reg_ptr();
        uint32_t reg_num_per_frag = AccType::kRegistersPerThread;
        for (uint32_t i=0; i < reg_num_per_frag; ++i) {
            asm volatile("" : "+f"(reg_ptr[i]) : : "memory");
        }

        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    }
};

} // namespace xpu

/**
 * @NOTE: This file is adapted from
 * https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_v32/topk_selector.py
 *
 * Author LEI WANG (yiakwang@ust.hk)
 *
 * Distributed Topk Radix Indexer for Decoding Acceleration in DeepSeek V3.2
 *
 * We studied how Radix should be used together with TopK from massive number (>64K):
 * 1. We reduce decoding latency (batch=1, batch=2,...,batch=64) by introducing distributed topk radix indexer,
      where multi-block cooperatively executes TopK prefix sum and communication via on-chip-netowrk (NoC on H800)
 * 2. Optimize the radix performance with Monotonic functions such as convert_to_monotonic_8bit, i.e. bin(x) >= bin(y)
 => x >= y,
 *    whereas previous top 8/11/13 bits from float's exponent and mantissa from float are not monotonic.
 * 3. This is probalistic good enough for topK selection for values ranging from (-1, 1) (e.g. DeepSeek V3.2 normalized
 attention scores), #    after first round of randix Topk
 */

// TODO (yiakwy) : using sglang headers to support ROCm data types
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <optional>
#include <stdexcept>

// The default (torch JIT) lowering path is built on ATen/libtorch. The optional
// TVM-FFI lowering path (compiled with -DUSE_TVM_FFI) drops the libtorch
// dependency entirely and exposes a raw C entry point instead.
#if !defined(USE_TVM_FFI)
// TODO (yiakwy) : Aten comes with libtorch, will be removed in favor of tvm-ffi raw C interface
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <torch/torch.h>

// TODO (yiakwy) : libC10 comes with libtorch, will be removed in favor of tvm-ffi raw C interface
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#else
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#ifndef C10_LIKELY
#define C10_LIKELY(x) (x)
#define C10_UNLIKELY(x) (x)
#endif
#endif

// enable cooperative blocks launch
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define ENABLE_HOPPER 1

#define USE_MONOLIC_RADIX 1

#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif

#ifndef MAX
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif

#ifndef CEILDIV
#define CEILDIV(x, y) (((x) + (y) - 1) / (y))
#endif

#ifndef SMs
#define SMs 132
#endif

#ifndef TOPK_HARNESSED_MAX_SPLIT
#define TOPK_HARNESSED_MAX_SPLIT 8
#endif

#if __CUDA_ARCH__ >= 900 && ENABLE_HOPPER  // Hopper or Blackwell
// NOTE(yiakwy) : Enable dshmem via NoC for faster decoding
#define ENABLE_SM90_FEATURES 1
#endif

namespace {

constexpr int TopK = 2048;

#ifndef TOPK_NUM_THREADS
#define TOPK_NUM_THREADS 512
#endif

constexpr int kThreadsPerBlock = TOPK_NUM_THREADS;
constexpr int RADIX = 256;
constexpr int WARP_SIZE = 32;

constexpr size_t MAX_BIN_CACHE = 4096 * 2;

#ifndef TOPK_MAX_CANDIDATES
#define TOPK_MAX_CANDIDATES MAX_BIN_CACHE
#endif
constexpr size_t SMEM_INPUT_SIZE = TOPK_MAX_CANDIDATES;

constexpr size_t kSmem = 2 * SMEM_INPUT_SIZE * (sizeof(float) + sizeof(unsigned int));

// BASE step radix prefix sum
template <int BASE>
__device__ __forceinline__ void radix_prefix_sum(int s_hist_buf[2][RADIX + 128], const int tx) {
  static_assert((BASE & (BASE - 1)) == 0, "BASE must be a power of 2");

  constexpr int ITERATIONS = (BASE == 2) ? 8 : (BASE == 4) ? 4 : (BASE == 16) ? 2 : 1;

  int base_pow = 1;

#pragma unroll
  for (int i = 0; i < ITERATIONS; ++i) {
    const int k = i & 1;
    if (C10_LIKELY(tx < RADIX)) {
      int value = s_hist_buf[k][tx];

#pragma unroll(BASE - 1)
      for (int j = 1; j < BASE; ++j) {
        int offset = j * base_pow;
        if (tx + offset < RADIX) {
          value += s_hist_buf[k][tx + offset];
        }
      }

      s_hist_buf[k ^ 1][tx] = value;
    }
    base_pow *= BASE;
    __syncthreads();
  }
}

template <typename Tval>
__device__ __forceinline__ void
atomicUpdateMaxIndex(int* index_ptr, const Tval* input_ptr, Tval* new_val_ptr, int* new_idx_ptr, int row_start) {
  const Tval& new_val = *new_val_ptr;
  const int& new_idx = *new_idx_ptr;

  int old_idx = *index_ptr;
  int assumed;

  Tval old_val = new_val;

  // printf("[before] [tx#%d] upding old_idx=%d to new_idx=%d....\n\n", threadIdx.x, old_idx, new_idx);

  do {
    assumed = old_idx;

    if (assumed != -1) {
      old_val = input_ptr[assumed + row_start];
      if (new_val <= old_val) {
        // change nothing
        return;
      }
    }

    old_idx = atomicCAS(index_ptr, assumed, new_idx);
  } while (assumed != old_idx);

  // printf("[after] [tx#%d] updated index from old_idx=%d to new_idx=%d\n\n", threadIdx.x, old_idx, new_idx);

  *new_idx_ptr = old_idx;
  *new_val_ptr = input_ptr[old_idx + row_start];
  
}

struct FastTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B]
  int32_t* __restrict__ indices;           // [B, TopK]
  int32_t* __restrict__ lengths;           // [B]
  int64_t input_stride;
};

// when length <= topk, we can directly write the indices
__device__ void naive_topk_cuda(const float* __restrict__ score, int32_t* __restrict__ indice, int32_t length, int topk) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < MIN(topk, length); i += kThreadsPerBlock) {
    // indice[i] = (i < length) ? i : -1;
    indice[i] = i;
  }
  if (topk >= length) {
    for (int i = tid + length; i < topk; i += kThreadsPerBlock) {
      indice[i] = -1;
    }
  }
}

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform(
    const float* __restrict__ score,
    int32_t length,
    int32_t* __restrict__ dst_page_table,
    const int32_t* __restrict__ src_page_table) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    dst_page_table[i] = (i < length) ? src_page_table[i] : -1;
  }
}

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform_ragged(
    const float* __restrict__ score, int32_t length, int32_t* __restrict__ topk_indices_ragged, int32_t offset) {
  const auto tid = threadIdx.x;
  for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
    topk_indices_ragged[i] = (i < length) ? static_cast<int32_t>(i) + offset : -1;
  }
}

__device__ __forceinline__ auto convert_to_monotonic_8bit(float x) -> uint8_t;

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
#if USE_MONOLIC_RADIX
  return convert_to_monotonic_8bit(x);
#else
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
#endif
}

// NOTE (yiakwy) : residual-domain key. 
__device__ __forceinline__ auto convert_to_monotonic_8bit(float x) -> uint8_t {
  int bin = __float2int_rd(x);
  return bin;
}

// Single-CTA short-sequence small-K specialization (top512/1024, L <= 16384).
#include "top_small_k.cuh"

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ int pad_index(int i) {
  return i + (i >> 5);
}

__device__ char* manual_itoa(int num, char* str) {
  if (num == 0) {
    *str++ = '0';
    return str;
  }
  char temp[12];
  int i = 0;
  while (num > 0) {
    temp[i++] = (num % 10) + '0';
    num /= 10;
  }
  while (i > 0) {
    *str++ = temp[--i];
  }
  return str;
}

__device__ __forceinline__ void parallel_reduce_histogram(int* s_histogram /*src and dest*/, int* g_scratch/*dest*/, const int& tx, const bool& is_split_mode, const int& num_splits, const int& split_idx, const int& BLOCK_SIZE) {
#if __CUDA_ARCH__ >= 900 && ENABLE_HOPPER
  auto cluster = cooperative_groups::this_cluster();
  // NOTE (yiakwy) : enable dshmem for faster decoding
  if (is_split_mode && num_splits > 1) {
    cluster.sync();

    if (split_idx == 0) {
      for (int bin = tx; bin < RADIX; bin += BLOCK_SIZE) {
        int total = s_histogram[bin];

#pragma unroll
        for (int r = 1; r < num_splits; ++r) {  // accumulate histogram from other blocks
          int* other_hist = cluster.map_shared_rank<int>(&s_histogram[0], r);
          total += other_hist[bin];
        }

        s_histogram[bin] = total;
      }
    }
    cluster.sync();

    if (split_idx != 0) {
      // broadcast
      for (int bin = tx; bin < RADIX; bin += BLOCK_SIZE) {
        int* dist_hist = cluster.map_shared_rank<int>(&s_histogram[0], 0);
        s_histogram[bin] = dist_hist[bin];
      }
    }
    cluster.sync();
  }
#else
  // NOTE(yiakwy) : rollback to the old way via L2 cache
  if (is_split_mode && num_splits > 1) {
    for (int bin = tx; bin < RADIX; bin += BLOCK_SIZE) {
      ::atomicAdd(&g_scratch[bin], s_histogram[bin]);
    }

    cooperative_groups::this_grid().sync();

    if (tx < RADIX) {
      // broadcast
      s_histogram[tx] = g_scratch[tx];
    }
    __syncthreads();
  }
#endif  // end of stage 2  
}

// TODO (yiakwy) : valid optimizatioin; applied to Top2048 pathway

// Harnessed variant: remove cluster barrier
__device__ __forceinline__ void parallel_reduce_histogram_harnessed(int* s_histogram /*src and dest*/, int* g_scratch/*dest*/, const int& tx, const bool& is_split_mode, const int& num_splits, const int& split_idx, const int& BLOCK_SIZE) {
#if __CUDA_ARCH__ >= 900 && ENABLE_HOPPER
  auto cluster = cooperative_groups::this_cluster();
  if (is_split_mode && num_splits > 1) {
    cluster.sync();

    if (split_idx == 0) {
      for (int bin = tx; bin < RADIX; bin += BLOCK_SIZE) {
        int total = s_histogram[bin];
#pragma unroll
        for (int r = 1; r < num_splits; ++r) {
          int* other_hist = cluster.map_shared_rank<int>(&s_histogram[0], r);
          total += other_hist[bin];
        }
        s_histogram[bin] = total;
      }
    }
    cluster.sync();

    if (split_idx != 0) {
      for (int bin = tx; bin < RADIX; bin += BLOCK_SIZE) {
        int* dist_hist = cluster.map_shared_rank<int>(&s_histogram[0], 0);
        s_histogram[bin] = dist_hist[bin];
      }
    }
    __syncthreads();
  }
#else
  parallel_reduce_histogram(s_histogram, g_scratch, tx, is_split_mode, num_splits, split_idx, BLOCK_SIZE);
#endif
}

__device__ __forceinline__ void local_calc_block_offset(int *s_block_count_ptr/*dest*/, int *s_block_offset_ptr/*dest*/, int* g_scratch/*dest*/, const int& tx, const int& lane_id, const int& split_idx, const int& num_splits, const int& threshold_bin, const uint8_t* bin_cache, const float* input, const int& start_offset, const int& end_offset, const int& row_start, const int& BLOCK_SIZE) {
  int local_count = 0;
  for (int idx = start_offset + tx; idx < end_offset; idx += BLOCK_SIZE) {
    // int bin = convert_to_uint8(input[idx + row_start]);
    const auto& _idx = idx - start_offset;
    int bin;

    if (_idx < MAX_BIN_CACHE) {
      bin = bin_cache[_idx];
    } else {
      bin = convert_to_uint8(input[idx + row_start]);
    }

    if (bin > threshold_bin) {
      local_count++;
    }
  }
  __syncwarp();

  // per warp reduce
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    local_count += __shfl_down_sync(0xffffffff, local_count, offset);
  }
  __syncthreads();

  // per block reduce
  if (lane_id == 0) {
    atomicAdd(s_block_count_ptr, local_count);
  }
  __syncthreads();

#if __CUDA_ARCH__ >= 900 && ENABLE_HOPPER
 auto cluster = cooperative_groups::this_cluster();

 cluster.sync();

 if (tx == 0) {
   int offset = 0;
   for (int r = 0; r < split_idx; ++r) {
     int* dst_cnt = cluster.map_shared_rank(s_block_count_ptr, r);
     offset += *dst_cnt;
   }
   *s_block_offset_ptr = offset;
 }

 __syncthreads();

#else
 // TODO (yiakwy) : fallback
 if (tx == 0) {
   g_scratch[blockIdx.x + split_idx * blockDim.x] = *s_block_count_ptr;  // write block_count to g_scratch for each block
 }

 cooperative_groups::this_grid().sync();

 if (tx == 0) {
   int offset = 0;
   for (int i = 0; i < split_idx; ++i) {
     offset += g_scratch[blockIdx.x + i * blockDim.x];
   }

   *s_block_offset_ptr = offset;  // write block_offset to shared memory for each block
 }
 __syncthreads();
#endif

}



// NOTE (yiakwy) : add "bin > threshold_bin" short path, to avoid re-computing
__device__ __forceinline__ void local_calc_block_offset_hist(int *s_block_count_ptr/*dest*/, int *s_block_offset_ptr/*dest*/, int* g_scratch/*dest*/, const int& tx, const int& lane_id, const int& split_idx, const int& num_splits, const int& threshold_bin, const int* s_local_hist, const int& BLOCK_SIZE) {
  int local_count = 0;
  for (int bin = tx; bin < RADIX; bin += BLOCK_SIZE) {
    if (bin > threshold_bin) local_count += s_local_hist[bin];
  }
  __syncwarp();
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    local_count += __shfl_down_sync(0xffffffff, local_count, offset);
  }
  __syncthreads();
  if (lane_id == 0) {
    atomicAdd(s_block_count_ptr, local_count);
  }
  __syncthreads();
#if __CUDA_ARCH__ >= 900 && ENABLE_HOPPER
  auto cluster = cooperative_groups::this_cluster();
  cluster.sync();
  if (tx == 0) {
    int offset = 0;
    for (int r = 0; r < split_idx; ++r) {
      int* dst_cnt = cluster.map_shared_rank(s_block_count_ptr, r);
      offset += *dst_cnt;
    }
    *s_block_offset_ptr = offset;
  }
  __syncthreads();
#else
  if (tx == 0) { g_scratch[blockIdx.x + split_idx * blockDim.x] = *s_block_count_ptr; }
  cooperative_groups::this_grid().sync();
  if (tx == 0) {
    int offset = 0;
    for (int i = 0; i < split_idx; ++i) { offset += g_scratch[blockIdx.x + i * blockDim.x]; }
    *s_block_offset_ptr = offset;
  }
  __syncthreads();
#endif
}

__device__ __forceinline__ void local_calc_block_offset_with_s_input(int *s_block_count_ptr/*dest*/, int *s_block_offset_ptr/*dest*/, int* g_scratch/*dest*/, const int& tx, const int& lane_id, const int& split_idx, const int& num_splits, const int& threshold_bin, const uint8_t* bin_cache, const float* s_input, const int& s_num_input, const int& BLOCK_SIZE) {
  int local_count = 0;
  for (unsigned int idx = tx; idx < s_num_input; idx += BLOCK_SIZE) {
    int bin;

    if (idx < MAX_BIN_CACHE) {
      bin = bin_cache[idx];
    } else {
      bin = convert_to_uint8(s_input[idx]);
    }

    if (bin > threshold_bin) {
      local_count++;
    }
  }
  __syncwarp();

  // per warp reduce
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    local_count += __shfl_down_sync(0xffffffff, local_count, offset);
  }
  __syncthreads();

  // per block reduce
  if (lane_id == 0) {
    atomicAdd(s_block_count_ptr, local_count);
  }
  __syncthreads();

#if __CUDA_ARCH__ >= 900 && ENABLE_HOPPER
 auto cluster = cooperative_groups::this_cluster();

 cluster.sync();

 if (tx == 0) {
   int offset = 0;
   for (int r = 0; r < split_idx; ++r) {
     int* dst_cnt = cluster.map_shared_rank(s_block_count_ptr, r);
     offset += *dst_cnt;
   }
   *s_block_offset_ptr = offset;
 }

 __syncthreads();

#else
 // TODO (yiakwy) : fallback
 if (tx == 0) {
   g_scratch[blockIdx.x + split_idx * blockDim.x] = *s_block_count_ptr;  // write block_count to g_scratch for each block
 }

 cooperative_groups::this_grid().sync();

 if (tx == 0) {
   int offset = 0;
   for (int i = 0; i < split_idx; ++i) {
     offset += g_scratch[blockIdx.x + i * blockDim.x];
   }

   *s_block_offset_ptr = offset;  // write block_offset to shared memory for each block
 }
 __syncthreads();
#endif

}

__device__ __forceinline__ void calc_global_remainder(int* s_last_block_write_offset_ptr/*dest*/, int* s_block_offset_ptr/*src & dest*/, int* s_write_ptr_p/*src & dest*/, int* g_scratch/*dest*/, const int& tx, const int& split_idx, const int& num_splits) {
#if __CUDA_ARCH__ >= 900 && ENABLE_HOPPER
  auto cluster = cooperative_groups::this_cluster();
  if (split_idx != num_splits - 1) {
    if (tx == 0) {
      int* dst_write_ptr = cluster.map_shared_rank(s_write_ptr_p, num_splits - 1);
      int* dst_block_offset = cluster.map_shared_rank(s_block_offset_ptr, num_splits - 1);

      *s_last_block_write_offset_ptr = *dst_block_offset + *dst_write_ptr;
    }
  } else {
    if (tx == 0) {
      *s_last_block_write_offset_ptr = *s_block_offset_ptr + *s_write_ptr_p;
    }
  }

  cluster.sync();
#else
  auto grid = cooperative_groups::this_grid();
  if (tx == 0) {
    g_scratch[blockIdx.x + split_idx * blockDim.x] = *s_block_offset_ptr + *s_write_ptr_p;
  }

  grid.sync();

  if (tx == 0) {
    *s_last_block_write_offset_ptr = g_scratch[blockIdx.x + (num_splits - 1) * blockDim.x];
  }

  grid.sync();
#endif
}

template <int TopK>
__device__ void fast_topk_split_kv_cuda_tl(
    const float* __restrict__ input,
    int* __restrict__ index,
    int row_start,
    int length,
    int topk = TopK,
    int* g_scratch = nullptr,
    bool is_split_mode = false) {
  // using Tval = half;
  using Tval = float;

  // We assume length > TopK here, or it will crash
  constexpr auto BLOCK_SIZE = kThreadsPerBlock;

  alignas(128) __shared__ uint8_t bin_cache[MAX_BIN_CACHE];

  // Sub-bin digits for the threshold-bin candidates.
  alignas(128) __shared__ uint8_t s_sub_bin[2][SMEM_INPUT_SIZE];

  extern __shared__ int shared_mem[][SMEM_INPUT_SIZE];

  Tval (*s_input)[SMEM_INPUT_SIZE] = (Tval(*)[SMEM_INPUT_SIZE]) &shared_mem[0][0];
  unsigned int (*s_input_idx)[SMEM_INPUT_SIZE] = (unsigned int(*)[SMEM_INPUT_SIZE]) (&shared_mem[0][0] + SMEM_INPUT_SIZE);

  Tval (*s_input_b)[SMEM_INPUT_SIZE] = (Tval(*)[SMEM_INPUT_SIZE]) (&shared_mem[0][0] + 2 * SMEM_INPUT_SIZE);
  unsigned int (*s_input_idx_b)[SMEM_INPUT_SIZE] = (unsigned int(*)[SMEM_INPUT_SIZE]) (&shared_mem[0][0] + 3 * SMEM_INPUT_SIZE);

  // double buffer
  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];

  // block-level radix threshold
  alignas(128) __shared__ int s_threshold_bin_id;

  // block-level elements drop in s_threshold_bin_id bin
  alignas(128) __shared__ unsigned int s_num_input[2];

  // block-level counters
  alignas(128) __shared__ int s_block_count;

  // block-level global TopK writing offsets
  alignas(128) __shared__ int s_block_offset;

  // block-level writing index
  alignas(128) __shared__ int s_write_ptr;

  alignas(128) __shared__ int s_last_block_write_offset;

  auto& s_histogram = s_histogram_buf[0];

  const int tx = threadIdx.x;

  const int lane_id = threadIdx.x % WARP_SIZE;
  const int warp_id = threadIdx.x / WARP_SIZE;

  int split_idx = -1;
  int num_splits = 1;

  // stage 0 : cross blocks histogram accumulation preparation
#if __CUDA_ARCH__ >= 900 && ENABLE_HOPPER
  auto cluster = cooperative_groups::this_cluster();

  if (is_split_mode) {
    split_idx = cluster.block_rank();
    num_splits = cluster.num_blocks();
  }
#else
  if (is_split_mode) {
    split_idx = blockIdx.y;
    num_splits = gridDim.y;
  }
#endif

  // stage 1: local 8bit coarse histogram
  const int chunk = (length + num_splits - 1) / num_splits;
  const int start_offset = split_idx * chunk;
  const int end_offset = MIN(start_offset + chunk, length);

  // if (tx == 0 && blockIdx.x == 0 && (blockIdx.y >= 0)) {
  //   printf("[Blk#%d] [Cooperative Blk#%d] start_offset=%d, end_offset=%d, length=%d, chunk_size=%d, split_idx=%d, num_splits=%d\n", blockIdx.x, blockIdx.y, start_offset, end_offset, length, chunk, split_idx, num_splits);
  // }
  // __syncthreads();

  if (tx < RADIX + 1) s_histogram[tx] = 0;
  if (tx == 0) {
    s_block_count = 0;
    s_write_ptr = 0;

    s_last_block_write_offset = 0;
    s_threshold_bin_id = 0;
  }
  __syncthreads();

  for (unsigned int idx = start_offset + tx; idx < end_offset; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx + row_start]);
    const auto& _idx = idx - start_offset;
    if (_idx < MAX_BIN_CACHE) {
      bin_cache[_idx] = bin;
    }
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  // stage 2 : aggregate radix histogram across blocks with NoC (requires compute arch >= 90) or L2 cache
  parallel_reduce_histogram(s_histogram/*src and dest*/, g_scratch/*dest*/, tx, is_split_mode, num_splits, split_idx, BLOCK_SIZE);

  // stage 3 : global prefix sum to cover the most likely topK (upper bound) in each block
  const auto run_cumsum = [&] { radix_prefix_sum<2>(s_histogram_buf, tx); };
  run_cumsum();

  if (tx < RADIX) {
    if (s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      ::atomicExch(&s_threshold_bin_id, tx);
    }
  }
  __syncthreads();

  const int threshold_bin = s_threshold_bin_id;
  int global_remainder = topk - s_histogram[threshold_bin + 1];

  // if (blockIdx.x == 0 && blockIdx.y == 0 && tx == 0) {
  //   printf("s_histogram[%d]=%d\n", threshold_bin - 1, s_histogram[threshold_bin - 1]);
  //   printf("s_histogram[%d]=%d\n", threshold_bin, s_histogram[threshold_bin]);
  //   printf("s_histogram[%d]=%d\n", threshold_bin + 1, s_histogram[threshold_bin + 1]);
  // }
  // __syncthreads();

  // stage 4: global offset calculation for each block
  local_calc_block_offset(&s_block_count, &s_block_offset, g_scratch, tx, lane_id, split_idx, num_splits, threshold_bin, bin_cache, input, start_offset, end_offset, row_start, BLOCK_SIZE);

  const int local_remainder = topk - s_block_count;

  // stage 5: write most likely topk indices onto g_mem and narrow down search elements
  if (global_remainder == 0) {

    for (unsigned int idx = start_offset + tx; idx < end_offset && s_write_ptr < topk; idx += BLOCK_SIZE) {
      // int bin = convert_to_uint8(input[idx + row_start]);
  
      const auto& _idx = idx - start_offset;
      int bin;
  
      if (_idx < MAX_BIN_CACHE) {
        bin = bin_cache[_idx];
      } else {
        bin = convert_to_uint8(input[idx + row_start]);
      }
  
      if (bin > threshold_bin) {
        int local_pos = atomicAdd(&s_write_ptr, 1);
        int global_pos = s_block_offset + local_pos;
  
        if (global_pos < topk) {
          index[global_pos] = idx;
        }
      }
    }

  } else {
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    if (tx == 0) {
      s_num_input[0] = 0;
    }
    __syncthreads();

    for (unsigned int idx = start_offset + tx; idx < end_offset && s_write_ptr < topk; idx += BLOCK_SIZE) {
      // int bin = convert_to_uint8(input[idx + row_start]);
  
      const auto& _idx = idx - start_offset;
      int bin;
      
      if (_idx < MAX_BIN_CACHE) {
        bin = bin_cache[_idx];

        bin_cache[_idx] = 0;

        if (bin > threshold_bin) {
          int local_pos = atomicAdd(&s_write_ptr, 1);
          int global_pos = s_block_offset + local_pos;
          
          if (global_pos < topk) {
            index[global_pos] = idx;
          }
        } else if (bin == threshold_bin) {
          Tval val = input[idx + row_start];
          Tval val_scale = (val - threshold_bin) * RADIX;
          const auto sub_bin = convert_to_uint8(val_scale);

          const unsigned int pos = ::atomicAdd(&s_num_input[0], 1);
          if (pos < SMEM_INPUT_SIZE) {
            s_input[0][pos] = val_scale;
            s_input_idx[0][pos] = idx;
            s_sub_bin[0][pos] = sub_bin;

            ::atomicAdd(&s_histogram[sub_bin], 1);
          }
        }
      } else {
        Tval val = input[idx + row_start];
        bin = convert_to_uint8(val);

        if (bin > threshold_bin) {
          int local_pos = atomicAdd(&s_write_ptr, 1);
          int global_pos = s_block_offset + local_pos;
    
          if (global_pos < topk) {
            index[global_pos] = idx;
          }
        } else if (bin == threshold_bin){
          Tval val_scale = (val - threshold_bin) * RADIX;
          const auto sub_bin = convert_to_uint8(val_scale);

          const unsigned int pos = ::atomicAdd(&s_num_input[0], 1);
          if (pos < SMEM_INPUT_SIZE) {
            s_input[0][pos] = val_scale;
            s_input_idx[0][pos] = idx;
            s_sub_bin[0][pos] = sub_bin;

            ::atomicAdd(&s_histogram[sub_bin], 1);
          }
        }
      }
    }    
    __syncthreads();

    // if (tx == 0) {
    //   printf("[before] [Blk#%d] count=%d; s_input_idx[0]=%d, s_input_idx[1]=%d\n\n", blockIdx.y, s_num_input[0], s_input_idx[0][0], s_input_idx[0][1]);
    // }
    // if (tx == 0 && blockIdx.y == 2) {
    //   for (int i=0; i < 28; i++) {
    //     if (s_input_idx[0][i] == 22134) {
    //       printf("[blk#%d] find it 22134 , pos=%d\n\n", blockIdx.y, i);
    //     }
    //   }
    // }
    // __syncthreads();

    int round = 0;
    int sub_cur = 0;
    index += topk - global_remainder;

    do {
      __syncthreads();

      const int scan_size = s_num_input[0];

      const int sub_next = sub_cur ^ 1;
      Tval (*s_in_cur)[SMEM_INPUT_SIZE] = (sub_cur == 0) ? s_input : s_input_b;
      unsigned int (*s_idx_cur)[SMEM_INPUT_SIZE] = (sub_cur == 0) ? s_input_idx : s_input_idx_b;

      Tval (*s_in_nxt)[SMEM_INPUT_SIZE] = (sub_cur == 0) ? s_input_b : s_input;
      unsigned int (*s_idx_nxt)[SMEM_INPUT_SIZE] = (sub_cur == 0) ? s_input_idx_b : s_input_idx;

      // stage 6 : repeat fine scale radix sort upon narrowed down elements in the threshold bin
      parallel_reduce_histogram(s_histogram, g_scratch, tx, is_split_mode, num_splits, split_idx, BLOCK_SIZE);
      run_cumsum();

      if (tx < RADIX) {
        if (s_histogram[tx] > global_remainder && s_histogram[tx + 1] <= global_remainder) {
          ::atomicExch(&s_threshold_bin_id, tx);
        }
      }
      __syncthreads();

      auto next_threshold_bin = s_threshold_bin_id;
      auto next_global_remainder = global_remainder - s_histogram[next_threshold_bin + 1];

      if (tx == 0) {
        s_block_count = 0;
        s_write_ptr = 0;
    
        s_last_block_write_offset = 0;
      }
      __syncthreads();

      local_calc_block_offset_with_s_input(&s_block_count, &s_block_offset, g_scratch, tx, lane_id, split_idx, num_splits, next_threshold_bin, s_sub_bin[sub_cur], s_in_cur[0], s_num_input[0], BLOCK_SIZE);
      __syncthreads();

      if (next_global_remainder == 0) {
        for (unsigned int idx = tx; idx < scan_size && s_write_ptr < global_remainder; idx += BLOCK_SIZE) {
          int bin = s_sub_bin[sub_cur][idx];

          if (bin > next_threshold_bin) {
            int local_pos = atomicAdd(&s_write_ptr, 1);
            int global_pos = s_block_offset + local_pos;
  
            if (global_pos < global_remainder) {
              index[global_pos] = s_idx_cur[0][idx];
            }
          }
        }
      } else {
        if (tx < RADIX + 1) {
          s_histogram[tx] = 0;
        }     
        if (tx == 0) {
          s_num_input[0] = 0;
        }
        __syncthreads();

        for (unsigned int idx = tx; idx < scan_size && s_write_ptr < global_remainder; idx += BLOCK_SIZE) {
          int bin = s_sub_bin[sub_cur][idx];

          s_sub_bin[sub_cur][idx] = 0;

          if (bin > next_threshold_bin) {
            int local_pos = atomicAdd(&s_write_ptr, 1);
            int global_pos = s_block_offset + local_pos;

            if (global_pos < global_remainder) {
              index[global_pos] = s_idx_cur[0][idx];
            }
          } else if (bin == next_threshold_bin) {
            Tval val = s_in_cur[0][idx];
            Tval val_scale = (val - next_threshold_bin) * RADIX;
            const auto sub_bin = convert_to_uint8(val_scale);

            const unsigned int pos = ::atomicAdd(&s_num_input[0], 1);
            if (pos < SMEM_INPUT_SIZE) {
              s_in_nxt[0][pos] = val_scale;
              s_idx_nxt[0][pos] = s_idx_cur[0][idx];
              s_sub_bin[sub_next][pos] = sub_bin;

              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }

      index += global_remainder - next_global_remainder;
      global_remainder = next_global_remainder;
      sub_cur ^= 1;
      __syncthreads();

#if __CUDA_ARCH__ >= 900 && ENABLE_HOPPER
      auto cluster = cooperative_groups::this_cluster();
      cluster.sync();
#else
      auto grid = cooperative_groups::this_grid();
      grid.sync();
#endif  

    } while (++round < 4 && global_remainder > num_splits); // end of do-while loop for global radix refinement

    // The last round wrote into the buffer selected by `sub_cur`.
    Tval (*s_in_fin)[SMEM_INPUT_SIZE] = (sub_cur == 0) ? s_input : s_input_b;
    unsigned int (*s_idx_fin)[SMEM_INPUT_SIZE] = (sub_cur == 0) ? s_input_idx : s_input_idx_b;

    if (tx == 0) {
      s_write_ptr = 0;
    }
    __syncthreads();

    if (global_remainder != 0) {
#if __CUDA_ARCH__ >= 900 && ENABLE_HOPPER

      auto cluster = cooperative_groups::this_cluster();
      cluster.sync();

      if (split_idx == 0) {
        if (tx == 0) {
          unsigned int count = s_num_input[0];
          for (int r=1; r < num_splits; r++) {
            unsigned int* dst_input_num_ptr = cluster.map_shared_rank(&s_num_input[0], r);
            int dst_input_num = *dst_input_num_ptr;


            if (dst_input_num > 0) {
              float* dst_input = cluster.map_shared_rank(&s_in_fin[0][0], r);
              unsigned int* dst_input_idx = cluster.map_shared_rank(&s_idx_fin[0][0], r);

              for (unsigned int i = 0; i < dst_input_num && count < SMEM_INPUT_SIZE; i++) {
                s_in_fin[0][count] = dst_input[i];
                s_idx_fin[0][count] = dst_input_idx[i];
                count++;
              }
            }
          }

          s_num_input[0] = count;
        }
        __syncthreads();

        Tval (*s_val)[SMEM_INPUT_SIZE] = (sub_cur == 0) ? s_input_b : s_input;
        for (unsigned int p = tx; p < s_num_input[0]; p += BLOCK_SIZE) {
          s_val[0][p] = static_cast<Tval>(input[s_idx_fin[0][p] + row_start]);
        }
        __syncthreads();

        if (tx < s_num_input[0]) {
          Tval val = s_val[0][tx];
          Tval cur_val = val;
          int cur_idx = tx;
          for (int i = 0; i < global_remainder; i++) {
            atomicUpdateMaxIndex(index + i, &s_val[0][0], &cur_val, &cur_idx, 0);

            if (cur_val == -1) { break; }
            if (cur_val != val) {
              val = cur_val;
            }
          }
        }
        __syncthreads();

        if (tx < global_remainder) {
          index[tx] = static_cast<int>(s_idx_fin[0][index[tx]]);
        }
        __syncthreads();
      } // end of split_idx == 0
      
      cluster.sync();
#else
      auto grid = cooperative_groups::this_grid();
      grid.sync();
      // NOTE (yiakwy) : we will support it soon, but definitely sycn via L2 cache will increase latency
#endif      
    }
  } // end of global_remainder > 0 case
}

#include "topk_harnessed.cuh"

template <int TopK>
__global__ __launch_bounds__(kThreadsPerBlock)  // topk
    void topk_kernel(const FastTopKParams params, int* g_scratch, bool use_split_kv, bool skip_short_len) {
  const auto& [input, row_starts, indices, lengths, input_stride] = params;

  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto indice = indices + bid * TopK;
  const auto score = input + bid * input_stride;

  // The single-CTA short path already handled this row (ragged dispatch).
  if (skip_short_len && static_cast<int>(length) <= SMALL_K_MAX_L) {
    return;
  }
  if (length <= TopK) {
    return naive_topk_cuda(score, indice, length, TopK);
  } else {
    return fast_topk_split_kv_cuda_tl<TopK>(score, indice, row_start, length, TopK, g_scratch, use_split_kv);
  }
}


__global__ __launch_bounds__(kThreadsPerBlock)  // decode
    void topk_transform_decode_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ dst_page_table,
        const int32_t* __restrict__ src_page_table,
        const int64_t src_stride) {
  const auto& [input, _1, _2, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = 0;
  const auto length = lengths[bid];
  const auto src_page_entry = src_page_table + bid * src_stride;
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;
  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[TopK];

    fast_topk_split_kv_cuda_tl<TopK>(score, s_indices, row_start, length);

    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);

    // static_assert(TopK / kThreadsPerBlock == 2);

    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = src_page_entry[pos_1];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // prefill
    void topk_transform_prefill_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ dst_page_table,
        const int32_t* __restrict__ src_page_table,
        const int64_t src_stride,
        const int32_t* __restrict__ cu_seqlens_q,
        const int64_t prefill_bs) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto length = lengths[bid];
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * input_stride;

  /// NOTE: prefill bs is usually small, we can just use a simple loop here
  /// We ensure that last cu_seqlens is equal to number of blocks launched
  __shared__ const int32_t* s_src_page_entry;
  if (C10_LIKELY(prefill_bs <= kThreadsPerBlock)) {
    if (tid < prefill_bs) {
      if (bid >= cu_seqlens_q[tid] && bid < cu_seqlens_q[tid + 1]) {
        s_src_page_entry = src_page_table + tid * src_stride;
      }
    }
  } else {
    for (int64_t i = tid; i < prefill_bs; i += kThreadsPerBlock) {
      if (bid >= cu_seqlens_q[i] && bid < cu_seqlens_q[i + 1]) {
        s_src_page_entry = src_page_table + i * src_stride;
      }
    }
  }
  __syncthreads();
  const auto src_page_entry = s_src_page_entry;

  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[TopK];

    fast_topk_split_kv_cuda_tl<TopK>(score, s_indices, row_start, length);

    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);

    // static_assert(TopK / kThreadsPerBlock == 2);

    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = src_page_entry[pos_1];
  }
}

__global__ __launch_bounds__(kThreadsPerBlock)  // prefill, ragged kv
    void topk_transform_prefill_ragged_kernel(
        const FastTopKParams params,
        int32_t* __restrict__ topk_indices_ragged,
        const int32_t* __restrict__ topk_indices_offset) {
  const auto& [input, row_starts, _, lengths, input_stride] = params;
  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto tid = threadIdx.x;
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto dst_indices_entry = topk_indices_ragged + bid * TopK;
  const auto score = input + bid * input_stride;
  const auto offset = topk_indices_offset[bid];

  if (length <= TopK) {
    return naive_topk_transform_ragged(score, length, dst_indices_entry, offset);
  } else {
    __shared__ int s_indices[TopK];

    fast_topk_split_kv_cuda_tl<TopK>(score, s_indices, row_start, length);

    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);

    // static_assert(TopK / kThreadsPerBlock == 2);

    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_indices_entry[idx_0] = pos_0 + offset;
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_indices_entry[idx_1] = pos_1 + offset;
  }
}

#if !defined(USE_TVM_FFI)

auto get_params(
    const at::Tensor& score,
    const at::Tensor& lengths,
    std::optional<at::Tensor> row_starts_opt = std::nullopt,
    std::optional<at::Tensor> indices_opt = std::nullopt) -> FastTopKParams {
  const auto B = score.size(0);
  TORCH_CHECK(score.dim() == 2 && score.stride(1) == 1);
  if (row_starts_opt.has_value()) {
    const auto& row_starts = row_starts_opt.value();
    TORCH_CHECK(row_starts.dim() == 1);
    TORCH_CHECK(row_starts.size(0) == B);
  }
  TORCH_CHECK(lengths.dim() == 1 && lengths.is_contiguous());
  TORCH_CHECK(lengths.size(0) == B);
  int32_t* indices_data_ptr = nullptr;
  if (indices_opt.has_value()) {
    const auto& indices = indices_opt.value();
    TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous());
    TORCH_CHECK(indices.size(0) == B);
    TORCH_CHECK(
        indices.size(1) == 512 || indices.size(1) == 1024 || indices.size(1) == 2048,
        "fast_topk currently supports topk in {512, 1024, 2048}, got ",
        indices.size(1));
    indices_data_ptr = indices.data_ptr<int32_t>();
  }

  return FastTopKParams{
      .input = score.data_ptr<float>(),
      .row_starts = row_starts_opt.has_value() ? row_starts_opt->data_ptr<int32_t>() : nullptr,
      .indices = indices_data_ptr,
      .lengths = lengths.data_ptr<int32_t>(),
      .input_stride = score.stride(0),
  };
}
#endif  // !USE_TVM_FFI

template <auto* f, size_t max_dynamic_smem>
void setup_kernel_smem_once() {
  [[maybe_unused]]
  static const auto result = [] {
#ifdef USE_ROCM
    // hipify will turn cudaFuncSetAttribute -> hipFuncSetAttribute. On ROCm,
    // hipFuncSetAttribute expects `const void*` and hipcc does not accept passing
    // a function pointer directly, so cast explicitly.
    return ::cudaFuncSetAttribute(
        reinterpret_cast<const void*>(f), ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#else
    // CUDA: keep original behavior (no cast needed).
    return ::cudaFuncSetAttribute(f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#endif
  }();
  if (result != cudaSuccess) {
#if !defined(USE_TVM_FFI)
    TORCH_CHECK(false, "set_up_kernel_once failed:", ::cudaGetErrorString(result));
#else
    throw std::runtime_error(
        std::string("set_up_kernel_once failed: ") + ::cudaGetErrorString(result));
#endif
  }
}

}  // namespace

#if !defined(USE_TVM_FFI)
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif  // !USE_TVM_FFI

template <int TopK>
void launch_topk_kernel(
    const FastTopKParams& params,
    dim3 grid,
    dim3 block,
    unsigned int split_kv,
    int* scratch_ptr,
    cudaStream_t stream,
    bool skip_short_len = false) {
  setup_kernel_smem_once<&topk_kernel<TopK>, kSmem>();

  cudaFuncSetAttribute(&topk_kernel<TopK>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
  // NOTE (yiakwy) : opt into clusters larger than the portable 8-CTA cap (Hopper max 16) 
  if (split_kv > 8) {
    cudaFuncSetAttribute(&topk_kernel<TopK>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }

#if ENABLE_HOPPER
  cudaLaunchConfig_t config = {0};

  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = kSmem;
  config.stream = stream;

  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim = {1, split_kv, 1};
  config.attrs = attr;
  config.numAttrs = 1;

  cudaLaunchKernelEx(&config, topk_kernel<TopK>, params, nullptr, split_kv > 1, skip_short_len);
#else
  bool use_split_kv = split_kv > 1;
  void* kernelArgs[] = {(void*)&params, (void*)&scratch_ptr, (void*)&use_split_kv, (void*)&skip_short_len};

  cudaLaunchCooperativeKernel((void*)&topk_kernel<TopK>, grid, block, kernelArgs, kSmem, stream);
#endif
}

// Single-CTA launcher for the short-sequence harnessed kernel.
template <int TopK>
void launch_topk_kernel_harnessed(
    const FastTopKParams& params,
    dim3 grid,
    dim3 block,
    unsigned int split_kv,
    int* scratch_ptr,
    cudaStream_t stream) {
  setup_kernel_smem_once<&topk_kernel_harnessed<TopK>, kSmem>();

  cudaFuncSetAttribute(&topk_kernel_harnessed<TopK>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
  // Allow clusters larger than the portable 8-CTA cap (Hopper max 16).
  cudaFuncSetAttribute(&topk_kernel_harnessed<TopK>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);

#if ENABLE_HOPPER
  cudaLaunchConfig_t config = {0};

  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = kSmem;
  config.stream = stream;

  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim = {1, split_kv, 1};
  config.attrs = attr;
  config.numAttrs = 1;

  cudaLaunchKernelEx(&config, topk_kernel_harnessed<TopK>, params, nullptr, split_kv > 1);
#else
  bool use_split_kv = split_kv > 1;
  void* kernelArgs[] = {(void*)&params, (void*)&scratch_ptr, (void*)&use_split_kv};

  cudaLaunchCooperativeKernel((void*)&topk_kernel_harnessed<TopK>, grid, block, kernelArgs, kSmem, stream);
#endif
}

// 16-CTA has better perforamnce in Small-K long-L workload.
static inline unsigned int smallk_widen_split_kv(int topk, int B, unsigned int split_kv) {
#if ENABLE_HOPPER
  if ((topk == 512 || topk == 1024) && B <= 4) {
    split_kv = MIN(CEILDIV(SMs, B), 16);
  }
#endif
  return split_kv;
}

#ifdef USE_TVM_FFI

void tvm_jit_fast_topk_launch(
    tvm::ffi::TensorView score,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView lengths,
    tvm::ffi::TensorView row_starts,
    int64_t has_row_starts) {
  const int B = static_cast<int>(score.size(0));
  const int topk = static_cast<int>(indices.size(1));
  if (!(topk == 512 || topk == 1024 || topk == 2048)) {
    throw std::runtime_error("fast_topk supports topk in {512, 1024, 2048}");
  }

  FastTopKParams params{
      static_cast<const float*>(score.data_ptr()),
      has_row_starts ? static_cast<const int32_t*>(row_starts.data_ptr()) : nullptr,
      static_cast<int32_t*>(indices.data_ptr()),
      static_cast<int32_t*>(lengths.data_ptr()),
      static_cast<int64_t>(score.stride(0)),
  };

  const auto device = score.device();
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));

  unsigned int split_kv = 1;
  if (B < SMs) {
    split_kv = CEILDIV(SMs, B);
    split_kv = MIN(split_kv, 8);
    if (B >= 64) {
      split_kv = MIN(split_kv, 2);
    } else if (B >= 32) {
      split_kv = MIN(split_kv, 4);
    } else {
      split_kv = MIN(split_kv, 8);
    }
  }
  split_kv = smallk_widen_split_kv(topk, B, split_kv);

  dim3 grid(B, split_kv, 1);
  dim3 block(kThreadsPerBlock, 1, 1);

#ifdef SMALL_K_CTA_CTRL

  auto launch_small_k = [&]() {
    if (topk == 512) {
      launch_topk_kernel_small_k<512>(params, B, stream);
    } else if (topk == 1024) {
      launch_topk_kernel_small_k<1024>(params, B, stream);
    } else {
      launch_topk_kernel_small_k<2048>(params, B, stream);
    }
  };
  auto throw_if_error = []() {
    const auto rc = cudaGetLastError();
    if (rc != cudaSuccess) {
      throw std::runtime_error(std::string("topk kernel failed: ") + ::cudaGetErrorString(rc));
    }
  };
  const int64_t L = static_cast<int64_t>(score.size(1));
  if ((topk == 512 || topk == 1024 || topk == 2048) && L > 0 && L <= SMALL_K_MAX_L) {
    launch_small_k();
    throw_if_error();
    return;
  }
  if ((topk == 512 || topk == 1024 || topk == 2048) && has_row_starts != 0) {
    // Ragged wide tensor: single-CTA for short rows, distributed (skip_short) for the rest.
    launch_small_k();
    switch (topk) {
      case 512:
        launch_topk_kernel<512>(params, grid, block, split_kv, nullptr, stream, /*skip_short_len=*/true);
        break;
      case 1024:
        launch_topk_kernel<1024>(params, grid, block, split_kv, nullptr, stream, /*skip_short_len=*/true);
        break;
      default:
        launch_topk_kernel<2048>(params, grid, block, split_kv, nullptr, stream, /*skip_short_len=*/true);
        break;
    }
    throw_if_error();
    return;
  }
#endif // SMALL_K_CTA_CTRL

  switch (topk) {
    case 512:
      launch_topk_kernel<512>(params, grid, block, split_kv, nullptr, stream);
      break;
    case 1024:
      launch_topk_kernel<1024>(params, grid, block, split_kv, nullptr, stream);
      break;
    default:
      launch_topk_kernel<2048>(params, grid, block, split_kv, nullptr, stream);
      break;
  }

  const auto result = cudaGetLastError();
  if (result != cudaSuccess) {
    throw std::runtime_error(
        std::string("topk kernel failed: ") + ::cudaGetErrorString(result));
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fast_topk, tvm_jit_fast_topk_launch);
#endif  // USE_TVM_FFI

#if !defined(USE_TVM_FFI)
void fast_topk_interface(
    const at::Tensor& score, at::Tensor& indices, const at::Tensor& lengths, std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  const auto B = score.size(0);
  const auto topk = static_cast<int>(indices.size(1));

  CHECK_CUDA(indices);

  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }
  CHECK_CUDA(lengths);

  const auto params = get_params(score, lengths, row_starts_opt, indices);

  // TODO (yiakwy) : use tvm ffi raw C interface to launch kernel, instead of Aten CUDAStream
  const auto stream = at::cuda::getCurrentCUDAStream().stream();

#if ENABLE_HOPPER
  // NOTE(yiakwy) : in hopper platform only maximum 8 blocks per cluster are supported to use NoC
  constexpr int max_kv_split = 8;
#else
  const int max_kv_split = SMs;
#endif

  // Ablation override (read once per process)
  static const int split_override = [] {
    const char* s = std::getenv("FLASH_FLOAT_TOPK_SPLIT");
    return s == nullptr ? 0 : std::atoi(s);
  }();

  unsigned int split_kv = 1;
  if (B < SMs) {  // we have enough batches of data to run in parallel
    split_kv = CEILDIV(SMs, B);
    split_kv = MIN(split_kv, max_kv_split);

#if ENABLE_HOPPER
    if (B >= 64) {
      split_kv = MIN(split_kv, 2);
    } else if (B >= 32) {
      split_kv = MIN(split_kv, 4);
    } else {
      split_kv = MIN(split_kv, 8);
    }
#else
    if (B >= 64) {
      split_kv = MIN(split_kv, 2);
    } else if (B >= 32) {
      split_kv = MIN(split_kv, 4);
    } else if (B >= 16) {
      split_kv = MIN(split_kv, 8);
    } else if (B >= 8) {
      split_kv = MIN(split_kv, 16);
    } else if (B >= 4) {
      split_kv = MIN(split_kv, 32);
    } else if (B >= 2) {
      split_kv = MIN(split_kv, 64);
    } else {
      split_kv = MIN(split_kv, 128);
    }
#endif
  }
  split_kv = smallk_widen_split_kv(topk, static_cast<int>(B), split_kv);
  if (split_override > 0) {
    // Software is limited to 8 CTAs per cluster; Hopper hard limit is 16 CTAs per cluster.
    split_kv = MIN(split_override, 16);
  }

  dim3 grid(B, split_kv, 1);
  dim3 block(kThreadsPerBlock, 1, 1);

  int* scratch_ptr = nullptr;

#if !ENABLE_HOPPER
  at::Tensor scratch;
  if (split_kv > 1) {
    scratch = at::zeros({(long)B, (long)split_kv, RADIX}, score.options().dtype(at::kInt));
    scratch_ptr = (int*)scratch.data_ptr();
  }
#else
  (void)scratch_ptr;
#endif

  const int L = static_cast<int>(score.size(1));

  const bool use_harnessed = (topk == 512 || topk == 1024) && L > 0 && L <= 32768;

  // Single-CTA short-sequence path for small K (top512/1024/2048, L <= 16384):
  //   - no fixed NoC-reduce
  //   - no cluster-barrier cost that dominates at short L
  //
  // See top_small_k.cuh. Long-L top2048 keeps the distributed cluster path.
#ifdef SMALL_K_CTA_CTRL
  auto launch_small_k = [&]() {
    if (topk == 512) {
      launch_topk_kernel_small_k<512>(params, static_cast<int>(B), stream);
    } else if (topk == 1024) {
      launch_topk_kernel_small_k<1024>(params, static_cast<int>(B), stream);
    } else {
      launch_topk_kernel_small_k<2048>(params, static_cast<int>(B), stream);
    }
  };

  const bool small_k_topk = (topk == 512 || topk == 1024 || topk == 2048);
  if (small_k_topk && L > 0 && L <= SMALL_K_MAX_L) {
    // Contiguous short row (or ragged whose tensor width already fits): one CTA.
    launch_small_k();
  } else if (small_k_topk && row_starts_opt.has_value()) {
    // Ragged wide tensor (window shorter than the tensor width): the single-CTA
    // path handles rows with length <= SMALL_K_MAX_L; the distributed kernel is
    // relaunched with skip_short_len so it only handles the long rows.
    launch_small_k();
    switch (topk) {
      case 512:
        launch_topk_kernel<512>(params, grid, block, split_kv, scratch_ptr, stream, /*skip_short_len=*/true);
        break;
      case 1024:
        launch_topk_kernel<1024>(params, grid, block, split_kv, scratch_ptr, stream, /*skip_short_len=*/true);
        break;
      default:
        launch_topk_kernel<2048>(params, grid, block, split_kv, scratch_ptr, stream, /*skip_short_len=*/true);
        break;
    }
  } else
#endif  // SMALL_K_CTA_CTRL

  if (use_harnessed) {
    // Independent split budget for the harnessed path (can exceed the 8-CTA
    // portable cluster cap, up to the non-portable max), so a single row can
    // occupy more SMs at short L. `bs`-dependent: CEILDIV(SMs, B).
    unsigned int split_h = CEILDIV(SMs, B);
    if (split_h > (unsigned int)TOPK_HARNESSED_MAX_SPLIT) split_h = (unsigned int)TOPK_HARNESSED_MAX_SPLIT;
    if (split_h < 1u) split_h = 1u;
    const dim3 grid_h(B, split_h, 1);
    if (topk == 512) {
      launch_topk_kernel_harnessed<512>(params, grid_h, block, split_h, scratch_ptr, stream);
    } else {
      launch_topk_kernel_harnessed<1024>(params, grid_h, block, split_h, scratch_ptr, stream);
    }
  } else
  switch (topk) {
    case 512:
      launch_topk_kernel<512>(params, grid, block, split_kv, scratch_ptr, stream);
      break;
    case 1024:
      launch_topk_kernel<1024>(params, grid, block, split_kv, scratch_ptr, stream);
      break;
    case 2048:
      launch_topk_kernel<2048>(params, grid, block, split_kv, scratch_ptr, stream);
      break;
    default:
      TORCH_CHECK(false, "fast_topk unsupported topk=", topk, " (expected 512, 1024 or 2048)");
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& dst_page_table,
    const at::Tensor& src_page_table,
    const at::Tensor& cu_seqlens_q,
    std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(dst_page_table);
  CHECK_CUDA(src_page_table);
  CHECK_CUDA(cu_seqlens_q);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }
  const auto params = get_params(score, lengths, row_starts_opt);
  const auto B = score.size(0);
  TORCH_CHECK(dst_page_table.dim() == 2 && dst_page_table.is_contiguous());
  TORCH_CHECK(src_page_table.dim() == 2 && src_page_table.stride(1) == 1);
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_q.is_contiguous());
  const auto prefill_bs = cu_seqlens_q.size(0) - 1;
  TORCH_CHECK(dst_page_table.size(0) == B);
  TORCH_CHECK(dst_page_table.size(1) == TopK);
  TORCH_CHECK(src_page_table.size(0) == prefill_bs);
  TORCH_CHECK(prefill_bs <= B);  // prefill_bs should be smaller than expanded bs

  // launch kernel
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  const auto src_stride = src_page_table.stride(0);

  // dispatch to decode or prefill
  // extend and draft extend: row_starts_opt is not null, invokes the prefill kernel
  // decode: row_starts_opt is null, invokes the decode kernel
  // target verify: row_starts_opt is null, invokes the prefill kernel
  const auto is_decode = !row_starts_opt.has_value() && prefill_bs == B;
  if (is_decode) {
    setup_kernel_smem_once<topk_transform_decode_kernel, kSmem>();
    topk_transform_decode_kernel<<<grid, block, kSmem, stream>>>(
        params, dst_page_table.data_ptr<int32_t>(), src_page_table.data_ptr<int32_t>(), src_stride);
  } else {
    setup_kernel_smem_once<topk_transform_prefill_kernel, kSmem>();
    topk_transform_prefill_kernel<<<grid, block, kSmem, stream>>>(
        params,
        dst_page_table.data_ptr<int32_t>(),
        src_page_table.data_ptr<int32_t>(),
        src_stride,
        cu_seqlens_q.data_ptr<int32_t>(),
        prefill_bs);
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

void fast_topk_transform_ragged_interface(
    const at::Tensor& score,
    const at::Tensor& lengths,
    at::Tensor& topk_indices_ragged,
    const at::Tensor& topk_indices_offset,
    std::optional<at::Tensor> row_starts_opt) {
  CHECK_CUDA(score);
  CHECK_CUDA(lengths);
  CHECK_CUDA(topk_indices_ragged);
  CHECK_CUDA(topk_indices_offset);
  if (row_starts_opt.has_value()) {
    CHECK_CUDA(row_starts_opt.value());
  }

  const auto params = get_params(score, lengths, row_starts_opt);
  const auto B = score.size(0);
  TORCH_CHECK(topk_indices_ragged.dim() == 2 && topk_indices_ragged.is_contiguous());
  TORCH_CHECK(topk_indices_offset.dim() == 1);

  TORCH_CHECK(topk_indices_ragged.size(0) == B);
  TORCH_CHECK(topk_indices_ragged.size(1) == TopK);
  TORCH_CHECK(topk_indices_offset.size(0) == B);

  // launch kernel
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};

  setup_kernel_smem_once<topk_transform_prefill_ragged_kernel, kSmem>();
  topk_transform_prefill_ragged_kernel<<<grid, block, kSmem, stream>>>(
      params, topk_indices_ragged.data_ptr<int32_t>(), topk_indices_offset.data_ptr<int32_t>());

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess, "topk kernel failed:", ::cudaGetErrorString(result));
}

#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fast_topk", &fast_topk_interface, "fast_topk");
  m.def("fast_topk_transform_fused", &fast_topk_transform_interface, "fast_topk_transform");
  m.def("fast_topk_transform_ragged_fused", &fast_topk_transform_ragged_interface, "fast_topk_transform_ragged");
}
#endif  // !USE_TVM_FFI

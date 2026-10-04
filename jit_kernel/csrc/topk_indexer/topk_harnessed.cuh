#pragma once

// ---------------------------------------------------------------------------
// HARNESSED COPY of fast_topk_split_kv_cuda_tl<TopK> (same multi-cluster NoC
// design). Optimizations for small k (512/1024) are applied in this copy only.
// ---------------------------------------------------------------------------
template <int TopK>
__device__ void fast_topk_split_kv_cuda_tl_harnessed(
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
  alignas(128) __shared__ int s_local_hist[RADIX];

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

// ---------------------------------------------------------------------------
// Short-sequence specialization for small k (512/1024): a SINGLE CTA per row
// (`is_split_mode=false`, no NoC cluster split). 
// 
// Proved by DeepSelect, also see our profiling results in benchmark, for small L 
// the cluster split (up to 8 CTAs + several cluster barriers) is pure overhead; 
// one block is faster and equally exact. Long sequences keep the original 
// `topk_kernel`.
// ---------------------------------------------------------------------------
template <int TopK>
__global__ __launch_bounds__(kThreadsPerBlock)  // topk (harnessed)
    void topk_kernel_harnessed(const FastTopKParams params, int* g_scratch, bool use_split_kv) {
  const auto& [input, row_starts, indices, lengths, input_stride] = params;

  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto indice = indices + bid * TopK;
  const auto score = input + bid * input_stride;

  if (length <= TopK) {
    return naive_topk_cuda(score, indice, length, TopK);
  } else {
    return fast_topk_split_kv_cuda_tl_harnessed<TopK>(
        score, indice, row_start, length, TopK, g_scratch, use_split_kv);
  }
}

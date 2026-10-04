#pragma once

// ---------------------------------------------------------------------------
// Single-CTA short-sequence small-K top-k (TopK in {512, 1024, 2048}, L <= 16384).
//
// Rationale : for short L the distributed (NoC cluster) radix spends a fixed 
// cross-CTA histogram-reduction + cluster barrier cost every refinement round, which 
// dominates the small amount of work
//
// This specialization runs ONE CTA per row: 
//   - no NoC reduce, 
//   - no cluster barriers. 
//
// After benchmarching we denied TMA streaming to hide partial hist reduction 
//
// The distributed path is kept for L > 16384.
// ---------------------------------------------------------------------------

// Maximum row length handled by this path.
constexpr int SMALL_K_MAX_L = 16384;

#ifdef SMALL_K_CTA_CTRL

// Fire-and-forget shared-memory increment
__device__ __forceinline__ void small_k_hist_inc(int* slot) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(slot));
  asm volatile("red.shared.add.u32 [%0], %1;" ::"r"(a), "r"(1) : "memory");
}

// NOTE (yiakwy) : adapted from our **convert_to_monotonic_8bit** with an explicit [0, 255] clamp
__device__ __forceinline__ uint8_t small_k_residual_key(float x) {
  const int b = __float2int_rd(x);
  return static_cast<uint8_t>(b < 0 ? 0 : (b > 255 ? 255 : b));
}

// Survivor buffer capacity (>= TopK + margin).
constexpr int SMALL_K_CAP = 8192;

#ifndef SMALL_K_PROFILE
#define SMALL_K_PROFILE 0
#endif // SMALL_K_PROFILE

__device__ __forceinline__ void small_k_warp_suffix_sum_256(const int* hist, int* out, int lane) {
  int loc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    loc[j] = hist[(lane << 3) + j];
  }
  int within[8];
  int acc = 0;
#pragma unroll
  for (int j = 7; j >= 0; --j) {
    acc += loc[j];
    within[j] = acc;
  }
  const int lane_total = acc;

  int x = lane_total;
#pragma unroll
  for (int off = 1; off < WARP_SIZE; off <<= 1) {
    const int t = __shfl_down_sync(0xffffffffu, x, off);
    if (lane + off < WARP_SIZE) {
      x += t;
    }
  }
  const int excl = x - lane_total;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    out[(lane << 3) + j] = within[j] + excl;
  }
  if (lane == 0) {
    out[RADIX] = 0;
  }
}

// Find the threshold bin such that (suffix[bin] > target) && (suffix[bin + 1] <= target).
__device__ __forceinline__ int small_k_find_threshold_bin(const int* suffix, int target, int tx) {
  if (tx < RADIX && suffix[tx] > target && suffix[tx + 1] <= target) {
    return tx;
  }
  return -1;
}

template <int TopK>
__global__ __launch_bounds__(kThreadsPerBlock)
void topk_kernel_small_k(const FastTopKParams params) {
  const auto& [input, row_starts, indices, lengths, input_stride] = params;

  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
  const auto length = lengths[bid];
  const auto indice = indices + bid * TopK;
  const auto score = input + bid * input_stride;

  if (length <= TopK) {
    return naive_topk_cuda(score, indice, length, TopK);
  }
  if (length > SMALL_K_MAX_L) {
    return;  // dispatched only for L <= SMALL_K_MAX_L
  }

  const int tx = threadIdx.x;
  const int lane_id = tx % WARP_SIZE;
  constexpr int BLOCK_SIZE = kThreadsPerBlock;

  // TODO (yiakwy) : use external memory in case of template instantiation for each TopK
  __shared__ int s_hist[2][RADIX + 128];
  __shared__ int s_threshold_bin;
  __shared__ int s_num_in[2];
  __shared__ int s_write_ptr;

  extern __shared__ __align__(16) char sk_smem[];

  char* sk_ptr = sk_smem;

  uint8_t* s_bin = reinterpret_cast<uint8_t*>(sk_ptr);
  sk_ptr += SMALL_K_MAX_L;
  
  float* s_val[2];
  s_val[0] = reinterpret_cast<float*>(sk_ptr);
  sk_ptr += (size_t)SMALL_K_CAP * sizeof(float);
  
  s_val[1] = reinterpret_cast<float*>(sk_ptr);
  sk_ptr += (size_t)SMALL_K_CAP * sizeof(float);
  
  unsigned* s_idx[2];
  s_idx[0] = reinterpret_cast<unsigned*>(sk_ptr);
  sk_ptr += (size_t)SMALL_K_CAP * sizeof(unsigned);
  s_idx[1] = reinterpret_cast<unsigned*>(sk_ptr);
  sk_ptr += (size_t)SMALL_K_CAP * sizeof(unsigned);

#if SMALL_K_PROFILE
  unsigned long long mt0 = 0, mt1 = 0, mt2 = 0, mt3 = 0, mt4 = 0;
  if (tx == 0) mt0 = clock64();
#endif // SMALL_K_PROFILE

  if (tx < RADIX + 1) {
    s_hist[0][tx] = 0;
  }
  if (tx == 0) {
    s_threshold_bin = 0;
    s_write_ptr = 0;
    s_num_in[0] = 0;
  }
  __syncthreads();

  const float* base = score + row_start;
  const bool aligned4 = ((reinterpret_cast<uintptr_t>(base) & 0xF) == 0);
  const int nvec = length >> 2;
  const int vec_end = aligned4 ? (nvec << 2) : 0;

  if (aligned4) {
    const float4* b4 = reinterpret_cast<const float4*>(base);
    for (int i = tx; i < nvec; i += BLOCK_SIZE) {
      const float4 v = b4[i];
      const int p = i << 2;

      const uint8_t k0 = convert_to_uint8(v.x);
      const uint8_t k1 = convert_to_uint8(v.y);
      const uint8_t k2 = convert_to_uint8(v.z);
      const uint8_t k3 = convert_to_uint8(v.w);

      *reinterpret_cast<uint32_t*>(&s_bin[p]) =
          (uint32_t)k0 | ((uint32_t)k1 << 8) | ((uint32_t)k2 << 16) | ((uint32_t)k3 << 24);
    
      small_k_hist_inc(&s_hist[0][k0]);
      small_k_hist_inc(&s_hist[0][k1]);
      small_k_hist_inc(&s_hist[0][k2]);
      small_k_hist_inc(&s_hist[0][k3]);
    }
  }
  for (int idx = vec_end + tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(base[idx]);
    s_bin[idx] = bin;
    small_k_hist_inc(&s_hist[0][bin]);
  }
  __syncthreads();

#if SMALL_K_PROFILE
  if (tx == 0) mt1 = clock64();
#endif // SMALL_K_PROFILE
  
  if (tx < WARP_SIZE) {
    small_k_warp_suffix_sum_256(s_hist[0], s_hist[1], lane_id);
  }
  __syncthreads();

  if (small_k_find_threshold_bin(s_hist[1], TopK, tx) >= 0) {
    s_threshold_bin = tx;
  }
  __syncthreads();

  const int threshold_bin = s_threshold_bin;
  int global_remainder = TopK - s_hist[1][threshold_bin + 1];

#if SMALL_K_PROFILE
  if (tx == 0) mt2 = clock64();
#endif // SMALL_K_PROFILE

  const bool want_cand = global_remainder > 0;
  if (tx < RADIX) {
    s_hist[0][tx] = 0;
  }
  if (tx == 0) {
    s_write_ptr = 0;
    s_num_in[0] = 0;
  }
  __syncthreads();

  auto collect_one = [&](int idx, int bin, float val) {
    if (bin > threshold_bin) {
      const int pos = s_hist[1][bin + 1] + ::atomicAdd(&s_hist[0][bin], 1);
      if (pos < TopK) {
        indice[pos] = idx;
      }
    } else if (want_cand && bin == threshold_bin) {
      const float val_scale = (val - threshold_bin) * RADIX;
      const int pos = ::atomicAdd(&s_num_in[0], 1);
      if (pos < SMALL_K_CAP) {
        s_val[0][pos] = val_scale;
        s_idx[0][pos] = idx;
      }
    }
  };

  if (aligned4) {
    const float4* b4 = reinterpret_cast<const float4*>(base);
    for (int i = tx; i < nvec; i += BLOCK_SIZE) {
      const int p = i << 2;
      const uint32_t packed = *reinterpret_cast<const uint32_t*>(&s_bin[p]);
      const float4 v = b4[i];

      collect_one(p, (int)(packed & 0xFFu), v.x);

      collect_one(p + 1, (int)((packed >> 8) & 0xFFu), v.y);
      collect_one(p + 2, (int)((packed >> 16) & 0xFFu), v.z);
      collect_one(p + 3, (int)((packed >> 24) & 0xFFu), v.w);
    }
  }

  for (int idx = vec_end + tx; idx < length; idx += BLOCK_SIZE) {
    collect_one(idx, s_bin[idx], base[idx]);
  }
  __syncthreads();

#if SMALL_K_PROFILE
  if (tx == 0) mt3 = clock64();
#endif // SMALL_K_PROFILE

  int* out_ptr = indice + (TopK - global_remainder);
  int rem = global_remainder;
  int sub_cur = 0;

  for (int round = 0; round < 4 && rem > 0; ++round) {
    const int scan_size = s_num_in[sub_cur];

    if (tx < RADIX + 1) {
      s_hist[0][tx] = 0;
    }
    if (tx == 0) {
      s_threshold_bin = 0;
      s_write_ptr = 0;
      s_num_in[sub_cur ^ 1] = 0;
    }
    __syncthreads();

    for (int i = tx; i < scan_size; i += BLOCK_SIZE) {
      const auto sub_bin = small_k_residual_key(s_val[sub_cur][i]);
      small_k_hist_inc(&s_hist[0][sub_bin]);
    }
    __syncthreads();

    if (tx < WARP_SIZE) {
      small_k_warp_suffix_sum_256(s_hist[0], s_hist[1], lane_id);
    }
    __syncthreads();

    if (small_k_find_threshold_bin(s_hist[1], rem, tx) >= 0) {
      s_threshold_bin = tx;
    }
    __syncthreads();

    const int next_threshold_bin = s_threshold_bin;
    const int next_remainder = rem - s_hist[1][next_threshold_bin + 1];
    const int sub_next = sub_cur ^ 1;

    for (int i = tx; i < scan_size; i += BLOCK_SIZE) {
      const int sub_bin = small_k_residual_key(s_val[sub_cur][i]);
      if (sub_bin > next_threshold_bin) {
        const int pos = ::atomicAdd(&s_write_ptr, 1);
        if (pos < rem) {
          out_ptr[pos] = s_idx[sub_cur][i];
        }
      } else if (sub_bin == next_threshold_bin && next_remainder > 0) {
        const float val_scale = (s_val[sub_cur][i] - next_threshold_bin) * RADIX;
        const int pos = ::atomicAdd(&s_num_in[sub_next], 1);
        if (pos < SMALL_K_CAP) {
          s_val[sub_next][pos] = val_scale;
          s_idx[sub_next][pos] = s_idx[sub_cur][i];
        }
      }
    }
    __syncthreads();

    out_ptr += rem - next_remainder;
    rem = next_remainder;
    sub_cur = sub_next;
  }

#if SMALL_K_PROFILE
  if (tx == 0) mt4 = clock64();
#endif // SMALL_K_PROFILE

  if (rem > 0) {
    float* s_val_fin = s_val[sub_cur];
    unsigned* s_idx_fin = s_idx[sub_cur];
    const int n = s_num_in[sub_cur];
    for (int i = tx; i < n; i += BLOCK_SIZE) {
      s_val_fin[i] = score[s_idx_fin[i] + row_start];
    }
    __syncthreads();

    if (tx < n) {
      float cur_val = s_val_fin[tx];
      int cur_idx = tx;
      for (int i = 0; i < rem; ++i) {
        atomicUpdateMaxIndex(out_ptr + i, s_val_fin, &cur_val, &cur_idx, 0);
        if (cur_val == -1) {
          break;
        }
      }
    }
    __syncthreads();

    if (tx < rem) {
      out_ptr[tx] = static_cast<int>(s_idx_fin[out_ptr[tx]]);
    }
  }

#if SMALL_K_PROFILE
  if (bid == 0 && tx == 0) {
    unsigned long long mt5 = clock64();
    printf("[smallk k=%d L=%d] pass1=%llu suffix=%llu collect=%llu refine=%llu final=%llu tot=%llu ticks\n",
           TopK, length, mt1 - mt0, mt2 - mt1, mt3 - mt2, mt4 - mt3, mt5 - mt4, mt5 - mt0);
  }
#endif // SMALL_K_PROFILE

}

template <int TopK>
void launch_topk_kernel_small_k(const FastTopKParams& params, int B, cudaStream_t stream) {
  constexpr size_t smem = (size_t)SMALL_K_MAX_L + 2 * (size_t)SMALL_K_CAP * sizeof(float) +
                          2 * (size_t)SMALL_K_CAP * sizeof(unsigned);
  static const bool once = [] {
    return cudaFuncSetAttribute(
               &topk_kernel_small_k<TopK>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess;
  }();
  (void)once;
  topk_kernel_small_k<TopK><<<B, kThreadsPerBlock, smem, stream>>>(params);
}

#endif  // SMALL_K_CTA_CTRL

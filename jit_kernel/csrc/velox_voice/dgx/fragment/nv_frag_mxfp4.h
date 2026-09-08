/* Copyright 2026 flashFloat authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
==============================================================================*/

/* ref to SM100 implementation https://github.com/Avarok-Cybersecurity/dgx-vllm/blob/main/cutlass_nvfp4/nvfp4_gemm_kernel_optimized.cuh */

/* ret to SM121a mma instruction https://github.com/ggml-org/llama.cpp/blob/8fe90e1fbfc065f17a0b233c9df239423cd24a75/ggml/src/ggml-cuda/mma.cuh#L1126 
 * Note our gemm implements standard wasp workload and extensively use TMA (NoC) for data loading.
 */

#pragma once

#include <cstdint>

namespace xpu {

__device__ inline void mxfp4_mma(float acc[4], const uint32_t a[4], const uint32_t b[2],
                                 uint32_t scale_a = 0x00007F7Fu,
                                 uint32_t scale_b = 0x00007F7Fu) {
  float c0 = acc[0], c1 = acc[1], c2 = acc[2], c3 = acc[3];
  const uint16_t z = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col."
      "f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
      "{%14},{%15,%16},{%17},{%18,%19};\n"
      : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3),
        "r"(scale_a), "h"(z), "h"(z), "r"(scale_b), "h"(z), "h"(z));
}

}  // namespace xpu

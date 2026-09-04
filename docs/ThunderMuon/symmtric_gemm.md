<h2 id="Thunder-Muon-Symm-Gemm">⚡ Thunder Muon : Symmetric GEMM</h2>

**Single Batch Symmetric GEMM for Muon**

| m | B | torch (ms) | triton_fp8_gemm_tril_ref (ms) | cuda_muon_symm_gemm (ms) | vs. torch (Speedup) |
|---:|---:|---:|---:|---:|---:|
| 2048.0 | 1.0 | 28.096000 | 64.864002 | 48.223998 | - |
| 4096.0 | 1.0 | 181.664005 | 230.991997 | 134.304002 | +26% (x1.35) |
| 8192.0 | 1.0 | 1567.391992 | 1642.271996 | 830.272019 | +47% (x1.88) |

When training vision models such as Hunyuan 3 under **Muon**, updating weight matrix using NS5 iterations, instead of using batched Symmetric GEMM over a multiple weights copies, can be more efficient for a large weight matrix with **batch-1** Symm GEMM, where more than **47%** speed up can be achieved under our innovated symmetric gemm design.

The result is attributed to the hardware and software co-design. We propsed a new swizzling algorithms for Hopper/Blackwell Platform to enable more efficient computation by utilizing L2 cache sharing among nearby blocks.

The symmetric gemm suffers from load balancing problem under `Zig-Zag` style scheduling :

<p align="center">
<img width="438" height="441" alt="Our thunder moun sym gemm illustration" src="https://github.com/user-attachments/assets/0cd067cd-841b-4d55-be4b-1d958cf931a4" />
<br>
<em>Our Thunder Muon under Zig-Zag style (group by group)</em>
</p>

We proposed ZigZag style **triangular linear swizzling** algorithm for better L2 cache locality. Morever, we proved that for **GROUP_SIZE_M** to be multiple of 4, we can safely open and close NoC multicast on condition to save HBM bandwidth.

**Batched Symmetric GEMM for Muon**

While single batch symmetric gemm is suitable for large NS5 iterations over large weight matrix, reproducing results on GPT-Nano (standard muon benchmark) requires batched symmetric gemm for small weight matrix such as `256 x 256`, `768 x 768`.

Our algorithm naturally supports batched gemm, where the batch dimension are required to be continous (this can be neatly achieved by splicing weights from a large tensor allocation before training).

| m | B | torch (ms) | triton_ref (ms) | gluon_muon_symm_gemm (ms) | cuda_muon_symm_gemm (ms) | vs. torch (Speedup) |
|---:|---:|---:|---:|---:|---:|---:|
| 2048.0 | 1.0 | 28.192000 | 42.080000 | 43.552000 | 47.936000 | 0.59x |
| 2048.0 | 4.0 | 94.527997 | 120.544001 | 108.543999 | 110.335998 | 0.86x |
| 2048.0 | 8.0 | 192.192003 | 226.047993 | 191.648006 | 193.312004 | 0.99x |
| 2048.0 | 16.0 | 391.328007 | 441.152006 | 357.279986 | 357.663989 | 1.09x |
| 4096.0 | 1.0 | 182.784006 | 178.415999 | 170.144007 | 135.008007 | 1.35x |
| 4096.0 | 4.0 | 789.632022 | 698.816001 | 637.088001 | 517.343998 | 1.53x |
| 4096.0 | 8.0 | 1594.752014 | 1392.944038 | 1272.896051 | 1028.704047 | 1.55x |
| 4096.0 | 16.0 | 3253.344059 | 2806.015968 | 2517.440081 | 1977.504015 | 1.65x |
| 8192.0 | 1.0 | 1752.368033 | 1276.031971 | 1289.759994 | 860.000014 | 2.04x |
| 8192.0 | 4.0 | 6354.944229 | 5169.951916 | 5165.328026 | 3469.984055 | 1.83x |
| 8192.0 | 8.0 | 12880.000114 | 10903.200150 | 10017.248154 | 6719.792128 | 1.92x |
| 8192.0 | 16.0 | 29570.655823 | 21074.111938 | 21681.424141 | 13627.840042 | 2.17x |

## Citation

If you use this codebase, or otherwise find our work valuable, please cite DistRadixTopK2026:

```bibtex
@misc{ThunderMuon2026,
  title   = {ThunderMuon : Bridging Specttral Optimization and Hardware Efficiency for Vision Tasks},
  author  = {LEI WANG, Mingzhe Zheng, Hao Gu, Hui Guo, Bei Liu, Sirui Han, Wei Xue, Qifeng Chen, Yike Guo},
  year    = {2026},
  url     = {https://github.com/yiakwy-xpu-ml-framework-team/flash-float-jit-kernels}
}
```
<h2 id="Thunder-Muon-Symm-Gemm">⚡ Thunder Muon : Symmetric GEMM</h2>

**Single Batch Symmetric GEMM for Muon**

| m | B | torch (ms) | triton_fp8_gemm_tril_ref (ms) | cuda_muon_symm_gemm (ms) | vs. torch (Speedup) |
|---:|---:|---:|---:|---:|---:|
| 2048.0 | 1.0 | 28.287999 | 42.272002 | 47.904000 | - |
| 4096.0 | 1.0 | 182.528004 | 178.752005 | 134.911999 | +26% (x1.35) |
| 8192.0 | 1.0 | 1651.120007 | 1300.160050 | 856.927991 | +48% (x1.93) |

When training vision models such as Hunyuan video 1.5 and MiniMax H3 under **Muon**, updating weight matrix using NS5 iterations, instead of using batched Symmetric GEMM over a multiple weights copies, can be more efficient for a large weight matrix with **batch-1** Symm GEMM, where more than **48%** speed up can be achieved under our innovated symmetric gemm design.

The result is attributed to the hardware and software co-design. We proposed new swizzling algorithms for Hopper/Blackwell Platform to enable more efficient computation by utilizing L2 cache sharing among nearby blocks.

The symmetric gemm suffers from load balancing problem under `Zig-Zag` style scheduling :

<p align="center">
<img width="438" height="441" alt="Our thunder moun sym gemm illustration" src="https://github.com/user-attachments/assets/0cd067cd-841b-4d55-be4b-1d958cf931a4" />
<br>
<em>Our Thunder Muon under Zig-Zag style (group by group)</em>
</p>

We proposed ZigZag style **triangular linear swizzling** algorithm for better L2 cache locality. Moreover, we proved that for **GROUP_SIZE_M** to be a multiple of 4, we can safely open and close NoC multicast on condition to save HBM bandwidth.

**Batched Symmetric GEMM for Muon**

While single batch symmetric gemm is suitable for large NS5 iterations over large weight matrix, reproducing results on GPT-Nano (standard muon benchmark) requires batched symmetric gemm for small weight matrix such as `256 x 256`, `768 x 768`.

Our algorithm naturally supports batched gemm, where the batch dimension is required to be continuous (this can be neatly achieved by splicing weights from a large tensor allocation before training).

| m | B | torch (ms) | triton_ref (ms) | gluon_muon_symm_gemm (ms) | cuda_muon_symm_gemm (ms) | vs. torch (Speedup) |
|---:|---:|---:|---:|---:|---:|---:|
| 2048.0 | 1.0 | 28.287999 | 42.272002 | 43.552000 | 47.904000 | 0.59x |
| 2048.0 | 4.0 | 94.272003 | 120.544001 | 108.960003 | 109.952003 | 0.86x |
| 2048.0 | 8.0 | 190.303996 | 224.703997 | 191.903993 | 193.039998 | 0.99x |
| 2048.0 | 16.0 | 386.319995 | 439.328000 | 357.167989 | 358.592004 | 1.08x |
| 4096.0 | 1.0 | 182.528004 | 178.752005 | 170.752004 | 134.911999 | 1.35x |
| 4096.0 | 4.0 | 797.599971 | 699.088007 | 637.983978 | 517.632008 | 1.54x |
| 4096.0 | 8.0 | 1599.135995 | 1395.887971 | 1261.824012 | 1027.199984 | 1.56x |
| 4096.0 | 16.0 | 3251.424074 | 2800.527930 | 2501.760006 | 1972.479999 | 1.65x |
| 8192.0 | 1.0 | 1651.120007 | 1300.160050 | 1292.032003 | 856.927991 | 1.93x |
| 8192.0 | 4.0 | 6572.080135 | 5140.128136 | 5169.935942 | 3413.984060 | 1.93x |
| 8192.0 | 8.0 | 12711.296082 | 10982.655525 | 10231.200218 | 6879.392147 | 1.85x |
| 8192.0 | 16.0 | 26802.463531 | 20702.303886 | 20535.840034 | 13805.695534 | 1.94x |

## Citation

If you use this codebase, or otherwise find our work valuable, please cite ThunderMuon2026:

```bibtex
@misc{ThunderMuon2026,
  title   = {ThunderMuon : Bridging Spectral Optimization and Hardware Efficiency for Vision Tasks},
  author  = {LEI WANG, Mingzhe Zheng, Erke Xia, Tillo Juraboev, Hao Gu, Hui Guo, Bei Liu, Sirui Han, Wei Xue, Qifeng Chen, Yike Guo},
  year    = {2026},
  url     = {https://github.com/yiakwy-xpu-ml-framework-team/flash-float-jit-kernels}
}
```

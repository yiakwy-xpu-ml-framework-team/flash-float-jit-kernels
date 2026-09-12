<h2 id="Thunder-Muon-Symm-Gemm">⚡ Thunder Muon : Symmetric GEMM</h2>

**Single Batch Symmetric GEMM for Muon**

| m | B | torch (ms) | triton_fp8_gemm_tril_ref (ms) | cuda_muon_symm_gemm (ms) | vs. torch (Speedup) |
|---:|---:|---:|---:|---:|---:|
| 2048.0 | 1.0 | 28.416000 | 41.855998 | 47.936000 | - |
| 4096.0 | 1.0 | 183.295995 | 178.335994 | 134.048000 | +27% (x1.37) |
| 5376.0 | 1.0 | 427.231997 | 397.376001 | 278.207988 | +53% (x1.54) |
| 8192.0 | 1.0 | 1584.752023 | 1257.120013 | 856.159985 | +46% (x1.85) |
| 14336.0 | 1.0 | 8706.048012 | 7634.687901 | 4289.920092 | +51% (x2.03) |

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
| 2048.0 | 1.0 | 27.807999 | 42.112000 | 43.104000 | 47.743998 | 0.58x |
| 2048.0 | 4.0 | 95.823999 | 120.320000 | 108.608000 | 114.752002 | 0.84x |
| 2048.0 | 8.0 | 191.648006 | 224.928007 | 191.520005 | 196.480006 | 0.98x |
| 2048.0 | 16.0 | 393.360004 | 436.607987 | 356.671989 | 353.376001 | 1.11x |
| 4096.0 | 1.0 | 182.912007 | 178.240001 | 170.719996 | 134.143993 | 1.36x |
| 4096.0 | 4.0 | 789.855987 | 708.959997 | 639.872015 | 514.335990 | 1.54x |
| 4096.0 | 8.0 | 1596.480012 | 1403.360009 | 1286.655962 | 1015.807986 | 1.57x |
| 4096.0 | 16.0 | 3253.151894 | 2793.823957 | 2512.943983 | 1997.248054 | 1.63x |
| 5376.0 | 1.0 | 429.807991 | 410.656005 | 388.687998 | 279.215991 | 1.54x |
| 5376.0 | 4.0 | 1744.655967 | 1598.111987 | 1522.400022 | 1066.975951 | 1.64x |
| 5376.0 | 8.0 | 3802.464008 | 3231.391907 | 2953.616023 | 2087.712049 | 1.82x |
| 5376.0 | 16.0 | 7414.112091 | 6507.071972 | 6013.983965 | 4134.208202 | 1.79x |
| 8192.0 | 1.0 | 1639.104009 | 1330.368042 | 1298.943996 | 872.255981 | 1.88x |
| 8192.0 | 4.0 | 6725.759983 | 5152.768135 | 5143.840075 | 3532.511950 | 1.90x |
| 8192.0 | 8.0 | 13215.919971 | 10984.895706 | 10224.960327 | 7009.376049 | 1.89x |
| 8192.0 | 16.0 | 26959.903717 | 21830.991745 | 22254.367828 | 13725.728035 | 1.96x |
| 14336.0 | 1.0 | 8705.151558 | 7718.751907 | 7038.047791 | 4331.791878 | 2.01x |
| 14336.0 | 4.0 | 39449.647903 | 31667.232513 | 29968.511581 | 18051.200867 | 2.19x |
| 14336.0 | 8.0 | 86586.334229 | 62748.958588 | 66641.372681 | 37815.328598 | 2.29x |
| 14336.0 | 16.0 | 174014.846802 | - | 151928.802490 | 78183.616638 | 2.23x |

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

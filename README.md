<div align="center">
  <p align="center">

  <picture>
    <img alt="Flash Float JIT KERNEL" src="https://raw.githubusercontent.com/yiakwy-xpu-ml-framework-team/flash-float-jit-kernels/main/assets/img/FlashFloat.png" width="50%">

  </p>
  <h3>Flash Float Ultra Low Latency Hardware Aware Decoding JIT Kernel Library</h3>
  <a href="#cite-us">📝 Papers</a> | <a href="#QuickStart">🚀 Quick Start</a> | <a href="#support-dits">🎯 Supported Flash-Float-JIT-Kernel</a> | <a href="#dev-guide">📚 Dev Guide </a> | <a href="https://github.com/yiakwy-xpu-ml-framework-team/flash-float-jit-kernels/discussions">📈  Discussion </a> | <a href="#Highlight">📝 Highlight </a></strong>
  <p></p>


</div>

This repository is complementary of "FlashFloat" with JIT Kernels

<h2 id="highlight"> Highlight </h2>

- April 27 2026, [🔥 ultra low latency topk , reduced maximum 50% 🚀 batch=1 latency for 1-M context 🎯!](#Ultra-Low-Latency-TopK-Indexer)


<h2 id="Ultra-Low-Latency-TopK-Indexer">🔥 Ultra Low Latency TopK Indexer</h2>

The introduction of the NSA (Native Sparse Attention) mechanism in DeepSeek V3.2 has become pivotal for mitigating inference latency in long-context language modeling. While the NSA top-k indexer playing a critical role in reducing computational overhead for DeepSeek V32 long context sequence modeling task [1], DeepSeek V4 [2] [3], recently, further pushes the context window limit to 1 million tokens where selecting top-2048 dimensions in hybrid sparse attention along from up to 1-M context is a prohibitive bottleneck up to **0.1** ms per layer per query token for agentic workflow.

Leveraging the latest SGLang (2026.3) as the benchmark, we investigated the root causes of this latency bottleneck: kernels for conventional throughput-optimized GPU designs suffer from low device utilization in low-batch yet long context decoding scenarios, due to insufficient inter-block coordination [4]. For example, on-chip network has been maturely adopted for many years in the processors such as Graphcore IPU, Cerebras WSE and Groq LPU, but only being introduced into Hopper lately since 2022. By exploiting the limited on-chip communication capabilities of the Hopper architecture  (up to 8 blocks per cluster), through hardware-aware algorithm-hardware co-design, we achieve more than **50%** latency reduction in low-batch yet long context decoding scenarios, demonstrating the effectiveness of synergistic optimization and **NoC** for long-context inference:

**H100/H800**

<picture>
  <img alt="dist-radix-radix-topk-indexer" src="https://raw.githubusercontent.com/yiakwy-xpu-ml-framework-team/flash-float-jit-kernels/main/assets/img/distributed-radix-topk-indexer.png">
</picture>

**GB300/B300**

| Sequence Length (L) | torch.topk (ms) | sgl fast_topk_v2 (ms) | flash fast_topk_v3 (ms) | vs. sgl (Speedup) | vs. torch (Speedup) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 16,384 | 0.0880 | 0.0144 | 0.0144 | 1.00x | 6.11x |
| 65,536 | 0.0879 | 0.0374 | 0.0182 | 2.05x | 4.81x |
| 98,304 | 0.0878 | 0.0457 | 0.0205 | 2.23x | 4.28x |
| 120,000 | 0.0976 | 0.0492 | 0.0236 | 2.09x | 4.14x |
| 262,144 | 0.1148 | 0.1106 | 0.0369 | 2.99x | 3.11x |
| 524,288 | 0.1536 | 0.1726 | 0.0620 | 2.79x | 2.48x |
| 1,048,576 | 0.2602 | 0.3753 | 0.1107 | 3.39x | 2.35x |

<br />

We hence propose **Distributed Radix Sort via NoC** to extremely reduce decoding latency in workload of low batch size, yet with up to **1-M** context length. 

- First we compute histogram in parallel to reduce collision rates per block and then accumulate the histogram via NoC network before N-ways prefix sum and prove this is an effective method to reduce latency for a throughput oriented hardware design. 

- Second, we enhance the linear mapping properties for radix sort in **NSA** problem for reduction of radix sorting iterations; instead of traditional top **8/11/13** bits [9] of IEEE FP32, FP16 format, we redesign the linear mapping such that $bin(x) >= bin(y)$, naturally deducing $x >= y$. 

  With this linear mapping design, we greatly reduced per block elements dropped in the threshold bin in radix sorting scheme and greatly reduce the residual numbers in later rounds.
  
  This further facilitate cache friendly re-visiting over 1-M context length : we hence enable less **SMEM** in revisiting more elements.
<br/>

- Finally, when remainder elements reduced to **8**/**16**, we can simply use **CAS** operations to perform a **neat parallel sorting** in few cycles. This further reduce the latency overhead in the last round.
<br/>

Previously, L2 cache was commonly used in NVGPU/AMD GPU to tackle the problem, for example in MoE Multi Block Block Size Align Sort Algorithm published 2025 [5], we tackle this problem by introducing mathematically equivalent **unaligned parallel prefix sum**. With distributed radix sort, we further prove that on-chip network can further reduce latency of our kernel, facilitating new design of algorithm and software for **1-M** context.

<br/>

#### Compared to other works at the time article is composing

###### TLE DSL Topk (preview article April 09, 2026, community article April 17, 2026)
Triton-TLE (Triton Language Extension) [TopK](https://github.com/flagos-ai/FlagTree/blob/f9a8d23602a65ec5c1af3b117e1faa46fe6f63b7/python/tutorials/tle/deepseek_v32/01-topk_selector.py#L3055) [6] [7], sits on the Pareto frontier of the productivity and performance, filling the gap between our CUDA and other high-level DSL such as triton-Gluon and TileLang. It is highly efficient and elegant to utilize the **DSHMEM** to reduce the cross blocks communications latency.

The triton extension introduce the semantics explicitly visiting remote (tle.remote) blocks within the clusters via block-level device mesh (**tle.device_mesh**), cluster barriers (**tle.distributed_barrier**) and close loop on chip processing. While our native CUDA implementation offers peak performance, this DSL drastically simplifies the implementation complexity compared to manual CUDA coding.

<br />

###### TileLang
Our CUDA implementation stems directly from the SGLang-optimized variant (fast_topk_v2) of the tileLang implementation. Built natively on the **TVM FFI Object** and tvm IR, tileLang significantly reduces CPU overhead and facilitates the rapid adoption of tiling-based programming advantages.

However, both the tileLang DSL and its generated CUDA kernels lack optimizations for cross-block communication via the Network-on-Chip (NoC). By relying on global memory for inter-block synchronization rather than hardware-accelerated features like DSMEM (Distributed Shared Memory), they exhibit significantly higher latency in our benchmarks compared to our CUDA implementation.

On the other hand, tileLang excels in its pipelining mechanism, which enables efficient I/O-compute overlap. This is particularly advantageous in scenarios where the computational load is heavy enough to hide the memory latency of TopK operations. We will further explore these trade-offs and our integration strategies in the discussion of our **ThunderMuon** optimizer.

<br />

###### Triton Gluon


###### Flashinfer TopK


###### TRT-LLM (April 27, 2026)

**Summary**
Distinct from distribution-agnostic algorithm with Monolithic Mapping, The Guess-Verify-Refine (GVR) algorithm [8] optimizes Top-K selection heuristically for DeepSeek-V3.2 on NVIDIA Blackwell GPUs by reducing the number dropping into the target threshold bin each round. 

Pivoting from a blind search to a data-aware prediction, the technique is built upon mathematic properties of Toeplitz matrix structure induced by RoPE in DS32 such that the prediction of radix step "t+1" highly relying on radix step "t".

**Single Pass Optimization**
The kernel of the paper digests up to 60KB shared memory (4x share memory we actually used) and verified on SWE-bench-derived LongSeqTasks dataset to pre-compute candidates. That is if the data distribution failed to pass **canUseHeuristic**, GVR TopK will not be called.

Both SGLang and our implementation deliberately avoid primitives such as `__popc` and `__ballot_sync`. This architectural choice stems from the fact that warp-level voting can incur significant latencies, sometimes exceeding 30,000 cycles in high-contention scenarios [8].

Traditional ballot-based voting relies on expensive bit-shifting operations (a mechanism we analyze in depth in our upcoming "Ultra-Low Bit Precision" paper). Despite these overheads, this pattern remains prevalent in the legacy TRT-LLM codebase:
```c++
    bool is_one = (val >> tid) & 1;
    // voting to count 1s
    unsigned int mask = __ballot_sync(0xFFFFFFFF, is_one); 
    int count = __popc(mask);
    __syncwarp();
```

**Compared to TRT-LLM production codebase**
The whole algorithm is based on the earlier [two stages solution: topKPerRowDecode](https://github.com/NVIDIA/TensorRT-LLM/blame/v1.3.0rc10/cpp/tensorrt_llm/kernels/indexerTopK.cu), see [the details](https://github.com/NVIDIA/TensorRT-LLM/blame/628bb566050d693894ddf22de03581dd101747c3/cpp/tensorrt_llm/kernels/indexerTopK.cu#L743) in TRT-LLM **v1.3.0rc10**, this largely limited its peak performance in 1-M context scenarios.

Recognizing this, the new approach [ballot-free kernel](https://github.com/NVIDIA/TensorRT-LLM/pull/12236) was integrated into TRT-LLM on March 16 2026. This implementation, written in cutedsl, leverages a Multi-CTA architecture with L2-cache-assisted synchronization to eliminate warp-level voting dependencies.

Later a cluster specific multiple CTA supported [version](https://github.com/NVIDIA/TensorRT-LLM/pull/12354) also written in cutedsl, was introduced into TRT-LLM in March 19 2026. The distributed shared memory accessing was supported via inline pseudo assembly PTX: 

```ptx
mapa.shared::cluster.u32 dest_reg, src_shmem_ptr, target_cta_id
```

It’s worth noting that while the research paper still highlights Single-Pass/Single-CTA modes in Chapter **5.1**, the actual production kernels have evolved toward Multi-CTA clusters. However, as noted in Chapter 5.1, using the L2 cache as a synchronization bridge across successive iterations still introduces significant latency, which remains a primary bottleneck for multi-round execution.

## Reference

[1] DeepSeek-AI V3.2 (2025.12). DeepSeek-V32 Technical Report: "Pushing the Frontier of Open Large Language Models", arXiv:2512.02556; Accessed on April 26, 2026

[2] DeepSeek-AI V4 (2026). DeepSeek-V4 Technical Report: "Towards Highly Efficient Million Token Context Intelligence", https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf;  Accessed on April 26, 2026

[3] Dissecting DeepSeek V4 ：https://www.zhihu.com/question/2030963929510310856/answer/2031157557008541232?share_code=1nP5rOshEmo63&utm_psn=2031815957111419327; Accessed on April 26, 2026

[4] SGLang 2026.4 (0.5.10.post2.dev419+g635e922eb), classical throughput optimized design of TopK : https://github.com/sgl-project/sglang/blob/c7878dbb6ddfc9c6721b9db20a876f2718b0e955/sgl-kernel/csrc/elementwise/topk.cu#L448; Accessed on April 26 2026

[5] MoE Align and Sort (2025.3), https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang, LEI (yiak.wy@gmail.com); Accessed on April 26 2026

[6] Triton-TLE TopK faster than Flashinfer : https://zhuanlan.zhihu.com/p/2025245870753429243?utm_source=wechat_session&utm_medium=social&s_r=0, Hui Guo (guohui@baai.ac.cn); Accessed on April 30 2026

[7] Technical Deep Dive : How to Use FlagOS’s New Triton-TLE Language to Build a TopK Selector Faster Than FlashInfer, https://medium.com/@baaiflagopen/technical-deep-dive-how-to-use-flagoss-new-triton-tle-language-to-build-a-topk-selector-faster-97d1c8354953; Accessed on April 30 2026

[8] Guess-Verify-Refine: Data-Aware Top-K for Sparse-Attention Decoding on Blackwell via Temporal Correlation, https://arxiv.org/pdf/2604.22312, Long Cheng, Ritchie Zhao, Timmy Liu, Mindy Li, Xianjie Qiao, Kefeng Duan, Yu-Jung Chen, Xiaoming Chen, Bita Darvish Rouhani, and June Yang; Accessed on April 27 2026

[9] (4-round 11-bit radix select) Parallel top-k algorithms on GPU:  A comprehensive study and new methods, Jingrong Zhang, Akira Naruse, Xipeng Li, and Yong Wang, SC23:  International Conference for High Performance Computing,
Networking, Storage and Analysis, 2023; codes : https://github.com/ZhangJingrong/gpu_topK_benchmark/tree/master/include; accessed on April 30 2026 


## Citation

If you use this codebase, or otherwise find our work valuable, please cite DistRadixTopK2026:

```bibtex
@misc{DistRadixTopK2026,
  title   = {Ultra Low Latency Distributed Radix TopK Indexer Via NoC},
  author  = {LEI WANG, Hao Gu, Hui Guo, Bei Liu, Dongjie Zou, Mingzhe Zheng, Sirui Han, Wei Xue, Qifeng Chen, Yike Guo},
  year    = {2026},
  url     = {https://github.com/yiakwy-xpu-ml-framework-team/flash-float-jit-kernels}
}
```
# GPU JIT Kernel Harnessing Agent Orchestrator (HAO)

A Python-based agent orchestrator that coordinates with a headless opencode
server (REST API) to write, verify, and optimize CUDA/Triton kernels.

## Architecture

```
┌──────────────────────┐  REST API  ┌──────────────────────┐
│  kernel_agent.py     │ ─────────► │  opencode serve      │
│  (Python CLI)        │            │  :8096               │
│                      │ ◄───────── │                      │
│  - session mgmt      │  SSE/REST  │  - reads .opencode   │
│  - prompt driver     │            │  - uses skills       │
│  - 5-stage harness   │            │  - tool calls        │
│  - benchmark         │            │  - code editing      │
│  - optimize loop     │            │                      │
└──────────────────────┘            └──────────────────────┘
        │                                    │
        │ local exec                         │ local exec
        ▼                                    ▼
┌──────────────────────┐            ┌──────────────────────┐
│  Safety Harness      │            │  Kernel Files        │
│  - smoke test        │            │  - .cu (CUDA)        │
│  - shape sweep       │            │  - .py (Triton)      │
│  - stability         │            │  - .metal            │
│  - determinism       │            │                      │
│  - edge cases        │            │                      │
└──────────────────────┘            └──────────────────────┘
```

## Components

### opencode_client.py
Python REST client for the opencode headless server API.
- Session CRUD (`/session`)
- Prompt sending (`/session/:id/message`, `/session/:id/prompt_async`)
- Slash commands (`/session/:id/command`)
- Message polling and SSE events (`/event`)
- File search and diff (`/find`, `/session/:id/diff`)

### kernel_agent.py
The main orchestrator with CLI interface.
- `KernelAgent` class: manages server lifecycle, sessions, prompts
- 5-stage safety harness (standalone, no server needed)
- Benchmark with SOL (Speed-of-Light) analysis
- Optimization loop: write -> verify -> bench -> keep/revert

## Usage

### Standalone (no server)
```bash
# Safety harness only
python tools/kernel_agent.py harness --kernel tools/example_kernel.py

# Benchmark only
python tools/kernel_agent.py benchmark --kernel tools/example_kernel.py -n 4096
```

### With opencode server
```bash
# Start headless server
python tools/kernel_agent.py serve --port 8096

# Full optimization loop (in another terminal)
python tools/kernel_agent.py harness \
    --kernel jit_kernel/thunder_moun.py \
    --task "Optimize symm_gemm for better SOL gap on H800" \
    --max-iter 5 \
    --port 8096

# Session management
python tools/kernel_agent.py session create
python tools/kernel_agent.py session prompt "write a vector add kernel"
```

### Via sandbox
```bash
./sandbox/run.sh headless --port 8096     # Start in sbx
./sandbox/run.sh kernel gemm --max-iter 20  # Run agent
```

## Safety Harness (5 Stages)

From the AutoKernel paper:

| Stage | What It Tests | Example |
|-------|--------------|---------|
| Smoke | Does it run? | Single call with small input |
| Shape sweep | Different sizes | 128, 512, 2048, 63, 4097 |
| Stability | Extreme values | Zeros, 1e4, 1e-6 |
| Determinism | Same output every time? | 3 identical calls |
| Edge cases | Non-power-of-2 | 1023 elements |

## SOL (Speed-of-Light) Analysis

```
T_compute = FLOPs / Peak_FLOPS
T_mem     = Bytes / Peak_BW
t_SOL     = max(T_compute, T_mem)
SOL_gap   = t_actual / t_SOL

Classification:
  T_compute > T_mem  →  compute-bound
  T_compute <= T_mem →  memory-bound
```

Reference: H800 SXM (SuperPod) = 989.5 TFLOPS FP16 tensor, 3352 GB/s HBM3, NVLink 400 GB/s (IB9700 NDR 400G)

## Optimization Loop

```
for i in 1..max_iterations:
    1. prompt opencode to optimize kernel
    2. run safety harness locally
    3. if harness fails → revert, continue
    4. benchmark locally
    5. if faster AND correct → keep
       else → revert
    6. log results to experiments/*.json
```

## Paper Connections

[1] J. Jaber and O. Jaber, "AutoKernel: Autonomous GPU Kernel Optimization via Iterative
Agent-Driven Search," arXiv:2603.21331, 2026.

[2] S. Guo, M. Lin, and T. Yang, "DRTriton: Large-Scale Synthetic Data Driven Reinforcement
Learning for Triton Kernel Generation," arXiv:2603.21465, 2026.

[3] S. K. S. Hari et al., "Improving EHiciency of GPU Kernel Optimization Agents using a DSL
and Speed-of-Light Guidance," arXiv:2603.29010, 2026.

[4] DeepSeek Harness : https://github.com/deepseek-ai/deepseek-harness. Accessed online on Sep 1st, 2026.

[5] Yifan Shi, Wei Zhang, Tianyi Cui, "A Programming Paradigm for Spatiotemporal Composabilit", https://arxiv.org/pdf/2608.25512. Accessed online on Sep 1st, 2026.

# Setup Guide: Flash Float JIT Kernel Development Agent

Last updated: July 2026

---

## Quick Start

### Step 1: Install opencode and Docker sbx

```bash
# Install opencode (see https://opencode.ai for latest)

# Install Docker sbx
# macOS:
brew trust docker/tap && brew install docker/tap/sbx

# Linux (Ubuntu):
curl -fsSL https://get.docker.com | sudo REPO_ONLY=1 sh
sudo apt-get install -y docker-sbx
sudo usermod -aG kvm $USER
newgrp kvm

# Windows:
winget install -h Docker.sbx

# Login (one-time, opens browser)
sbx login
```

### Step 2: Launch opencode in a GPU-ready sandbox

```bash
cd flash-float-jit-kernels

# Launch opencode with the flash kernel kit
sbx run opencode --kit ./sbx-kits/flash-kernel-kit/

# Or give it a name for reconnection
sbx run opencode --kit ./sbx-kits/flash-kernel-kit/ --name flash-dev
```

**What happens:**
1. sbx creates an isolated microVM
2. The kit installs PyTorch + CUDA + Triton (~2 min first time, cached after)
3. Your repo is mounted into the sandbox
4. opencode starts inside, ready to work on kernels

**Reconnect later:**
```bash
sbx run opencode --name flash-dev
```

---

## What opencode Can Do Here

This repo is configured with GPU kernel development knowledge from three papers
(AutoKernel, DRTriton, SOLAR). Ask opencode:

| What you type | What happens |
|--------------|-------------|
| `/kernel-opt optimize the topk kernel` | Runs the full optimization pipeline |
| `write a Triton softmax kernel for shape (4096, 4096)` | Uses the Triton skill to generate the kernel |
| `debug why this kernel gives wrong results` | Uses CUDA/Metal skill + safety harness patterns |
| `profile symm_gemm.cu and report the bottleneck` | Runs SOL analysis and classifies compute/memory-bound |
| `convert this PyTorch matmul to an optimized Triton kernel` | Uses DRTriton patterns (fusion, tiling, autotune) |
| `write a 1-bit quantized GEMM kernel for Apple Silicon` | Uses Metal skill (SIMD groups, StreamK, bit unpacking) |

---

## Managing Sandboxes

```bash
sbx                      # Interactive dashboard
sbx ls                   # List sandboxes
sbx stop flash-dev       # Pause (keeps installed packages)
sbx rm flash-dev         # Delete (wipes everything inside)
sbx exec -it flash-dev bash    # Shell inside sandbox
```

### Clone Mode (Safer)

Agent works in its own Git clone, host repo is read-only:

```bash
sbx run --clone opencode --kit ./sbx-kits/flash-kernel-kit/
```

### Parallel Sandboxes

```bash
sbx run opencode --kit ./sbx-kits/flash-kernel-kit/ --name topk-work
sbx run opencode --kit ./sbx-kits/flash-kernel-kit/ --name gemm-work
```

---

## The Flash Kernel Kit

Located at `sbx-kits/flash-kernel-kit/spec.yaml`, the kit automatically:

| On first launch | On every launch |
|----------------|-------------|
| Installs PyTorch + CUDA 12.4 | Shows GPU info |
| Installs Triton 3.4 | Verifies installed packages |
| Installs SGLang (for benchmarks) | |
| Installs build tools | |

Kit commands:
```bash
sbx kit validate ./sbx-kits/flash-kernel-kit/     # Check validity
sbx kit inspect ./sbx-kits/flash-kernel-kit/      # Show details
```

---

## Where Config Files Live

```
.opencode/
├── opencode.json           # Project config (model, permissions, agent registry)
├── agent/
│   └── kernel-dev.md       # Kernel developer subagent (MANTIS loop)
├── command/
│   └── kernel-opt.md       # /kernel-opt command
└── skills/
    ├── gpu-kernel-dev/     # General GPU optimization + correctness harness
    ├── cuda-kernel/        # CUDA patterns (JIT, WMMA, TMA, bank conflicts)
    ├── triton-kernel/      # Triton patterns (autotuning, epilogue fusion)
    └── metal-kernel/       # Metal patterns (SIMD groups, 1-bit GEMM, StreamK)

sbx-kits/
└── flash-kernel-kit/
    └── spec.yaml           # sbx kit: auto-installs GPU dev environment
```

## Paper References

| Paper | What we took |
|-------|-------------|
| **AutoKernel** | 5-stage safety harness, keep/revert loop, Amdahl's Law prioritization |
| **DRTriton** | CSP-DAG synthetic data, curriculum RL, test-time kernel search |
| **SOLAR** | Speed-of-Light analysis, MANTIS optimization loop, integrity checking |

## Troubleshooting

### sbx: command not found
Install sbx (see Quick Start above).

### Package install fails in sandbox
Check network policy: `sbx policy ls`. Allow pypi.org if blocked.

### Kernel crashes inside sandbox
The crash is contained. Restart: `sbx rm flash-dev && sbx run opencode --kit ./sbx-kits/flash-kernel-kit/ --name flash-dev`

## Cheat Sheet

```bash
# ─── START ───────────────────────────────────────────────────────
sbx run opencode --kit ./sbx-kits/flash-kernel-kit/               # launch
sbx run opencode --kit ./sbx-kits/flash-kernel-kit/ --name dev    # named

# ─── MANAGE ──────────────────────────────────────────────────────
sbx                     # dashboard
sbx ls                  # list
sbx stop dev            # pause
sbx rm dev              # delete
sbx exec -it dev bash   # shell inside

# ─── KIT ─────────────────────────────────────────────────────────
sbx kit validate ./sbx-kits/flash-kernel-kit/
sbx kit inspect ./sbx-kits/flash-kernel-kit/

# ─── INSIDE OPENCODE ─────────────────────────────────────────────
/kernel-opt optimize the topk CUDA kernel
write a Triton matmul kernel for (4096, 4096) with epilogue fusion
profile symm_gemm.cu and report SOL gap
```

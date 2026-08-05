# Kernel Development Agent

A simple CPU-first agent loop for testing and improving Triton kernels.
Based on the AutoKernel + SOLAR papers. 160 lines, runs anywhere.

## Architecture

```
┌──────────┐    edit    ┌──────────┐    test     ┌──────────┐
│  Agent    │ ────────► │  Kernel   │ ──────────► │  5-Stage  │
│  (LLM)    │           │  file     │             │  Harness  │
└──────────┘           └──────────┘             └─────┬─────┘
     ▲                                                │
     │                    ┌──────────┐                │
     └──── (feedback) ────│ results  │◄───────────────┘
                          └──────────┘
```

## Usage

```bash
# Quick test with progress output
python tools/kernel_agent.py --kernel-file tools/example_kernel.py

# Generate snapshot report for mentor review
python tools/kernel_agent.py --kernel-file tools/example_kernel.py --snapshot

# Save snapshot to file
python tools/kernel_agent.py --kernel-file tools/example_kernel.py --snapshot --save report.md

# More iterations
python tools/kernel_agent.py --kernel-file tools/example_kernel.py --max-iterations 20 --snapshot
```

## How to Write a Testable Kernel

Your kernel file needs a function that takes `torch.Tensor` inputs and returns
a `torch.Tensor` output. The agent will call it with different shapes and values.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    # ... kernel code ...

def fused_operator(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """This is the function the agent will test."""
    N = x.numel()
    out = torch.empty(N, dtype=x.dtype)
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    from triton.runtime.interpreter import InterpretedFunction
    InterpretedFunction(my_kernel.fn).run(...)
    return out
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

## Paper Connections

- **AutoKernel**: 5-stage harness, keep/revert loop, consecutive revert stopping
- **SOLAR**: SOL gap analysis (extensible, hooks in later)
- **DRTriton**: Designed to work with LLM-generated kernel edits

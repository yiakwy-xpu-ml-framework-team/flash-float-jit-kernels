"""
Example kernel for kernel_agent.py testing.
A simple vector add — runs on CPU via Triton interpreter.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def fused_operator(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Vector add — the function the agent optimizes."""
    N = x.numel()
    out = torch.empty(N, dtype=x.dtype)
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    from triton.runtime.interpreter import InterpretedFunction
    InterpretedFunction(add_kernel.fn).run(
        x, y, out, N, BLOCK_SIZE, grid=grid, warmup=False,
    )
    return out

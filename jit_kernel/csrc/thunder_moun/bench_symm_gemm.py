from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import torch
import triton
import triton.testing

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jit_kernel.thunder_moun import symm_gemm_block_scaled  # noqa: E402


SEED = 42
DEBUG = False

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


def calculate_diff(m: int, dummy: int) -> None:
    del dummy
    torch.manual_seed(SEED)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    x = torch.arange(m, dtype=torch.float16, device="cuda").view(1, -1) / (
        m - 1
    ) + torch.arange(m, dtype=torch.float16, device="cuda").view(-1, 1) / (m - 1)
    xs_0 = torch.ones((m, triton.cdiv(m, 128)), dtype=torch.float32, device="cuda")
    xs_1 = torch.ones(
        (triton.cdiv(m, 128), triton.cdiv(m, 128)),
        dtype=torch.float32,
        device="cuda",
    )

    x_fp8 = x.to(torch.float8_e4m3fn)
    w_fp8 = x.to(torch.float8_e4m3fn)

    o_torch_ref = x @ x.T
    o_cuda = symm_gemm_block_scaled(x_fp8, w_fp8, xs_0, xs_1)

    torch.testing.assert_close(o_cuda, o_torch_ref, rtol=5e-01, atol=1e-03)
    print(f"PASS: {m}x{m}x{m} CUDA path matches the PyTorch baseline")
    torch.cuda.synchronize()


M = [2048, 4096, 8192]
dummy = [1]
configs = list(itertools.product(M, dummy))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["m", "dummy"],
        x_vals=configs,
        line_arg="provider",
        line_vals=[
            "torch",
            "cuda_muon_symm_gemm",
        ],
        line_names=[
            "torch",
            "cuda_muon_symm_gemm",
        ],
        styles=[
            ("red", "-"),
            ("purple", "-"),
        ],
        ylabel="Latency",
        plot_name="cuda-muon-symm-gemm-performance",
        args={},
    )
)
def benchmark(m: int, dummy: int, provider: str):
    del dummy
    torch.manual_seed(SEED)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    x = torch.randn(m, m, dtype=torch.bfloat16, device="cuda")
    xs_0 = torch.ones((m, triton.cdiv(m, 128)), dtype=torch.float32, device="cuda")
    xs_1 = torch.ones(
        (triton.cdiv(m, 128), triton.cdiv(m, 128)),
        dtype=torch.float32,
        device="cuda",
    )

    x_fp8 = x.to(torch.float8_e4m3fn)
    w_fp8 = x.to(torch.float8_e4m3fn)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        fn = lambda: x @ x.T
    elif provider == "cuda_muon_symm_gemm":
        fn = lambda: symm_gemm_block_scaled(x_fp8, w_fp8, xs_0, xs_1)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    return ms * 1000, min_ms * 1000, max_ms * 1000


if __name__ == "__main__":
    test_configs = [configs[0]] if IS_CI else configs

    for cfg in test_configs:
        print(f"cfg : {cfg}")
        calculate_diff(*cfg)

    print("\n" + "=" * 60)
    if not DEBUG:
        print("Starting performance benchmark...")
        benchmark.run(print_data=True)

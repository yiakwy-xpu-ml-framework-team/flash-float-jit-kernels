import itertools

import torch
import triton

from jit_kernel.triton3_5.gluon.symm_gemm import GluonXXT
from jit_kernel.triton3_5.symm_gemm import XXT


SEED = 42

M = [2048, 4096, 8192]
dummy = [1]

configs = list(itertools.product(M, dummy))

BATCH_BENCH_CONFIGS = list(itertools.product([2048, 4096], [1, 4, 8, 16]))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["m", "dummy"],
        x_vals=configs,
        line_arg="provider",
        line_vals=[
            "triton_impl_ref",
            "triton_tvm_ffi",
            "gluon_native",
            "gluon_tvm_ffi",
        ],
        line_names=[
            "triton_impl_ref",
            "triton_tvm_ffi",
            "gluon_native",
            "gluon_tvm_ffi",
        ],
        styles=[
            ("blue", "-"),
            ("cyan", "-"),
            ("green", "-"),
            ("orange", "-"),
        ],
        ylabel="Latency",
        plot_name="tvm-ffi-symm-gemm-performance",
        args={},
    )
)
def benchmark(m: int, dummy: int, provider) -> None:
    torch.manual_seed(SEED)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    x = torch.randn(m, m, dtype=torch.bfloat16, device="cuda")

    quantiles = [0.5, 0.2, 0.8]

    symm_gemm_op = GluonXXT()

    if provider == "triton_impl_ref":
        fn = lambda: XXT(x, use_tvm_ffi=False)
    elif provider == "triton_tvm_ffi":
        fn = lambda: XXT(x, use_tvm_ffi=True)
    elif provider == "gluon_native":
        fn = lambda: symm_gemm_op(x, use_tvm_ffi=False)
    elif provider == "gluon_tvm_ffi":
        fn = lambda: symm_gemm_op(x, use_tvm_ffi=True)

    # warm up
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    return ms * 1000, min_ms * 1000, max_ms * 1000


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["m", "B"],
        x_vals=BATCH_BENCH_CONFIGS,
        line_arg="provider",
        line_vals=[
            "triton_impl_ref",
            "triton_tvm_ffi",
            "gluon_native",
            "gluon_tvm_ffi",
        ],
        line_names=[
            "triton_impl_ref",
            "triton_tvm_ffi",
            "gluon_native",
            "gluon_tvm_ffi",
        ],
        styles=[
            ("blue", "-"),
            ("cyan", "-"),
            ("green", "-"),
            ("orange", "-"),
        ],
        ylabel="Latency",
        plot_name="tvm-ffi-symm-gemm-batch-performance",
        args={},
    )
)
def benchmark_batch(m: int, B: int, provider) -> None:
    torch.manual_seed(SEED)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    x = torch.randn(B, m, m, dtype=torch.bfloat16, device="cuda")

    quantiles = [0.5, 0.2, 0.8]

    symm_gemm_op = GluonXXT()

    if provider == "triton_impl_ref":
        fn = lambda: XXT(x, use_tvm_ffi=False)
    elif provider == "triton_tvm_ffi":
        fn = lambda: XXT(x, use_tvm_ffi=True)
    elif provider == "gluon_native":
        fn = lambda: symm_gemm_op(x, use_tvm_ffi=False)
    elif provider == "gluon_tvm_ffi":
        fn = lambda: symm_gemm_op(x, use_tvm_ffi=True)

    # warm up
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    return ms * 1000, min_ms * 1000, max_ms * 1000


if __name__ == "__main__":
    benchmark.run(print_data=True)
    benchmark_batch.run(print_data=True)

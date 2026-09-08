import itertools
import os
import time
from typing import Any, Optional, Tuple

import torch
import triton
import triton.language as tl
import triton.testing


from jit_kernel.triton3_5.gluon.power_mel_log_no_tma import GluonPowerMelLog


SEED = 42

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


DEBUG = False


def power_mel_log_einsum(spec, mel, cmvn_mean=None, cmvn_istd=None):
    """
    log(( |spec_real|^2 + |spec_imag|^2 ) * mel.T) + CMVN

    Args:
        spec: complex64 [T, F]
        mel: float32 [M, F]
        out: optional float32 [T, M]
        cmvn_mean / cmvn_istd: optional float32 [M]
    """
    power = spec.real ** 2 + spec.imag ** 2
    power_mel = torch.einsum('tf,mf->tm', power, mel)

    kLogFloor = 1.1920929e-7
    out = torch.log(torch.clamp(power_mel, min=kLogFloor))

    if cmvn_mean is not None and cmvn_istd is not None:
        out = (out - cmvn_mean) * cmvn_istd

    return out, power_mel


COND_THRESHOLD = 1e-2


def calculate_diff(T, F, M, atol=5e-2):
    torch.manual_seed(SEED)
    spec = torch.randn(T, F, device="cuda", dtype=torch.cfloat)
    mel = torch.randn(M, F, device="cuda", dtype=torch.float32)

    ref, power_mel_ref = power_mel_log_einsum(spec, mel)
    ref = ref.cpu()
    mask = (power_mel_ref > COND_THRESHOLD).cpu()

    power_mel_log_op = GluonPowerMelLog()
    cuda_out = power_mel_log_op(spec, mel, None, None).cpu()

    abs_err = (cuda_out - ref).abs()
    diff = abs_err[mask].max().item()

    mean_diff = abs_err[mask].mean().item()
    diff_all = abs_err.max().item()

    ok = diff < atol

    print(f"✅ {T}x{F}x{M} power_mel_log: max_diff={diff:.4f} "
          f"(all-entries={diff_all:.4f}, well-conditioned={mask.double().mean() * 100:.1f}%) ok={ok}")
    return diff, mean_diff, ok


# NOTE (yiakwy) : sampled from 1 hrs video transcribing task
T = [1024, 2048, 9956, 349012]
F = [257, 513]
M = [80]


configs = list(itertools.product(T, F, M))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["T", "F", "M"],
        x_vals=configs,
        line_arg="provider",
        line_vals=[
            "torch",
            "gluon_power_mel_log_ref",
            "gluon_power_mel_log_fp16x3",
            # "cuda_power_mel_log",
        ],
        line_names=[
            "torch",
            "gluon_power_mel_log tf32x3",
            "gluon_power_mel_log fp16x3",
            # "cuda_power_mel_log",
        ],
        styles=[
            ("red", "-"),
            ("blue", "-"),
            ("green", "-"),
            # ("cyan", "-"),
        ],
        ylabel="Latency",
        plot_name="velox-voice-power_mel_log-performance",
        args={},
    )
)
def benchmark(T: int, F: int, M: int, provider) -> None:
    torch.manual_seed(SEED)
    spec = torch.randn(T, F, device="cuda", dtype=torch.cfloat)
    mel = torch.randn(M, F, device="cuda", dtype=torch.float32)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    power_mel_log_op = GluonPowerMelLog()
    power_mel_log_op_fp16 = GluonPowerMelLog(precision="fp16x3")

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        fn = lambda: power_mel_log_einsum(spec, mel)
    elif provider == "gluon_power_mel_log_ref":
        fn = lambda: power_mel_log_op(spec, mel)
    elif provider == "gluon_power_mel_log_fp16x3":
        fn = lambda: power_mel_log_op_fp16(spec, mel)
    elif provider == "cuda_power_mel_log":
        pass

    # warm up
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)

    return ms * 1000, min_ms * 1000, max_ms * 1000


if __name__ == "__main__":
    # Correctness check - simplified for CI
    if IS_CI:
        # Only test one configuration in CI
        test_configs = [configs[0]]
    else:
        test_configs = configs

    for cfg in test_configs:
        print(f"cfg : {cfg}")
        calculate_diff(*cfg)

    print("\n" + "=" * 60)
    if not DEBUG:
        print("Starting performance benchmark...")
        benchmark.run(print_data=True)


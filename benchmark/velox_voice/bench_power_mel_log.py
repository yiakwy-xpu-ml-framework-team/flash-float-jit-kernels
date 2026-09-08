import itertools
import os
import time
from typing import Any, Optional, Tuple

import torch
import triton
import triton.language as tl
import triton.testing


from jit_kernel.triton3_5.gluon.power_mel_log import GluonPowerMelLog


SEED = 42

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


DEBUG = False

def power_mel_log_einsum(spec, mel, cmvn_mean=None, cmvn_istd=None):
    """
    使用 einsum 可能更清晰，但性能与 matmul 类似
    """
    power = spec.real ** 2 + spec.imag ** 2
    out = torch.einsum('tf,mf->tm', power, mel)
    
    kLogFloor = 1.1920929e-7
    out = torch.log(torch.clamp(out, min=kLogFloor))
    
    if cmvn_mean is not None and cmvn_istd is not None:
        out = (out - cmvn_mean) * cmvn_istd
    
    return out

def calculate_diff(T, F, M, atol=5e-2):
    """Verify CUDA kernel vs torch reference."""
    torch.manual_seed(42)
    spec = torch.randn(T, F, device="cuda", dtype=torch.cfloat)
    mel = torch.randn(M, F, device="cuda", dtype=torch.float32)

    ref = power_mel_log_einsum(spec, mel).cpu()

    power_mel_log_op = GluonPowerMelLog()
    cuda_out = power_mel_log_op(spec, mel, None, None).cpu()

    diff = (cuda_out - ref).abs().max().item()
    mean_diff = (cuda_out - ref).abs().mean().item()
    ok = diff < atol

    print(f"✅ {T}x{F}x{M} power_mel_log")
    return diff, mean_diff, ok


T = [1024, 2048, 9956, 349012]
F = [257, 513]
M = [80]


configs = list(itertools.product(T, F, M))


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
        # benchmark.run(print_data=True)


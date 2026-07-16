import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import triton
import triton.testing


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jit_kernel.triton3_5.symm_gemm import thunder_moun_gemm


DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def parse_sizes(value: str) -> List[int]:
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("expected at least one matrix size")
    return sizes


def make_problem(
    m: int,
    k: int,
    dtype: torch.dtype,
    scale_block_size_n: int,
    scale_block_size_k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    xq = x.to(torch.float8_e4m3fn)

    xs_0 = torch.ones(
        (m, triton.cdiv(k, scale_block_size_k)),
        dtype=torch.float32,
        device="cuda",
    )
    xs_1 = torch.ones(
        (triton.cdiv(m, scale_block_size_n), triton.cdiv(k, scale_block_size_k)),
        dtype=torch.float32,
        device="cuda",
    )
    return xq, xq, xs_0, xs_1


def reference(xq_lhs: torch.Tensor, xq_rhs: torch.Tensor) -> torch.Tensor:
    return torch.matmul(xq_lhs.float(), xq_rhs.float().T)


def check_result(
    xq_lhs: torch.Tensor,
    xq_rhs: torch.Tensor,
    xs_0: torch.Tensor,
    xs_1: torch.Tensor,
    split_k: int,
) -> None:
    out = thunder_moun_gemm(xq_lhs, xq_rhs, xs_0, xs_1, SPLIT_K=split_k)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        out.float(),
        reference(xq_lhs, xq_rhs),
        rtol=5e-2,
        atol=1.0,
    )


def bench_one(
    m: int,
    k: int,
    dtype: torch.dtype,
    warmup: int,
    rep: int,
    do_check: bool,
    split_k: int,
    scale_block_size_n: int,
    scale_block_size_k: int,
) -> float:
    xq_lhs, xq_rhs, xs_0, xs_1 = make_problem(
        m,
        k,
        dtype,
        scale_block_size_n,
        scale_block_size_k,
    )

    if do_check:
        check_result(xq_lhs, xq_rhs, xs_0, xs_1, split_k)

    thunder_moun_gemm(xq_lhs, xq_rhs, xs_0, xs_1, SPLIT_K=split_k)
    torch.cuda.synchronize()

    ms = triton.testing.do_bench(
        lambda: thunder_moun_gemm(xq_lhs, xq_rhs, xs_0, xs_1, SPLIT_K=split_k),
        warmup=warmup,
        rep=rep,
    )

    grid_y = triton.cdiv(m // 2, 128)
    print(
        f"thunder_moun m={m:<5d} k={k:<5d} "
        f"grid=({split_k},{grid_y},1) ms={ms:.6f}"
    )
    return float(ms)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark jit_kernel.triton3_5.symm_gemm.thunder_moun_gemm."
    )
    parser.add_argument("--sizes", type=parse_sizes, default=parse_sizes("2048,4096,8192"))
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--dtype", choices=DTYPES.keys(), default="fp16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--split-k", type=int, default=1)
    parser.add_argument("--scale-block-size-n", type=int, default=128)
    parser.add_argument("--scale-block-size-k", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    print("torch", torch.__version__, torch.version.cuda)
    print("triton", triton.__version__)
    print("gpu", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))

    dtype = DTYPES[args.dtype]
    for m in args.sizes:
        k = args.k if args.k is not None else m
        bench_one(
            m=m,
            k=k,
            dtype=dtype,
            warmup=args.warmup,
            rep=args.rep,
            do_check=args.check,
            split_k=args.split_k,
            scale_block_size_n=args.scale_block_size_n,
            scale_block_size_k=args.scale_block_size_k,
        )


if __name__ == "__main__":
    main()

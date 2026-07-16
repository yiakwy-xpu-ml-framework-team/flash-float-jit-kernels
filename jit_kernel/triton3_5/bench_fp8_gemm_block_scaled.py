import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import triton
import triton.testing


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jit_kernel.triton3_5.symm_gemm import fp8_gemm_block_scaled


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


def selected_modes(mode: str) -> Iterable[str]:
    if mode == "both":
        return ("full", "symm")
    return (mode,)


def make_problem(
    m: int,
    n: int,
    k: int,
    mode: str,
    dtype: torch.dtype,
    scale_block_size_n: int,
    scale_block_size_k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device="cuda", dtype=dtype)
    xq = x.to(torch.float8_e4m3fn)

    if mode == "symm":
        wq = xq
        n = m
    else:
        w = torch.randn((n, k), device="cuda", dtype=dtype)
        wq = w.to(torch.float8_e4m3fn)

    xs = torch.ones(
        (m, triton.cdiv(k, scale_block_size_k)),
        dtype=torch.float32,
        device="cuda",
    )
    ws = torch.ones(
        (triton.cdiv(n, scale_block_size_n), triton.cdiv(k, scale_block_size_k)),
        dtype=torch.float32,
        device="cuda",
    )
    return xq, wq, xs, ws


def reference(xq: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
    return torch.matmul(xq.float(), wq.float().T)


def check_result(
    xq: torch.Tensor,
    wq: torch.Tensor,
    xs: torch.Tensor,
    ws: torch.Tensor,
    is_symm: bool,
    scale_block_size_n: int,
    scale_block_size_k: int,
) -> None:
    out = fp8_gemm_block_scaled(
        xq,
        wq,
        xs,
        ws,
        SCALE_BLOCK_SIZE_N=scale_block_size_n,
        SCALE_BLOCK_SIZE_K=scale_block_size_k,
        is_symm=is_symm,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        out.float(),
        reference(xq, wq),
        rtol=5e-2,
        atol=1.0,
    )


def launch_grid_hint(m: int, n: int, mode: str) -> str:
    if mode == "symm":
        grid_m = triton.cdiv(m, 128)
        grid_n = triton.cdiv(n, 128)
        total_tiles = (grid_m * grid_n + grid_m) // 2
    else:
        # The full kernel autotunes between two block-N values, so this is a
        # stable upper-level task count rather than a selected-config detail.
        grid_m = triton.cdiv(m, 64)
        grid_n = triton.cdiv(n, 128)
        total_tiles = grid_m * grid_n
    return f"total_tiles={total_tiles}"


def bench_one(
    mode: str,
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    warmup: int,
    rep: int,
    do_check: bool,
    scale_block_size_n: int,
    scale_block_size_k: int,
) -> float:
    if mode == "symm":
        n = m

    xq, wq, xs, ws = make_problem(
        m,
        n,
        k,
        mode,
        dtype,
        scale_block_size_n,
        scale_block_size_k,
    )
    is_symm = mode == "symm"

    if do_check:
        check_result(
            xq,
            wq,
            xs,
            ws,
            is_symm,
            scale_block_size_n,
            scale_block_size_k,
        )

    fp8_gemm_block_scaled(
        xq,
        wq,
        xs,
        ws,
        SCALE_BLOCK_SIZE_N=scale_block_size_n,
        SCALE_BLOCK_SIZE_K=scale_block_size_k,
        is_symm=is_symm,
    )
    torch.cuda.synchronize()

    ms = triton.testing.do_bench(
        lambda: fp8_gemm_block_scaled(
            xq,
            wq,
            xs,
            ws,
            SCALE_BLOCK_SIZE_N=scale_block_size_n,
            SCALE_BLOCK_SIZE_K=scale_block_size_k,
            is_symm=is_symm,
        ),
        warmup=warmup,
        rep=rep,
    )
    print(
        f"{mode:5s} m={m:<5d} n={n:<5d} k={k:<5d} "
        f"{launch_grid_hint(m, n, mode)} ms={ms:.6f}"
    )
    return float(ms)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark jit_kernel.triton3_5.symm_gemm.fp8_gemm_block_scaled."
    )
    parser.add_argument("--sizes", type=parse_sizes, default=parse_sizes("2048,4096,8192"))
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--dtype", choices=DTYPES.keys(), default="fp16")
    parser.add_argument("--mode", choices=("full", "symm", "both"), default="both")
    parser.add_argument("--symm", action="store_true", help="Shortcut for --mode symm.")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--scale-block-size-n", type=int, default=128)
    parser.add_argument("--scale-block-size-k", type=int, default=128)
    args = parser.parse_args()

    if args.symm:
        args.mode = "symm"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    print("torch", torch.__version__, torch.version.cuda)
    print("triton", triton.__version__)
    print("gpu", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))

    dtype = DTYPES[args.dtype]
    for mode in selected_modes(args.mode):
        for m in args.sizes:
            n = args.n if args.n is not None else m
            k = args.k if args.k is not None else m
            bench_one(
                mode=mode,
                m=m,
                n=n,
                k=k,
                dtype=dtype,
                warmup=args.warmup,
                rep=args.rep,
                do_check=args.check,
                scale_block_size_n=args.scale_block_size_n,
                scale_block_size_k=args.scale_block_size_k,
            )


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import torch
import triton
import triton.testing


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jit_kernel.triton3_5.symm_gemm as symm_gemm


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


def make_input(m: int, k: int, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
    shape = (batch_size, m, k) if batch_size > 1 else (m, k)
    return torch.randn(shape, device="cuda", dtype=dtype)


def make_output(a: torch.Tensor, m: int, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
    shape = (batch_size, m, m) if batch_size > 1 else (m, m)
    return torch.empty(shape, device=a.device, dtype=dtype)


def check_result(a: torch.Tensor, out: torch.Tensor, use_tvm_ffi: bool) -> None:
    out.zero_()
    symm_gemm.XXT(a, out, use_tvm_ffi=use_tvm_ffi)
    torch.cuda.synchronize()

    ref = torch.matmul(a.float(), a.float().transpose(-1, -2))
    torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=5e-1)


def sentinel_check(a: torch.Tensor, out: torch.Tensor) -> None:
    sentinel = -1234.0
    symm_gemm.tvm_ffi_modules["XXT"] = None

    out.fill_(sentinel)
    symm_gemm.XXT(a, out, use_tvm_ffi=True)
    torch.cuda.synchronize()
    first_remaining = int((out == sentinel).sum().item())

    out.fill_(sentinel)
    symm_gemm.XXT(a, out, use_tvm_ffi=True)
    torch.cuda.synchronize()
    cached_remaining = int((out == sentinel).sum().item())

    print(
        f"sentinel: first_remaining={first_remaining}, "
        f"cached_remaining={cached_remaining}, total={out.numel()}"
    )


def bench_one(
    launcher: str,
    m: int,
    k: int,
    batch_size: int,
    dtype: torch.dtype,
    warmup: int,
    rep: int,
    do_check: bool,
    do_sentinel: bool,
) -> float:
    use_tvm_ffi = launcher == "tvm_ffi"
    a = make_input(m, k, batch_size, dtype)
    out = make_output(a, m, batch_size, dtype)

    block_m, block_n, block_k, num_stages, num_warps = symm_gemm._xxt_config(k)
    grid = batch_size * triton.cdiv(m, block_m) * triton.cdiv(m, block_n)

    if do_check:
        check_result(a, out, use_tvm_ffi)

    if do_sentinel:
        if not use_tvm_ffi:
            raise ValueError("--sentinel only applies to --launcher tvm_ffi or both")
        sentinel_check(a, out)

    symm_gemm.XXT(a, out, use_tvm_ffi=use_tvm_ffi)
    torch.cuda.synchronize()

    ms = triton.testing.do_bench(
        lambda: symm_gemm.XXT(a, out, use_tvm_ffi=use_tvm_ffi),
        warmup=warmup,
        rep=rep,
    )
    print(
        f"{launcher:8s} m={m:<5d} k={k:<5d} batch={batch_size:<3d} "
        f"grid=({grid},1,1) block=({num_warps * 32},1,1) "
        f"bm={block_m} bn={block_n} bk={block_k} ms={ms:.6f}"
    )
    return float(ms)


def selected_launchers(name: str) -> Iterable[str]:
    if name == "both":
        return ("native", "tvm_ffi")
    return (name,)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark jit_kernel.triton3_5.symm_gemm.XXT."
    )
    parser.add_argument("--sizes", type=parse_sizes, default=parse_sizes("2048,4096,8192"))
    parser.add_argument("--k", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=DTYPES.keys(), default="fp16")
    parser.add_argument("--launcher", choices=("native", "tvm_ffi", "both"), default="native")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--sentinel", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    print("torch", torch.__version__, torch.version.cuda)
    print("triton", triton.__version__)
    print("gpu", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
    print("USE_TVM_FFI default", symm_gemm.USE_TVM_FFI)

    dtype = DTYPES[args.dtype]
    for launcher in selected_launchers(args.launcher):
        for m in args.sizes:
            bench_one(
                launcher=launcher,
                m=m,
                k=args.k,
                batch_size=args.batch_size,
                dtype=dtype,
                warmup=args.warmup,
                rep=args.rep,
                do_check=args.check,
                do_sentinel=args.sentinel and launcher == "tvm_ffi",
            )


if __name__ == "__main__":
    main()

import torch
import triton

import jit_kernel.triton3_4.symm_gemm as s

SEED = 42


def test(m=2048):

    torch.manual_seed(SEED)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    xq = torch.arange(m, dtype=torch.float16, device="cuda").view(1, -1) / (
        m - 1
    ) + torch.arange(m, dtype=torch.float16, device="cuda").view(-1, 1) / (m - 1)

    x_fp8 = xq.to(torch.float8_e4m3fn)

    xs_0 = torch.ones((m, triton.cdiv(m, 128)), dtype=torch.float32, device="cuda")
    xs_1 = torch.ones(
        (triton.cdiv(m, 128), triton.cdiv(m, 128)), dtype=torch.float32, device="cuda"
    )

    out = torch.empty((m, m), dtype=torch.float16, device="cuda")

    sentinel = float("NaN")

    # first call: populate TVM-FFI cache
    print("first call : ...")
    out.fill_(sentinel)
    torch.cuda.synchronize()

    s.thunder_moun_gemm(x_fp8, x_fp8, xs_0, xs_1, out=out)
    torch.cuda.synchronize()
    remaining = (out == sentinel).sum().item()
    print("[1st Write] sentinel remaining:", remaining)

    assert (
        remaining == 0
    ), f"[1st Write] Expected kernel to overwrite all elements, but {remaining} remained sentinel"

    sentinel = float("-1234")

    # second call: cached TVM-FFI path only
    print("second call : ...")
    out.fill_(sentinel)
    torch.cuda.synchronize()

    s.thunder_moun_gemm(x_fp8, x_fp8, xs_0, xs_1, out=out)
    torch.cuda.synchronize()
    remaining = (out == sentinel).sum().item()

    print("[2rd Write], sentinel remaining:", remaining)

    assert (
        remaining == 0
    ), f"[2rd Write] Expected kernel to overwrite all elements, but {remaining} remained sentinel"


if __name__ == "__main__":
    test()

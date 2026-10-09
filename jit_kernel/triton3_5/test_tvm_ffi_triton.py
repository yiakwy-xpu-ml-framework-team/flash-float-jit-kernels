import torch

import jit_kernel.triton3_5.symm_gemm as s

SEED = 42


def test(m=2048, k=2048):
    torch.manual_seed(SEED)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    ref = torch.matmul(x.float(), x.float().T).to(torch.float16)
    out = torch.empty((m, m), dtype=torch.float16, device="cuda")

    s.tvm_ffi_modules["XXT"] = None

    sentinel = -1234.0

    # first call: compile, populate TVM-FFI cache, and launch once
    print("first call : ...")
    out.fill_(sentinel)
    torch.cuda.synchronize()

    s.XXT(x, out=out, use_tvm_ffi=True)
    torch.cuda.synchronize()
    remaining = (out == sentinel).sum().item()
    print("[1st Write] sentinel remaining:", remaining)

    assert (
        remaining == 0
    ), f"[1st Write] Expected kernel to overwrite all elements, but {remaining} remained sentinel"
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=1e-1)

    # second call: cached TVM-FFI path
    print("second call : ...")
    out.fill_(sentinel)
    torch.cuda.synchronize()

    s.XXT(x, out=out, use_tvm_ffi=True)
    torch.cuda.synchronize()
    remaining = (out == sentinel).sum().item()
    print("[2nd Write] sentinel remaining:", remaining)

    assert (
        remaining == 0
    ), f"[2nd Write] Expected cached kernel to overwrite all elements, but {remaining} remained sentinel"
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=1e-1)


if __name__ == "__main__":
    test()

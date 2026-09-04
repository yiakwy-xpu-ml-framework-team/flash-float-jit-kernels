import torch
import triton

import jit_kernel.triton3_5.gluon.symm_gemm as s

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

    symm_gemm_op = s.GluonXXT()
    sentinel = float("NaN")

    # first call: native Gluon launch
    print("first call : ...")
    out.fill_(sentinel)
    torch.cuda.synchronize()

    out = symm_gemm_op(xq, out)
    torch.cuda.synchronize()
    remaining = (out == sentinel).sum().item()
    print(f"[1st Write] sentinel ({sentinel}) remaining:", remaining)

    assert (
        remaining == 0
    ), f"[1st Write] Expected kernel to overwrite all elements, but {remaining} remained sentinel ({sentinel})."

    # reset the op
    # symm_gemm_op = s.GluonXXT()

    sentinel = float("-1234")

    # second call: native Gluon cache workaround
    print("second call : ...")
    out.fill_(sentinel)
    torch.cuda.synchronize()

    # out = symm_gemm_op(xq, out=out)
    out = symm_gemm_op(xq)
    torch.cuda.synchronize()
    remaining = (out == sentinel).sum().item()

    print(f"[2rd Write] sentinel ({sentinel}) remaining:", remaining)

    assert (
        remaining == 0
    ), f"[2rd Write] Expected kernel to overwrite all elements, but {remaining} remained sentinel ({sentinel})."


def test_tvm_ffi(m=2048):

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

    symm_gemm_op = s.GluonXXT()
    s.tvm_ffi_modules["XXT"] = None
    sentinel = float("NaN")

    # first call: populate TVM-FFI cache
    print("first call : ...")
    out.fill_(sentinel)
    torch.cuda.synchronize()

    out = symm_gemm_op(xq, out, use_tvm_ffi=True)
    torch.cuda.synchronize()
    remaining = (out == sentinel).sum().item()
    print(f"[1st Write] sentinel ({sentinel}) remaining:", remaining)

    assert (
        remaining == 0
    ), f"[1st Write] Expected kernel to overwrite all elements, but {remaining} remained sentinel ({sentinel})."

    cache = s.tvm_ffi_modules["XXT"]
    assert cache is not None and len(cache) == 1
    cached_module = next(iter(cache.values()))[0]

    # reset the op
    # symm_gemm_op = s.GluonXXT()

    sentinel = float("-1234")

    # second call: cached TVM-FFI path only
    print("second call : ...")
    out.fill_(sentinel)
    torch.cuda.synchronize()

    # out = symm_gemm_op(xq, out=out, use_tvm_ffi=True)
    out = symm_gemm_op(xq, use_tvm_ffi=True)
    torch.cuda.synchronize()
    remaining = (out == sentinel).sum().item()

    print(f"[2rd Write] sentinel ({sentinel}) remaining:", remaining)

    assert (
        remaining == 0
    ), f"[2rd Write] Expected kernel to overwrite all elements, but {remaining} remained sentinel ({sentinel})."

    assert len(s.tvm_ffi_modules["XXT"]) == 1
    assert next(iter(s.tvm_ffi_modules["XXT"].values()))[0] is cached_module


if __name__ == "__main__":
    test()
    test_tvm_ffi()

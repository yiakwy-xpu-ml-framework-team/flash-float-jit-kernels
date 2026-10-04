"""Benchmark & correctness harness for the Distributed Radix TopK indexer.

Covers:
  * flash-float ``fast_topk_v3`` (this repo, NoC distributed radix, top512/1024/2048)
  * sglang ``fast_topk_v2`` (reference sibling implementation, top2048 only)
  * DeepSelect (``deep_select.topk``, external comparison baseline)
  * ``torch.topk`` ground truth

Score domain
------------
The indexer was initially designed for DeepSeek V3.2 normalized attention scores, 
i.e. the *raw* score lives in ``[0, 1]``. The radix mapper (``to_key``) maps that domain
onto an 8-bit monotonic key, so all providers here are fed ``[0, 1]`` scores.

The row stride of a DeepSelect input must be a multiple of 1024 bytes
(``get_stride_requirement()[0]``); we therefore materialize DeepSelect inputs in
a padded, stride-aligned buffer.
"""

import itertools
import os
from typing import Optional

import torch
import triton
import triton.testing
from sgl_kernel import fast_topk_v2

from jit_kernel.topk_indexer import fast_topk_v3

SEED = 42
MAX_SEQ_LEN = 131072

USE_TORCH_ORI = True

# DeepSelect (Sm90a) from https://github.com/yiakwy-xpu-ml-framework-team/DS-DeepSelect-fork
try:
    import deep_select

    HAS_DEEP_SELECT = True
except Exception as _e:  # pragma: no cover - depends on the environment
    print(f"[bench] deep_select unavailable, skipping baseline: {_e}")
    HAS_DEEP_SELECT = False


# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


VALUE_ATOL = float(os.getenv("TOPK_VALUE_ATOL", "1e-3"))
MAX_PERMIT_ERROR = int(os.getenv("TOPK_MAX_PERMIT_ERROR", "5"))


# ---------------------------------------------------------------------------
# Score distributions (all defined on the native [0, 1] domain)
# ---------------------------------------------------------------------------
def _dist_uniform(n: int, dev) -> torch.Tensor:
    return torch.rand(n, device=dev)


def _dist_sigmoid(n: int, dev) -> torch.Tensor:
    # typical post-sigmoid / normalized attention scores
    return torch.sigmoid(torch.randn(n, device=dev))


def _dist_beta(n: int, dev) -> torch.Tensor:
    # U-shaped: mass near both 0 and 1
    return torch.distributions.Beta(0.5, 0.5).sample((n,)).to(dev)


def _dist_power(n: int, dev) -> torch.Tensor:
    # heavily skewed towards small scores
    return torch.rand(n, device=dev).pow(6)


def _dist_bimodal(n: int, dev) -> torch.Tensor:
    u = torch.rand(n, device=dev)
    return torch.where(
        u < 0.5, 0.25 * torch.rand(n, device=dev), 0.75 + 0.25 * torch.rand(n, device=dev)
    )


def _dist_gaussian(n: int, dev) -> torch.Tensor:
    # matches the original benchmark: randn normalized to [0, 1]
    x = torch.randn(n, device=dev)
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def _dist_sparse(n: int, dev) -> torch.Tensor:
    # ~98% small continuous background + ~2% large continuous spikes. 
    spikes = (torch.rand(n, device=dev) < 0.02).float()
    return spikes * (0.5 + 0.5 * torch.rand(n, device=dev)) + (
        1.0 - spikes
    ) * (0.01 * torch.rand(n, device=dev))


DISTRIBUTIONS = {
    "gaussian": _dist_gaussian,
    "uniform": _dist_uniform,
    "sigmoid": _dist_sigmoid,
    "beta": _dist_beta,
    "power": _dist_power,
    "bimodal": _dist_bimodal,
    "sparse": _dist_sparse,
}


def make_scores(bs, seq_len, has_row_starts, dist="uniform", device="cuda"):
    """Build a (bs, L) score tensor in [0, 1] plus `lengths` and `row_starts`."""
    L = MAX_SEQ_LEN if has_row_starts else seq_len

    # native distribution is [0, 1]; the indexer's 8-bit radix key uses the
    # linear 0..255 mapping, so scale to [0, 255] (original convention).
    scores = DISTRIBUTIONS[dist](bs * L, device).reshape(bs, L).clamp(0.0, 1.0) * 255.0

    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device=device)

    if has_row_starts:
        row_starts = torch.randint(0, 2048, (bs,), dtype=torch.int32, device=device)
    else:
        row_starts = None
    return scores, lengths, row_starts


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def _ref_torch_impl_ori(
    score: torch.Tensor,
    seq_len: int,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert score.dim() == 2
    if row_starts is None:
        return torch.topk(score[:, :seq_len], topk, dim=-1, sorted=False).indices
    else:
        ks = row_starts.cpu().tolist()
        ke = (row_starts + seq_len).tolist()

        scores = [score[i, s:e].unsqueeze(0) for i, (s, e) in enumerate(zip(ks, ke))]

        score = torch.cat(scores, dim=0)
        return torch.topk(score, topk, dim=-1, sorted=False).indices


def _ref_torch_impl(
    score: torch.Tensor, 
    seq_len: int, 
    topk: int, 
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if row_starts is None:
        return torch.topk(score[:, :seq_len], topk, dim=-1, sorted=False).indices
    else:
        idx = torch.arange(seq_len, device=score.device)
        idx = idx.unsqueeze(0) + row_starts.unsqueeze(1)
        sliced = torch.gather(score, 1, idx)
        return torch.topk(sliced, topk, dim=-1, sorted=False).indices


# def assert_equal(
#     score: torch.Tensor,
#     indices_ref: torch.Tensor,
#     indices_our: torch.Tensor,
#     bs: int,
#     k: int,
#     seq_len: int,
#     topk_indices_offset: Optional[torch.Tensor] = None,
#     max_permit_error: int = 0,
# ):
#     indices_our_cpu = indices_our.cpu().tolist()
#     indices_ref_cpu = indices_ref.cpu().tolist()

#     wrong_values = 0
#     for i in range(bs):
#         indices_ref_set_i = set(indices_ref_cpu[i])
#         indices_our_set_i = set(indices_our_cpu[i])
#         more = indices_our_set_i - indices_ref_set_i
#         less = indices_ref_set_i - indices_our_set_i

#         offset = topk_indices_offset[i].item() if topk_indices_offset is not None else 0

#         if len(more) > 0 or len(less) > 0:
#             # check whether more values are the same with less values
#             # if so, either one is acceptable, since their values are the same
#             more_values = sorted(score[i, idx - offset].item() for idx in more)
#             less_values = sorted(score[i, idx - offset].item() for idx in less)
#             if more_values != less_values:
#                 wrong_values += len(more)
#                 print(
#                     f"{bs=}, {k=}, {seq_len=}, {i=}, {more=}, {less=} failed, with {more_values=}, {less_values=}"
#                 )
#         assert wrong_values <= max_permit_error, f"{wrong_values=}, {max_permit_error=}"


def assert_equal(
    score: torch.Tensor,
    indices_ref: torch.Tensor,
    indices_our: torch.Tensor,
    bs: int,
    k: int,
    seq_len: int,
    row_starts: Optional[torch.Tensor] = None,    
    max_permit_error: int = 0,
    value_atol: float = 0.0,
    tag: str = "",
):
    # indices_our_cpu = indices_our.cpu().tolist()
    # indices_ref_cpu = indices_ref.cpu().tolist()

    starts = row_starts.cpu().tolist() if row_starts is not None else [0] * bs

    wrong_values = 0
    max_abs_diff = 0.0
    for i in range(bs):
        sc = score[i, starts[i] : starts[i] + seq_len]

        ref_vals = torch.sort(sc[indices_ref[i].long()].float()).values
        our_vals = torch.sort(sc[indices_our[i].long()].float()).values

        diff = (ref_vals - our_vals).abs()

        max_abs_diff = max(max_abs_diff, float(diff.max().item()) if diff.numel() else 0.0)
        wrong_values += int((diff > value_atol).sum().item())
    assert wrong_values <= max_permit_error, (
        f"[{tag}] wrong_values={wrong_values} > {max_permit_error} "
        f"(value_atol={value_atol}, max_abs_diff={max_abs_diff:.3e})"
    )
    return wrong_values, max_abs_diff


# def calculate_diff(bs, k, seq_len, has_row_starts):
#     torch.manual_seed(SEED)

#     stream = torch.cuda.Stream()
#     torch.cuda.set_stream(stream)

#     if has_row_starts:
#         score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
#     else:
#         score = torch.randn(bs, seq_len, dtype=torch.float32, device="cuda")

#     score_max = score.max()
#     score_min = score.min()

#     score = (score - score_min) / (score_max - score_min + 1e-6) * 255

#     # score = torch.arange(MAX_SEQ_LEN, dtype=torch.float32, device="cuda").view(1, -1).expand(bs, -1)

#     lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")

#     if has_row_starts:
#         row_starts = torch.randint(0, 2048, (bs,), dtype=torch.int32, device="cuda")
#     else:
#         row_starts = None

#     if USE_TORCH_ORI:
#         indices_ref = _ref_torch_impl_ori(score, seq_len, k, row_starts=row_starts)
#     else:
#         indices_ref = _ref_torch_impl(score, seq_len, k, row_starts=row_starts)

#     indices_old = fast_topk_v2(score, lengths, k, row_starts=row_starts)

#     indices_our = fast_topk_v3(score, lengths, k, row_starts=row_starts)

#     # sort and compare
#     indices_ref = torch.sort(indices_ref, dim=-1).values
#     indices_old = torch.sort(indices_old, dim=-1).values
#     indices_our = torch.sort(indices_our, dim=-1).values

#     # Tests can pass with max_permit_error=3, set to 5 for safety
#     # assert_equal(score, indices_ref, indices_old, bs, k, seq_len, max_permit_error=5)

#     assert_equal(score, indices_ref, indices_our, bs, k, seq_len, max_permit_error=5)


# ---------------------------------------------------------------------------
# DeepSelect input preparation (stride aligned, window relative indices)
# ---------------------------------------------------------------------------
_DS_ALIGN_ELEMS = None


def _ds_align_elems(dtype: torch.dtype) -> int:
    global _DS_ALIGN_ELEMS
    if _DS_ALIGN_ELEMS is None:
        _DS_ALIGN_ELEMS = deep_select.get_stride_requirement()[0] // torch.tensor(
            [], dtype=dtype
        ).element_size()
    return _DS_ALIGN_ELEMS


def _deepselect_input(score, seq_len, row_starts):
    align = _ds_align_elems(score.dtype)
    stride = ((seq_len + align - 1) // align) * align
    padded = torch.empty((score.shape[0], stride), device=score.device, dtype=score.dtype)
    if row_starts is None:
        padded[:, :seq_len] = score[:, :seq_len]
    else:
        idx = torch.arange(seq_len, device=score.device).unsqueeze(0) + row_starts.unsqueeze(1)
        padded[:, :seq_len] = torch.gather(score, 1, idx)
    return padded[:, :seq_len]


def _run_deepselect(inp, k):
    return deep_select.topk(
        inp, k, sorted_index=False, indices_type=torch.int32, return_value=False
    )[1]


def calculate_diff(bs, k, seq_len, has_row_starts, dist="uniform"):
    torch.manual_seed(SEED)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    score, lengths, row_starts = make_scores(bs, seq_len, has_row_starts, dist)

    if USE_TORCH_ORI:
        indices_ref = _ref_torch_impl_ori(score, seq_len, k, row_starts=row_starts)
    else:
        indices_ref = _ref_torch_impl(score, seq_len, k, row_starts=row_starts)

    tag = f"v3 {dist}"
    indices_our = fast_topk_v3(score, lengths, k, row_starts=row_starts)
    n_v3, diff_v3 = assert_equal(
        score,
        indices_ref,
        indices_our,
        bs,
        k,
        seq_len,
        row_starts=row_starts,
        max_permit_error=MAX_PERMIT_ERROR,
        value_atol=VALUE_ATOL,
        tag=tag,
    )

    if k == 2048:
        # sglang fast_topk_v2 only supports k==2048
        _ = fast_topk_v2(score, lengths, k, row_starts=row_starts)

    n_ds, diff_ds = 0, 0.0
    if HAS_DEEP_SELECT:
        ds_in = _deepselect_input(score, seq_len, row_starts)
        indices_ds = _run_deepselect(ds_in, k)
        n_ds, diff_ds = assert_equal(
            score,
            indices_ref,
            indices_ds,
            bs,
            k,
            seq_len,
            row_starts=row_starts,
            max_permit_error=MAX_PERMIT_ERROR,
            value_atol=VALUE_ATOL,
            tag=f"ds {dist}",
        )

    print(
        f"  OK  k={k:<5} bs={bs} seq_len={seq_len:<7} row_starts={int(has_row_starts)} "
        f"dist={dist:8s} v3[mismatch={n_v3} maxdiff={diff_v3:.2e}] "
        f"ds[mismatch={n_ds} maxdiff={diff_ds:.2e}]"
    )


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
def _ms(fn, cudagraph=True):
    """Median latency in milliseconds."""
    quantiles = [0.5, 0.2, 0.8]
    if cudagraph:
        r = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)
    else:
        r = triton.testing.do_bench(fn, quantiles=quantiles)
    return r[0] if isinstance(r, (list, tuple)) else r


PROVIDERS = ["torch", "radix_2602", "radix", "deep_select"]


def bench_one(bs, k, seq_len, has_row_starts, dist="uniform"):
    torch.manual_seed(SEED)
    score, lengths, row_starts = make_scores(bs, seq_len, has_row_starts, dist)
    results = {}

    if USE_TORCH_ORI:
        results["torch"] = _ms(
            lambda: _ref_torch_impl_ori(score, seq_len, k, row_starts=row_starts),
            cudagraph=False,
        )
    else:
        results["torch"] = _ms(
            lambda: _ref_torch_impl(score, seq_len, k, row_starts=row_starts)
        )

    results["radix_2602"] = _ms(
        lambda: fast_topk_v3(score, lengths, k, row_starts=row_starts)
    )

    if k == 2048:
        results["radix"] = _ms(
            lambda: fast_topk_v2(score, lengths, k, row_starts=row_starts)
        )

    if HAS_DEEP_SELECT:
        ds_in = _deepselect_input(score, seq_len, row_starts)
        results["deep_select"] = _ms(lambda: _run_deepselect(ds_in, k))

    return results


bs = [1, 2, 4, 8]
k = [512, 1024, 2048]  # we only support 2048 now

# 32k smem
seq_len = [
    16384,
    65536,
    98304,
    120000
]
has_row_starts = [True, False]


configs = list(itertools.product(bs, k, seq_len, has_row_starts))


def run_benchmark():
    global configs

    dist = "gaussian"  # matches the original benchmark workload

    if IS_CI:
        configs = configs[:2]

    header = f"{'bs':>3} {'k':>5} {'seq_len':>8} {'row':>4} " + " ".join(
        f"{p:>12}" for p in PROVIDERS
    )
    print("\n" + "=" * len(header))
    print("Performance (ms, median; '-' = unsupported)")
    print(header)
    print("-" * len(header))

    for bs, k, seq_len, has_row_starts in configs:
        res = bench_one(bs, k, seq_len, has_row_starts, dist)
        row = f"{bs:>3} {k:>5} {seq_len:>8} {int(has_row_starts):>4} "
        for p in PROVIDERS:
            v = res.get(p)
            row += f"{v:>12.5f} " if v is not None else f"{'-':>12} "
        print(row)
        if "deep_select" in res and "radix_2602" in res:
            spd = res["deep_select"] / res["radix_2602"]
            print(f"{'':>23}  speedup vs deep_select: {spd:>6.2f}x")
    print("=" * len(header))


def _print_kernel_provenance():
    """Print which jit_kernel / .cu the run actually uses (guards against a
    stale installed copy shadowing the repo)."""
    try:
        import jit_kernel
        from jit_kernel.utils import KERNEL_PATH

        import pathlib
        import re

        cu = KERNEL_PATH / "csrc" / "topk_indexer" / "topk_indexer_radix.cu"
        src = cu.read_text() if cu.exists() else ""
        print(f"[bench] jit_kernel = {jit_kernel.__file__}")
        print(f"[bench] KERNEL_PATH = {KERNEL_PATH}")
        print(f"[bench] kernel .cu lines={len(src.splitlines())} has_fix={'s_input_b' in src}")

        # Inspect the JIT build dir: a stale build.ninja/.so is the usual cause
        # of "ninja: no work to do" compiling the wrong (old) source.
        try:
            from torch.utils.cpp_extension import _get_build_directory

            bdir = pathlib.Path(_get_build_directory("topk_indexer_radix", verbose=False))
            nj = bdir / "build.ninja"
            sos = list(bdir.glob("*.so"))
            print(f"[bench] ext_build_dir = {bdir}")
            if nj.exists():
                ref = sorted(set(re.findall(r"(\S*topk_indexer_radix\.cu)", nj.read_text())))
                print(f"[bench] build.ninja .cu -> {ref}")
            if sos:
                import os as _os

                so_m = _os.path.getmtime(sos[0])
                cu_m = _os.path.getmtime(cu) if cu.exists() else 0
                print(
                    f"[bench] .so={sos[0].name} so_mtime={so_m:.0f} cu_mtime={cu_m:.0f} "
                    f"{'STALE' if so_m < cu_m else 'fresh'}"
                )
        except Exception as e:  # pragma: no cover
            print(f"[bench] ext-dir check failed: {e}")
    except Exception as e:  # pragma: no cover
        print(f"[bench] provenance check failed: {e}")


if __name__ == "__main__":
    _print_kernel_provenance()
    run_correctness = os.getenv("TOPK_SKIP_CORRECTNESS", "0") != "1"
    run_perf = os.getenv("TOPK_SKIP_PERF", "0") != "1"

    if run_correctness:
        print("=" * 60)
        print("Correctness (score domain [0, 1])")
        print("=" * 60)

        corr_configs = configs

        if IS_CI:
            corr_configs = corr_configs[:1]
        for bs, k, seq_len, has_row_starts in corr_configs:
            for dist in DISTRIBUTIONS:
                calculate_diff(bs, k, seq_len, has_row_starts, dist)

    if run_perf:
        print("\n" + "=" * 60)
        print("Starting performance benchmark...")
        run_benchmark()

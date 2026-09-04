"""
kernel_lint.py — static landmine checks for CUDA/Triton kernels.

check:
  * smem buffer aliasing overlaps 
    - reinterpret_cast regions crossing declared sizes
  * hardcoded cluster rank math
    - 1-D clusters along split-k
  * illegal PTX
    - push only for smem to smem bulk copy
  * gluon/triton 
    - smem allocation inside persistent loops
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

ROOT = pathlib.Path(__file__).parent.parent.resolve()


RE_SMEM_REGION = re.compile(
    r"(?P<var>shmem_[A-Za-z_]+|smem_buffer\s*\+\s*[\w()*+\- ]+)\s*=\s*"
    r"reinterpret_cast<(?P<ty>[^*]+)\*>\(\s*smem_buffer\s*\+\s*(?P<off>[^)]+)\)",
)
RE_ALISED_PTR = re.compile(
    r"OutDtype\s*\*\s*(?P<alias>\w+)\s*=\s*reinterpret_cast<[^>]+>\(\s*(?P<base>shmem_\w+)\s*\)",
)
RE_HARDCODED_RANK = re.compile(
    r"(map_shared_rank|mbar_arrive_cluster_release|cluster\.map_shared_rank)[^;]*,\s*(?P<rank>[0-9]+)\s*[,)]",
)
RE_ILLEGAL_BULK_DIR = re.compile(r"cp\.async\.bulk\.shared::cta\.shared::cluster")
RE_LOOP_LOCAL_SMEM_LIKE = re.compile(
    r"gl\.allocate_shared_memory\([^\n]*\)  #.*note:(?!.*function.scope)|"
    r"gl\.allocate_shared_memory\(",
)


def _read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""


def _find_csrc_headers(start: pathlib.Path) -> list[pathlib.Path]:
    csrc = start / "jit_kernel" / "csrc"
    if csrc.exists():
        out: list[pathlib.Path] = []
        for ext in (".cu", ".h", ".cuh", ".cc"):
            out.extend(csrc.rglob(f"*{ext}"))
        return out
    return []


def lint_smem_alias_overlap(root: pathlib.Path) -> list[dict]:
    """Warn when a reinterpret_cast'ed pointer aliases another smem scratch region
    without an explicit NOTE acknowledging the overlap/lifetime."""
    findings: list[dict] = []
    for hdr in _find_csrc_headers(root):
        text = _read_text(hdr)
        for m in RE_ALISED_PTR.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            window = text[max(0, m.start() - 600):m.start()]
            if "alias" not in window.lower() and "scratch" not in window.lower():
                findings.append({
                    "rule": "smem-alias-overlap",
                    "level": "warn",
                    "detail": (f"{hdr.name}:{line_no} pointer `{m.group('alias')}` aliases "
                               f"`{m.group('base')}` — confirm the aliased region is dead at "
                               "this point and document it (see kernel-debug-probes L1)"),
                    "line": line_no,
                })
    return findings


def lint_hardcoded_cluster_rank(root: pathlib.Path) -> list[dict]:
    """Numeric cluster ranks are only valid for a 1-D cluster along split-k.
    Flag them when not accompanied by the documented guard note."""
    findings: list[dict] = []
    for hdr in _find_csrc_headers(root):
        text = _read_text(hdr)
        if "split_k" not in text:
            continue
        for m in RE_HARDCODED_RANK.finditer(text):
            rank = m.group("rank")
            if rank == "0":
                continue
            line_no = text.count("\n", 0, m.start()) + 1
            window = text[max(0, m.start() - 800):m.start() + 200]
            if "csm == 1" in window or "cluster_size_m == 1" in window or \
               "1-D cluster" in window or "rank == blockIdx.x" in window:
                continue
            findings.append({
                "rule": "hardcoded-cluster-rank",
                "level": "warn",
                "detail": (f"{hdr.name}:{line_no} hardcoded cluster rank {rank} — valid only "
                           "for a 1-D cluster along split-k; either parameterize by "
                           "(blockIdx.y % csm) * split_k or assert csm == 1 in the host "
                           "(kernel-debug-probes L2)"),
                "line": line_no,
            })
    return findings


def lint_illegal_ptx_direction(root: pathlib.Path) -> list[dict]:
    findings: list[dict] = []
    for hdr in _find_csrc_headers(root):
        text = _read_text(hdr)
        if RE_ILLEGAL_BULK_DIR.search(text):
            # The pull form does not exist; only push (dst=cluster, src=cta) is legal.
            for i, line in enumerate(text.splitlines(), 1):
                if "cp.async.bulk.shared::cta.shared::cluster" in line:
                    findings.append({
                        "rule": "illegal-ptx-direction",
                        "level": "error",
                        "detail": (f"{hdr.name}:{i} `cp.async.bulk.shared::cta.shared::cluster` "
                                   "is not a valid state-space combination (ptxas rejects it); "
                                   "smem→smem bulk copies are push-only: use "
                                   ".shared::cluster.shared::cta"),
                        "line": i,
                    })
    return findings


def lint_loop_scoped_async_smem(root: pathlib.Path) -> list[dict]:
    """gluon/triton: allocating shared-memory inside a persistent loop or an
    epilogue branch can clash with an in-flight async TMA read of another buffer
    (compiler does not model async liveness)."""
    findings: list[dict] = []
    tdir = root / "jit_kernel"
    for py in list(tdir.rglob("*.py")):
        if "triton" not in str(py) and "gluon" not in str(py):
            continue
        text = _read_text(py)
        for m in re.finditer(r"for pid in range\([^\n]+\):", text):
            body_start = m.end()
            body = text[body_start:body_start + 6000]
            for alloc in re.finditer(r"gl\.allocate_shared_memory\(", body):
                line_no = text.count("\n", 0, body_start + alloc.start()) + 1
                window = body[max(0, alloc.start() - 400):alloc.end() + 200]
                if "function scope" in window.lower():
                    continue
                findings.append({
                    "rule": "loop-scoped-async-smem",
                    "level": "warn",
                    "detail": (f"{py.name}:{line_no} allocate_shared_memory inside a persistent "
                               "loop body — the compiler does not model in-flight async TMA reads "
                               "as liveness; hoist to function scope unless you proved disjoint "
                               "slots (kernel-debug-probes L1)"),
                    "line": line_no,
                })
    return findings


_RULE_SCANS = (
    lint_smem_alias_overlap,
    lint_hardcoded_cluster_rank,
    lint_illegal_ptx_direction,
    lint_loop_scoped_async_smem,
)


def lint_kernel(kernel_path: str, root: pathlib.Path | None = None) -> list[dict]:
    """Run all rules. `kernel_path` focuses report context; scans cover the csrc
    trees the kernel depends on."""
    root = root or ROOT
    findings: list[dict] = []
    for scan in _RULE_SCANS:
        try:
            findings.extend(scan(root))
        except Exception as exc:  # lint must never break the loop
            findings.append({
                "rule": scan.__name__,
                "level": "info",
                "detail": f"scan error: {exc}",
                "line": 0,
            })
    findings.sort(key=lambda f: ({"error": 0, "fatal": 0, "warn": 1, "info": 2}[f["level"]], f["line"]))
    return findings


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "jit_kernel/thunder_moun.py"
    rows = lint_kernel(target)
    if not rows:
        print("lint: clean")
    for r in rows:
        print(f"[{r['level']:>5}] {r['rule']:>26}  {r['detail']}")
    sys.exit(1 if any(r["level"] in ("error", "fatal") for r in rows) else 0)

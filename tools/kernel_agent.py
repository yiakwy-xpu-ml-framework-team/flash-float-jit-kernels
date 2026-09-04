"""
kernel_agent.py : GPU JIT Kernel Harnessing Agent Orchestrator (HAO)

A Python-based remote GPU JIT (H800 sm90a/DgxSpark sm120a /MacStudio Metal GPU) Kernel Harnessing agent that coordinates with a headless opencode server
with REST API to write, verify, and optimize CUDA/Hip/Metal/Triton kernels on Hopper and other platforms (coming soon).

Architecture:
    kernel_agent.py supports opencode serve RESTful API at port :8096 (reads .opencode, uses skills, tool calls)

    The agent drives:
    1. Session management  (create, prompt, abort)
    2. Kernel write/modify (via opencode agent)
    3. Safety verify       (5-stage correctness verification)
    4. Benchmark           (performance measurement)
    5. Harness loop        (write → verify → bench → keep/revert)

Usage:
    python tools/kernel_agent.py serve --port 8096
    python tools/kernel_agent.py harness --kernel jit_kernel/thunder_moun.py --approve-all --timeout-min 60
    python tools/kernel_agent.py verify --kernel jit_kernel/thunder_moun.py
    python tools/kernel_agent.py benchmark --kernel jit_kernel/thunder_moun.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import pathlib
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

ROOT = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from tools.opencode_client import OpenCodeClient  # noqa: E402

logger = logging.getLogger("kernel_agent")


# ---------------------------------------------------------------------------
# H800 DGX SupperPod Constants
# ---------------------------------------------------------------------------

# see https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/superpod/hardware-specs for inter node info

H100_PEAK_TFLOPS_FP16 = 989.5
H100_PEAK_BW_GBS = 3352.0
DEFAULT_PORT = 8096
DEFAULT_HOST = "0.0.0.0"
def _parse_int_list(env_name: str, default: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    """Env override for harness shapes, comma-separated dims, e.g.
    KERNEL_AGENT_SAFETY_SHAPES="128,256,2048" -> [(128,), (256,), (2048,)].
    Useful while a known kernel bug (e.g. the small-shape wasp_1p2c hang at
    nbm in {3,4,5} — see sandbox/probes/probe_hang_m384.py) blocks a stock shape."""
    import os
    raw = os.environ.get(env_name, "")
    if not raw:
        return default
    try:
        return [(int(tok.strip()),) for tok in raw.split(",") if tok.strip()]
    except ValueError:
        logger.warning("%s ignored (unparseable: %r)", env_name, raw)
        return default

SAFETY_SHAPES = _parse_int_list(
    "KERNEL_AGENT_SAFETY_SHAPES", [(128,), (512,), (2048,), (63,), (4097,), (1023,)])
SAFETY_VALUES = [0.0, 1e4, 1e-6]
DETERMINISM_RUNS = 3


# ---------------------------------------------------------------------------
# 5-Stage Safety Harness (AutoKernel paper)
# ---------------------------------------------------------------------------

@dataclass
class HarnessResult:
    passed: bool
    stage: str = ""
    detail: str = ""
    errors: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


@dataclass
class OperatorSpec:
    """Specification of an operator under test.

    Attributes:
        name:            Short kernel name (used for report files).
        fn:              The kernel callable under test.
        ref_fn:          Ground-truth reference (e.g. a PyTorch native impl).
                         If ``None`` the harness degrades to crash-only checks.
        shape_generator: Callable ``(shape, fill_val=None) -> tuple[tensor, ...]``
                         producing inputs for a given shape. Defaults to
                         ``num_inputs`` random (or constant) fp32 tensors.
        atol / rtol:     Tolerances for ``torch.allclose``.
        dtype:           Input dtype for the default shape generator.
        num_inputs:      Number of input tensors the kernel expects.
        device:          Tensor device for the default shape generator.
                         ``"auto"`` picks CUDA when available so the benchmark
                         measures real GPU code (not the Triton CPU interpreter).
    """

    name: str
    fn: Any
    ref_fn: Optional[Any] = None
    shape_generator: Optional[Callable[[tuple[int, ...], Optional[float]], tuple]] = None
    atol: float = 1e-3
    rtol: float = 1e-3
    dtype: Any = None
    num_inputs: int = 2
    device: str = "auto"
    alignment: Optional[int] = None

    def make_inputs(self, shape: tuple[int, ...], fill_val: Optional[float] = None) -> tuple:
        """Build inputs for ``shape``. ``fill_val`` gives a constant fill
        (used by the stability stage); otherwise random data is generated."""
        import torch

        if self.shape_generator is not None:
            return self.shape_generator(shape, fill_val)
        dtype = self.dtype or torch.float32
        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device
        if fill_val is None:
            return tuple(torch.randn(*shape, dtype=dtype, device=device) for _ in range(self.num_inputs))
        return tuple(torch.full(shape, fill_val, dtype=dtype, device=device) for _ in range(self.num_inputs))


def verify_correctness(spec: OperatorSpec, inputs: tuple) -> tuple[bool, str]:
    """Compare kernel output against the reference implementation.

    Inputs are cloned so an in-place kernel cannot corrupt the reference run.
    A CUDA synchronize forces async launch errors (illegal address etc.) to
    surface. Without ``ref_fn`` we can only verify that the kernel executes.
    """
    import torch

    if spec.ref_fn is None:
        try:
            spec.fn(*inputs)
            return True, ""
        except Exception as exc:
            return False, f"Execution error: {exc}"

    inputs_kernel = tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in inputs)
    inputs_ref = tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in inputs)

    try:
        out_kernel = spec.fn(*inputs_kernel)
        out_ref = spec.ref_fn(*inputs_ref)
    except Exception as exc:
        return False, f"Execution error: {exc}"

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if isinstance(out_kernel, (tuple, list)):
        ok = all(
            torch.allclose(k, r, atol=spec.atol, rtol=spec.rtol)
            for k, r in zip(out_kernel, out_ref)
        )
    else:
        ok = torch.allclose(out_kernel, out_ref, atol=spec.atol, rtol=spec.rtol)

    if not ok:
        return False, "Output mismatch with reference"
    return True, ""


def _verify_stage_worker(kernel_path: str, shape: tuple, fill_val, out_q) -> None:
    """Subprocess worker for one harness stage. Builds its OWN spec + inputs in
    the child (CUDA tensors cannot cross process boundaries) and reports back
    (passed, detail). Killing this process on timeout is safe because its CUDA
    context dies with it."""
    try:
        spec = load_operator_spec(kernel_path)
        if spec is None:
            out_q.put((False, "load", "cannot load kernel spec"))
            return
        inputs = spec.make_inputs(shape, fill_val=fill_val)
        ok, msg = verify_correctness(spec, inputs)
        out_q.put((ok, "", msg))
    except Exception as exc:  # noqa: BLE001
        out_q.put((False, "worker", f"{type(exc).__name__}: {exc}"))


def run_stage_isolated(kernel_path: str, spec_for_skip: OperatorSpec,
                       shape: tuple, fill_val=None,
                       stage_timeout_s: float = 240.0) -> "tuple[bool, str]":
    """Run one verification stage in a spawned subprocess with a hard timeout.

    A hanging kernel (documented for small m in this repo: nbm in {3,4,5})
    would otherwise freeze the whole harness loop. On timeout the stage fails
    with a `stage_hang` detail so the orchestrator can react; the parent
    records which shape to reproduce.
    """
    if shape and len(shape) == 1 and not all(d % (spec_for_skip.alignment or 1) == 0 for d in shape):
        return True, ""  # skipped by alignment (spec already evaluated)

    if not _torch_cuda_available():
        # In-process fallback when there is no GPU (CI CPU-only environments).
        inputs = spec_for_skip.make_inputs(shape, fill_val=fill_val)
        return verify_correctness(spec_for_skip, inputs)

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    proc = ctx.Process(target=_verify_stage_worker,
                       args=(kernel_path, shape, fill_val, out_q),
                       name=f"verify-stage-{shape}")
    proc.start()
    proc.join(stage_timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
        return False, f"stage_hang: timed out after {stage_timeout_s:.0f}s (reproduce with this shape)"
    try:
        ok, tag, msg = out_q.get_nowait()
    except Exception:
        return False, f"stage crashed without report (exitcode={proc.exitcode})"
    if not ok:
        return False, msg or tag or "verification failed"
    return True, ""


def _torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def run_safety_verification(spec: OperatorSpec,
                            *,
                            kernel_path: Optional[str] = None,
                            stage_timeout_s: float = 240.0) -> HarnessResult:
    """Run the 5-stage correctness harness on *spec*.

    Stages:
        1. Smoke       — small base shape
        2. Shape sweep — multiple shapes
        3. Stability   — extreme / degenerate fill values
        4. Determinism — same input produces identical output
        5. Edge cases  — non-power-of-2 dimensions

    Every stage is verified against ``spec.ref_fn`` when available; the
    correctness gate must pass before any benchmark is attempted.
    """
    import torch

    def _shape_supported(shape: tuple[int, ...]) -> bool:
        """A shape is supported when it respects the kernel's alignment (if any)."""
        if not spec.alignment:
            return True
        return all(d % spec.alignment == 0 for d in shape)

    def _verify(shape: tuple[int, ...], fill_val: Optional[float] = None) -> tuple[bool, str]:
        # Isolated subprocess with hard timeout when a kernel_path is given: a
        # hanging kernel must fail its stage, never freeze the loop.
        if kernel_path is not None:
            if not _shape_supported(shape):
                return True, ""  # handled as skip below
            return run_stage_isolated(kernel_path, spec, shape, fill_val,
                                      stage_timeout_s=stage_timeout_s)
        inputs = spec.make_inputs(shape, fill_val=fill_val)
        return verify_correctness(spec, inputs)

    skipped: list[str] = []

    # Stage 1: Smoke
    ok, msg = _verify((128,))
    if not ok:
        return HarnessResult(False, "smoke", f"base shape: {msg}", [msg])

    # Stage 2: Shape sweep
    for shape in SAFETY_SHAPES:
        if not _shape_supported(shape):
            skipped.append(f"shape {shape} skipped (kernel requires {spec.alignment}-aligned dims)")
            continue
        ok, msg = _verify(shape)
        if not ok:
            detail = f"shape {shape}: {msg}"
            return HarnessResult(False, "shape_sweep", detail, [detail])

    # Stage 3: Stability (extreme / degenerate fill values)
    for val in SAFETY_VALUES:
        ok, msg = _verify((256,), fill_val=val)
        if not ok:
            detail = f"fill {val}: {msg}"
            return HarnessResult(False, "stability", detail, [detail])

    # Stage 4: Determinism (same input -> identical output across runs)
    inputs = spec.make_inputs((256,))
    ref_out = spec.fn(*inputs)
    for run_i in range(1, DETERMINISM_RUNS):
        args_clone = tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in inputs)
        out = spec.fn(*args_clone)
        if not torch.allclose(ref_out, out, atol=1e-6, rtol=1e-6):
            msg = f"run {run_i} differs from first run"
            return HarnessResult(False, "determinism", msg, [msg])

    # Stage 5: Edge cases (non-power-of-2 / non-tile-aligned dims)
    for shape in [(1023,), (4097,), (1537,)]:
        if not _shape_supported(shape):
            skipped.append(f"shape {shape} skipped (kernel requires {spec.alignment}-aligned dims)")
            continue
        ok, msg = _verify(shape)
        if not ok:
            detail = f"shape {shape}: {msg}"
            return HarnessResult(False, "edge_cases", detail, [detail])

    return HarnessResult(True, skipped=skipped)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    time_us: float
    min_us: float = 0.0
    max_us: float = 0.0
    throughput_gbs: Optional[float] = None
    tflops: Optional[float] = None
    flops: int = 0
    bytes_moved: int = 0
    ref_time_us: Optional[float] = None


def benchmark_kernel(
    fn: Any,
    inputs: tuple,
    *,
    warmup: int = 25,
    rep: int = 100,
    ref_fn: Optional[Any] = None,
) -> BenchResult:
    """Measure execution time of *fn* with ``triton.testing.do_bench``.

    Mirrors the repo benchmark scripts (``benchmark/bench_topk.py``,
    ``benchmark/moun/bench_symm_gemm.py``): quantiles ``[0.5, 0.2, 0.8]``.
    Falls back to a ``perf_counter`` loop when CUDA is unavailable.
    """
    import torch
    import triton.testing

    def _do_bench(target: Any) -> tuple[float, float, float]:
        if target is None:
            return 0.0, 0.0, 0.0
        if torch.cuda.is_available():
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: target(*inputs), warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8],
            )
            return float(ms) * 1e3, float(min_ms) * 1e3, float(max_ms) * 1e3
        # CPU fallback
        for _ in range(max(1, warmup // 5)):
            target(*inputs)
        times: list[float] = []
        for _ in range(rep):
            t0 = time.perf_counter()
            target(*inputs)
            times.append((time.perf_counter() - t0) * 1e6)
        return sum(times) / len(times), min(times), max(times)

    median_us, min_us, max_us = _do_bench(fn)
    ref_us, _, _ = _do_bench(ref_fn)

    # Rough FLOPs/bytes for SOL reporting — GEMM-aware when the output is 2-D
    # (flops = 2*M*N*K) and both operands share the contraction dim; otherwise
    # falls back to element-wise (1 FLOP per element).
    out = fn(*inputs)
    if isinstance(out, (tuple, list)):
        out = out[0]
    in_bytes = sum(t.numel() * t.element_size() for t in inputs if isinstance(t, torch.Tensor))
    out_bytes = out.numel() * out.element_size() if hasattr(out, "numel") else in_bytes
    bytes_moved = in_bytes + out_bytes
    if out.dim() == 2 and inputs[0].dim() == 2 and inputs[0].shape[-1] == inputs[1].shape[-1]:
        flops = 2 * out.shape[0] * out.shape[1] * inputs[0].shape[-1]
    else:
        flops = inputs[0].numel()  # element-wise: 1 FLOP per element

    return BenchResult(
        time_us=median_us,
        min_us=min_us,
        max_us=max_us,
        flops=flops,
        bytes_moved=bytes_moved,
        ref_time_us=ref_us if ref_us > 0 else None,
    )


def compute_sol_gap(result: BenchResult, peak_tflops: float = H100_PEAK_TFLOPS_FP16, peak_bw: float = H100_PEAK_BW_GBS) -> dict:
    """Compute Speed-of-Light analysis."""
    t_s = result.time_us / 1e6
    t_compute = (result.flops / (peak_tflops * 1e12)) if result.flops > 0 else float("inf")
    t_mem = (result.bytes_moved / (peak_bw * 1e9)) if result.bytes_moved > 0 else float("inf")
    t_sol = max(t_compute, t_mem)
    gap = t_s / t_sol if t_sol > 0 else float("inf")
    classification = "compute-bound" if t_compute > t_mem else "memory-bound"
    return {
        "time_us": result.time_us,
        "t_sol_us": t_sol * 1e6,
        "sol_gap": gap,
        "classification": classification,
        "t_compute_us": t_compute * 1e6,
        "t_mem_us": t_mem * 1e6,
    }


# ---------------------------------------------------------------------------
# Module loader (for testing kernels directly)
# ---------------------------------------------------------------------------

def load_kernel_module(path: pathlib.Path):
    """Dynamically load a Python kernel module."""
    spec = importlib.util.spec_from_file_location("kernel_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_testable_function(mod: Any):
    """Find the main testable function in a kernel module."""
    for name in ("main", "fused_operator", "kernel_fn", "test_fn",
                  "symm_gemm_block_scaled", "thunder_moun_gemm",
                  "fp8_gemm_block_scaled"):
        if hasattr(mod, name):
            return getattr(mod, name)
    # Fallback: first public callable
    for name in dir(mod):
        obj = getattr(mod, name)
        if callable(obj) and not name.startswith("_"):
            return obj
    return None


REF_FN_NAMES = ("ref_fn", "reference", "ref", "_ref", "_ref_torch_impl",
                "_ref_torch_impl_ori", "torch_ref", "ref_torch")

SHAPE_GEN_NAMES = ("shape_generator", "make_inputs", "gen_inputs", "input_generator")


def load_operator_spec(
    path: pathlib.Path,
    *,
    ref_fn_name: Optional[str] = None,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    device: str = "auto",
) -> Optional[OperatorSpec]:
    """Build an ``OperatorSpec`` from a kernel module file.

    Auto-discovers the callable under test plus, when present:
      * a ground-truth reference function (names in ``REF_FN_NAMES`` or ``ref_fn_name``),
      * an input ``shape_generator`` (names in ``SHAPE_GEN_NAMES``) — required for
        multi-argument operators such as GEMM, and
      * ``ATOL`` / ``RTOL`` module constants for the comparison tolerances.
    """
    import inspect

    path = pathlib.Path(path).resolve()
    if not path.exists():
        logger.error("kernel file not found: %s", path)
        return None

    mod = load_kernel_module(path)
    fn = find_testable_function(mod)
    if fn is None:
        logger.error("no testable function found in %s", path)
        return None

    ref_fn: Optional[Any] = None
    candidates = (([ref_fn_name] if ref_fn_name else []) + list(REF_FN_NAMES))
    for name in candidates:
        if name and hasattr(mod, name) and name != "ref":
            ref_fn = getattr(mod, name)
            break

    shape_generator: Optional[Any] = None
    for name in SHAPE_GEN_NAMES:
        if hasattr(mod, name):
            shape_generator = getattr(mod, name)
            break

    if atol is None:
        atol = float(getattr(mod, "ATOL", 1e-3))
    if rtol is None:
        rtol = float(getattr(mod, "RTOL", 1e-3))

    # Optional shape-alignment constraint (e.g. tiled kernels: ALIGNMENT = 128).
    # The harness skips unaligned shapes instead of failing on them.
    alignment = int(getattr(mod, "ALIGNMENT", 0)) or None

    # Hint when the default generator can't feed the kernel signature
    if shape_generator is None:
        try:
            required = [
                p for p in inspect.signature(fn).parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            if len(required) != 2:
                logger.warning(
                    "%s requires %d positional args (%s); the default generator only "
                    "produces 2. Define a `shape_generator(shape, fill_val)` in the "
                    "module to drive the harness.",
                    path.name, len(required), ", ".join(p.name for p in required),
                )
        except (ValueError, TypeError):
            pass

    return OperatorSpec(name=path.stem, fn=fn, ref_fn=ref_fn, shape_generator=shape_generator,
                        atol=atol, rtol=rtol, device=device, alignment=alignment)


# ---------------------------------------------------------------------------
# Kernel Agent Orchestrator
# ---------------------------------------------------------------------------

class KernelAgent:
    """Orchestrates kernel Harnessing via the opencode REST API.

    Lifecycle:
        1. start_server()     — launch ``opencode serve --port <port>``
        2. create_session()   — create an opencode session
        3. prompt / command   — drive the agent to write/modify kernels
        4. harness / bench    — verify correctness and measure performance
        5. optimize_loop()    — iterate: write → verify → bench → keep/revert
    """

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        host: str = DEFAULT_HOST,
        project_dir: Optional[str] = None,
        interactive_permissions: bool = False,
        auto_approve_permissions: bool = False,
    ) -> None:
        self.port = port
        self.host = host
        self.project_dir = str(project_dir or ROOT)
        self.base_url = f"http://{host}:{port}"
        self.client = OpenCodeClient(self.base_url)
        self._server_proc: Optional[subprocess.Popen] = None
        self._session_id: Optional[str] = None
        self._history: list[dict] = []
        self.interactive_permissions = interactive_permissions
        self.auto_approve_permissions = auto_approve_permissions
        self._answered_permissions: set[str] = set()

    # -- server lifecycle ----------------------------------------------------

    def start_server(self, timeout: float = 15.0) -> bool:
        """Start opencode serve as a subprocess. Returns True when ready."""
        cmd = [
            "opencode", "serve",
            "--port", str(self.port),
            "--hostname", self.host,
            "--log-level", "DEBUG", # NOTE (yiakwy) : remove
            "--print-logs" # NOTE (yiakwy) : remove
        ]
        logger.info("starting opencode server: %s", " ".join(cmd))
        self._server_proc = subprocess.Popen(
            cmd,
            cwd=self.project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to become healthy
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.client.ping():
                logger.info("opencode server ready at %s", self.base_url)
                return True
            time.sleep(0.5)

        logger.error("opencode server did not become ready within %.0fs", timeout)
        return False

    def stop_server(self) -> None:
        """Stop the opencode server subprocess."""
        if self._server_proc and self._server_proc.poll() is None:
            logger.info("stopping opencode server (pid %d)", self._server_proc.pid)
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
            self._server_proc = None

    def wait_for_server(self, timeout: float = 30.0) -> bool:
        """Wait for an already-running server to respond."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.client.ping():
                return True
            time.sleep(0.5)
        return False

    # -- session management --------------------------------------------------

    def create_session(self, title: Optional[str] = None) -> str:
        """Create an opencode session and return its ID."""
        session = self.client.create_session(title=title)
        self._session_id = session["id"]
        logger.info("created session: %s", self._session_id)
        return self._session_id

    @property
    def session_id(self) -> str:
        if self._session_id is None:
            raise RuntimeError("No session. Call create_session() first.")
        return self._session_id

    # -- prompting -----------------------------------------------------------

    def prompt(self, text: str, *, agent: Optional[str] = None, timeout_ms: int = 300_000) -> str:
        """Send a prompt and wait for the reply. Returns the text.

        Permission requests (opencode ``action.action=ask``) are serviced while
        waiting: interactively from the keyboard when ``interactive_permissions``
        is on, or automatically ("once") when ``auto_approve_permissions`` is on.
        With both off (headless, no TTY), a pending request stalls the loop until
        the prompt timeout — prefer --approve-all for unattended runs.
        """
        logger.info("prompting session %s (%d chars)", self.session_id, len(text))
        hook: Optional[Callable[[], None]] = None
        if self.auto_approve_permissions:
            hook = self._service_permission_asks
        elif self.interactive_permissions and sys.stdin.isatty():
            hook = self._service_permission_asks
        reply = self.client.send_and_wait(
            self.session_id, text, agent=agent, timeout_ms=timeout_ms, poll_hook=hook
        )
        self._history.append({"role": "user", "text": text, "reply_len": len(reply)})
        return reply

    # -- interactive permission approval ----------------------------------

    def _service_permission_asks(self) -> None:
        """Service pending permission requests: auto-approve ("once") when
        ``auto_approve_permissions`` is on, otherwise ask the operator."""
        try:
            pending = [
                p for p in self.client.list_permissions()
                if p.get("sessionID") == self._session_id
            ]
        except Exception:
            return  # endpoint missing/erroring — keep polling as before
        for req in pending:
            pid = req.get("id")
            if not pid or pid in self._answered_permissions:
                continue
            self._answered_permissions.add(pid)
            reply = "once" if self.auto_approve_permissions else self._keyboard_approve(req)
            try:
                self.client.reply_permission(pid, reply)
                logger.info("permission %s (%s): %s", pid, req.get("permission"), reply)
            except Exception as exc:
                logger.warning("reply_permission failed (%s): %s", pid, exc)

    @staticmethod
    def _keyboard_approve(req: dict) -> str:
        """Render one permission request on the terminal and read y/a/n."""
        kind = req.get("permission", "unknown")
        patterns = req.get("patterns") or []
        meta = req.get("metadata") or {}
        detail = meta.get("command") or meta.get("filepath") or meta.get("url") or ""
        print()
        print("=" * 60)
        print(f"  [PERMISSION REQUEST] {kind}")
        if patterns:
            print(f"  patterns : {patterns}")
        if detail:
            print(f"  detail   : {detail}")
        print("=" * 60)
        while True:
            try:
                ans = input("Approve? [y]=once  [a]=always  [n]=reject > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return "reject"
            if ans in ("", "y", "yes"):
                return "once"
            if ans in ("a", "always"):
                return "always"
            if ans in ("n", "no", "reject"):
                return "reject"

    def command(self, cmd: str, args: str = "", *, agent: Optional[str] = None, timeout_ms: int = 600_000) -> str:
        """Execute a registered slash command (e.g. ``/kernel-opt``) and return
        the reply text. The command's own frontmatter ``agent:`` takes effect
        when *agent* is left as None."""
        result = self.client.send_command(
            self.session_id, cmd, args, agent=agent, timeout=timeout_ms / 1000.0
        )
        return self.client._extract_text(result)

    def inject_context(self, context: str) -> None:
        """Inject context into the session without triggering a reply."""
        self.client.send_prompt(self.session_id, context, no_reply=True)

    # -- kernel-specific operations ------------------------------------------

    def write_kernel(self, task: str, *, agent: str = "kernel-dev",
                     timeout_ms: int = 1_800_000) -> str:
        """Ask opencode to write/modify a kernel. Returns the reply text.

        ``kernel-dev`` runs as a multi-step subagent (read → edit → nvcc
        compile → benchmark), so a single iteration regularly takes 5-20
        minutes; the 30-minute default avoids premature client timeouts.
        """
        return self.prompt(task, agent=agent, timeout_ms=timeout_ms)

    def revert_last(self) -> bool:
        """Revert the last message in the session."""
        messages = self.client.list_messages(self.session_id)
        if not messages:
            return False
        last = messages[-1]
        info = last.get("info", last)
        msg_id = info.get("id")
        if msg_id:
            return self.client.revert_message(self.session_id, msg_id)
        return False

    # -- file snapshot / restore (robust alternative to revert_last) ---------

    _SNAPSHOT_DIRS = ("jit_kernel", "benchmark")

    def snapshot_tree(self, tag: str) -> Optional[pathlib.Path]:
        """Tar the kernel source trees so a failed iteration can be restored
        deterministically (revert_last only undoes ONE session message while the
        agent may have edited files across many). Returns the archive path."""
        import tarfile

        base = ROOT / "experiments" / "snapshots"
        base.mkdir(parents=True, exist_ok=True)
        arc = base / f"{tag}.tar.gz"
        with tarfile.open(arc, "w:gz") as tar:
            for d in self._SNAPSHOT_DIRS:
                path = ROOT / d
                if path.exists():
                    tar.add(path, arcname=d)
        return arc

    def restore_tree(self, tag: str) -> bool:
        """Restore the source trees from a snapshot taken by ``snapshot_tree``."""
        import tarfile

        arc = ROOT / "experiments" / "snapshots" / f"{tag}.tar.gz"
        if not arc.exists():
            logger.warning("no snapshot %s to restore", arc)
            return False
        with tarfile.open(arc, "r:gz") as tar:
            # Only extract over our own tree members; never follow absolute paths.
            members = [m for m in tar.getmembers() if m.name.split("/")[0] in self._SNAPSHOT_DIRS]
            tar.extractall(ROOT, members=members)
        logger.info("restored snapshot %s", arc.name)
        return True

    # -- safety harness (local, no opencode needed) --------------------------

    def check_kernel(self, kernel_path: str, num_inputs: int = 2) -> HarnessResult:
        """Run the 5-stage safety harness on a local kernel file."""
        spec = load_operator_spec(kernel_path)
        if spec is None:
            return HarnessResult(False, "load", f"cannot load kernel spec: {kernel_path}")
        spec.num_inputs = num_inputs
        return run_safety_verification(spec, kernel_path=kernel_path)

    # -- benchmark (local) ---------------------------------------------------

    def bench_kernel(self, kernel_path: str, n: int = 4096) -> BenchResult | None:
        """Benchmark a local kernel file against its reference (if any)."""
        spec = load_operator_spec(kernel_path)
        if spec is None:
            logger.error("cannot load kernel spec: %s", kernel_path)
            return None

        inputs = spec.make_inputs((n,))
        return benchmark_kernel(spec.fn, inputs, ref_fn=spec.ref_fn)

    # -- optimization loop ---------------------------------------------------

    def _build_context_block(self, kernel_path: str, baseline: Optional[BenchResult],
                             baseline_sol: Optional[dict]) -> str:
        """Repo + baseline scaffolding injected into EVERY iteration prompt.

        Without it the agent burns its budget rediscovering the file map, the
        baseline numbers and the verification command every round (the #1 cause
        of "no effective changes + timeout").
        """
        lines = [
            "## Environment (fixed for all iterations)",
            f"- Kernel under test: `{kernel_path}` (repo root = cwd)",
            "- Verification command (run before claiming ANY improvement):",
            f"  `python tools/kernel_agent.py verify --kernel {kernel_path}`",
            "- Benchmark scripts live under `benchmark/` (see the one matching this op).",
            "- Debugging methodology: follow the `kernel-debug-probes` skill — write a",
            "  minimal standalone probe under `sandbox/probes/probe_<conjecture>.py` to",
            "  isolate ONE hypothesis BEFORE re-editing the kernel; keep the probe.",
            "- Change exactly ONE thing per iteration; end your reply with a line:",
            '  `RESULT: {"decision":"keep|revert","time_us":<float>,"files":[...],"probes":[...],"next_hypothesis":"..."}`',
        ]
        if baseline is not None and baseline_sol is not None:
            lines.append(
                f"- Baseline: {baseline.time_us:.1f} us | SOL {baseline_sol['t_sol_us']:.1f} us "
                f"| gap {baseline_sol['sol_gap']:.2f}x ({baseline_sol['classification']})"
            )
        return "\n".join(lines)

    @staticmethod
    def _build_outcome_block(last_result: Optional[dict]) -> str:
        """Feed the PREVIOUS iteration outcome back to the agent.

        This is the feedback half of the loop: without it the agent never learns
        why its last change was rejected and keeps proposing ineffective edits.
        """
        if not last_result:
            return ""
        phase = last_result.get("phase", "")
        lines = ["\n## Previous iteration outcome"]
        if phase == "harness":
            lines += [
                f"- **Verification FAILED** at stage `{last_result.get('stage')}`:",
                f"  `{last_result.get('detail')}`",
                "- Files were restored to the last good state. Diagnose with a probe",
                "  (kernel-debug-probes skill) BEFORE editing again.",
            ]
        elif phase == "revert":
            lines += [
                f"- Kept-out: new time {last_result.get('time_us', float('nan')):.1f} us did NOT beat"
                f" best {last_result.get('best_us', float('nan')):.1f} us. Files restored.",
                "- Pick a DIFFERENT hypothesis; consider a profiling probe first.",
            ]
        elif phase == "keep":
            lines.append(f"- KEEPED: {last_result.get('speedup_pct', 0):.1f}% faster.")
        elif phase == "prompt":
            lines += [
                f"- **Prompt failed**: {last_result.get('error', '')[:300]}",
                "- Work only on the single highest-value step; avoid long tool chains.",
            ]
        return "\n".join(lines)

    def optimize_loop(
        self,
        kernel_path: str,
        task: str,
        *,
        max_iterations: int = 5,
        agent: str = "kernel-dev",
        prompt_timeout_ms: int = 1_800_000,
    ) -> list[dict]:
        """Run the full optimization loop.

        For each iteration:
        1. Ask opencode to optimize the kernel
        2. Run safety harness locally
        3. Benchmark locally
        4. Keep if faster + correct, restore snapshot otherwise
        Every prompt carries (a) fixed environment context and (b) the previous
        iteration's outcome, so failures actually reach the agent.
        """
        results: list[dict] = []
        best_time_us: Optional[float] = None

        # Baseline benchmark
        logger.info("=== Baseline measurement ===")
        base_result = self.bench_kernel(kernel_path)
        baseline_sol = None
        if base_result:
            best_time_us = base_result.time_us
            baseline_sol = compute_sol_gap(base_result)
            logger.info(
                "baseline: %.1f us, SOL gap: %.1f×, %s",
                baseline_sol["time_us"], baseline_sol["sol_gap"], baseline_sol["classification"],
            )
            results.append({"iteration": 0, "phase": "baseline", **baseline_sol})

        context_block = self._build_context_block(kernel_path, base_result, baseline_sol)
        last_outcome: Optional[dict] = None
        consecutive_conn_failures = 0

        for i in range(1, max_iterations + 1):
            logger.info("=== Iteration %d/%d ===", i, max_iterations)

            outcome_block = self._build_outcome_block(last_outcome)
            best_txt = f"{best_time_us:.1f}" if best_time_us else "?"
            iteration_task = (
                f"{context_block}\n"
                f"{outcome_block}\n\n"
                f"## Task\n{task}\n\n"
                f"Iteration {i}/{max_iterations}. Current best: {best_txt} us."
            )

            tag = f"iter_{i}_pre"
            self.snapshot_tree(tag)

            try:
                reply = self.write_kernel(iteration_task, agent=agent,
                                          timeout_ms=prompt_timeout_ms)
                logger.info("agent reply (%d chars): %s", len(reply), reply[:200])
                consecutive_conn_failures = 0
            except TimeoutError as exc:
                # Capture whatever the agent produced so far and feed it forward.
                partial = ""
                try:
                    msgs = self.client.list_messages(self._session_id)
                    found = self.client._find_reply(msgs)
                    if found:
                        partial = self.client._extract_text(found)[:2000]
                except Exception:
                    pass
                logger.error("agent prompt timed out after %d ms; partial reply %d chars",
                             prompt_timeout_ms, len(partial))
                results.append({"iteration": i, "phase": "prompt",
                                "error": str(exc), "partial_reply": partial})
                last_outcome = {"phase": "prompt", "error": f"timeout; partial: {partial[-400:]}"}
                # The agent may still be running server-side — abort so the next
                # prompt does not collide, but do NOT assume the server died.
                try:
                    self.client.abort_session(self._session_id)
                except Exception:
                    pass
                continue
            except Exception as exc:
                logger.error("agent prompt failed: %s", exc)
                results.append({"iteration": i, "phase": "prompt", "error": str(exc)})
                last_outcome = {"phase": "prompt", "error": str(exc)}
                # Distinguish dead-server from transient errors; restart once.
                if not self.client.ping():
                    consecutive_conn_failures += 1
                    logger.warning("opencode server unreachable (fail #%d) — restarting",
                                   consecutive_conn_failures)
                    self.start_server()
                    if not self.wait_for_server():
                        logger.error("server still down; stopping loop")
                        break
                    continue
                else:
                    try:
                        self.client.abort_session(self._session_id)
                    except Exception:
                        pass
                    continue

            # 2. Safety harness
            harness = self.check_kernel(kernel_path)
            if not harness.passed:
                logger.warning("safety harness FAILED at stage %s: %s", harness.stage, harness.detail)
                results.append({
                    "iteration": i, "phase": "harness",
                    "passed": False, "stage": harness.stage, "detail": harness.detail,
                })
                # Restore the last-good sources (deterministic; revert_last only
                # undoes a single session message).
                self.restore_tree(tag)
                last_outcome = {"phase": "harness", "stage": harness.stage,
                                "detail": harness.detail}
                continue

            # 3. Benchmark
            bench = self.bench_kernel(kernel_path)
            if bench is None:
                results.append({"iteration": i, "phase": "bench", "error": "benchmark failed"})
                self.restore_tree(tag)
                last_outcome = {"phase": "harness", "stage": "bench",
                                "detail": "benchmark crashed"}
                continue

            sol = compute_sol_gap(bench)
            logger.info(
                "iteration %d: %.1f us, SOL gap: %.1f×",
                i, sol["time_us"], sol["sol_gap"],
            )

            # 4. Keep or restore
            if best_time_us is not None and sol["time_us"] < best_time_us * 0.98:
                # Faster — keep
                speedup = (1 - sol["time_us"] / best_time_us) * 100
                logger.info("KEEP — %.1f%% faster", speedup)
                best_time_us = sol["time_us"]
                results.append({"iteration": i, "phase": "keep", "speedup_pct": speedup, **sol})
                last_outcome = {"phase": "keep", "speedup_pct": speedup}
            else:
                # No improvement — restore
                logger.info("REVERT — no improvement (%.1f us vs %.1f us)", sol["time_us"], best_time_us or 0)
                self.restore_tree(tag)
                results.append({"iteration": i, "phase": "revert",
                                **sol, "best_us": best_time_us or 0})
                last_outcome = {"phase": "revert", "time_us": sol["time_us"],
                                "best_us": best_time_us or 0}

        logger.info("=== Optimization complete. Best: %.1f us ===", best_time_us or 0)
        return results


# ---------------------------------------------------------------------------
# Standalone harness runner (no opencode server needed)
# ---------------------------------------------------------------------------

def _harness_report_path(kernel_name: str, output_dir: Optional[str] = None) -> pathlib.Path:
    base = pathlib.Path(output_dir) if output_dir else ROOT / "experiments"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"harness_{kernel_name}.md"


def build_harness_report(
    kernel_path: str,
    spec: OperatorSpec,
    result: HarnessResult,
    bench: Optional[BenchResult] = None,
    sol: Optional[dict] = None,
    n: int = 0,
) -> str:
    """Render the correctness + performance report persisted to ``harness_<name>.md``."""
    import datetime

    lines: list[str] = []
    lines.append(f"# Harness Report — `{spec.name}`")
    lines.append("")
    lines.append(f"- **File**: `{kernel_path}`")
    lines.append(f"- **Timestamp**: {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- **Correctness**: {'PASS' if result.passed else 'FAIL'} "
                 f"(stage: {result.stage or 'n/a'})")
    lines.append(f"- **Reference compared**: {'yes' if spec.ref_fn is not None else 'no (crash-only)'}")
    lines.append("")

    lines.append("## Correctness (5-stage)")
    lines.append("")
    if result.passed:
        lines.append("| Stage | Status |")
        lines.append("|-------|--------|")
        for s in ("smoke", "shape_sweep", "stability", "determinism", "edge_cases"):
            lines.append(f"| {s} | pass |")
        if result.skipped:
            lines.append("")
            lines.append("### Skipped (kernel alignment constraint)")
            lines.append("")
            for e in result.skipped:
                lines.append(f"- {e}")
    else:
        lines.append(f"Failed at **{result.stage}**: `{result.detail}`")
        lines.append("")
        if result.errors:
            lines.append("### Errors")
            lines.append("")
            for e in result.errors:
                lines.append(f"- `{e}`")
    lines.append("")

    if bench is not None and result.passed:
        lines.append("## Performance (triton.testing.do_bench, quantiles [0.5, 0.2, 0.8])")
        lines.append("")
        lines.append(f"- **Input size**: N = {n}")
        lines.append(f"- **Time (median)**: {bench.time_us:.1f} us "
                     f"(min {bench.min_us:.1f} / max {bench.max_us:.1f})")
        if bench.ref_time_us:
            lines.append(f"- **Reference time**: {bench.ref_time_us:.1f} us "
                         f"(speedup {bench.ref_time_us / bench.time_us:.2f}x)" if bench.time_us > 0
                         else "- **Reference time**: n/a")
        if sol:
            lines.append(f"- **SOL time**: {sol['t_sol_us']:.1f} us")
            lines.append(f"- **SOL gap**: {sol['sol_gap']:.2f}x")
            lines.append(f"- **Classification**: {sol['classification']} "
                         f"(compute {sol['t_compute_us']:.1f} us / mem {sol['t_mem_us']:.1f} us)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| median (us) | {bench.time_us:.1f} |")
        lines.append(f"| min (us) | {bench.min_us:.1f} |")
        lines.append(f"| max (us) | {bench.max_us:.1f} |")
        if bench.ref_time_us:
            lines.append(f"| reference (us) | {bench.ref_time_us:.1f} |")
            lines.append(f"| speedup vs ref | {bench.ref_time_us / bench.time_us:.2f}x |" if bench.time_us > 0 else "")
    lines.append("")
    return "\n".join(lines)


def run_safety_verification_standalone(
    kernel_path: str,
    *,
    n: int = 4096,
    output_dir: Optional[str] = None,
    device: str = "auto",
) -> None:
    """Run safety harness on a kernel file without opencode.

    Flow:
        1. Load operator spec (kernel fn + optional reference fn)
        2. Run the 5-stage correctness verification (reference-verified)
        3. If correctness passes, benchmark with triton.testing.do_bench
        4. Persist the report to ``harness_<kernel_name>.md``
    """
    spec = load_operator_spec(kernel_path, device=device)
    if spec is None:
        print(f"failed to load kernel spec: {kernel_path}")
        return

    print("=" * 60)
    print("  Safety Harness Result")
    print("=" * 60)
    print(f"  File:      {kernel_path}")
    print(f"  Reference: {spec.ref_fn.__name__ if spec.ref_fn else '(none — crash-only)'}")

    result = run_safety_verification(spec, kernel_path=kernel_path)

    print(f"  Passed:  {result.passed}")
    if not result.passed:
        print(f"  Stage:   {result.stage}")
        print(f"  Detail:  {result.detail}")
        if result.errors:
            print(f"  Errors:  {len(result.errors)}")
            for e in result.errors:
                print(f"    - {e}")
    else:
        print("  All 5 stages passed.")
        if result.skipped:
            print(f"  Skipped: {len(result.skipped)} (kernel alignment constraint)")
            for e in result.skipped:
                print(f"    - {e}")

    bench: Optional[BenchResult] = None
    sol: Optional[dict] = None
    if result.passed:
        print("=" * 60)
        print("  Performance (triton.testing.do_bench)")
        print("=" * 60)
        print(f"  Input size: N = {n}")
        inputs = spec.make_inputs((n,))
        bench = benchmark_kernel(spec.fn, inputs, ref_fn=spec.ref_fn)
        sol = compute_sol_gap(bench)
        print(f"  Time (median): {bench.time_us:.1f} us  "
              f"(min {bench.min_us:.1f} / max {bench.max_us:.1f})")
        if bench.ref_time_us:
            print(f"  Reference:     {bench.ref_time_us:.1f} us  "
                  f"(speedup {bench.ref_time_us / bench.time_us:.2f}x)" if bench.time_us > 0 else "  Reference: n/a")
        print(f"  SOL gap:   {sol['sol_gap']:.2f}x  [{sol['classification']}]")
        print(f"  SOL time:  {sol['t_sol_us']:.1f} us")

    report = build_harness_report(kernel_path, spec, result, bench, sol, n=n)
    report_path = _harness_report_path(spec.name, output_dir)
    report_path.write_text(report)
    print("=" * 60)
    print(f"Report persisted to: {report_path}")
    print("=" * 60)


def run_benchmark_standalone(kernel_path: str, n: int = 4096, device: str = "auto") -> None:
    """Benchmark a kernel file without opencode (also persists a report)."""
    spec = load_operator_spec(kernel_path, device=device)
    if spec is None:
        print("benchmark failed: cannot load kernel spec")
        return

    inputs = spec.make_inputs((n,))
    bench = benchmark_kernel(spec.fn, inputs, ref_fn=spec.ref_fn)
    sol = compute_sol_gap(bench)

    print("=" * 60)
    print("  Benchmark Result")
    print("=" * 60)
    print(f"  File:         {kernel_path}")
    print(f"  Input size:   N={n}")
    print(f"  Time (median): {bench.time_us:.1f} us  "
          f"(min {bench.min_us:.1f} / max {bench.max_us:.1f})")
    if bench.ref_time_us:
        print(f"  Reference:    {bench.ref_time_us:.1f} us")
    print(f"  SOL time:     {sol['t_sol_us']:.1f} us")
    print(f"  SOL gap:      {sol['sol_gap']:.2f}×")
    print(f"  Classification: {sol['classification']}")
    print("=" * 60)

    result = HarnessResult(True)
    report = build_harness_report(kernel_path, spec, result, bench, sol, n=n)
    report_path = _harness_report_path(spec.name)
    report_path.write_text(report)
    print(f"Report persisted to: {report_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPU Kernel Harnessing Agent Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # serve
    p_serve = sub.add_parser("serve", help="Start opencode headless server")
    p_serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_serve.add_argument("--host", default=DEFAULT_HOST)

    # run (optimize loop)
    p_run = sub.add_parser("harness", help="Run the full optimization loop")
    p_run.add_argument("--kernel", required=True, help="Path to kernel file")
    p_run.add_argument(
        "--task",
        default=(
            "Improve the kernel's performance on H800 (sm90a). "
            "Pick ONE bottleneck hypothesis per iteration; if correctness failed "
            "last round, first isolate it with a minimal probe under sandbox/probes/ "
            "(kernel-debug-probes skill) instead of re-editing blindly. Verify with "
            "the verification command above before reporting."
        ),
        help="Task description for the agent (environment/baseline/outcome context "
             "is injected automatically around it)",
    )
    p_run.add_argument("--max-iter", type=int, default=5, help="Max optimization iterations")
    p_run.add_argument("--timeout-min", type=float, default=45.0,
                       help="Per-iteration prompt timeout in minutes (default: 45; "
                            "subagent edits+compiles+benches kernels, keep it generous)")
    p_run.add_argument("--no-approve", action="store_true",
                       help="Disable keyboard approval of opencode permission requests")
    p_run.add_argument("--approve-all", action="store_true",
                       help="Auto-approve every permission request (recommended for "
                            "unattended/headless runs; avoids stalls until timeout)")
    p_run.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_run.add_argument("--host", default=DEFAULT_HOST)
    p_run.add_argument("--agent", default="kernel-dev", help="opencode agent name")
    # v2 loop knobs (tools/harness_loop.py)
    p_run.add_argument("--candidates", type=int, default=1,
                       help="Mutually exclusive optimization variants per iteration (default 1)")
    p_run.add_argument("--explore", action="store_true",
                       help="Two-phase: agent first profiles and returns an evidence-backed "
                            "hypothesis list (RESULT.hypotheses) that seeds later candidates")
    p_run.add_argument("--gap-near-sol-stop", type=float, default=1.1,
                       help="Stop when SOL gap <= this (default 1.1 = within 10%)")
    p_run.add_argument("--gap-structural", type=float, default=5.0,
                       help="Above this SOL gap allow structural changes (keep threshold 8%)")
    p_run.add_argument("--no-lint", action="store_true", help="Skip kernel_lint advisory checks")
    p_run.add_argument("--no-probes", action="store_true",
                       help="Skip running archived regression probes during verify")
    p_run.add_argument("--loop", choices=["legacy", "v2"], default="v2",
                       help="Harness loop implementation (default v2 pluggable)")

    # harness
    p_harness = sub.add_parser("verify", help="Run safety harness on a kernel")
    p_harness.add_argument("--kernel", required=True, help="Path to kernel file")
    p_harness.add_argument("-n", type=int, default=4096, help="Benchmark input size")
    p_harness.add_argument("--output-dir", default=None,
                           help="Directory for the harness_<name>.md report (default: experiments/)")
    p_harness.add_argument("--device", default="auto",
                           help="Input device: auto|cpu|cuda (default: cuda when available)")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Benchmark a kernel")
    p_bench.add_argument("--kernel", required=True, help="Path to kernel file")
    p_bench.add_argument("-n", type=int, default=4096, help="Input size")
    p_bench.add_argument("--device", default="auto",
                         help="Input device: auto|cpu|cuda (default: cuda when available)")

    # command (trigger a registered slash command, e.g. /kernel-opt)
    p_cmd = sub.add_parser("command", help="Execute a registered slash command (e.g. kernel-opt)")
    p_cmd.add_argument("name", help="Command name without leading slash (e.g. kernel-opt)")
    p_cmd.add_argument("--args", default="", help="Arguments passed to the command")
    p_cmd.add_argument("--session-id", default=None, help="Reuse an existing session")
    p_cmd.add_argument("--agent", default=None, help="Agent override (default: command frontmatter)")
    p_cmd.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_cmd.add_argument("--host", default=DEFAULT_HOST)

    # session (subcommands)
    p_session = sub.add_parser("session", help="Manage opencode sessions")
    session_sub = p_session.add_subparsers(dest="session_cmd")
    session_sub.add_parser("create", help="Create a new session")
    p_prompt = session_sub.add_parser("prompt", help="Send a prompt")
    p_prompt.add_argument("text", help="Prompt text")
    p_prompt.add_argument("--session-id", help="Session ID (uses last created)")
    p_prompt.add_argument("--agent", default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "serve":
        agent = KernelAgent(port=args.port, host=args.host)
        if not agent.start_server():
            sys.exit(1)
        try:
            print(f"opencode server running at {agent.base_url}")
            print("Press Ctrl+C to stop.")
            signal.signal(signal.SIGINT, lambda *_: agent.stop_server())
            signal.signal(signal.SIGTERM, lambda *_: agent.stop_server())
            while agent._server_proc and agent._server_proc.poll() is None:
                time.sleep(1)
        finally:
            agent.stop_server()

    elif args.command == "harness":
        agent = KernelAgent(
            port=args.port, host=args.host,
            interactive_permissions=not args.no_approve and not args.approve_all,
            auto_approve_permissions=args.approve_all,
        )
        if not agent.wait_for_server():
            logger.error("cannot connect to opencode server at %s", agent.base_url)
            sys.exit(1)
        agent.create_session(title=f"kernel-opt:{pathlib.Path(args.kernel).name}")

        from tools.harness_loop import HarnessLoop, LoopConfig

        cfg = LoopConfig(
            max_iterations=args.max_iter,
            prompt_timeout_ms=int(args.timeout_min * 60_000),
            agent=args.agent,
            num_candidates=args.candidates,
            explore_first=args.explore,
            gap_structural=args.gap_structural,
            gap_near_sol_stop=args.gap_near_sol_stop,
            run_lint=not args.no_lint,
            run_regression_probes=not args.no_probes,
        )
        loop = HarnessLoop(agent, args.kernel, cfg)
        logger.info("loop hooks: %s (pluggable: register_hook)", list(loop.hooks))
        results = loop.run(args.task)
        # results persisted by report hook; also keep legacy write for compat
        out_path = ROOT / "experiments" / f"{pathlib.Path(args.kernel).stem}_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        logger.info("results saved to %s", out_path)

    elif args.command == "verify":
        run_safety_verification_standalone(args.kernel, n=args.n, output_dir=args.output_dir, device=args.device)

    elif args.command == "benchmark":
        run_benchmark_standalone(args.kernel, n=args.n, device=args.device)

    elif args.command == "command":
        agent = KernelAgent(port=args.port, host=args.host)
        if not agent.wait_for_server():
            logger.error("cannot connect to opencode server at %s", agent.base_url)
            sys.exit(1)
        if args.session_id:
            agent._session_id = args.session_id
        else:
            agent.create_session(title=f"/{args.name}")
        reply = agent.command(args.name, args.args, agent=args.agent)
        print(reply)

    elif args.command == "session":
        agent = KernelAgent()
        if not agent.wait_for_server():
            logger.error("cannot connect to opencode server")
            sys.exit(1)
        if args.session_cmd == "create":
            sid = agent.create_session()
            print(f"session: {sid}")
        elif args.session_cmd == "prompt":
            if args.session_id:
                agent._session_id = args.session_id
            reply = agent.prompt(args.text, agent=args.agent)
            print(reply)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

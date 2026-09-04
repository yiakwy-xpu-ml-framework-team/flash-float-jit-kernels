"""
harness_loop.py — pluggable rule-based harnessing loop for the kernel agent.

The loop's "root nodes" are a registry of swappable callables. A general LLM
agent (e.g. opencode headless) can inject replacements at runtime — plain Python
monkey-patching is the plugin protocol:

    from tools.harness_loop import HarnessLoop, LoopConfig
    loop = HarnessLoop(agent, "jit_kernel/thunder_moun.py", LoopConfig())
    loop.register_hook("policy", my_policy_fn)   # replace any root node
    loop.register_hook("propose", my_proposer)
    loop.unregister_hook("policy")               # back to default
    results = loop.run(task)

Root nodes (all replaceable):
    build_context  (iter_idx, feedback) -> str        prompt scaffolding
    propose        (iter_idx, feedback) -> list[dict]  candidate descriptors [{tag, prompt}]
    snapshot       (tag) -> pathlib.Path               baseline/candidate copy
    verify         () -> HarnessResult                 correctness gate (lint+safety+probes)
    benchmark      () -> BenchResult                    perf gate
    policy         (sol_gap, time_us, best_us, iter) -> PolicyDecision
    restore        (tag_or_path) -> None               rollback (called on ANY failure)
    report         (results) -> None                   final account

Snapshot discipline (non-negotiable):
  * baseline snapshot BEFORE any candidate is verified
  * each candidate ALSO gets its own snapshot right after the proposer edits
  * ANY failure (verify/bench) restores the last-good (baseline or best-keep) copy
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from tools.kernel_agent import (
    BenchResult,
    HarnessResult,
    KernelAgent,
    compute_sol_gap,
    run_safety_verification,
    benchmark_kernel,
    load_operator_spec,
)

logger = logging.getLogger("harness_loop")

ROOT = pathlib.Path(__file__).parent.parent.resolve()

HOOK_NAMES = (
    "build_context", "propose", "snapshot", "verify",
    "benchmark", "policy", "restore", "report",
)


@dataclass
class PolicyDecision:
    action: str            # "keep" | "restore" | "stop"
    threshold: float = 0.02
    note: str = ""


@dataclass
class LoopConfig:
    max_iterations: int = 5
    prompt_timeout_ms: int = 2_700_000
    agent: str = "kernel-dev"
    
    # NOTE (yiakwy) : mutually exclusive variants per iteration
    num_candidates: int = 1
    # NOTE (yiakwy) : defaults to explore+report to harvest hypothesis queue
    explore_first: bool = False

    # kernel development agent tunning parameters

    gap_structural: float = 5.0       # gap > this → allow structural, need >=8%
    keep_ratio_structural: float = 0.92
    keep_ratio_default: float = 0.98  # today's 2%
    keep_ratio_tight: float = 0.995   # gap <= gap_tight: keep small wins too
    gap_tight: float = 1.5
    gap_near_sol_stop: float = 1.1    # stop once baseline/best is within 10% of SOL
    run_lint: bool = True
    lint_fatal: bool = False
    run_regression_probes: bool = True
    probe_timeout_s: int = 300
    bench_n: int = 4096


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RESULT_RE = re.compile(r"RESULT:\s*(\{.*?\})\s*(?:\n|$)", re.S)


def parse_result_line(reply: str) -> dict:
    """Extract the RESULT json from an agent reply. Tolerant: returns {} when absent."""
    for m in reversed(list(_RESULT_RE.finditer(reply or ""))):
        try:
            data = json.loads(m.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return {}


def probe_registry_path(kernel_stem: str) -> pathlib.Path:
    base = ROOT / "experiments" / "probe_registry"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{kernel_stem}.json"


class ProbeRegistry:
    def __init__(self, kernel_stem: str) -> None:
        self.stem = kernel_stem
        self.path = probe_registry_path(kernel_stem)
        self.entries: list[dict] = json.loads(self.path.read_text()) if self.path.exists() else []

    def register_from_result(self, result: dict, iteration: int) -> list[str]:
        added: list[str] = []
        for p in result.get("probes", []) or []:
            if not isinstance(p, str) or not p.endswith(".py"):
                continue
            if any(e["path"] == p for e in self.entries):
                continue
            if not (ROOT / p).exists():
                continue
            self.entries.append({"path": p, "iteration": iteration,
                                 "conjecture": result.get("next_hypothesis", "")})
            added.append(p)
        if added:
            self.path.write_text(json.dumps(self.entries, indent=2))
            logger.info("probe registry +%d (%s)", len(added), added)
        return added


# ---------------------------------------------------------------------------
# HarnessLoop
# ---------------------------------------------------------------------------

class HarnessLoop:
    """self-envolved harnessing loop with runtime monkey patches"""

    def __init__(self, agent: KernelAgent, kernel_path: str, config: LoopConfig) -> None:
        self.agent = agent
        self.kernel_path = kernel_path
        self.config = config
        self.stem = pathlib.Path(kernel_path).stem
        self.registry = ProbeRegistry(self.stem)
        self.hypothesis_queue: list[dict] = []
        self._defaults: dict[str, Callable] = {
            "build_context": self._default_build_context,
            "propose": self._default_propose,
            "snapshot": self._default_snapshot,
            "verify": self._default_verify,
            "benchmark": self._default_benchmark,
            "policy": self._default_policy,
            "restore": self._default_restore,
            "report": self._default_report,
        }
        self.hooks: dict[str, Callable] = dict(self._defaults)

    def register_hook(self, name: str, fn: Callable) -> None:
        if name not in HOOK_NAMES:
            raise KeyError(f"unknown hook {name!r}; expected one of {HOOK_NAMES}")
        self.hooks[name] = fn
        logger.info("registered custom %s: %s", name, getattr(fn, "__name__", fn))

    def unregister_hook(self, name: str) -> None:
        if name not in HOOK_NAMES:
            raise KeyError(f"unknown hook {name!r}")
        self.hooks[name] = self._defaults[name]

    def run(self, task: str) -> list[dict]:
        """Main loop: baseline ➜ [propose candidate(s) → snapshot → verify → bench
        → policy → keep/restore] × iterations."""
        cfg = self.config
        results: list[dict] = []

        self.hooks["snapshot"]("baseline")
        base_bench = self.hooks["benchmark"]()
        if base_bench is None:
            logger.error("baseline benchmark failed; aborting")
            return [{"phase": "baseline", "error": "benchmark failed"}]
        baseline_sol = compute_sol_gap(base_bench)
        best_us = base_bench.time_us
        logger.info("baseline: %.1f us | SOL gap %.2f× (%s)",
                    best_us, baseline_sol["sol_gap"], baseline_sol["classification"])
        results.append({"phase": "baseline", "time_us": best_us, **baseline_sol})

        context = self.hooks["build_context"](0, {"baseline": baseline_sol, "task": task})

        if cfg.explore_first and not self.hypothesis_queue:
            self._explore_phase(task, context, results)

        feedback: dict = {"baseline": baseline_sol, "task": task}

        for i in range(1, cfg.max_iterations + 1):
            logger.info("=== iteration %d/%d ===", i, cfg.max_iterations)
            candidates = self.hooks["propose"](i, feedback)
            if not candidates:
                logger.warning("proposer returned no candidates; stopping")
                break

            best_tag = "baseline"

            for cand in candidates:
                tag = cand["tag"]
                reply = cand.get("reply", "")
                result_line = parse_result_line(reply)
                probes_added = self.registry.register_from_result(result_line, i)
                if probes_added:
                    cand.setdefault("probes", probes_added)

                self.hooks["snapshot"](tag)

                harness = self.hooks["verify"]()
                if not harness.passed:
                    logger.warning("candidate %s FAILED at %s: %s", tag, harness.stage, harness.detail)
                    results.append({"phase": "verify", "iteration": i, "candidate": tag,
                                    "stage": harness.stage, "detail": harness.detail,
                                    "probes": cand.get("probes", [])})
                    feedback = {"phase": "verify", "stage": harness.stage, "detail": harness.detail}
                    self.hooks["restore"](best_tag)
                    continue

                bench = self.hooks["benchmark"]()
                if bench is None:
                    results.append({"phase": "bench", "iteration": i,
                                    "candidate": tag, "error": "benchmark failed"})
                    feedback = {"phase": "bench", "detail": "benchmark crashed"}
                    self.hooks["restore"](best_tag)
                    continue

                sol = compute_sol_gap(bench)
                decision: PolicyDecision = self.hooks["policy"](
                    sol["sol_gap"], sol["time_us"], best_us, i)
                logger.info("candidate %s: %.1f us (gap %.2f×) → policy: %s (%s)",
                            tag, sol["time_us"], sol["sol_gap"], decision.action, decision.note)
                results.append({"phase": "bench", "iteration": i, "candidate": tag,
                                "time_us": sol["time_us"], "sol_gap": sol["sol_gap"],
                                "decision": decision.action, "note": decision.note,
                                "probes": cand.get("probes", [])})

                if decision.action == "stop":
                    logger.info("policy stop: %s", decision.note)
                    results.append({"phase": "stop", "note": decision.note, "iteration": i})
                    self.hooks["restore"](best_tag)
                    return results

                if decision.action == "keep":
                    best_us = sol["time_us"]
                    best_tag = tag
                    feedback = {"phase": "keep", "candidate": tag, "time_us": best_us}
                else:
                    feedback = {"phase": "revert", "candidate": tag,
                                "time_us": sol["time_us"], "best_us": best_us}

                # restore to CURRENT best so the next candidate builds on it
                self.hooks["restore"](best_tag)

        self.hooks["report"](results)
        return results

    # ------------------------------------------------------------------
    # default root-node implementations
    # ------------------------------------------------------------------

    def _explore_phase(self, task: str, context: str, results: list[dict]) -> None:
        """open exploration: no edits; harvest an evidence-backed
        hypothesis list consumed by later iterations."""

        prompt = (
            f"{context}\n\n"
            f"## Exploration phase (Plan Mode without edits)\nTask: {task}\n"
            "Profile this kernel (benchmark script, ncu if available, code reading) and "
            "produce an evidence-backed hypothesis LIST, worst first.\n"
            "For each:\n"
            "title, rationale (with numbers), and a probe_hint for the kernel-debug-probes skill. "
            "End with:\n"
            'RESULT: {"decision":"explore","time_us":null,'
            '"hypotheses":[{"title":"...","rationale":"...","probe_hint":"..."}]}'
        )
        try:
            reply = self.agent.prompt(prompt, agent=self.config.agent,
                                      timeout_ms=self.config.prompt_timeout_ms)
        except Exception as exc:
            logger.warning("explore phase failed: %s", exc)
            results.append({"phase": "explore", "error": str(exc)})
            return
        data = parse_result_line(reply)
        hyps = data.get("hypotheses") or []
        self.hypothesis_queue.extend(h for h in hyps if isinstance(h, dict) and h.get("title"))
        logger.info("explore harvested %d hypotheses", len(self.hypothesis_queue))
        results.append({"phase": "explore", "hypotheses": self.hypothesis_queue})

    def _default_build_context(self, iter_idx: int, feedback: dict) -> str:
        """Build the prompt scaffolding (kernel path, verify command, probes, SOL)."""
        sol = feedback.get("baseline") or {}
        lines = [
            "## Environment (fixed for all iterations)",
            f"- Kernel under test: `{self.kernel_path}` (repo root = cwd)",
            f"- Verify: `python tools/kernel_agent.py verify --kernel {self.kernel_path}`",
            f"- Regression probes for this kernel: `sandbox/probes/` + registry",
            f"  `experiments/probe_registry/{self.stem}.json`",
            "- Methodology: kernel-debug-probes skill — minimal probe per hypothesis before edits.",
            "- Each candidate reply MUST end with:",
            '  RESULT: {"decision":"keep|revert","time_us":<float>,"files":[...],"probes":[...],"next_hypothesis":"..."}',
        ]
        if sol:
            lines.append(f"- Baseline: {feedback.get('baseline', {}).get('t_sol_us', 0):.1f} us SOL "
                         f"| gap {sol.get('sol_gap', 0):.2f}× ({sol.get('classification', '?')})")
        return "\n".join(lines)

    def _default_propose(self, iter_idx: int, feedback: dict) -> list[dict]:
        """Return candidate descriptors. Default behaviours:
        * num_candidates == 1: one refinement prompt (today's loop semantics);
        * num_candidates  > 1: sequential candidates — implement variant k, we
          snapshot + verify + restore between variants;
        * if the explore phase harvested a hypothesis queue, seeds are consumed
          in order before fresh ideas are requested."""

        cfg = self.config
        seed: Optional[dict] = self.hypothesis_queue.pop(0) if self.hypothesis_queue else None
        seeds: list[Optional[dict]] = [seed] if seed else [None] * cfg.num_candidates

        def one_prompt(k: int, total: int, seed_: Optional[dict]) -> str:
            outcome = ""
            if feedback.get("phase") == "verify":
                outcome = (f"\n## Previous outcome\n- Verification FAILED at"
                           f" `{feedback.get('stage')}`: `{feedback.get('detail')}` — diagnose"
                           " with a probe first; do NOT repeat this approach.\n")
            elif feedback.get("phase") == "revert":
                outcome = (f"\n## Previous outcome\n- candidate {feedback.get('candidate')}:"
                           f" {feedback.get('time_us', 0):.1f} us did NOT beat best"
                           f" {feedback.get('best_us', 0):.1f} us — pick a DIFFERENT angle.\n")
            elif feedback.get("phase") == "keep":
                outcome = (f"\n## Previous outcome\n- KEEPED candidate {feedback.get('candidate')}"
                           f" (best now {feedback.get('time_us', 0):.1f} us).\n")
            seed_txt = ""
            if seed_:
                seed_txt = (f"\n## Approved hypothesis (from exploration)\n"
                            f"- title: {seed_.get('title')}\n"
                            f"- rationale: {seed_.get('rationale')}\n"
                            f"- probe_hint: {seed_.get('probe_hint')}\n")
            v_txt = f" (variant {k}/{total})" if total > 1 else ""
            return (f"{self.hooks['build_context'](iter_idx, feedback)}\n"
                    f"{outcome}{seed_txt}\n"
                    f"## Task\n{feedback.get('task')}\n\n"
                    f"Iteration {iter_idx}, candidate{v_txt}. Implement ONLY this variant; "
                    "change exactly one thing.")

        candidates: list[dict] = []
        for k in range(cfg.num_candidates):
            prompt = one_prompt(k + 1, cfg.num_candidates, seeds[k % len(seeds)])
            try:
                reply = self.agent.prompt(prompt, agent=cfg.agent,
                                          timeout_ms=cfg.prompt_timeout_ms)
            except Exception as exc:
                logger.error("candidate %d prompt failed: %s", k + 1, exc)
                try:
                    self.agent.client.abort_session(self.agent.session_id)
                except Exception:
                    pass
                if not self.agent.client.ping():
                    self.agent.start_server()
                    if not self.agent.wait_for_server():
                        logger.error("server unrecoverable; aborting loop")
                        break
                continue
            candidates.append({"tag": f"iter{iter_idx}_cand{k+1}", "reply": reply})
        return candidates

    def _default_snapshot(self, tag: str) -> pathlib.Path:
        arc = self.agent.snapshot_tree(tag)
        assert arc and arc.exists(), f"snapshot failed for tag {tag}"
        return arc

    def _default_restore(self, tag: str) -> None:
        ok = self.agent.restore_tree(tag)
        if not ok:
            logger.error("restore failed for tag %s — tree may be left dirty", tag)

    def _default_verify(self) -> HarnessResult:
        cfg = self.config

        if cfg.run_lint:
            from tools import kernel_lint
            findings = kernel_lint.lint_kernel(self.kernel_path)
            fatal = [f for f in findings if f["level"] in ("error", "fatal")]
            if fatal and cfg.lint_fatal:
                first = fatal[0]
                return HarnessResult(False, "lint", f"{first['rule']}: {first['detail']}", [first["detail"]])
            if findings:
                for f_ in findings[:8]:
                    logger.info("lint[%s] %s: %s", f_["level"], f_["rule"], f_["detail"])

        from tools.kernel_agent import load_operator_spec
        spec = load_operator_spec(self.kernel_path)
        if spec is None:
            return HarnessResult(False, "load", f"cannot load kernel spec: {self.kernel_path}")
        result = run_safety_verification(spec)

        if result.passed and cfg.run_regression_probes and self.registry.entries:
            result = self._run_regression_probes()
        return result

    def _run_regression_probes(self) -> HarnessResult:
        import os
        import subprocess
        import sys

        for e in self.registry.entries:
            probe = ROOT / e["path"]
            if not probe.exists():
                continue
            env = dict(os.environ, PYTHONPATH=str(ROOT))
            try:
                proc = subprocess.run(
                    [sys.executable, str(probe)], cwd=ROOT, env=env,
                    capture_output=True, text=True, timeout=self.config.probe_timeout_s)
                out = (proc.stdout or "") + (proc.stderr or "")
            except subprocess.TimeoutExpired:
                return HarnessResult(False, "regression_probe",
                                     f"{e['path']}: timed out after {self.config.probe_timeout_s}s")
            bad = proc.returncode != 0 or re.search(r"fail|mismatch|error", out, re.I)
            if bad:
                logger.warning("regression probe FAILED: %s\n%s", e["path"], out[-1500:])
                return HarnessResult(False, "regression_probe", f"{e['path']}: rc={proc.returncode}")
        return HarnessResult(True, stage="regression_probe")

    def _default_benchmark(self) -> Optional[BenchResult]:
        return self.agent.bench_kernel(self.kernel_path, n=self.config.bench_n)

    def _default_policy(self, sol_gap: float, time_us: float, best_us: float,
                        iter_idx: int) -> PolicyDecision:
        cfg = self.config
        if sol_gap <= cfg.gap_near_sol_stop:
            return PolicyDecision("stop", note=f"within {cfg.gap_near_sol_stop - 1:.0%} of SOL")
        if sol_gap > cfg.gap_structural:
            thr_keep = cfg.keep_ratio_structural
        elif sol_gap <= cfg.gap_tight:
            thr_keep = cfg.keep_ratio_tight
        else:
            thr_keep = cfg.keep_ratio_default
        if time_us < best_us * thr_keep:
            return PolicyDecision("keep", threshold=thr_keep,
                                  note=f">|{(1 - thr_keep):.0%} better than best")
        return PolicyDecision("restore", threshold=thr_keep,
                              note=f"needs {(1 - thr_keep):.0%} improvement to keep")

    def _default_report(self, results: list[dict]) -> None:
        out_path = ROOT / "experiments" / f"{self.stem}_results.json"
        out_path.write_text(json.dumps(results, indent=2))
        logger.info("results persisted to %s", out_path)

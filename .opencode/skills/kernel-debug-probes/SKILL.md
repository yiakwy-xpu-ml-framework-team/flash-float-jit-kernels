---
name: kernel-debug-probes
description: Probe-first debugging for GPU kernels (CUDA/Triton/Gluon/Hopper). Write a minimal standalone repro BEFORE touching the big kernel whenever results are wrong, a launch crashes/hangs, or you must attribute an error to quantization vs accumulation vs logic. Covers 9 validated probe patterns (primitive isolation, NaN-sentinel coverage, bit-exactness vs single-calls, fp64 arbiter, block-error maps, pure-python index decode checks, in-kernel A/B guards, fresh-process isolation, sanitizer interpretation) plus Hopper-specific gotchas learned the hard way (async-TMA liveness overlap, cluster rank math, PTX direction constraints). Use when editing jit_kernel/csrc/**, jit_kernel/triton*/**, or benchmark correctness fails.
---

# Probe-First Kernel Debugging

Core rule: **never debug inside the production kernel**. Isolate the suspect
primitive in a standalone script first; edit the kernel only after the probe has
proven which line is wrong.

## Why (measured payoff)

One working session turned three "unsolvable" bugs into one-line fixes purely by
probing:

| Bug | Wrong guess (hours) | Probe verdict (minutes) |
|-----|--------------------|-------------------------|
| batched symm-GEMM wrong for b>0 | "fold lands in stride-1 dim", rewrote descriptors 3x | host descriptor dim1 not batch-expanded (`{K,M}` → `{K,B*M}`); fold math was correct all along |
| gluon mirrored tiles corrupt | "triton 3.6 broken" (true but vague) | 40-line single-CTA repro proved `smem.permute→TMA store` silently writes untransposed data; `permute→ld.shared` load is exact |
| split-k reduce random corruption | barrier protocol re-derived twice | compute-sanitizer clean + hang ⇒ NOT OOB; compiler assigned a loop-scoped smem buffer OVERLAPPING a buffer still read by an in-flight async TMA store |

## The 9 patterns

Probes live in `sandbox/probes/probe_<conjecture>.py`. Each prints PASS/FAIL plus
numbers, takes shapes via argv, and is kept afterwards (regression value).

### P1 — Primitive isolation (smallest standalone repro)
Reproduce ONLY the suspect instruction/layout/barrier in a tiny kernel with an
iota-pattern input and exact-equality check. If the primitive is wrong, no amount
of kernel-side reasoning helps.
Example (`tma_permute_probe.py`): single CTA, TMA-load a 128×128 tile, store it
identity + permuted-view side by side into one destination; compare both halves.
Verdict: identity exact, permuted garbage ⇒ primitive broken upstream.

### P2 — NaN-sentinel coverage test
Pre-fill the output with NaN, run the kernel, count survivors and map them to
blocks. Proves full-write coverage or finds exactly which tiles never landed.
```python
out = torch.full(shape, float("nan"), ...)
run_kernel(..., out=out)
print(torch.isnan(out).sum())          # 0 == full coverage
nan.view(B, nb, blk, nb, blk).any(...) # block-level heatmap
```
⚠️ Always sentinel-test before trusting a benchmark pass: `torch.empty`
allocations recycle freed blocks, so a kernel that skips regions can still
"match" a reference whose memory it happens to reuse (bit us twice).

### P3 — Bit-exactness vs single-unit calls
Run the fused/batched path and compare bitwise against N independent small calls.
diff==0 ⟹ plumbing correct (any remaining error is numerics); diff>0 ⟹ real
addressing/fold bug. This separated batch-fold bugs from fp8 noise instantly.

### P4 — fp64 arbiter (attribute the error)
Compare BOTH the kernel and the naive reference against an fp64 recomputation of
the quantized inputs. Whichever moves toward truth tells you whether the gap is
input quantization (expected), accumulation order (benign, ~1 ulp), or logic
(structural, huge). Never argue about tolerance without it.

### P5 — Block-error map
`(out - ref).abs().view(nb,blk,nb,blk).amax(dim=(1,3))` heatmap localizes
corruption to specific tiles — diagonal-only = rounding; stripes = partial tile
write; corners = scheduling tail; symmetric violation set = mirror path.

### P6 — Pure-python decode validation
Replicate index/swizzle math in numpy/python and assert: every pair in range,
bijective coverage, no duplicates. Costs minutes and eliminates an entire class
of "maybe the scheduler is wrong" speculation without touching the GPU.

### P7 — In-kernel A/B bisect
Guard the suspected region with `if False:` (or an `#ifdef`) and flip ONE
variable at a time: store2 on/off, staging buffer hoisted/not, multicast
gated/not. Re-run the SAME probe after each flip.
⚠️ One GPU fault poisons the CUDA context: run each variant in a FRESH process,
never loop variants inside one interpreter.

### P8 — Process & tooling discipline
- `CUDA_LAUNCH_BLOCKING=1` to pin faults to the right launch.
- `compute-sanitizer --tool memcheck`: OOB shows as Invalid access. A
  **launch-failure/hang with ZERO memcheck errors means barriers/races/liveness —
  go look at sync protocol, not pointers**.
- Device printf does NOT flush while a kernel hangs — printf tracing cannot find
  deadlocks. Prefer state dumps from the host side or sanitizer synccheck.
- Hangs: bisect by shape/config matrix first (cheap), source second.

### P9 — Env-gated kernel variants
Wire experimental paths behind `-D` flags injected from env vars (e.g.
`FLASH_FLOAT_BULK_SPLITK=1` adds `-DUSE_BULK_SPLITK_REDUCE=1`). The JIT cache
keys on flags, so A/B needs no source churn and defaults stay safe. Always print
which variant is active from the harness script.

## Hopper/triton-specific landmines (learned 2026-08)

1. **Async liveness hole**: the compiler does NOT treat an in-flight async TMA
   read/store as a use of the source smem buffer. A buffer allocated LATER in the
   same scope can be placed OVERLAPPING one whose TMA transfer is still pending.
   Symptom: data corrupted only in some compiled specializations, "works then
   breaks when you touch unrelated code". Fix: hoist the buffer to function
   scope (or prove disjoint slots).
2. **Cluster rank math**: any hardcoded rank (arrive rank 0, map_shared_rank(r))
   is only valid for a 1-D cluster along that grid axis. With cluster
   `{split_k, csm}` ranks are x-major: same-row peer = `r + (blockIdx.y % csm) * split_k`.
   Either parameterize or force the degenerate layout (host sets csm=1 when sk>1).
3. **PTX direction constraints**: `cp.async.bulk` smem↔smem exists ONLY as
   dst=`.shared::cluster` / src=`.shared::cta` (PUSH; source must be local).
   There is NO pull form. `cp.reduce.async.bulk.add.f16x2` with shared::cluster
   dst + mbarrier completion is rejected by ptxas 12.8.
4. **TMA coordinate order**: helpers take `(inner_dim_offset, outter_dim_offset)`
   = (dim0 stride-1 coord, dim1 coord). Batch folding belongs in the OUTER
   coordinate whose stride is the row stride; expanding descriptor shape to
   `{K, B*M}` (not moving strides!) makes `off_m += b*M` jump planes exactly.
5. **mbarrier tx accounting**: `expect_bytes` must be posted before completions
   can land, donors/receivers flip phases the same number of times, and donors
   must hold their epilogue buffer until the receiver signals release
   (readable-barrier handshake) — exiting early frees smem mid-read.
6. **Transposed smem views are asymmetric**: `memdesc.permute((1,0))` +
   ld.shared LOAD is exact on triton 3.6, but the same view fed to a TMA STORE
   is silently corrupted (and deadlocks back-to-back with another store).
   Route transposes through registers into a normal-layout buffer.

## Loop contract with the harness

When driven by `tools/kernel_agent.py`, each iteration ends with a machine-checkable
summary block so the orchestrator can feed it forward:

```
RESULT: {"decision":"keep|revert","time_us":<float>,"files":["..."],
         "probes":["sandbox/probes/<name>.py"],"next_hypothesis":"..."}
```

If a verification stage fails, name the stage and quote its detail verbatim;
attach the probe that isolates it rather than re-describing the whole kernel.

# mamba-ssm verify-before-lock — GPU half: PASS (2026-06-17, step b)

**The verdict the W4 CPU half deferred.** Forward/backward fused-kernel numerics on an
A100 under the EXACT locked stack. Machine evidence:
`reports/2026-06-17-mamba-gpu-half-probe.json` (Leonardo job 47092267, branch
`phase-2-mamba-lock-gpu`, probe `scripts/probe_mamba_gpu_half.sh` +
`scripts/mamba_gpu_numerics.py`). The locked GPU stack computes mamba correctly →
`env_lock.py` extended; the mamba-hybrid backbone is now authorized to be built (a
separate downstream task).

## What was verified (teeth, not a smoke test)

**Reference** = mamba-ssm's own pure-PyTorch ops (`selective_scan_ref`,
`causal_conv1d_ref`) — the package's correctness baseline.
**torch held the comparability lock at runtime:** `2.5.1+cu121` on `NVIDIA
A100-SXM-64GB`; versions exactly the candidate pin (`triton 3.1.0`, `mamba-ssm 2.3.1`,
`causal-conv1d 1.6.2.post1`). The GPU env did not drift the version.

- **fp32 (direct relative tol) — essentially exact:** fused kernels match the fp32
  reference to **~1e-5 relative** (selective_scan fwd err 0.0051 on values to 317;
  causal_conv1d fwd err **1.4e-6**). Pure reduction-order difference — correct, not
  "barely within tolerance."
- **bf16 (the TRAINING dtype) — the fp32-ground-truth ratio test:** fused error vs the
  fp32 truth is **no worse than the naive bf16 reference's** (often identical: grads
  δ/B/C/z equal to naive; D and δ_bias fused *better*; grad A 1% relative on magnitude
  1.3M). The fused kernel adds **zero error beyond bf16 rounding**.
- **Mamba module:** trains end to end (437,760 params, finite grads, loss 0.00186).

A genuinely wrong kernel would blow the fp32 ~1e-5 bound and show bf16 `fused_err` orders
of magnitude above `naive_err` — the test discriminates.

## Instrument honesty (why the PASS is trustworthy)

The first two probe runs returned `verdict_PASS: false` — both were the **instrument**,
not the kernel: (1) absolute tolerances are the wrong metric for a fused-vs-PyTorch
comparison over a 2048-step scan; (2) the reference ops reject fp64 (internal fp32
assumptions); (3) `causal_conv1d_ref` was imported from the wrong path; (4) an unguarded
None grad. Each was diagnosed against the evidence (small relative errors + the module
training fine) and the TEST was corrected to mamba-ssm's own fp32-truth methodology —
not loosened until green, not false-HALTed. The passing test is the one that would fail
a wrong kernel; the earlier miscalibrated versions DID fail.

## What landed: the env_lock extension (the lock the PASS authorizes)

`src/cfm/training/env_lock.py`:
- `LOCKED_TRITON = "3.1.0"` joins the **shared** `_EXPECTED` (torch.compile uses triton on
  every backbone; the repo `.venv` carries exactly 3.1.0 — verified).
- `LOCKED_MAMBA_SSM = "2.3.1"`, `LOCKED_CAUSAL_CONV1D = "1.6.2.post1"` go in
  `_MAMBA_EXPECTED`, enforced by `assert_mamba_env_locked()` (reads dist metadata, no
  import → no gcc-12 preload needed), **conditionally** — because the transformer-ar /
  discrete-diffusion runs use the repo `.venv` with no mamba installed.
- `build_backbone("mamba-hybrid", …)` calls `assert_mamba_env_locked()` at the
  construction site ahead of the (still-present) `BackboneNotYetBuilt` gate;
  discrete-diffusion gates directly (no mamba dependency). Guards updated in the same
  change (`test_env_lock.py`, `test_backbone_identity_lock.py`).

## Carried preconditions (the three that MUST ride every mamba GPU entry point)

From `reports/2026-06-12-mamba-candidate-pin.md`, verified in the probe sbatch:
1. `module load gcc/12.2.0` + `CC`/`CXX`/`CUDAHOSTCXX` at build.
2. The torch+triton constraints file on every install (here: reuse the W4 probe venv →
   no install → torch cannot drift; the runtime tooth re-checked `2.5.1+cu121`).
3. gcc-12 `libstdc++` (`GLIBCXX_3.4.29`) visible at import (`LD_PRELOAD`).

## STILL DOWNSTREAM — not part of step b (first things (c)'s planning must cover)

1. **Build the mamba-hybrid backbone** — implement `MambaHybrid`, remove the
   `BackboneNotYetBuilt` raise. The lock verdict is in; the module is not.
2. **Wire the three runtime preconditions** (gcc/12.2.0 module, `LD_PRELOAD` gcc-12
   libstdc++) into the mamba **run** sbatch entry points.
3. **The gcc/compile eval-path blocker** found at Step 18.5: the run sbatch loads
   `cuda/12.2` but no modern gcc, so torch.compile's CPU inductor codegen crashed in
   post-train eval (gcc-8.5). The same `module load gcc/12.2.0` fix resolves it.
4. `--scored-run` + `--shard-cache` + `eval_max_new ≥ 13,312` (budget already locked).

**Verdict: PASS. env_lock extended. Backbone build + the above are (c)'s planning input.**

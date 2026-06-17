# Phase-2 bake-off — 2-backbone delta design (2026-06-17)

**Status:** DELTA on the locked `2026-06-02-phase-2-bakeoff-design.md` (+ its
`2026-06-09-phase-2-bakeoff-delta-design.md`). Everything in those holds **unchanged**
except where this delta says otherwise. This is an execution delta + three
reconciliations, not a re-design.

**Predecessors discharged (2026-06-17):**
- **18.5** F8 across-job resume — PASS (`reports/2026-06-17-step-18.5-f8-resume-proof.md`).
- **(b) mamba-lock** GPU verify-before-lock — PASS; `env_lock.py` extended
  (`reports/2026-06-17-mamba-gpu-half-verdict.md`). Roster's mamba kernel is verified;
  the backbone MODULE is the build below.

## 1. Why this delta exists

The bake-off was designed 2026-06-02; the readiness-closure re-scope and the GPU-wait
CPU drain then intervened. Four things now differ from the locked spec: a **scope
narrowing** (2 backbones) and **three reconciliations** (the 13,312 eval-budget lock, the
env-lock extension, the gcc/compile eval-path fix). This delta absorbs them.

## 2. Scope — two backbones (the spec's own §9 fallback)

**`transformer-ar` + `mamba-hybrid`.** `discrete-diffusion` is **DEFERRED, not dropped.**

This is **not a deviation**: locked spec §9 names it — *"Defer diffusion to a second wave
(run 2 AR backbones first) = named build-risk fallback."* Rationale: the decision layer
needs only ≥2 architectures to rank; transformer-ar exists today and mamba-hybrid is a
bounded mixer-swap (§3) reusing everything else by identity — the two cleanest builds;
discrete-diffusion is the largest/riskiest net-new build (a new denoising loss + a new
T-step generation path, neither of which transfers from the AR family) and folding it in
now would substantially delay the founding-question signal. Diffusion re-opens as a
well-motivated follow-on once the 2-backbone signal lands (§10).

## 3. The shared-scaffold refactor + the mamba-hybrid build

The locked spec §9 mandates *"one SHARED scaffold + a swappable backbone… backbones
differ only in their sequence-mixing layers."* `MicroAR` (transformer-ar) currently
**bakes the `TransformerEncoder` into the class** — there is no mixer seam. We create one.

### 3.1 Extract a shared base (the decision)

Refactor `src/cfm/models/micro_ar.py` behavior-preservingly into:
- a **shared scaffold** — embedding (`n_subf_vocab + conditioning_id_span()` rows),
  positional embedding, the char-carrier `Linear(n_char_stats → d_model)` at
  `char_position=CONDITIONING_PREFIX_LEN`, the sub-F head (`→ subf_vocab_size()` = 1508),
  the AR next-token training loss, and AR generation; plus
- a **swappable mixer** interface (the only divergence point).

`transformer-ar` = shared base + a `TransformerEncoder` mixer (behavior unchanged).
`mamba-hybrid` = shared base + the §3.3 interleave mixer.

`build_backbone` (`src/cfm/models/backbone.py`) gains the real `mamba-hybrid` branch
(removing its `BackboneNotYetBuilt` raise; `assert_mamba_env_locked()` stays, now ahead of
construction not the gate). `discrete-diffusion` keeps its gate.

### 3.2 Behavior-preservation — sealed-code-touch, W2-grade care

`MicroAR` is working, tested, sealed-by-use code. The refactor MUST be **behavior-
preserving** — the same discipline applied to the W2 `locked_yaml` extraction. Teeth:

- **(a) Existing tests stay green** — `tests/models/test_micro_ar.py`,
  `tests/models/test_backbone_identity_lock.py`, `tests/models/test_character_prefix.py`
  all pass unchanged.
- **(b) Bit-identical behavior-preservation test (new)** — refactored transformer-ar
  produces **identical** forward logits AND training loss to pre-refactor `MicroAR` on a
  fixed seed + fixed input (capture the pre-refactor output as a golden tensor; assert
  `torch.equal`). This is the external-source-of-truth check (protocol Gate-6): the new
  abstraction is cross-referenced against the old module's actual output, not its
  description.

### 3.3 mamba-hybrid = 7:1 Jamba interleave (NOT pure mamba)

Per locked spec §9: an **interleaved Mamba + transformer stack, ~7:1 (Jamba-style)**, via
the `mamba-ssm` package (CLAUDE.md mandate — never a custom SSM). A **drop-in swap**: same
head / mask (causal) / AR loss / AR generation.

- `MambaHybridConfig`: mamba params (`d_state=16, d_conv=4, expand=2` — the (b)-verified
  shape) + `n_layers` + the interleave ratio.
- **Interleave rule across the scale ladder** (layer counts vary by scale): place 1
  transformer layer per ~7 Mamba layers; **at small layer counts keep ≥1 transformer
  layer** (so attention is never absent); record the exact per-scale layer composition in
  the run config. Concrete mapping is a plan-time table, derived from each scale's
  `n_layers`, not improvised per run.

### 3.4 Gate-6 identity test (trivial by construction)

Because the shared base is literally **the same objects** (shared by reference, not
reconstructed), the identity test is: `MambaHybrid`'s embedding / head / conditioning
builder **ARE** transformer-ar's (`is`, not `==`) — the locked identity constraint
(`test_backbone_identity_lock.py`'s `shared_conditioning_builder() is …`) extended to the
new backbone.

### 3.5 Non-scored GPU smoke — the pre-scored gate (NEVER folded into a scored run)

Before any scored hour, `mamba-hybrid` must pass a **non-scored** 4×A100 smoke:
- **trains** — finite grads, loss decreasing over a short run;
- **evals** — generate → decode → score completes without crash (this is also the **first
  real verification of the gcc/compile eval-path fix**, §4.1/§5);
- **compile stability** — the §5 quantified check.

## 4. Dependency graph

```
0. (b) mamba-lock ........................................ DONE (this session)
1. Shared sbatch fix (NO GPU)  ─────────────────────────────────────────────┐
2. Backbone refactor + mamba-hybrid build → §3 teeth + §3.5 smoke  ──┐       │ (1 gates 2 & 3)
3. Task-1 diagnostic RUN (transformer-ar — exists today) ───────────┘∥ 2     │
4. Scored runs over the feasible matrix → run_bakeoff_decision → verdict ────┘
```

**Critical-path win:** the diagnostic runs on the **existing** transformer-ar, so **step 3
parallelizes with step 2** (the mamba build) — there is no dependency between them. Step 1
(sbatch fix) gates both; step 4 needs all of 2 + 3.

### 4.1 Step 1 — shared sbatch fix (the gcc/compile eval-path reconciliation)

The 18.5 finding: `bakeoff_run.sbatch` loads `cuda/12.2` but **no modern gcc**, so
`torch.compile`'s inductor CPU codegen in eval crashes on gcc-8.5. `bakeoff_run.sbatch` is
the **shared** run sbatch (parameterized by `--backbone "$BACKBONE"`), so one fix serves
both backbones. Fix:
- `module load gcc/12.2.0` + `export CC/CXX/CUDAHOSTCXX` in **`bakeoff_run.sbatch`** (the
  compiled scored path, all backbones) — this same load also satisfies mamba runtime
  precondition #3.
- **`LD_PRELOAD` the gcc-12 `libstdc++`** (GLIBCXX_3.4.29) in the same sbatch (per
  `reports/2026-06-12-mamba-candidate-pin.md`): **required** for the mamba run's
  `mamba_ssm`/`causal_conv1d` import, **harmless** for the transformer-ar run — so it rides
  the shared entry unconditionally rather than branching on `$BACKBONE`. The non-scored
  mamba smoke (§3.5) carries the identical toolchain.
- **Verified by an `sbatch --test-only` dry submit** against account `AIFAC_P02_548` — not
  assumed. (`bakeoff_diagnostic.sbatch` runs `--no-compile`, so it cannot hit the crash;
  it gets the gcc load **precautionarily**, not load-bearingly.)

### 4.2 Step 3 — the Task-1 diagnostic (UNRUN; gates the matrix)

`scripts/bakeoff_diagnostic.sbatch` (transformer-ar, 90M, `--no-compile`,
`--eval-max-new 2048`). Triple-duty per locked spec §5/§6: emergence floor, geometry-`r`,
per-scale eval-cost-per-cell. **Extended here (§6):** also capture **per-token generation
cost** so the 13,312 scored-eval budget can be projected. Its output **sets the feasible
scale ladder** (`feasible_ladder`, `r·N ≤ TRAIN_TOKENS·E`, `TRAIN_TOKENS=623,900,790`) and
the **decision basis** (step-function of feasible-scale count).

## 5. Compile-ON for scored runs — smoke-gated with a QUANTIFIED threshold

Scored runs **compile** (gcc-12 fixes the codegen; production-fidelity, faster steady-state
than eager). But `torch.compile` keys on tensor shape, and the bake-off's cell-token
sequences are **variable-length** — a per-shape recompilation storm could make compile
**slower** than `--no-compile`. This is a real, unmeasured uncertainty: **measure it in the
mamba smoke before any scored hour** (same measure-then-decide discipline as the W3 tier
probe and the SDPA flash probe — never "it seemed fine").

**The smoke's compile-stability check (explicit, quantified):** run a representative
window (≥ ~200 steps) whose inputs **span the real cell-length distribution** (the actual
variable lengths from the singapore/EU training shards), instrumented for `torch._dynamo`
recompilation events and cumulative compile time. **Keep compile ON iff BOTH hold:**
1. **Recompilations plateau** — bounded and stop early (no new recompiles in the final
   half of the window; automatic-dynamic-shape bucketing should converge to a small
   bounded count, target ≤ ~10), AND
2. **Compile overhead < 10%** of the window's wall-clock.

If either fails → **`--no-compile` for scored runs**, finding recorded (compile demoted to
a later optimization). The threshold and window size are pinned at smoke-design; the
verdict is data, not assumption.

## 6. Eval-dominated budget — the inversion (projected AT the diagnostic)

The 2026-06-02 spec sized eval at `eval_max_new ≈ 2048` (~200 node-h, 3 backbones). W1
(2026-06-11) locked **scored** eval at `eval_max_new ≥ 13,312` (full-cell generation for
cross-backbone commensurability) — **~6.5× the AR-generation cost** (AR generation is one
sequential forward per token). **Eval now DOMINATES the budget, not training** — sharpening
the locked spec §2 "eval binds."

- The diagnostic (§4.2) captures **per-token generation cost** → project the real 13,312
  scored-eval budget for **2 backbones × the feasible scales × seed-repeats**.
- **`n_cells` is the knob**, minimized to the must-rank resolution gap (locked spec §6/§8,
  in the per-feature unit).
- **Present the revised 2-backbone budget vs the 5000 node-h allocation AT the diagnostic,
  BEFORE committing any scored run.** Expect a **tighter feasible-scale ladder** than the
  2048-sized estimate (eval eats the envelope; the joint `feasible_ladder` ∩ eval-budget
  bound may drop 1B or the top rung). If the ladder collapses toward 1–2 feasible scales,
  the decision basis shifts to `FIXED_SCALE_PLUS_S13` per the locked delta — handled, not a
  surprise.

## 7. Reconciliations absorbed (summary)

| reconciliation | where |
|---|---|
| 13,312 eval-budget lock (W1) | §6 + the `--scored-run` gate `assert_scored_commensurate` (max_len == 13,312 ∧ eval_max_new ≥ 13,312) |
| env-lock extension (b, done) | `assert_mamba_env_locked()` at mamba construction; `triton` in shared `_EXPECTED` |
| gcc/compile eval-path fix (18.5) | §4.1 (sbatch) + §5 (compile decision) |

## 8. Unchanged from the locked spec (cited, not re-derived)

Geometry-fidelity decision axis (NOT NLL; lexicographic ranking, §7); per-city worst-case
over the 4 held-out EU cities; the resolution seam + **binding-city power gate +
munich→manchester** escalation (§8); compute-optimal `D∝N`, **measured node-h** x-axis,
param-matched `N` {30/100/300M/1B} (§5); the decision layer — `decide()` 5 teeth +
`pick_winner` (**memorization-first hard-halt** → structural → power-gated worst-case); the
scored mechanics (`--scored-run`, `--shard-cache` streaming + stale-HALT, USR1
verified-resubmit, end-state markers, buildability dry-run).

## 9. Seeds

Per locked spec §8: seed-repeats sized so **per-run score noise < the must-rank gap**. The
concrete count is set at plan time from the diagnostic's measured per-run noise — not
guessed here.

## 10. Deferred — not dropped

`discrete-diffusion` — the spec §9 "second wave": a minimal MDLM-family absorbing-state
diffusion with **three quarantined divergences** (denoising/ELBO loss, T-step generation,
bidirectional mask). The absorbing mask (`src/cfm/models/diffusion/mask.py`) already
exists. Re-opens as its own design→implement→verify once the 2-backbone signal is in.

## 11. Risks + mitigations

- **Compile thrash on variable shapes** → §5 measure-then-decide with a stated threshold.
- **Eval-budget overrun** → §6 projection-before-commit + `n_cells` minimized to the
  must-rank gap.
- **Feasible-ladder collapse** (eval eats the envelope) → locked decision-basis step
  function absorbs it (`FIXED_SCALE_PLUS_S13`); reported, not silent.
- **Sealed-code regression** (the MicroAR refactor) → §3.2 bit-identical behavior-
  preservation teeth + the existing sealed tests.
- **Per-token cost mis-projection** → measure on the real length distribution at the
  diagnostic, not extrapolated from the 2048 diagnostic eval blindly.

## 12. Out of scope (named, not vaguely deferred)

discrete-diffusion (§10); tile cell-to-cell coherence / macro-planner (the locked spec's
Phase-3-opening winner-preview gate); a second extraction region (the locked spec's
parallel-escalation trigger if the pilot resolution gap lands within noise).

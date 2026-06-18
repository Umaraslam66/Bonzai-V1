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

## 1A. LOCKED DECISION (2026-06-18) — single fixed scale, N ≈ 53M (clean-1:7 ratio-constrained; supersedes the scale ladder)

**Umar's word, 2026-06-18.** Phase 2 is a **SINGLE fixed-scale bake-off, NOT a scaling
curve.** One rung: **N ≈ 53M** (the original aim was ~50M; the rung LANDED at ~53M — see the
"WHY ~53M" rationale below — and is **never labeled "50M"**), both backbones
(`transformer-ar`, `mamba-hybrid`), **`--no-compile`** (T7 verdict). There is no ladder:
`feasible_ladder` / `decision_basis` and the "drop 1B / the top rung" language below (§6, §8,
§11) are **superseded** — the decision basis is fixed-scale **by choice** (the
`FIXED_SCALE_PLUS_S13` family: decide at the single scale + §13), not by ladder collapse.

**Why ~53M and clean 1:7 (rationale to preserve):** Chinchilla ≈ 20 tok/param puts the
fully-trained optimum near **30M** for our **~624M unique EU train tokens**
(`TRAIN_TOKENS = 623,900,790`). **100M** was the measured diagnostic rung. **~50M was the
chosen middle** — but the rung was re-derived under a **clean-Jamba-ratio constraint** and
**landed at ~53M, not 50M**. The reason: a pure param-match to 50M instead picked `d640/14L` =
**1 transformer + 13 mamba (13:1)** — **attention-starved, below Jamba's validated 1:7 sweet
spot.** A clean **1:7 interleave** within the **≤2% param-match gate** is **unreachable near
50M**, so the rung moved to **`d512` / ~53M** (param-match-to-50M and clean-1:7 are **mutually
exclusive near 50M**; clean 1:7 wins). At **N ≈ 53M**, r≈20 → **~1.06B training tokens →
~1.7× data reuse** to its flattening horizon, inside the safe ≤~4× reuse band → **nearly
fully-cooked**, while **more meaningful than 30M**. The T9 diagnostic loss curve (110k steps,
r≈40 via ~5.8× reuse) **flattened** (mean train_loss 2.60→2.09; last 30k steps moved ~0.026
nats), confirming the data saturates well before r=40 — so a ~53M rung at r≈20 is well-cooked,
not data-starved.

**~53M is a NEW param-matched rung — NOT in the locked table {30/100/300M/1B}.** It carries the
FULL §3.3 param-match discipline (a VERIFIED gate): derive transformer-ar AND mamba-hybrid ~53M
configs, **count ACTUAL built params**, assert **≤2%** — never eyeballed (equal-depth ≠
equal-capacity). Appended to `bakeoff_scales.py` (append-only; the existing {30/100/300M/1B}
rungs are untouched).

**LOCKED configs (DONE 2026-06-18 — actual-build counts, `d_model=512` shared, ratio-constrained
clean 1:7 Jamba; see handoff `2026-06-18-t9-gate-53m-locked.md`):**
- transformer-ar: **`d_model=512, n_layers=14, n_heads=8` → 52,798,948.**
- mamba-hybrid: **`d_model=512, n_layers=24, transformer_every=7` → 53,733,348** — a **clean
  1:7 Jamba** interleave (**21 mamba + 3 transformer**, transformer layers at **8 / 16 / 24**).
  delta = **1.77% ≤ 2%.**
- **DONE (2026-06-18):** the actual-count derivation was run **ratio-constrained** by
  `scripts/rederive_53m_ratio.py` on the unified Leonardo env; the verified pair is **LOCKED**
  into `bakeoff_scales.py` under the **`"53M"` key** (append-only). The real
  `tests/models/test_bakeoff_param_match.py` **passes at 53M and is non-vacuous** (a >2%
  perturbation **REDS** it). The param-match rung is locked — the only thing still deferred is
  the **eval-sharding GPU equivalence golden** (plan Task 11 Step 4).

**UNITS (confirmed 2026-06-18 against `saldo -b` + `scontrol show node`):** the grant
(AIFAC_P02_548) is **40,000 core-h** (`saldo` "local h"); a Booster node = **32 cores + 4×A100**
(`CPUTot=32, Gres=gpu:a100:4`, full node bills `billing=32`). So **40,000 core-h = 1,250 node-h
= 5,000 GPU-h.** The "5,000" ceiling is **GPU-HOURS** (= the full 548 grant); earlier "5,000
node-h" was a MISLABEL. `gen_seconds_per_token` is a **single-GPU** rate — post-train eval runs
on **rank 0 only** (1 GPU works; the node's other 3 GPUs are allocated-and-billed but idle).

**Budget re-scaled to ~53M** (interpolation from the 100M diagnostic — NOT a direct ~53M
measurement): single-GPU per-token `0.026779 s/tok` (@100M) × ~53M scale (≈0.47–0.51) × **13,312**
full-cap × **Σ per-city held-out 1,859 cells** (523/579/156/601, glasgow/eisenhüttenstadt/
munich/krakow) × **2 backbones**, with the transformer full-context correction (tf ~×2.7,
mamba-hybrid ~×1.2 — attention/KV grows with the 13,312 context; an architecture-dependent
eval-cost gap to carry into the verdict). **Per-seed eval ≈ 336 node-h wall-clock** (tf 233 +
mamba 103):

| seeds | AS-IS (rank-0 eval, 3 GPUs idle) | with 4-GPU eval sharding |
|---|---|---|
| 1 | 336 node-h = **1,344 GPU-h (27%)** | 84 node-h = 336 GPU-h (7%) |
| 2 | 672 node-h = **2,688 GPU-h (54%)** | 168 node-h = 672 GPU-h (13%) |
| 3 | 1,008 node-h = **4,032 GPU-h (81%)** | 252 node-h = 1,008 GPU-h (20%) |

vs the **5,000 GPU-h (= 1,250 node-h)** grant. Training (r≈20 @~53M ≈ 1.06B tokens) is small
(~<15 node-h total; DDP uses all 4 GPUs, no waste). **AS-IS, 3 seeds eats 81% of the entire
3-month grant** — the rank-0 eval wastes 3/4 of billed GPU-h, so **4-GPU eval sharding is the
decisive ~4× lever** (and is what makes the 7/13/20% figures real). 1 seed AS-IS (27%) is
comfortable; 2–3 seeds want the sharding.

**Seeds + eval-sharding — LOCKED (Umar's word, 2026-06-18):** **3 seeds** per backbone +
**4-GPU eval-sharding**. Rationale: the verdict needs seed-repeats to separate skill from
luck on param-matched (near-tie) models; AS-IS rank-0 eval bills 4 GPUs for 1 GPU's work
(3 seeds = 81% of the grant), and sharding recovers ~4× → 3 seeds ≈ 20%, leaving headroom
for the eval-heavy held-out workload. The **seed→verdict combination rule** and the
**sharding equivalence golden (2 teeth)** are specified in the plan's Task 10 / new
eval-sharding task; `bakeoff_scales.py` stays untouched.

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
  capacity target, not improvised per run.
- **Param-matched, NOT layer-matched — a VERIFIED gate (locked spec §8).** The per-scale
  table must land `transformer-ar` and `mamba-hybrid` at the **same parameter count** at
  each scale {30/100/300M/1B}, **not the same layer count.** A Jamba 7:1 interleave has
  different per-layer params than a pure-transformer layer, so **equal-depth = unequal-
  capacity**, which **confounds architecture with capacity and breaks the bake-off's
  validity condition** — a transformer-at-90M vs mamba-at-110M comparison measures
  capacity, not architecture. Param-matching is what isolates the variable. This is
  teeth-bearing: the plan's table is a gate that **counts the actual built model's
  parameters** (`_param_count` on the constructed module — never the eyeballed mapping) and
  asserts each backbone-pair's counts match **within a stated tolerance per scale**; the
  mixer's width/depth knobs are tuned to hit the param target at each scale.

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

> **Resolved (2026-06-18, §1A):** the diagnostic ran; the budget is re-scaled to the locked
> **single ~53M rung** (~336–1008 node-h eval for 2 backbones × {1,2,3} seeds, ≪ 5000). The
> "feasible ladder / drop 1B" language below is the pre-diagnostic framing — superseded by the
> fixed-scale decision. The per-token measurement + full-context caveat below still apply.

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
guessed here. **LOCKED (2026-06-18): 3 seeds + 4-GPU eval-sharding.** Against the **5,000
GPU-h (=1,250 node-h)** grant, AS-IS rank-0 eval costs 27/54/81% for {1,2,3} seeds; sharding
(~4×) drops 3 seeds to **20%**, so 3 seeds is affordable WITH sharding. The seed-repeats are
what let the power gate separate skill from luck on the near-tie param-matched models — the
seed→verdict rule is in plan Task 10.

**Two-floor closure (LOCKED 2026-06-18; the MIDDLE band is the likely near-tie outcome, not an
edge case):** at each city the winner-vs-runner-up mean-KS gap must clear
`effective_floor = max(C/√n, seed_noise)` — `C/√n` = statistical *resolvability*
(`single_region_floor_gap`), `seed_noise` = *reproducibility* (max of the two backbones' seed-SEM).
Three bands: **DECISIVE** (gap > both → crown), **LUCK** (gap ≤ seed_noise), **MIDDLE** (clears one
floor but not the other). A winner is declared **ONLY** when DECISIVE; LUCK and MIDDLE are both
non-decisive (demoted, worst-first per the #21 gate). **Neither floor dominates — the larger binds
per-city; either failing alone blocks the crown.** If NO city is DECISIVE → **`NO_DECISIVE_WINNER`**
(S13 / `FIXED_SCALE_PLUS_S13` family), a NAMED verdict — never improvised in a later session, never
a bare exception.

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

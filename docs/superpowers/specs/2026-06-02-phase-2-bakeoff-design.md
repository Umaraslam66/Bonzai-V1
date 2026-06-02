# Phase-2 Architecture Bake-off — design

**Status:** design locked via brainstorm 2026-06-02 (9 topics, topic-by-topic gate discipline); awaiting spec review → implementation plan.
**Branch:** `phase-2-bakeoff` (local-first; merge to `main` + push at sub-project end once suite-green + `reports/` summary written; PR optional per CLAUDE.md).
**Protocol:** `docs/protocols/sub-project-planning-protocol-v3.md` (six gates + six principles + §9 construction-identity exclusion + §10 freeze-gate principles).
**Predecessor:** Phase-1 training scaffold (CLOSED + merged to `main`, tip `78ddc18`) + the 300M de-risking probe (`50b8b01`). The scaffold closed the loop `tokens → train → decode → eval` on 4×A100; this sub-project reuses its plumbing and answers PRD §6 (which architecture scales best).
**Authoritative:** PRD §6 (candidates) + §6.4/§10 (budget) + §9 (eval) + §11 (phases). Where experiments disagree with the PRD, the PRD is updated (flagged inline; "experiments win").

---

## 1. Goal & scope boundary

Decide **which sequence-model backbone to scale up for the production run**, by training each candidate at several sizes, scoring its generated geometry against the frozen Singapore holdout, fitting a scaling curve of *geometry-fidelity vs measured compute*, and picking the backbone whose curve extrapolates best to production compute. The methodology is PRD §6's scaling-curve extrapolation ("right by construction — if you can't extrapolate the curve, you can't make a confident scaling decision").

### The load-bearing reframe (open here, keep it front of mind everywhere)

The 300M probe relocated the binding constraint. **Training is NOT the constraint; the eval pass is.** But the probe's headline number was measured in an under-trained regime and must be corrected (see §2). The whole bake-off is sized and budgeted around the **eval pass** (generate → decode → score), not training.

### PRD §6 update — roster reshaped from "4 architectures × 3 scales" to "3 backbones × 4 scales" (experiments win)

Two locked decisions reshape the PRD §6 roster:
- **Cell-level collapse (Topic 5):** the bake-off compares CELL-level generators. At cell level, PRD candidate 1 (pure-AR) and candidate 2's micro-generator (hierarchical-AR) are the **same backbone** — their only difference (the macro/micro split) lives at the *tile* level. So the four PRD candidates collapse to **three backbones**: `transformer-AR`, `mamba-hybrid`, `discrete-diffusion`. The pure-vs-hierarchical *packaging* is no longer a bake-off candidate; it is the Phase-3 question on the winning backbone.
- **Ladder extension (Topic 9):** because training is cheap *enough* and production is a separate/later allocation, the ladder extends UP to ~1B to shorten the extrapolation. Scales = **{30M, 100M, 300M, 1B}**.

→ **Roster = 3 backbones × 4 scales = 12 runs** + 1 task-1 diagnostic. (Coincidentally 12, but the axes differ from PRD's 4×3.)

### Explicitly OUT of scope — named follow-ons (named, not vaguely-deferred)

- **Tile cell-to-cell coherence / macro-planner / boundary-contract stitching** → a **named, committed Phase-3-OPENING winner-preview gate**. Phase 3 opens by building the macro-planner + stitching for the winning backbone and running a tile-coherence preview *before* committing the full hierarchical build. NOT a vague "later."
- **Full cross-dimensional conditioning-compliance** (PRD §9 European-vs-Asian) → deferred to the **second region** (single-region Singapore only varies density meaningfully).
- **Generalization metric** (train-A-eval-D) → deferred to the **second region**.
- **The production run** (~500 node-h) → a separate/later allocation (confirmed by PI 2026-06-02); NOT drawn from the current June-11 window.

### UNSCORED → SCORED transitions (Topic 5 — named, never implied by a passing number)

| Scaffold follow-on | Bake-off disposition |
|---|---|
| Learned right-angle bar (PoC 95%) | **SCORED**, emergence-gated; *reported*, not a hard gate at small scale |
| Value-bearing conditioning | **BUILT** (wire `read_tile_labels` values into the conditioning vector; scoped to within-Singapore-varying dims, primarily density) |
| Conditioning-compliance scoring | **SCORED secondary/reported** (density dim only; full version → second region) |
| Tile cell-to-cell coherence | **DEFERRED** to the Phase-3-opening winner-preview gate |

---

## 2. The eval-cost reframe & budget (the corrected headline goes FIRST)

**Number correction — do not inherit the probe's headline.** "12 runs ≈ 4.7 node-h, ~80× under budget" was measured at the probe's **r ≈ 0.11 tokens/param — ~190× under compute-optimal**. At compute-optimal **r ≈ 20**, the 3-scale (30/100/300M) training is **≈ 49 node-h (~7.6× under the 375-node-h PRD envelope)**, not 80×. **The reframe's *direction* holds (training ≪ eval; training under envelope), but the *magnitude* was an under-trained-regime artifact.** Every downstream cost estimate in this spec is computed against ~7.6×, NOT the inherited 80×. "Training is free, spend freely" is wrong by ~10×; headroom is **comfortable, not infinite.**

**Why eval binds.** Eval cost ∝ `n_cells` (cells generated + scored per run), and `n_cells` is set by *how finely we must resolve the architecture-to-architecture geometry gap* (§8 resolution, in the correct unit). Training is fixed-and-cheap; **eval scales with the discrimination we demand.** The eval-cost model's central job is to find the **minimum (`n_cells`, `eval-max-new`, seed-repeats)** that still resolves the must-rank gap AND still reaches buildings.

**Budget (current allocation `AIFAC_P02_222`, ends 2026-06-11, ~1,222 node-h remaining; production is a SEPARATE/later allocation):**

| Item | Est. node-h | Notes |
|---|---|---|
| Task-1 diagnostic | < 1 | one 30M/100M run; triple-duty (emergence, geometry-r, eval-cost/scale) |
| Bake-off training, 3 scales (30/100/300M × 3 backbones) | ~49 | corrected compute-optimal figure |
| 1B extension (× 3 backbones) | ~477 | the ladder-shortening points; the biggest training line |
| Eval pass (binding) | ~200 | dominated by the large-scale evals; sized by §6/§8 |
| **Subtotal** | **~726** | of ~1,222 |
| Reserve (second-region extraction, re-runs, contingency) | ~496 | second region ≈ its own data sub-project (~8h+ cold fetch) |

---

## 3. Topic 1 — Decision axis (what "win" measures)

**Geometry-fidelity is the cross-architecture decision axis.** Curves are fit on a geometry score computed from *generated + decoded output*; a single winner is chosen by best value **extrapolated to production scale**.

**Why not eval-loss (NLL):** NLL is *incomparable across families* — an AR backbone reports an *exact* per-token surprise; discrete diffusion reports only a *variational upper bound (ELBO)*. Different quantities on different rulers; a pure-NLL curve literally cannot rank diffusion against the AR backbones. Geometry-fidelity scores decoded *output*, so every backbone is measured on one ruler — that is the bake-off's comparison-validity condition. This also resolves the PRD §6-vs-§9 tension (§6 says "evaluation loss"; §9 defines evaluation as the geometry suite) toward the metric that can validly rank all candidates. (Reframe-coherence: eval is the binding cost *because* the only architecture-agnostic honest metric requires generation.)

**NLL is retained as an AR-family-only diagnostic, with a structural guard:** it is a *cross-check* on the AR-family geometry ranking; if NLL and geometry disagree on AR ranking, that is a flag to **investigate the geometry measurement**, never a reason to switch axes. NLL **never** ranks across families and **never** overrides the geometry decision. (The trap is asymmetric — NLL exists for the 2 AR backbones, not diffusion, so leaning on it structurally advantages the AR family in the one place diffusion can't follow.)

Metric set/priority → §7. Tie-break → §11.

---

## 4. Topic 2 — Compute axis

- **Rule = compute-optimal `D ∝ N`** (tokens proportional to params, ~`r` tokens/param). All scales sit on the efficient frontier, so up-extrapolation predicts a *properly-trained* production model — the textbook Chinchilla method. Fixed-absolute-D was rejected: it under-feeds larger models, sits off the frontier, and up-extrapolating it predicts an under-fed production model and can flip the ranking toward data-thrifty backbones. Fixed-wall-clock rejected: it rations a constraint the probe proved absent.
- **x-axis = MEASURED compute (node-h)**, NOT the `C ≈ 6ND` formula. 6ND is a *transformer* FLOP approximation; Mamba and diffusion have different FLOP/param/token ratios (the 6ND-across-backbones regime-transfer catch, §9). Theoretical FLOPs recorded as a **hardware-agnostic secondary** (guards against penalizing Mamba for kernel immaturity). Measured node-h is decision-relevant: it is what the production run will cost.
- **Size knob = param-matched N** (30/100/300M/1B identical across backbones); the curve's x lands at each backbone's own measured node-h.
- **Extrapolation target = production compute** (~500 node-h; §11).
- **`r`'s magnitude is geometry-verified, not borrowed.** r ≈ 20 is the LLM ratio; our tokens are sub-F geometry (vocab 1508, different redundancy/structure). r's magnitude is `max(Chinchilla, emergence-floor)`, **MEASURED in §5**, not locked here.
- **Diffusion-r deviation:** "same r tokens/param" is same-data-*exposure* fairness, not same-*convergence* fairness; diffusion's objective may need more tokens. If it under-trains at the shared r, raise r for all diffusion runs and log the deviation (§10).

---

## 5. Topic 3 — Building-emergence floor + geometry-r (MEASURED as bake-off task 1)

The probe's `n_polygons=0` is **confounded across three causes** — ~190× under-training, value-agnostic conditioning, and possible eval-truncation — and the data only firmly establishes under-training. (`ogc_valid=1.0` next to `n_polygons=0` is the §2 "broken-but-in-range" vacuous top score, in the flesh.) The probe cannot bound geometry-r (it stopped at r≈0.11). **So both numbers are measured, not guessed, as the first plan task.**

**Task-1 diagnostic** — one cheap run (30M/100M), staged to isolate the confound in **cost-to-rule-out order**:
1. **Rule out eval-truncation FIRST** (cheapest; can fake `n_polygons=0` at any training level). Set `eval-max-new` generous enough that generation provably *can* reach polygon tokens; instrument: does the generated sequence contain building-class tokens that simply didn't close into polygons, vs no building tokens at all?
2. **Then training** (the diagnostic's main axis): train with the *existing value-agnostic conditioning*. If buildings emerge by r≈20–40 → under-training was the cause; **value-bearing conditioning demotes to a §7 quality lever, not an emergence gate**; geometry-r reads off the same loss curve.
3. **Then conditioning** (only if buildings absent after 1–2 are cleared): build value-bearing conditioning and re-test.

**Two definitions, one source each:**
- **"Buildings emerged" = a non-vacuous, holdout-density-tied threshold** (a count/rate meaning "the model reliably produces buildings where real data has them"), NOT `n_polygons>0` (one stray polygon is noise). This threshold is the **same source** as the §2 structural guard.
- **§2 structural guard:** `n_polygons < floor` → **floor score** (failure to produce buildings) + loud flag, NEVER a vacuous pass like `ogc_valid=1.0`. (This is §2 threshold-pairing — we surface the failure — NOT §9 exclusion.)

**Stopping rule = "train until loss flattens OR buildings emerge, report where each happens"** — NOT a fixed r≈40 ceiling. If neither has happened by r≈40, that is a finding (*geometry-r is high*), not license to conclude conditioning is required. r≈40 must not become the new borrowed-constant ceiling the way r≈20 was the borrowed floor.

The diagnostic does **triple duty**: emergence floor, geometry-r, and per-scale eval-cost-per-cell (it already generates→decodes→scores).

---

## 6. Topic 4 — Eval-cost model (per-family; reframe AR-scoped)

**Two cost profiles, because diffusion is not autoregressive:**
- **AR-family (transformer-AR, mamba-hybrid):** generation cost ∝ `L` sequential forwards/cell (one token at a time). The binding cost. (Mamba is AR/sequential but cheaper per step — constant recurrent state vs growing KV cache.)
- **Discrete diffusion:** generation cost ∝ `T` denoising full-sequence forwards. **T is a chosen knob decoupled from L.** The "per-token slow" framing does NOT apply. Sized independently when diffusion is built (§9).

**The reframe is scoped to the AR family;** diffusion has a separate parallel-denoising profile. What is **identical across families is the eval CONTENT** — same cells, same conditioning, same holdout comparison — so the Topic-1 one-ruler axis holds; only production *cost* differs.

**Diffusion T fairness (hard constraint, → §9):** **T is set by quality-convergence** (denoise until the geometry score plateaus); cost follows. Never cap T to fit a budget and then compare a capped (artificially-weak) diffusion model against fully-generated AR models — that turns the cost split into a silent quality handicap.

**Eval frequency = final checkpoint only (per run), overriding PRD §11's "every checkpoint"** (eval is binding). Each run is one (measured-compute, score) point. **"Converged" = loss-flat AND past the §5 emergence floor** (one source with §5) — NOT loss-flat alone, because loss can sit flat at `n_polygons=0`. Convergence itself is confirmed cheaply by the loss curve (no generation needed); generation is only paid at the final, converged checkpoint.

**Sample size = a measured constrained minimum, set jointly with §8.** `n_cells` bounded below by §8 resolution (in the per-feature unit) and above by cost; `eval-max-new` bounded below by §5 emergence; seed-repeats sized so per-run score noise < the must-rank gap (§8). Per-scale AR eval node-h **measured by the task-1 diagnostic**; diffusion eval cost measured when diffusion is built.

---

## 7. Topic 5 — Scored metric set & ranking composition

**Cell-level 3-backbone comparison** (see §1 collapse). **Lexicographic ranking — NOT a blended composite** (blend weights are the arbitrary knob rejected in Topic-1 Option C):

1. **Gates:** emergence floor (`n_polygons ≥ floor`, §5) + basic validity (decodability, OGC-valid above a floor). Right-angle is *reported*, not a hard gate at small scale (95% is too high to gate small models or the curve vanishes).
2. **Rank the curve** on a continuous **statistical-realism KS distance vs the holdout** (building-size / road-length distributions) — continuous + smooth (good for curve-fitting), and it IS what §8's resolution seam measures. This is the curve's y-axis.
3. **Report** right-angle rate (its march toward the 95% PoC bar across scales is itself a finding) and density-compliance.

**Value-bearing conditioning is BUILT** (wire `read_tile_labels`-derived values into the conditioning vector — the derivation already exists, scaffold carry-forward trigger #2 — scoped to within-Singapore-varying dims, primarily density bucket). **Density-compliance SCORED secondary.** Full cross-dimensional compliance → second region.

---

## 8. Topic 6 — Resolution seam / Trigger-3 (LIVE at ≥2 backbones)

The seam (`assert_resolution_sufficient`) is wired from the scaffold; the bake-off makes it live by feeding it **real architecture-to-architecture KS gaps**.

**The resolution is RE-DERIVED in the per-feature unit — NOT the inherited 0.076.** The frozen 0.076 (resolved) / 0.049 (single-region floor) were derived for the **per-cell density-representativeness** of the held-out set. The bake-off ranks on a **per-feature geometry-realism KS** (building-size / road-length). Different observation unit, different sample size, different noise floor. Inheriting 0.076 is the §10.3 "vacuous pass relocates into the sample size" trap. (Same class as the emergence-confound — checking what a number IS, not what it's labeled.)

**Re-derivation discipline (so the trap doesn't move one level down):**
- Re-derive the per-feature resolution **on the actual generated + holdout feature populations** (building-size, road-length samples), NOT a per-feature number borrowed from a KS power table — the same "measured on the real distribution" discipline that produced 0.076 properly.
- **Check against the BINDING (worst-resolved / coarsest) feature distribution.** Building-size and road-length may have different per-feature resolutions; one seam, checked against the worst-resolved feature, or a pair resolvable on roads but not buildings gets falsely green-lit.

**Firing condition = winner-vs-runner-up gap unresolved — NOT any-pair.** The decision is "pick the winner," so the load-bearing pair is winner-vs-runner-up. A tie for last place is irrelevant to the decision and must NOT trigger an 8h+ extraction.

**Early-warning + parallel escalation:** estimate the per-feature resolution and a **pilot winner-vs-runner-up gap early** (from the task-1 diagnostic / a pilot backbone pair, available after backbones 1–2 — before diffusion finishes). Action contract: pilot gap within noise → **kick off second-region extraction in parallel** rather than blocking after all runs (extraction is slow — a new region ≈ its own data sub-project). The 3↔4↔6 loop: per-feature resolution ∝ `n_cells × features-per-cell` → `n_cells` (§6) bounded below by the must-rank resolution; features require emergence (§5).

**Escalation = second region (Sweden or Sri Lanka, the locked de-risking set) — TRIPLE DUTY:** resolves the gap **+** enables the PRD-§9 generalization test **+** unlocks full cross-dimensional conditioning-compliance (§7 deferral).

---

## 9. Topic 7 — Candidate builds (3 backbones, risk-staged)

**Build structure = one SHARED scaffold + a swappable backbone.** Shared: embedding, value-bearing conditioning prefix, vocab head (1508), training harness, eval. The backbones differ only in their sequence-mixing layers (+ diffusion's quarantined divergences).

- **transformer-AR** — EXISTS (`micro_ar.py`). Baseline; the task-1 diagnostic vehicle; the §11 tie-break default.
- **mamba-hybrid** — interleaved Mamba + transformer (~7:1, Jamba-style) via the **`mamba-ssm` package** (CLAUDE.md mandate — not a custom SSM). A **drop-in backbone swap** (same head/mask/loss/AR-generation).
- **discrete-diffusion** — **minimal MDLM-family absorbing-state (masked) diffusion**, implemented against the shared scaffold (chosen over adapting SEDD/MDLM research code: comparability-is-the-bake-off's-validity-condition, and a ported codebase risks *silent* comparability breakage via diverging conditioning/vocab/normalization assumptions). **Three divergences quarantined to dedicated modules:** loss (denoising/ELBO, not next-token CE), generation (T denoising passes, not AR), mask (bidirectional, not causal). "Defer diffusion to a second wave" (run 2 AR backbones first) = **named build-risk fallback**, not the plan.

**Identity-lock test (the comparability proof, → §10):** assert the SHARED parts are *literally shared* — all 3 backbones consume the **same** conditioning-prefix builder, the **same** vocab/head object, the **same** eval-content path — **by identity** (`is`-lock, like the scaffold's conditioning derivation), not "they produce equal output today." The quarantine is the design; the identity-lock is the proof the quarantine held. (A diffusion run scoring differently could be the architecture OR a forked conditioning path — the identity-lock is the only way to tell which.)

**Risk staging:** transformer-AR (exists; runs the diagnostic) → mamba-hybrid → discrete-diffusion. Dovetails with §8: the pilot winner-vs-runner-up gap is available after backbones 1–2, kicking parallel extraction *before* diffusion finishes.

**Diffusion's three accommodations (all documented in the §10 deviation log):** T by quality-convergence (§6); raise-r-if-under-trained (§4); geometry-only ranking (no NLL cross-check, §3) — one more reason the geometry metric must be solid.

---

## 10. Topic 8 — Comparability lock & DDP discipline

**Carried unchanged (all runs):** the version lock asserted at every GPU entrypoint (`assert_training_env_locked`); DDP gotchas ([[feedback_ddp_determinism_gotchas]]) — **seed before model init**, `save_checkpoint` is **collective (all ranks)**, **`WorldSizeGuard` non-vacuous** (`world_size==4`, real ranks), functional identity at **float32-ε (atol 1e-4)**, not bit-identity; CSVLogger for loss curves; every run is a `reports/` entry (config + commit + data-snapshot fully determine it). All on 4×A100 DDP ([[feedback_leonardo_full_node]]).

**New requirement at 1B scale — across-job-boundary resume (the scaffold did not exercise this).** The scaffold proved 30-min-checkpoint + resume-after-*failure* at small scale. But 1B runs are long, Slurm jobs have hard wall-clock limits, so **a 1B run will SPAN MULTIPLE JOBS** and needs **automatic checkpoint-and-resume across job boundaries** (a job hitting its time limit relaunches and continues from the last checkpoint), not just crash recovery. Checkpoints land on **`$WORK`** (allocation-independent storage that survives allocation expiry), so a couple-day renewal gap means "resume when the new allocation lands," NOT "start over." This is the insurance that makes the §11/§15 June-11 wall-clock risk *survivable rather than lost*. Tested at Task 10: kill a job mid-run → relaunch → continues from the last checkpoint, not step 0.

**New lock event — `mamba-ssm` (the biggest operational risk).** `mamba-ssm` + kernel companions (`causal-conv1d`, often `triton`) have tight torch/CUDA constraints and may not sit on the locked **torch 2.5.1+cu121**. Discipline (verify-before-lock, [[feedback_verify_before_lock_not_after]]): **verify `mamba-ssm` imports + runs a forward/backward on Leonardo under the EXACT locked stack BEFORE locking it.** If it forces *any* torch/CUDA change → **re-lock event: ALL backbones (including the already-run transformer-AR) re-run under the new lock** (a mid-bake-off version change breaks the comparability that makes the curves comparable). **Resolve this right after the diagnostic, before Mamba** — an explicit early plan task with the re-lock-all contingency stated.

**Optimizer recipe = identical across all runs + documented-deviation valve.** One recipe (lr/schedule/warmup; **effective batch held constant via gradient accumulation** so per-scale memory limits aren't an uncontrolled variable), tuned once on the task-1 diagnostic. If the diagnostic shows the largest scale needs a different lr, apply a **principled width-scaling rule identically across backbones**. Per-architecture deviation governed by three constraints so it stays principled, not outcome-driven:
1. **Trigger = demonstrably FAILS TO TRAIN** (diverge / NaN / flatline), never *scores lower*. Bright line: fix a backbone that *can't* train; never tune one that *trains-but-loses*.
2. **Decided on the diagnostic, pre-scored-runs** — never a reaction to a ranking already seen (recipes locked before the scored runs start).
3. **Same principled rule applied uniformly** (e.g. loss-scale-normalized lr / width-scaling law), not a bespoke per-backbone number (a bespoke number is the rejected per-run-tuning confound in disguise).

Each deviation logged in the **comparability-deviation log** (same register as diffusion-r / diffusion-T).

**Seed discipline extended to diffusion:** mask-sampling (training) and denoising-sampling (generation) are both seeded; the generation seed is part of the §6 eval-seed-repeat count.

---

## 11. Topic 9 — Curve robustness & the "doesn't separate" branch

- **Extrapolation target = production compute ~500 node-h** (PRD §10's 2000 GPU-h; implies N_prod ≈ 1.75B, in PRD's 30M–3B range), refined in Phase 3 when production scale is actually estimated.
- **Ladder extended UP to ~1B** (4 points: 30/100/300M/1B) to shrink the extrapolation from ~34× (3 points) to ~3×; the 1B eval is the highest-value point (closest to target). Production is a separate allocation, so this fits (~726 of 1,222; §2).
- **Fit with uncertainty** — bootstrap CIs over cells/seeds; the extrapolated C_prod score carries a confidence interval, not a bare point.
- **"Doesn't separate" = winner-vs-runner-up EXTRAPOLATED CIs overlap at C_prod** — i.e. **reuse §8's resolution seam at the extrapolated point**, not just the measured points. One mechanism guards both.
- **Pre-committed §13 tie-break (decided NOW, not post-hoc): pick the simplest backbone = `transformer-AR`** (baseline; no mamba-ssm kernels; no diffusion loss/gen complexity; CLAUDE.md default-to-simplicity; PRD §13 "pick simplest"). Documented with evidence when invoked. Deciding it now prevents a "we picked transformer-AR because nothing separated" outcome from masquerading as "transformer-AR won."
- **§2 structural check on any crowning:** the winner needs a *valid* curve — monotonic-improving fit, acceptable residuals/CI — **AND** a C_prod gap exceeding the resolution. A garbage / non-monotonic fit cannot crown a winner; that routes to the §13 branch.

---

## 12. Run roster & budget (consolidated)

| Backbone | 30M | 100M | 300M | 1B | NLL cross-check? |
|---|---|---|---|---|---|
| transformer-AR (exists) | ✓ | ✓ | ✓ | ✓ | yes (diagnostic only) |
| mamba-hybrid | ✓ | ✓ | ✓ | ✓ | yes (diagnostic only) |
| discrete-diffusion | ✓ | ✓ | ✓ | ✓ | **no** (geometry-only) |

12 scored runs + 1 task-1 diagnostic. Each run: compute-optimal `D ∝ N` at geometry-verified r (§5), 4×A100 DDP, identical recipe (§10), final-checkpoint eval (§6) on the frozen holdout, one (measured-node-h, KS-realism) point + CI. Budget: §2 table (~726 used / ~1,222; ~496 reserve incl. second region).

---

## 13. Architecture, data flow & new code surface

```
frozen holdout (132 tiles) + sub-G validated Singapore (494) ──► [reused scaffold plumbing]
        build_training_shards (byte-deterministic) · CellDataModule (fail-closed holdout audit, all ranks, halt-before-batch-0)
                                       │
        ┌──────────────────────────────┴───────────────────────────────┐
        ▼                              ▼                                 ▼
  SHARED scaffold:  embedding · value-bearing conditioning prefix (read_tile_labels) · vocab head 1508 · harness · eval
        │  (identity-locked across backbones — §9)
        ├─► transformer-AR backbone   (exists; causal mask, next-token CE, AR generate)
        ├─► mamba-hybrid backbone     (mamba-ssm; ~7:1 interleave; drop-in)
        └─► discrete-diffusion        (absorbing-state; quarantined: ELBO loss · T-denoising generate · bidirectional mask)
                                       │
                          per scale {30M,100M,300M,1B}, compute-optimal D∝N, identical recipe, 4×A100 DDP
                                       ▼
        FINAL-checkpoint eval (converged = loss-flat AND past-emergence): generate n_cells on holdout conditioning
                                       ▼  sub-F sealed decoder → geometry
        gates (emergence + validity) → KS-realism distance vs holdout (curve y) + report right-angle/density-compliance
                                       ▼
        assert_resolution_sufficient(winner-vs-runner-up gap, per-feature unit, worst-resolved feature)
                                       ▼
        fit score vs MEASURED node-h (+FLOPs secondary), bootstrap CI → extrapolate to ~500 node-h → winner | §13 branch
```

**New / extended code surface:**
- `src/cfm/models/mamba_hybrid.py`, `src/cfm/models/discrete_diffusion.py` (+ quarantined diffusion loss/generation/mask modules).
- `src/cfm/training/env_lock.py` — extend the version lock with pinned `mamba-ssm`/`causal-conv1d`/`triton`; the comparability-deviation log.
- Value-bearing conditioning wiring (`read_tile_labels` values into the conditioning prefix).
- `src/cfm/eval/` — per-feature KS-realism distance; re-derived per-feature resolution; `assert_resolution_sufficient` fed real pairwise gaps; bootstrap-CI curve fit + extrapolation + the §13 structural check.
- Reuse `build_trainer(max_time=)`, per-step-node-h cost reporting, `--d-model/--n-layers/--n-heads/--batch-size/--max-time/--no-compile/--eval-cells/--eval-max-new`, `scripts/scaffold_scaleup_probe.sbatch`.
- `configs/experiments/bakeoff-*.yaml` — pydantic-validated; config + commit + snapshot determine each run.

---

## 14. Task decomposition (gate→test matrix — the spec backbone)

| # | Task | Gate | Discrimination test(s) | Depends on |
|---|---|---|---|---|
| 1 | **Task-1 diagnostic** (transformer-AR): rule out truncation → measure emergence floor + geometry-r + per-scale eval-cost; lock the recipe | §2 + §5 + §10.3 | truncation ruled out (building-class tokens present?) before any n_polygons conclusion; "emerged" = density-tied non-vacuous threshold (one source w/ guard); stopping = loss-flat-or-emerge, not fixed r; per-scale eval node-h reported | — |
| 2 | **`mamba-ssm` verify-before-lock** on Leonardo under the exact locked stack; re-lock-all contingency | verify-before-lock | import + fwd/bwd pass under torch 2.5.1+cu121; if any version change forced → re-lock event documented, all runs re-run under new lock | 1 |
| 3 | Value-bearing conditioning wiring (`read_tile_labels` values → prefix) | 6 + §3 | identity assertion (shared builder, fork-detection) + hand-enum value cross-ref (no builder in expected) | 1 |
| 4 | §2 emergence guard + lexicographic ranking + per-feature KS-realism metric | 2 + §9 | `n_polygons<floor` → floor score (NOT vacuous `ogc_valid=1.0`); guard FIRES on a real defect in the kept set; KS-realism continuous | 1, 3 |
| 5 | Per-feature resolution re-derivation + `assert_resolution_sufficient` fed real gaps | 2 + §10.3 | resolution derived on REAL generated+holdout feature populations (not borrowed); checked vs worst-resolved feature; fires on winner-vs-runner-up only; marker-absent raises | 4 |
| 6 | mamba-hybrid backbone (drop-in) + identity-lock test | 6 (identity) | `is`-lock: shares conditioning builder + vocab/head + eval path with transformer-AR; only sequence-mixing differs | 2, 3 |
| 7 | discrete-diffusion backbone (absorbing-state; quarantined loss/gen/mask) + identity-lock | 6 (identity) + §9 | identity-lock on shared parts; T-by-quality-convergence (not budget-capped); mask+denoising seeds set; geometry-only ranking | 2, 3, 6 |
| 8 | Comparability lock + deviation valve + grad-accum effective batch | §10 + threshold-pairing | recipe identical; deviation only on fails-to-train, pre-scored, uniform rule, logged; effective batch constant across scales | 1, 6, 7 |
| 9 | **Pilot winner-vs-runner-up gap** (after backbones 1–2) → conditional parallel second-region extraction | §8 early-warning | pilot gap estimated in per-feature unit; within-noise → extraction kicked in parallel (action contract fires) | 5, 6 |
| 10 | Run the 12-run ladder (3 backbones × {30/100/300M/1B}), final-checkpoint eval; **1B: across-job-boundary resume** | small-before-big + resumability | "converged" = loss-flat AND past-emergence; one (measured-node-h, KS-realism) point + seed-repeat noise < gap; **1B auto checkpoint-and-resume ACROSS JOB BOUNDARIES on `$WORK`** (survives Slurm wall-clock limits + allocation-expiry/renewal gap) — **test: kill a job mid-run → relaunch → continues from last checkpoint, not step 0** | 8 |
| 11 | Bootstrap-CI curve fit + extrapolate to ~500 node-h + §13 structural check + tie-break | 2 + §9 | valid monotonic fit + residuals/CI; winner C_prod gap > resolution; "doesn't separate" → pre-committed transformer-AR, documented | 10, 5 |
| 12 | `reports/` summary + PRD §6 update (roster reshape, cost correction) + Phase-3-opening winner-preview gate named | reproducibility | config+commit+snapshot per run; PRD edits flagged; Phase-3 gate written as a committed checkpoint | 11 |

---

## 15. Risks

- **`mamba-ssm` forces a torch/CUDA re-lock** (Task 2) → all runs re-run. Mitigated by verify-before-lock at the front, before any scored run.
- **Diffusion build risk** (loss/generation/mask from scratch) → named fallback: run the 2 AR backbones first, add diffusion only if build risk proves too high (§9).
- **Curves don't separate even at ~3× extrapolation** → pre-committed transformer-AR tie-break (§11); a real outcome, documented, not a failure.
- **Second-region extraction is slow** (~8h+ cold fetch; ≈ its own data sub-project) → early-warning pilot (§8/Task 9) kicks it in parallel; ~496 node-h reserve holds it.
- **June-11 WALL-CLOCK window, not just budget** (the ~477 node-h 1B extension must *complete* before allocation expiry, and a renewal gap may follow). The §2 table confirms budget *fit*; the plan must also confirm wall-clock *fit* and **sequence the ladder so the most decision-relevant points land first if the window tightens** (the winner-vs-runner-up-distinguishing scales before the curve-shape filler). Worst case (renewal slips past June 11 mid-1B-run) is recoverable *only because* of the §10 across-job `$WORK` resume — the two are linked: across-job resume is the insurance, wall-clock sequencing is the mitigation.
- **iCloud editable-install gotcha** ([[project_repo_location_icloud]]): confirm the venv / `pythonpath=["src"]` before the first run.
- **The five regime-transfer catches are a meta-pattern, not one-offs** — a number/method valid in one regime does not transfer to another: (1) NLL exact-vs-ELBO across families; (2) the emergence-confound (probe under-trained 190×); (3) the resolution unit (per-cell 0.076 ≠ per-feature); (4) 6ND across backbones; (5) the budget magnitude (80×→7.6×). **Audit every inherited number in this spec for "what regime was it measured in?" before trusting it downstream.**

---

## 16. Decisions log (load-bearing, for future readers)

- **Decision axis = geometry-fidelity** (not NLL); NLL incomparable across AR/diffusion families (exact vs ELBO); NLL kept as AR-only non-overriding diagnostic.
- **Compute = compute-optimal D∝N; x-axis = MEASURED node-h** (6ND is transformer-only); r geometry-verified in §5, not borrowed.
- **Emergence floor + geometry-r MEASURED as task 1**, staged truncation→training→conditioning; "emerged" = density-tied non-vacuous threshold (one source w/ §2 guard); stopping = loss-flat-or-emerge.
- **Eval-cost per-family** (AR ∝ L; diffusion ∝ T-by-quality-convergence); reframe AR-scoped; identical eval content; final-checkpoint-only with converged = loss-flat AND past-emergence.
- **Bake-off is CELL-level → 3 backbones** (pure/hierarchical collapse to one backbone at cell level); tile-coherence → named Phase-3-opening winner-preview gate.
- **Resolution re-derived in per-feature unit** on real populations, checked vs worst-resolved feature; seam fires on winner-vs-runner-up only; escalation = second region (triple duty).
- **Diffusion = minimal MDLM-family absorbing-state under the shared scaffold**, divergences quarantined, shared parts identity-locked; defer-diffusion = named fallback.
- **Comparability:** identical recipe + fails-to-train-only deviation valve (decided pre-scored, uniform rule, logged); `mamba-ssm` verify-before-lock + re-lock-all; seeds extended to diffusion.
- **Ladder extended to ~1B (4 scales)**; extrapolate to ~500 node-h; pre-committed transformer-AR tie-break; §2 structural check on crowning.
- **Cost correction:** compute-optimal training ≈ 49 node-h (~7.6× under envelope), NOT the probe's 4.7 / 80× (a 190×-under-trained-regime artifact); production is a separate/later allocation.

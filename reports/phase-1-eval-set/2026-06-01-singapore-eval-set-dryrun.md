# Eval-set generation — Singapore dry-run measurement (NOT FROZEN)

**Date:** 2026-06-01 · **Status:** DRY-RUN — measured + proposed, **manifest NOT frozen** · **δ review:** RESOLVED (relative form, below) · **Freeze pending:** final review of these v2 numbers.
**Code commit:** `48614ee` (branch `phase-1-eval-set-generation`) · **Spec:** `docs/superpowers/specs/2026-06-01-eval-set-generation-design.md` · **Plan:** `docs/superpowers/plans/2026-06-01-eval-set-generation.md`

## How to reproduce

```bash
uv run python -c "from cfm.eval.holdout import pipeline, paths; \
r=pipeline.generate_eval_set(release=paths.DEFAULT_RELEASE, region=paths.DEFAULT_REGION, lock=False); \
print(r.n, r.residual, r.ceiling_overall, r.per_stratum_bref_rate, r.underpowered_feature_strata)"
```

Data snapshot: `data/processed/{sub_c,sub_d,sub_f}/2026-04-15.0/singapore/` (494 tiles, sub-G `_PHASE1_VALIDATED`). Runtime ≈ 12 s (decodes all 494 tiles' round-tripped-real geometry).

## Config

| Parameter | Value | Where | Basis |
|---|---|---|---|
| release / region | `2026-04-15.0` / `singapore` | `paths.py` | spec |
| **ρ (`RHO_BREF_REGIME`)** | **0.5** | `sizing.py` | relative over-emission boundary (below) |
| **δ_floor (`DELTA_FLOOR_BREF`)** | **0.005** | `sizing.py` | near-zero backstop (below) |
| KS effect size (`_KS_EFFECT`) | 0.15 | `pipeline.py` | provisional; gates nothing yet (deferred) |
| residual cap (`n_cap_fraction`) | 0.50 | `pipeline.py` | did not bind |

## Measurement

| Quantity | Value |
|---|---|
| **Proposed held-out N** | **53 tiles** (441 training residual, ~89%) |
| Round-tripped-real geometric-validity ceiling | **0.967992** (≈3.2% bref-placeholder collapse — v1 tokenizer limitation, consistent with H3) |
| Underpowered strata (rate-detection / cell-reference) | **none / none** |

| cell_density_bucket | faithful bref-rate | over-emit threshold (abs) | **relative tol** | feature pop (pool) | feature floor | held-out features |
|---|---|---|---|---|---|---|
| 0 (sparsest) | 6.79% | 0.0340 | **+50%** | 76,013 | 211 | 10,138 |
| 1 | 3.82% | 0.0191 | **+50%** | 126,201 | 388 | 17,713 |
| 2 | 2.85% | 0.0143 | **+50%** | 451,555 | 524 | 54,292 |
| 3 (densest) | 2.33% | 0.0116 | **+50%** | 219,689 | 646 | 55,780 |

Total features across the pool: **873,458**.

## The chosen numbers — justified independently (NOT on joint feasibility)

### 1. The over-emission threshold — δ review RESOLVED: relative-to-base-rate

The first dry-run used an **absolute** δ=0.03. Against the measured per-stratum faithful rates (2.3–6.8%) that gave **per-stratum relative tolerances of +44%/+79%/+105%/+129%** — the dense-bucket guard was vacuous: a model could **more than double** the bucket-3 degenerate rate (2.33%→4.66%, +2.33pp < 3pp) and still pass. That is the per-stratum-vacuous-pass pattern one level up, in the threshold.

**Fix (form approved 2026-06-01):** `over_emission_threshold(faithful) = max(ρ·faithful, δ_floor)`.

- **ρ = 0.5** — a model is over-emitting iff its per-stratum bref-rate exceeds the faithful rate by **>50% relative**. The discrimination is now **uniform across strata** (the table's relative-tol column is a flat +50%), while the **absolute threshold varies per stratum** (0.0340→0.0116), tracking each base rate. The dense-bucket doubling that absolute-0.03 waved through now trips (regime-distinguishing guard `test_GD2_GUARD_dense_bucket_doubling_trips_under_relative_but_absolute_missed_it`). ρ is set on meaning, not to fit N (it does not move N — see item 3). It is **revisitable toward the data-supported ~0.25** once the model's natural over-emission variation is observed (model side deferred, spec §7); the data supports a tighter ρ because feature power is abundant.
- **δ_floor = 0.005** — a backstop for **genuinely near-zero strata only**. Verified: `δ_floor < ρ·faithful` for every current bucket (min ρ·faithful = 0.5×0.0233 = 0.01163 > 0.005), so the relative term governs **including the dense bucket**; δ_floor would bind only below ~1% faithful (no current bucket).

**Per-stratum POWER check (the review's deepest point):** detecting a rate excess is per-**feature** (each feature is a Bernoulli collapse/not). Feature populations are enormous (76k–452k per stratum; the densest bucket has the *most* features, not the fewest). The feature floor to detect a 50%-relative excess is 211–646 features/stratum; the held-out set carries **10,138–55,780** — ≥15× margin in every stratum. So the vacuous pass **cannot** move into the sample size: `underpowered_feature_strata == []`, verified on the selected set, not assumed.

### 2. KS effect size = 0.15 — provisional; gates nothing yet

Sizes the per-stratum **cell-density reference target** only. The model-vs-baseline KS/Wasserstein **distance is deferred** (no model — spec §7), so 0.15 currently gates no verdict. Re-derive against observed architecture-to-architecture distributional gaps when the bake-off runs.

### 3. residual cap = 50% — did not bind; moot

N=53 is ~11% of the pool, far below the 247-tile cap. The cap played no role.

## A measured finding that contradicts the spec (experiments win)

The spec calls D's stratified floor "the binding one." **Measurement shows it is NOT binding:** D's rate-detection floor is per-feature and features are abundant (floors 211–646 vs tens of thousands available). What actually drives N=53 is the **provisional cell-density reference target** (a uniform ~164-cell/stratum sample for the *deferred* KS scoring), not D's power floor. Reported honestly; the spec's expectation was based on a pre-measurement guess about the binding constraint.

## Status & next step

- **NOT FROZEN.** No `holdout_manifest.yaml`, no `_EVAL_SET_LOCKED`.
- δ review **resolved** (relative form; uniform relative discrimination; per-stratum thresholds + feature power both adapt to base rate — no new uniform-hides-variation problem).
- **Freeze when approved:** `generate_eval_set(..., lock=True)` (write-once). N=53, ρ=0.5, δ_floor=0.005.

## Deferred to the eval-harness / training-scaffold successor (spec §7)

Model-scoring orchestration · simulation-viability execution · the tokenizer-on-**model** side of R2 · the Wasserstein/KS **distance** against model output · the training loader's actual holdout exclusion (calls **this** manifest + `lineage_audit.audit_no_holdout_leak`, one source).

## Spec-coverage checklist

| Spec | Implemented (commit `48614ee`) |
|---|---|
| A — scope (R2 real-side in / model-side deferred) | task headers + `degeneracy.py` |
| B — region partition + generalization UNSCORED | `manifest.py` region-keyed + `lineage_audit` region-scaling |
| C — core/full baselines + ceiling + gap | `baselines.py` |
| §2 — one shared bref-rate | `bref_rate.py` (identity-locked to sub-G, `is`-asserted) |
| D — per-instance exclude + G-D1/G-D2 | `degeneracy.py` (relative threshold; dense-doubling guard) |
| E — labels one-source + morphology collision + density-aggregate | `labels.py` (Gate-6 vocab cross-ref) |
| F — lock + G-F1..F4 + manifest | `manifest.py` + `lineage_audit.py` |
| G — per-stratum floors + ordered degradation + single (ρ, δ_floor) | `sizing.py` + `pipeline.co_optimize` |

47 eval-set tests pass (incl. the slow real-data dry-run); full project suite 960 passed, 1 pre-existing xfail, no regressions.

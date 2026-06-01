# Eval-set generation — Singapore dry-run measurement (NOT FROZEN)

**Date:** 2026-06-01 · **Status:** DRY-RUN — measured + proposed, **manifest NOT frozen** · **Freeze blocked on:** the δ review (item 1 below).
**Code commit:** `8e1c024` (branch `phase-1-eval-set-generation`) · **Spec:** `docs/superpowers/specs/2026-06-01-eval-set-generation-design.md` · **Plan:** `docs/superpowers/plans/2026-06-01-eval-set-generation.md`

## How to reproduce

```bash
uv run python -c "from cfm.eval.holdout import pipeline, paths; \
r=pipeline.generate_eval_set(release=paths.DEFAULT_RELEASE, region=paths.DEFAULT_REGION, lock=False); \
print(r.n, r.residual, r.ceiling_overall, r.per_stratum_bref_rate, r.underpowered_cell_density_strata)"
```

Data snapshot: `data/processed/{sub_c,sub_d,sub_f}/2026-04-15.0/singapore/` (494 tiles, sub-G `_PHASE1_VALIDATED`). Runtime ≈ 11 s (decodes all 494 tiles' round-tripped-real geometry).

## Config (the inputs that determined the result)

| Parameter | Value | Where | Spec? |
|---|---|---|---|
| release / region | `2026-04-15.0` / `singapore` | `paths.py` | yes |
| δ (`DELTA_BREF_REGIME`) | **0.03** | `sizing.py` | chosen (spec leaves to impl) |
| KS effect size (`_KS_EFFECT`) | **0.15** | `pipeline.py` | chosen |
| residual cap (`n_cap_fraction`) | **0.50** | `pipeline.py` | chosen |

## Measurement

| Quantity | Value |
|---|---|
| **Proposed held-out N** | **53 tiles** |
| Training residual (494 − N) | **441** (~89%) |
| Round-tripped-real geometric-validity ceiling (overall) | **0.967992** (≈3.2% bref-placeholder collapse — v1 tokenizer limitation, consistent with H3) |
| Underpowered cell-density strata | **none** |

| cell_density_bucket | faithful bref-rate | cell population (of 494 tiles) | cell floor |
|---|---|---|---|
| 0 (sparsest) | 6.79% | 8538 | 271 |
| 1 | 3.82% | 2504 | 164 |
| 2 | 2.85% | 4476 | 164 |
| 3 (densest) | 2.33% | 1531 | 164 |

All populations clear their floors; the co-optimization met every floor at N=53. **The spec's deep feasibility tension (can a single-region 494-tile pool power every stratum AND leave a viable training set?) is answered YES by measurement.** Numbers are rough, not round (rough-numbers heuristic).

## The three chosen numbers — justified independently (NOT on joint feasibility)

Joint feasibility ("they produced a clean N=53") is *not* a justification — a δ/effect/cap triple can always be found to fit N. Each number is interrogated on its own merits.

### 1. δ = 0.03 — **OPEN, blocks freeze**

δ is THE regime-distinguishing threshold with three consumers (D's faithful-vs-over-emitting boundary, G's δ-relaxation bound, C's R2 tolerance — one number). The claim it must earn: *"this is the rate-excess that distinguishes a model that learned the bref limitation from one over-emitting degenerate stubs."*

**An absolute 0.03 does not hold that meaning uniformly across strata.** Against the measured per-stratum faithful rates:

| bucket | faithful | δ=0.03 trips only at | relative tolerance |
|---|---|---|---|
| 0 | 6.79% | 9.79% | **+44%** |
| 1 | 3.82% | 6.82% | +79% |
| 2 | 2.85% | 5.85% | +105% |
| 3 | 2.33% | 5.33% | **+129%** |

A model can **more than double** the dense-bucket (bucket 3) degenerate-stub rate and still sit "within tolerance." Since D's purpose is *distribution-matching* ("learned the limitation" = reproduces the faithful rate), a rate-doubling has clearly **not** learned the limitation — yet the absolute guard waves it through. This is the per-stratum-vacuous-pass pattern one level up: G-D2 stratifies *where* the rate is measured (per `cell_density_bucket`), but the *trip threshold* stayed a global absolute 0.03, so the dense-bucket guard is relatively lax.

**Recommendation (resolve BEFORE freeze):** make δ **relative-to-base-rate** (trip if `model_rate > faithful_rate · (1 + ρ)`) with an **absolute floor** for near-zero strata (so a stratum with faithful≈0 doesn't get an infinitely tight guard). E.g. trip if `model_rate − faithful_rate > max(ρ·faithful_rate, δ_floor)`. This makes "learned vs over-emitting" consistent across strata. Choosing ρ (and δ_floor) is the review item; it will change the per-stratum floors slightly and therefore possibly the selection — which is exactly why it must precede the write-once lock.

### 2. KS effect size = 0.15 — provisional; gates nothing yet

Used only to size the per-stratum cell floor. The model-vs-baseline KS/Wasserstein **distance is deferred** (no model yet — spec §7), so 0.15 currently gates **no verdict**; it only influences N. The principled basis it *will* need: tie it to the smallest distributional gap between two architectures that would change a bake-off ranking — if 0.15 is coarser than that gap, the substrate can't distinguish the architectures it exists to compare. **Re-derive against observed architecture-to-architecture gaps when the bake-off runs.** For now: provisional sizing input.

### 3. residual cap = 50% — **did not bind; moot for this result**

The cap (247 tiles) is a ceiling that pushes N down only when floors demand many tiles. N=53 is ~11% of the pool, far below 247, and all floors were met without approaching it. **The cap played no role in N=53.** It remains a sensible guardrail for future regions/parameters but is not load-bearing here.

## Status & next step

- **NOT FROZEN.** No `holdout_manifest.yaml`, no `_EVAL_SET_LOCKED` written.
- **Blocker:** resolve δ (item 1) — relative/per-stratum threshold + chosen ρ/floor. This may shift the per-stratum floors and the selection.
- After δ is settled: re-run the dry-run, confirm the numbers, then freeze with `generate_eval_set(..., lock=True)` (write-once).

## Deferred to the eval-harness / training-scaffold successor (spec §7)

Model-scoring orchestration · simulation-viability execution (model + CARLA) · the tokenizer-on-**model** side of R2 · the Wasserstein/KS **distance** computation against model output · the training loader's actual holdout exclusion (it calls **this** manifest + `lineage_audit.audit_no_holdout_leak`, one source).

## Spec-coverage checklist

| Spec | Implemented (commit `8e1c024`) |
|---|---|
| A — scope boundary (R2 real-side in / model-side deferred) | task headers + `degeneracy.py` docstring |
| B — region partition + generalization UNSCORED | `manifest.py` region-keyed + `lineage_audit` region-scaling test; generalization not scored |
| C — core/full baselines + ceiling + gap | `baselines.py` |
| §2 — one shared bref-rate | `bref_rate.py` (identity-locked to sub-G, `is`-asserted) |
| D — per-instance exclude + G-D1/G-D2 | `degeneracy.py` (**δ refinement open, item 1**) |
| E — labels one-source + morphology collision + density-aggregate | `labels.py` (Gate-6 vocab cross-ref) |
| F — lock + G-F1..F4 + manifest | `manifest.py` + `lineage_audit.py` |
| G — per-stratum floors + ordered degradation + single δ | `sizing.py` + `pipeline.co_optimize` (**δ open**) |

47 eval-set tests pass (incl. the slow real-data dry-run); full project suite 958 passed, 1 pre-existing xfail, no regressions.

# Eval-set generation — Singapore held-out set (FROZEN)

**Date:** 2026-06-01 · **Status:** **FROZEN** (write-once lock written + verified) · **N = 132 held-out tiles** · **KS target gap 0.08 (resolves 0.076).**
**Code commit:** `1c9c122` (branch `phase-1-eval-set-generation`) · **Spec:** `docs/superpowers/specs/2026-06-01-eval-set-generation-design.md` · **Plan:** `docs/superpowers/plans/2026-06-01-eval-set-generation.md`

## The lock (verified against the written files, not the result object)

- `data/processed/eval_set/2026-04-15.0/holdout_manifest.yaml` — 132 region-keyed tiles (`region=singapore`), each with `provenance_sha256`; `manifest_sha256` recorded and recompute-matches (`2b4da67d…`). Force-added to git (durable past `/data/` gitignore); write-once.
- `data/processed/eval_set/2026-04-15.0/_EVAL_SET_LOCKED` — records: `ks_target_gap=0.08`, `ks_resolved_gap_binding=0.07597`, `ks_single_region_floor=0.04908`, `n_held_out=132`, `rho_bref_regime=0.5`, `delta_floor_bref=0.005`.
- **Determinism verified:** two independent `co_optimize` runs produced an identical 132-tile selection, equal to the frozen manifest — the lock is not run-order dependent.

## Frozen parameters and the reasoning each earned

| Parameter | Value | Basis (on meaning, not workability) |
|---|---|---|
| N (held-out tiles) | **132** | sized to the KS target gap below; 362 training residual (~73%) |
| KS target gap | **0.08** | PI call on a single-region tradeoff curve (below); over-provisioned vs the write-once asymmetry |
| KS resolved gap (actual, binding stratum) | **0.076** | the gap the frozen set actually resolves (bucket 3 = 639 held-out cells) |
| KS single-region hard floor | **0.049** | finest gap this region can EVER resolve (full pool, binding bucket) — the v1 ceiling |
| ρ (over-emission, relative) | **0.5** | uniform +50% relative discrimination; per-stratum absolute thresholds track base rate |
| δ_floor | **0.005** | near-zero backstop; binds no current bucket (min ρ·faithful = 0.0116 > 0.005) |
| geometric-validity ceiling | **0.968** | round-tripped-real (≈3.2% bref collapse, consistent with H3) |

## Two freeze-gate catches, both resolved

**(1) The over-emission threshold — relative, not absolute.** The first dry-run used an absolute δ=0.03, which against per-stratum faithful rates (2.3–6.8%) gave +44%/+79%/+105%/+129% relative tolerance — the dense-bucket guard was vacuous (a model could >2× the bucket-3 rate and pass). Replaced by `max(ρ·faithful, δ_floor)` with ρ=0.5: uniform relative discrimination, per-stratum absolute thresholds (0.034→0.012). Regime-distinguishing guard `test_GD2_GUARD_dense_bucket_doubling_trips_under_relative_but_absolute_missed_it` proves the dense-bucket doubling now trips. Per-stratum **feature** power verified (detection is per-feature; held-out features 26k–138k vs floors 211–646; `underpowered_feature_strata == []`) — the vacuous pass cannot hide in the sample size.

**(2) The freeze must not be bound by a provisional number.** N was initially driven by a provisional KS effect (0.15), but the KS *distance* is model-facing and deferred — and the freeze is write-once. Measurement reframed it: the KS-resolvability ceiling is a **single-region pool property (0.049)**, not a tunable knob. So the freeze is a point on a bounded tradeoff curve:

| target gap | cells/stratum | N | residual |
|---|---|---|---|
| 0.15 | 164 | 53 | 441 |
| 0.10 | 369 | 76 | 418 |
| **0.08** | **577** | **132** | **362** |
| 0.06 | 1025 | 247 | 247 |
| 0.05 | 1476 | 394 | 100 |

**Chosen 0.08 / N=132** on the write-once asymmetry: under-provisioning is unrecoverable (can shrink, never grow); over-provisioning costs ~11% training tiles that degrade all four architectures **equally** (so it does not distort the bake-off comparison — what the substrate exists for); and capable models on the same data commonly differ by sub-0.10 gaps — exactly where discrimination matters. 0.08 sits inside that band; the resolved 0.076 is comfortably above the 0.049 wall.

## Carry-forward obligations → training-scaffold / eval-harness successor (LOAD-BEARING)

These are explicit trigger conditions, not advisories:

1. **Holdout exclusion (one source):** the training loader MUST call `cfm.eval.holdout.lineage_audit.audit_no_holdout_leak(manifest, training_reachable)` against this frozen manifest. Fail-closed on absent lineage (G-F4).
2. **Eval-harness fail-loud on resolution:** when models exist, assert the architecture-distinguishing gap the bake-off needs is **≥ 0.076** (the frozen set's resolved gap). If finer is needed → this is the documented **second-region extraction trigger** (B-decision), NOT silent under-power.
3. **Single-region ceiling = 0.049:** no single-region held-out set can resolve finer; a finer requirement is categorically a second-region need.
4. **Conditioning vector (one source):** the model's conditioning MUST consume the same sub-C/sub-D quantities `labels.py` reads (`population_density_bucket`, `cell_density_bucket`, the `morphology_stratum` = sub-D `road_skeleton_class`+`zoning_class` — NOT sub-C's constant `morphology_class`), or conditioning-compliance is apples-to-oranges.
5. **ρ is tunable downward (~0.25) once the model's natural over-emission variation is observed** — it does not move N (feature power abundant), so tightening later is free.

## Deferred (spec §7, unchanged)

Model-scoring orchestration · simulation-viability execution · tokenizer-on-**model** side of R2 · the Wasserstein/KS **distance** against model output · generalization (UNSCORED v1, gated by a Phase-2 multi-region-training-corpus decision).

## Spec-coverage

A (scope) · B (region-keyed + generalization unscored) · C (core/full + ceiling) · §2 (one shared bref-rate, identity-locked) · D (per-instance exclude + G-D1/G-D2, relative threshold) · E (labels one-source, morphology collision, density-aggregate) · F (lock + G-F1..F4 + frozen manifest) · G (per-stratum floors + ordered degradation + (ρ,δ_floor)) — all implemented. 47 eval-set tests pass (incl. slow real-data); full suite 960 passed, 1 pre-existing xfail, no regressions.

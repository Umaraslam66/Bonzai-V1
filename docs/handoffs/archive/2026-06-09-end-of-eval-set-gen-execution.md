# Handoff — eval-set-generation: EXECUTION COMPLETE, awaiting merge (2026-06-09)

You are a fresh, context-free agent. The eval-set-generation sub-project was **executed in full**
(12 tasks, subagent-driven, two-stage review per task, Umar as chat reviewer). This file is the
cold-start carry-forward for the **bake-off** session. Read it, then the spec it points to.

---

## ⛳ STATE

- **Branch `phase-2-eval-set-gen`** (29 commits off `main`, tip `4cb74dd`). **NOT merged, NOT pushed.**
  Final whole-branch review = **READY TO MERGE** (no Critical/Important issues). Suite: `tests/eval` +
  `tests/training` = **164 passed**, 4 deselected (`@slow`, real-data on Leonardo); ruff clean.
- Merge stays gated on **Umar's explicit word + `--no-ff`**. Local-first; a PR is optional.
- Contract docs (on the branch): spec `docs/superpowers/specs/2026-06-08-eval-set-gen-design.md`
  (§7 **amended 2026-06-09** — see below), plan `docs/superpowers/plans/2026-06-08-eval-set-gen.md`.

## 1. What was built (12 tasks → spec sections)

- **T1** `eval/holdout/paths.py` — per-region CRS label (`epsg_label_for_region`) + distinct
  `multiregion/` eval-set paths; SG paths byte-untouched (§5).
- **T2** `eval/resolution.py` — generic multi-region escalation (no "second region"/"munich"; swap
  is T12's) (§5/§7).
- **T3** `eval/holdout/labels.py` — regression-lock test: no SG constant scored (§4.1/§5).
- **T4** `eval/holdout/manifest.py` — `build_holdout_manifest_multiregion` + §2.2 correct-by-
  construction assertions; **schema constants INDEPENDENT** (SG `MANIFEST_SCHEMA_VERSION="1.0"`,
  multiregion `MULTIREGION_MANIFEST_SCHEMA_VERSION="2.0"`).
- **T5** `eval/holdout/macro_graph.py` (the ONE shared `interior_road_graph`) + `eval/usable_tiles.py`
  + `scripts/eval/measure_usable_tiles.py` (§2.1/§3.5b).
- **T6** `scripts/eval/build_multiregion_manifest.py` + freeze → **FROZEN manifest committed**.
- **T7/T8** `eval/holdout/lineage_audit.py` city-identity guard (`whole_city`) + 4-case HALT teeth (§6).
- **T8.5** `data/training/holdout_guard.py` fail-closed schema-2.0 backstop + `CellDataModule`
  `expected_holdout_schema="2.0"` (§6 trigger-1, closes #16).
- **T9** `eval/holdout/coherence.py` — S1 metric (continuity + giant-component + active-active zoning,
  interior-permutation null, **NO density** §3.4).
- **T10** coherence 3-way HALT teeth (fragmentation non-redundancy, mutation-proven) (§3.3).
- **T11** `scripts/eval/measure_coherence_reference.py` + `eval/holdout/coherence_reference.py` —
  **reference committed** (both measures + structural dense-core exclusion) (§3.1/§10.2).
- **T12** `eval/resolution.py::assert_coherence_power_sufficient` — §7 gate; **consumes** opaque
  `model_vs_real_effect`, does NOT define it.

## 2. Committed artifacts (force-added; all verified-end-state, internally consistent)

- `data/processed/eval_set/2026-04-15.0/multiregion/holdout_manifest.yaml` + `_EVAL_SET_LOCKED`
  (commit `be83f03`): schema **2.0**, 4 cities, `manifest_sha256=ae4d5af6011585c9ae1121af71066e46e4fde5bc1750c6220b61e54108a97d1e`
  (stored==recomputed-from-disk, Leonardo+Mac), held_out 46,130,102 / train 623,900,790. **SG set byte-untouched**
  (SG manifest sha `c676e21e…`/28467 before==after).
- `reports/2026-06-08-usable-n.yaml` (`1361314`): glasgow 523/549, eisenhuttenstadt 579/616,
  munich 156/171, krakow 601/616 (n_unreadable=0 all).
- `reports/2026-06-08-coherence-reference.yaml` (`df65102`, n_shuffle=200): per stratum BOTH the
  absolute band (continuity_real/giant_real/zoning_real) AND the shuffle-gap (+sd) + mean_road_edges
  + dense_core_saturated.

## 3. NEW finding (2026-06-09) — munich dense-core saturation (RESOLVED: stay+record)

munich's held-out tooth-3 real-vs-permuted separation is **0.43** (the 3 moderate strata 0.81–0.95).
**Not weak — SATURATED:** munich's tiles are **dense-core** (#21 inner-core bbox; mean **47.8** road-
edges > **40** = 2/3 of the 60-edge interior capacity), so a random rearrangement is itself near-
complete and the shuffle-null saturates; munich's *absolute* coherence is in fact the HIGHEST
(0.92/0.98). **Source-settled:** the §7 gate consumes resolution's KS number + the opaque first-model
`model_vs_real_effect`, **NOT** the shuffle-gap → this is a metric-VALIDATION limitation, not a power
failure. **PI ruling: munich STAYS** — no swap (manchester is itself dense at 40 edges; swap = cosmetic
+ write-once re-freeze + z32 loss), **no re-freeze**. tooth-3 gated ≥0.70 on moderate strata only;
dense-core (`mean_road_edges > 40`, **structural not by city name**) reported-not-gated
(`coherence_reference.py::assert_validation_separation`, non-vacuity mutation-proven). Spec §7 amended
(`ba6f74e`). The §2 planning-handoff locked decisions all still hold.

## 4. THREE DORMANT bake-off obligations (the critical carry-forward — do NOT drop)

1. **T8.5 datamodule re-point + DDP schema flips.** The EU bake-off datamodule MUST be constructed with
   `multiregion_holdout_manifest_path("2026-04-15.0")` (default `expected_holdout_schema="2.0"` backstops
   a forgotten re-point — fails loud, never silently audits EU against the SG manifest). When the 2 legacy
   DDP scripts (`scripts/train_scaffold.py`, `scripts/ddp_resume_check.py`) are reused for EU, their
   explicit `expected_holdout_schema="1.0"` MUST flip to `"2.0"` AND re-point the path — else they silently
   audit EU against the SG holdout (#16 one layer over). The 3 `"1.0"` sites carry inline comments saying so.
2. **§7 `model_vs_real_effect` is OPEN (first-model) and MUST be anti-leak-proven.** Its computation is
   deliberately NOT pinned (`resolution.py:91-95` docstring + spec §7). Binding constraint: whatever it is
   defined as, a model that merely ECHOES the handed tile-mode conditioning (high *absolute* continuity/giant
   without real structure) MUST FAIL it — the absolute band ALONE is conditioning-contaminated (the shuffle-gap
   exists to subtract the echo). **munich's saturated shuffle-gap cannot be its munich reference.** Both measures
   are recorded per stratum so first-model has both. The gate `assert_coherence_power_sufficient` (consumes the
   effect, owns the swap) fires at the first-model checkpoint; dormant until then. NB the gate's quantitative use
   of `usable_n` is likewise a first-model decision deferred with the effect definition.
3. **munich→manchester swap reserve + EU-train-split resolved-gap recompute.** The swap is the §7 POWER reserve
   (fires at first-model only if `usable_n` can't resolve the effect) — distinct from the §3 saturation finding
   (which did NOT trigger a swap). And `assert_resolution_sufficient` still reads the **SG** marker's KS fields;
   the EU-train-split resolved-gap recompute is a first-model obligation per §5/§7 (the EU `_EVAL_SET_LOCKED`
   intentionally carries no KS number per §2.3).

## 5. Leonardo infra (for the bake-off)

- Isolated clone `/leonardo_work/AIFAC_P02_222/eval-set-gen-wt` (cloned from bundle, corpus `sub_d` symlinked);
  the main repo stayed pristine on `phase-2-corpus-completion @ 212e3ed`. Run scripts via the main
  `.venv/bin/python`; **GOTCHA**: ad-hoc python needs `PYTHONPATH=<clone>/src` (editable install otherwise
  shadows the clone). Build/measure scripts self-inject. Artifacts produced on Leonardo → transferred byte-exact
  → committed on the Mac.

## ▶️ One-liner to start the bake-off session

> eval-set-generation EXECUTION is COMPLETE on `phase-2-eval-set-gen` (tip `4cb74dd`, READY TO MERGE, NOT merged).
> Frozen multiregion manifest + usable-n + coherence reference committed. THREE dormant first-model obligations
> (T8.5 re-point + DDP schema flips; §7 `model_vs_real_effect` OPEN+anti-leak; munich→manchester reserve + EU
> resolved-gap recompute) — read §4 here before wiring the bake-off. munich STAYS (dense-core #21 saturates the
> shuffle-null, structural exclusion, no swap). Merge gated on Umar's word + `--no-ff`.

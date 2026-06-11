# RESUME: readiness-closure — segment 3 (Phase 6 onward), 2026-06-10

**This is a thin continuation handoff.** The authoritative documents are:
- Spec (LOCKED, 4 PI-calls resolved incl. δ=0.15 + character-anchored):
  `docs/superpowers/specs/2026-06-10-readiness-closure-and-conditioning-enrichment-design.md`
- Plan (APPROVED; 27 tasks / 10 phases; F16 spine = hard ordering):
  `docs/superpowers/plans/2026-06-10-readiness-closure-and-conditioning-enrichment.md`
- Audit artifacts (the WHY): `reports/2026-06-10-readiness-audit-{execution-surface-map,failure-class-enumeration}.md`
- Prior segment handoffs: `2026-06-10-readiness-closure-segment-2-resume.md` (segment-1 close).

## State
- **Branch `phase-2-readiness-closure`** (off main@`77fdb6d`). Tip at segment close = the commit
  containing this file. Segment-2 commits (Tasks 10–21 + review follow-ups):
  Task 10 `6538c28`; Task 11 `c0f19c5`+`3cce491`; Task 12 `c787477`+`85b05bd`+`75651c3`;
  Task 13 `618ea53`+`5f11e8e`; Task 14 `ca54e7c`+`c54f413`; Task 15 `4b4935d`+`a7b3141`;
  Task 16 `de8b22c`; Task 17 `50009e5`+`ee08a47`; Task 18 `2c6241b`; Task 19 `5e57b95`+`fec6018`;
  Task 20 `f108685`+`deeed98`; Task 21 `289075d`. **NOT pushed** (local; push/merge only on
  Umar's word, --no-ff at sub-project end).
- **Suite: 1482 passed, 2 skipped (Leonardo-only), 36 deselected, 2 xfailed** (both pre-existing:
  Phase-0 ENTRY marker; Task-12 `--config` gap — resolves at plan Task 26). Slow e2e green.
- **DONE: Phases 3–5 (Tasks 10–21).** Every task went through implementer → spec reviewer →
  quality reviewer; review findings were fixed in named follow-up commits, never deferred
  silently. Task 20's implementer stalled at final verification; orchestrator verified
  end-state (full+slow suites, ruff unpiped) and committed — noted in that commit body.
- **NEXT: plan Task 22** (Phase 6 — recalibrated verdict: δ=0.15 effect floor + bref
  exclusion-by-construction-identity), then 23 (localization diagnostic; step 5 GATED, step 6
  PI HALT — the character feature for Task 24 is chosen THERE, by the data, with Umar's word),
  Phase 7 (24), Phase 8 (25 — GATED + HALT), Phase 9 (26).

## Open gates / pending words (NONE may be assumed)
- **No Leonardo, no GPU, no re-derive** without Umar's explicit per-step word. Pending
  [LEONARDO — GATED] steps, all CPU except Task 18's: Task 12 step 5 (CRS check over 42
  cities — `scripts/check_crs_consistency.py --report …`), Task 13 step 5 (EU emergence
  floors — `scripts/measure_emergence_floor.py`), Task 15 step 5 (38-city token lengths —
  `scripts/measure_cell_token_lengths.py`, HALT-gate frac>5760 > 0.5%), Task 18 step 5
  (kill→resubmit resume proof; FIRST post-renewal GPU job, additionally gated on T0 closure),
  Task 23 step 5 + Task 25 (diagnostic + re-derive; Task 23 step 6 and Task 25 step 3 are
  PI HALT-gates).
- Allocation soft-ended 2026-06-11; T0 work is CPU-local and proceeds through the renewal gap
  (PI-call #4).

## Execution discipline (carried, non-negotiable)
Subagent-driven (fresh implementer per task; implementer ≠ reviewer; two-stage review: spec
compliance then code quality; orchestrator verifies end-state before accepting); TDD with
red-run shown; ruff UNPIPED; halt-on-defect (no implementer improvisation); stop-before-commit
at gates/forks; verified-end-state never exit codes; no version-skewed partial fixes; subagents
never branch/push/PR.

## Corrections discovered during segments 1–2 (hand these forward at dispatch)
Segment-1 set (still live):
1. `tile_conditioning` dict keys are `dominant_zoning_class` / `modal_road_skeleton_class` /
   `admin_region` — the kwarg mapping in `datamodule._cell_prefix_ids` is the reference.
2. The plan's "tests/data/training/test_build_shards.py" is actually `tests/training/test_build_shards.py`.
3. Slow tests need `-m slow` explicitly.
4. Pre-existing lint debt: `src/cfm/data/sub_c/io.py` (I001+F401) and a large block in
   `scripts/sub_f/analyze_geometry_primitives.py` (~120 errors) — left deliberately; bundle
   into a later `chore:` sweep, never silently.
5. `flatten_shards_to_cells` has `seed: int = 0` (constant-bucketed, inert in ids).

New in segment 2:
6. `resume_ckpt_path(ckpt_dir: Path)` takes a DIRECTORY (plan snippet `resume_ckpt_path(backbone,
   scale)` is stale). `work_checkpoint_dir(backbone, scale, *, region, seed, work_root=None)` →
   `$WORK/Bonzai-OSM/checkpoints/bakeoff/{backbone}-{scale}/{region}-seed{seed}` (Task-17
   follow-up widened the key; train_set deliberately NOT in the key — DECISION + trigger in
   resume.py). The scale label is PARAM-DERIVED (`_scale_label`, e.g. "311M"), may differ from
   the nominal sbatch $SCALE ("300M").
7. `ScaffoldConfig` now carries `conditioning_scheme`, `train_set`, `eval_cells`, `eval_max_new`.
   The `--emergence-floor` CLI flag is GONE — floors resolve fail-closed from
   `configs/eval/emergence_floors.yaml` by `cfg.region` (`_resolve_emergence_floor`); the
   singapore seed hand-carries floor 1.96 (vs 1.9625 exact) — floor is authoritative; a future
   SG re-derive writes 1.9625 and deliberately fails the seeded-value test.
8. `extract_features_by_city_stratum_metric` now returns `ExtractionResult(features,
   tile_coverage)` — Task 25's re-derive runner must unpack it (the existing gate-(i) runner
   already does, via `dataclasses.replace` threading).
9. Positional capacity = `max_len + CONDITIONING_PREFIX_LEN` (8 POSITIONS); `n_cond=512` is
   embedding-table ROWS, not positions — don't conflate the axes. Generation crash boundary is
   `max_new >= max_len + 2` (last token appended, never fed back). Pinned by
   `test_generation_at_exact_positional_capacity_through_production_build`.
10. The diagnostic sbatch at `--max-len 2048 / --eval-max-new 2048` renders the emergence
    verdict INCOMMENSURATE by construction vs the 5760 floor regime (honest; commensurate eval
    needs a re-planned model budget) — the diagnostic re-run parameters are a Phase-6/PI
    decision; surface this at Task 23/25 planning.
11. The drop-rate action contract (`DropRateExceeded`, >0.5% too-long over non-empty) enforces
    only at `max_cell_tokens >= DEFAULT_MAX_CELL_TOKENS`; run_smoke + three 2048 sbatch
    opt-downs are known-exempt (INFO log is the visibility there); revisit trigger = sub-design
    budget in a SCORED bake-off run.
12. Union path: G4 rollup path one-sourced at `build_shards.DEFAULT_G4_ROLLUP`;
    `verify_union_manifests(release, g4_rollup=…, holdout_manifest=…)` is the testable
    preamble verifier (raise-only, no asserts). Strict `holdout["held_out_cities"]` reads at
    every caller (train_cities itself still `.get`s — callers must be strict).
13. Reader-side integrity (Task 20): `run_holdout_audit(..., manifest_path=…)` verifies
    stored-vs-recomputed `manifest_sha256` + `_EVAL_SET_LOCKED` beside the manifest; synthetic
    holdout fixtures must stamp the real freeze grammar + touch the marker (shared fixture
    writers already do).
14. For Task 18's gated Step-5 checklist: stale $WORK checkpoints under the OLD flat
    `{backbone}-{scale}` dirs are invisible to the new nested key (fresh-run = safe direction);
    Lightning version-suffixed step files (`-v1`) tie under `_STEP_RE` (glob-order tie-break);
    a torn `last.ckpt` passes the existence check and fails loudly at `torch.load` — the
    kill→resubmit proof should exercise these.
15. Watch for plan tests that are mathematically unsatisfiable (Task 15's union-threshold test
    was — convex combination; replaced by a direction-discriminating pair). Screen Task 22–26
    test specs the same way before RED.

## Capability side-note
Keep harvesting to `reports/2026-06-10-capability-observations.md` (never scope-expanding).
Segment-2 observations appended (plan-test unsatisfiability; reviewer wrong-construction-path
reproduction; subagent stall + orchestrator end-state coverage; n_cond axis conflation).

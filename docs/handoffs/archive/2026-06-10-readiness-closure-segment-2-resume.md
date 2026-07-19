# RESUME: readiness-closure — segment 2 (Phase 3 onward), 2026-06-10

**This is a thin continuation handoff.** The authoritative documents are:
- Spec (LOCKED, 4 PI-calls resolved incl. δ=0.15 + character-anchored):
  `docs/superpowers/specs/2026-06-10-readiness-closure-and-conditioning-enrichment-design.md`
- Plan (APPROVED; 27 tasks / 10 phases; F16 spine = hard ordering):
  `docs/superpowers/plans/2026-06-10-readiness-closure-and-conditioning-enrichment.md`
- Audit artifacts (the WHY): `reports/2026-06-10-readiness-audit-{execution-surface-map,failure-class-enumeration}.md`

## State
- **Branch `phase-2-readiness-closure`** (off main@`77fdb6d`). Tip at segment close = the commit
  containing this file; preceding tips: Task 9 `fc54064`, Task 8 `e891f85`, Task 7 `6719921`,
  Task 6 `ba487d4`, Task 5 `89b87c3`, Task 4 `38ee7b8`, Task 3 `cc19480`, Task 2 `3df5d0e`,
  Task 1 `b4a2a2f`, Task 0 `368fd5b`. **NOT pushed** (local branch; push/merge only on Umar's
  word, --no-ff at sub-project end).
- **Suite: 1356 passed, 2 skipped (Leonardo-only), 36 deselected, 2 xfailed** (both pre-existing:
  Phase-0 ENTRY marker; Task-12 `--config` gap — resolves at plan Task 26).
- **DONE: Phase 1 (Tasks 1–4) + Phase 2 (Tasks 5–9).** The model's training path now carries
  VALUE-BEARING conditioning (`datamodule._cell_prefix_ids`, explicit field mapping, strict
  indexing); generation is matched-conditioning with real strata (slow e2e re-proven PASSED);
  `conditioning_scheme: Literal["slot","value"]="value"` rides every checkpoint/report;
  `tile_conditioning` key collision renamed (`admin_region`); conditioning_gate tombstoned;
  delta-spec §4 prior corrected; promotion live-call guard landed (red-by-reversion proven);
  city-name `_value_bucket` aliasing pinned (8 pairs / 38 cities — Task 24 gives city_identity
  its own field instead).
- **NEXT: plan Task 10** (Phase 3 — `--region`/`--release` CLI + sbatch de-Singaporization +
  content test), then 11–15 (union caller, CRS checker, emergence floors, commensurability,
  token-length stats), then Phase 4 (16–19 resume/isolation), Phase 5 (20–21 integrity).

## Open gates / pending words (NONE may be assumed)
- **No Leonardo, no GPU, no re-derive** without Umar's explicit per-step word. The plan's five
  [LEONARDO — GATED] steps: Task 12 step 5 (CRS check over 42 cities), Task 13 step 5 (EU
  emergence floors), Task 15 step 5 (EU token lengths — HALT-gate >0.5% drop), Task 18 step 5
  (resume kill→resubmit proof; FIRST post-renewal GPU job, additionally gated on T0 closure),
  Task 23 step 5 + Task 25 (diagnostic + re-derive; Task 23 step 6 and Task 25 step 3 are
  PI HALT-gates).
- Allocation soft-ended 2026-06-11; T0 work is CPU-local and proceeds through the renewal gap
  (PI-call #4).

## Execution discipline (carried, non-negotiable)
Subagent-driven (fresh implementer per task; implementer ≠ reviewer; orchestrator verifies
end-state before accepting); TDD with red-run shown; ruff UNPIPED; halt-on-defect (no implementer
improvisation); stop-before-commit at gates/forks; verified-end-state never exit codes; no
version-skewed partial fixes; subagents never branch/push/PR.

## Corrections discovered during segment 1 (hand these forward at dispatch)
1. `tile_conditioning` dict keys are `dominant_zoning_class` / `modal_road_skeleton_class` /
   `admin_region` (NOT the plan snippets' bare names) — the kwarg mapping lives in
   `datamodule._cell_prefix_ids` and is the reference.
2. The plan's "tests/data/training/test_build_shards.py" is actually `tests/training/test_build_shards.py`.
3. Slow tests need `-m slow` explicitly (default addopts deselect by marker even when the file
   is named by path).
4. Pre-existing lint debt in `src/cfm/data/sub_c/io.py` (I001 + F401, on HEAD before this
   branch) — left alone deliberately; bundle into some later cleanup commit, never silently.
5. `flatten_shards_to_cells` gained `seed: int = 0` (wired from `CellDataModule.seed`);
   seed is constant-bucketed so inert in ids.

## Capability side-note
Keep harvesting to `reports/2026-06-10-capability-observations.md` (never scope-expanding).

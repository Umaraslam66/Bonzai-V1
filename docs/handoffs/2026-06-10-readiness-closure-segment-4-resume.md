# RESUME: readiness-closure — segment 4 (Task 23 step 5 onward), 2026-06-10

**This is a thin continuation handoff.** The authoritative documents are:
- Spec (LOCKED): `docs/superpowers/specs/2026-06-10-readiness-closure-and-conditioning-enrichment-design.md`
- Plan (APPROVED; 27 tasks / 10 phases): `docs/superpowers/plans/2026-06-10-readiness-closure-and-conditioning-enrichment.md`
- Audit artifacts: `reports/2026-06-10-readiness-audit-{execution-surface-map,failure-class-enumeration}.md`
- Prior segment handoffs: `2026-06-10-readiness-closure-segment-{2,3}-resume.md`

## State
- **Branch `phase-2-readiness-closure`** (off main@`77fdb6d`). Segment-3 commits:
  Task 22 `d1f49c8`+`f56ff08` (polish); Task 23 steps 1–4 `4cd0dbb`+`0169af2` (hardening).
  **NOT pushed** (local; push/merge only on Umar's word, --no-ff at sub-project end).
- **Suite: 1503 passed, 2 skipped (Leonardo-only), 36 deselected, 2 xfailed** (both
  pre-existing: Phase-0 ENTRY marker; Task-12 `--config` gap — resolves at plan Task 26).
- **DONE: Phase 6 (Tasks 22 + 23 steps 1–4).** Both tasks went implementer → spec reviewer →
  quality reviewer; findings fixed in named follow-up commits. Task-23 re-review verified the
  new F3 guard tests by MUTATION (each test shown to fail with its guard neutered, tree
  restored clean). Plan checkboxes are deliberately not ticked — tracking lives in handoffs.
- **HALTED AT: Task 23 step 5 [LEONARDO — GATED] + step 6 (PI HALT).** Nothing on Leonardo
  has been run for this segment. The diagnostic instrument exists and is synthetic-proven only.

## NEXT (in order, each on Umar's explicit word — NONE may be assumed)
1. **Task 23 step 5 [LEONARDO — GATED]:** run on the 4 held-out cities, CPU, zero GPU:
   `uv run python scripts/run_localization_diagnostic.py --release 2026-04-15.0` (defaults:
   cities=eisenhuttenstadt/glasgow/krakow/munich, min-n 50, alpha 0.05, effect-size-floor
   0.15, report-out `reports/2026-06-10-localization-diagnostic.yaml`). Verified-end-state:
   re-read YAML; rough-numbers check; per-variant n within ±20% of V0's (per-city totals are
   in the YAML for exactly this); commit the report.
2. **Task 23 step 6 — PI HALT-GATE:** bring the variant table to Umar; the character feature
   for Task 24 is chosen THERE, by the data, with Umar's word (criterion: largest drop in
   n_significant_effect at δ=0.15). Surface at this gate: (a) the diagnostic's embedded
   DECISION points (V1 read literally as stratum-REPLACEMENT `(per_cell_zoning,
   per_cell_density)`; V2 equal-width buckets over [0,1] top-inclusive; V3 sea buckets
   {≤EPS_RATIO→0, (0,0.5]→1, >0.5→2}) — serialized in the YAML methodology block;
   (b) the carried correction-#10 question (the GPU diagnostic re-run parameters /
   2048-vs-5760 commensurability are a Phase-6/PI decision).
3. Then Task 24 (Phase 7; steps finalized at dispatch with the chosen feature), Task 25
   (Phase 8 — GATED step 2 + HALT step 3, the reopened-T5 verdict), Task 26 (Phase 9).

## Other open gates / pending words (unchanged from segment 2; NONE assumed)
Task 12 step 5 (CRS check, 42 cities), Task 13 step 5 (EU emergence floors), Task 15 step 5
(38-city token lengths, HALT frac>5760 > 0.5%), Task 18 step 5 (kill→resubmit proof; FIRST
post-renewal GPU job, additionally gated on T0 closure). Allocation soft-ended 2026-06-11;
T0 work is CPU-local through the renewal gap (PI-call #4).

## Execution discipline (carried, non-negotiable)
Subagent-driven (fresh implementer per task; implementer ≠ reviewer; two-stage review;
orchestrator verifies end-state); TDD with red-run shown; ruff UNPIPED; halt-on-defect;
stop-before-commit at gates/forks; verified-end-state never exit codes; no version-skewed
partial fixes; subagents never branch/push/PR; no Leonardo/GPU/re-derive without Umar's
explicit per-step word.

## Corrections (segments 1–2 set, items 1–15, still live — hand forward verbatim from
`2026-06-10-readiness-closure-segment-3-resume.md` §"Corrections") **plus new in segment 3:**
16. `conditioning_discrimination_verdict` now REQUIRES keyword `effect_size_floor` (no
    default in the pure fn; 0.15 default lives at the runner CLI). Result carries
    `effect_size_floor`, `n_significant_raw_bh`, `n_significant_effect` (the latter drives
    the verdict); `TileCoverage` gained `n_bref_excluded: int = 0`.
17. `_tile_features` (gate-(i) module) now returns `(features, n_bref_excluded)`; outbound-
    bref roads excluded by IDENTITY via `_has_outbound_bref` imported from
    `cfm.data.sub_g.seam_decodability` (private import sanctioned; a public re-export was
    reviewed and DEFERRED to a later `chore:` sweep — recorded, not silent). Bref token
    authority: `cfm.data.sub_f.decoder._is_bref_token`, band 1500..1507; fixtures anchor
    with `assert _is_bref_token(_BREF)`.
18. Localization diagnostic facts: pure layers (`variant_features`, `diagnose`,
    `_tile_cell_features`) take in-memory `TileRecord`s; the IO walk mirrors the gate-(i)
    reference (F3 skip+count, shrinkage/zero-tile halts shared via `_SHRINKAGE_CEILING`/
    `TileCoverage`); missing `derivation_evidence.parquet`/sub-C `cells.parquet` on a tile
    that HAS sub-F cells is LOUD by design. `CELL_GRID_SIZE` imports from
    `cfm.data.sub_d.lattice`; sub-C io.py has WRITERS only — read cells.parquet via
    `pq.ParquetFile(path).read()` (Hive-inference correction). V0 parity vs the real
    extraction is pinned by exact equality on a real-writer fixture; the bref path is
    vacuous there (disclosed in the docstring; mocked twin test is the defense).
19. Synthetic token fixtures must anchor ids to the vocab authority — a plausible literal
    collided with the zero-length road block `[0, 5, 0]` (`min(building_token_ids()) == 5`).

## Capability side-note
Keep harvesting to `reports/2026-06-10-capability-observations.md` (never scope-expanding).
Segment-3 observations appended (mutation-verified re-review; weak-red audit; fixture-id
collision; required-keyword threshold surfacing all callers).

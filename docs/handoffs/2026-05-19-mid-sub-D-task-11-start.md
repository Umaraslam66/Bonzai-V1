# Session handoff — mid-sub-D, Task 11 start (2026-05-19)

> **For the new session:** read this doc, then start Task 11 per the plan.

## Status

Sub-D Tasks 1-10 complete (10 of 16). Branch
`phase-1-sub-D-macro-plan-derivation`. Task 11 (per-tile provenance) is next.

## Commit history (this session)

| Task | Commit | Subject |
|---|---|---|
| T1 | `5a15fa3` | `refactor(data): extract shared determinism helpers` (Gate 1 closed) |
| T2 | `e9345d4` | `feat(sub_d): add version namespace helper` |
| T3 | `5fd543e` | `feat(sub_d): add fixed macro lattice utilities` (incl. Side enum amend) |
| T4 | `d127f59` | `feat(sub_d): add sub-C sidecar reader` |
| T5 | `2d9573d` | `feat(sub_d): add derivation evidence primitives` |
| T6 | `a2162f3` | `feat(sub_d): add macro frequency analysis artifacts` |
| T7 | `8a8f129` | `expt(sub_d): generate macro vocab proposal artifacts` |
| (notes) | `f10fa5e` | `docs(sub_d): record Phase B tension flags during Gate 2B halt` |
| T7→T8 | `12b1cdf` | `refactor(sub_d): restructure proposal artifacts for Gate 2 (A1+A2+A3)` |
| T7→T8 | `dfe9a50` | `refactor(sub_d): drop top-level candidate_strategies mirror; pin tile-pop evidence contract (F1+F2)` |
| **T8** | **`6e411cf`** | **`data(sub_d): lock macro plan vocab v1 (Task 8)`** — **Gate 2B closed.** Byte-identity-modulo-status-marker verified on real Singapore: 392309-byte proposal → 392307-byte locked (delta = -2, exact `'proposal' → 'locked'` substitution). |
| (notes) | `db28f5d` | `docs(known_issues): file 3 sub-D follow-ups from Gate 2B review` |
| T9 | `3f89599` | `feat(sub_d): add macro core artifact writers` |
| T9-fix | `ba9e0f6` | `fix(sub_d): reject SlotKind.TILE rows in macro_core writer` |
| T10 | `e12dae5` | `feat(sub_d): add effective conditioning overlay` |

Full fast suite at T10 commit: **430 passed, 10 deselected, 1 xfailed**. 53 sub-D tests.

## Locked vocab + artifacts (committed at T8 `6e411cf`)

Verbatim paths:

- `configs/macro_plan/v1/macro_plan_vocab.yaml`
- `reports/phase-1-sub-D/macro_vocab_proposal.yaml`
- `reports/phase-1-sub-D/zoning_analysis.yaml`
- `reports/phase-1-sub-D/cell_density_analysis.yaml`
- `reports/phase-1-sub-D/tile_population_density_analysis.yaml`
- `reports/phase-1-sub-D/road_skeleton_analysis.yaml`
- `tests/golden/sub_d/frequency_analysis/zoning_analysis.yaml`
- `tests/golden/sub_d/frequency_analysis/cell_density_analysis.yaml`
- `tests/golden/sub_d/frequency_analysis/tile_population_density_analysis.yaml`
- `tests/golden/sub_d/frequency_analysis/road_skeleton_analysis.yaml`

## Reviewer-confirmed design decisions (not in spec/plan; in tension-flags doc)

Source of truth: `docs/superpowers/notes/2026-05-19-sub-D-phase-B-tension-flags.md`.

- **A1** — per-namespace derivation versions (`ZONING_DERIVATION_VERSION`,
  `CELL_DENSITY_DERIVATION_VERSION`,
  `TILE_POPULATION_DENSITY_DERIVATION_VERSION`,
  `ROAD_SKELETON_DERIVATION_VERSION`). Analysis dict carries
  `derivation_versions` (plural, dict).
- **A2** — 4 namespace files + 1 index file Gate 2 layout. Reviewer edits
  the index only; namespace files are content-pinned by sha256 in the
  index's `namespace_files`. Byte-identity-modulo-status-marker test
  applies to the **index file only**.
- **A3** — `tile_population_density` evidence is in Tasks 5/6/7
  (Layer 1 emits all four candidate proxies as separate metric rows;
  reviewer picks one at Gate 2). Pinned in `evidence.py` docstring.
- **D1** — Layer-3 cached-Singapore tile IDs read from the locked vocab
  artifact's `selected_layer3_tiles` field (committed at T8). Task 15
  consumes them verbatim.

## Reviewer hand-edits at Gate 2B (locked vocab — do NOT re-litigate)

- `zoning.locked_buckets`: **top_4_categories (default kept)**. Reviewer
  overrode agent recommendation to cut `base` — argument: locking vocab
  from one region's data for a globally-generalizing model. Singapore
  `base` rarity does not extrapolate globally.
- `cell_density.locked_buckets`: `[0.0, 0.05, 0.15, 0.35) + [0.35, ∞)`.
- `road_skeleton.locked_buckets`: `[0, 1, 4, 9) + [9, ∞)`.
- `tile_population_density.locked_proxy`: `p75_building_footprint_ratio`.
- `tile_population_density.locked_buckets`:
  `[0.0, 0.02, 0.15, 0.31) + [0.31, ∞)`.

## Open known_issues filed this session

All deferred, **none are Phase 1 sub-D blockers**.

- **#9** — `cell_density` ratio > 1.0 invariant violation in real Singapore
  data (sub-C root cause investigation).
- **#10** — bucket-merge marginal-cost-of-cut metric degenerate (coverage
  stays at 1.0 across all bucket-merge strategies; needs replacement
  before next region enrollment).
- **#11** — Layer-3 sparse-side dimension scoring negates positive values,
  causing 3 of 13 dimensions to never pick a tile.

See `docs/known_issues.md`.

## Tension flags still applicable to Tasks 11-16

Source: `docs/superpowers/notes/2026-05-19-sub-D-phase-B-tension-flags.md`.

- **B1** — applied at T9-fix `ba9e0f6`. `SlotKind.TILE` rejected by
  `write_macro_core_parquet`.
- **B2** — applied at T9 `3f89599`. `isinstance(True, int)` trap; bool
  checked before int in `_dispatch_value`;
  `test_derivation_evidence_value_type_dispatches_bool_before_int` pins it.
- **B3** — applies to **Task 13**. Every version comparison in
  `src/cfm/data/sub_d/validator.py` (and any `validator_*.py`) must use
  `compare_version`; AST meta-test scans these files for direct `==`/`!=`
  on names/attrs/subscripts containing `version`.
- **B4** — applies to **Task 11**. Sub-D uses two digest semantics
  simultaneously: bytes-sha for the upstream view (sub-C input digests
  carried verbatim from `SubCTileInputs.digests`); excluding-timestamp
  for sub-D's own provenance self-integrity. **`rerun_reason` MUST NOT
  be in `SUB_D_EXCLUDED_FROM_SHA`** (audit-trail purpose, per sub-C F2
  precedent).
- **B5** — applied at T10 `e12dae5`. Conditioning copy is schema-driven
  via `_is_owner_marker` + `SUB_D_OWNED_FIELDS`. Future sub-C fields
  forward automatically.
- **B6** — applies to **Task 12**. Sub-D manifest copies the **entire**
  `manifest["config"]` dict from sub-C verbatim, not a hand-picked
  subset. Validator checks the full dict matches.
- **B7** — applied at T10 `e12dae5`. Sub-D YAML artifacts use
  namespaced versions (`effective_conditioning_schema_version`,
  `provenance_schema_version`, `manifest_schema_version`), never bare
  `schema_version`. Task 11 + Task 12 must inherit this discipline.
- **C1** — applied throughout. Sub-D uses consolidated `evidence.py` /
  (future) `validator.py` per the plan File Map, NOT the spec's
  `zoning.py`/`density.py`/`validator_inline.py` split. Plan wins.

## Task 11 starting context

Per the plan:

- Files: `src/cfm/data/sub_d/provenance.py`, `tests/data/sub_d/test_provenance.py`.
- 5 named tests (verbatim from plan):
  - `test_provenance_schema_uses_provenance_schema_version_not_bare_schema_version`
  - `test_provenance_records_sub_c_input_digests`
  - `test_provenance_records_locked_vocab_and_derivation_versions`
  - `test_provenance_sha_excludes_extracted_utc_and_output_sha_fields`
  - `test_provenance_sha_includes_rerun_reason`
- Functions (verbatim from plan):
  - `build_tile_provenance(tile_i, tile_j, extraction, inputs, versions, outputs) -> dict`
  - `provenance_sha256(data: dict) -> str`
  - `write_provenance(data: dict, path: Path) -> None`
- `SUB_D_EXCLUDED_FROM_SHA` table:
  - Includes `"extraction.extracted_utc"` (sub-D's own timestamp).
  - Includes final-segment `"*_sha256"` (chain-of-custody convention from sub-C).
  - **Excludes `extraction.rerun_reason`** (audit-trail; test pins it).
- B4 and B7 apply here. Re-read those entries before writing the
  exclusion table.

## Pointers (do NOT paraphrase; read directly)

- Spec: `docs/superpowers/specs/2026-05-19-phase-1-sub-D-macro-plan-derivation-design.md`
- Plan: `docs/superpowers/plans/2026-05-19-phase-1-sub-D-macro-plan-derivation.md`
- Tension flags: `docs/superpowers/notes/2026-05-19-sub-D-phase-B-tension-flags.md`
- Previous handoff: `docs/handoffs/2026-05-19-start-of-sub-D-implementation.md`
- Known issues: `docs/known_issues.md`

## Branch discipline (carried verbatim from session)

> Do NOT create new branches. Do NOT push to remote. Do NOT open pull
> requests. Commit task-by-task to the `phase-1-sub-D-macro-plan-derivation`
> branch via the user's git config.

Atomic commits per task. Reviewer approves between tasks. Halt on
surprises (failing invariant, unexpected data shape) — do not weaken
tests or assumptions to make broken code pass.

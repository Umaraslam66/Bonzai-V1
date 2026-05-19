# Session handoff — end of Phase 1 sub-D (2026-05-19)

> **For the reviewer:** the branch is ready for merge review. The merge
> decision is yours — this doc describes the state of the branch, not a
> request to merge. Do NOT merge to main without explicit approval.

## Branch state

- Branch: `phase-1-sub-D-macro-plan-derivation`
- Final commit: `7ae392e` (`test(sub_d): add cached Singapore integration coverage`)
- Working tree: clean (no staged or unstaged changes outside `.claude/`).
- Diverges from `main` by 27 commits (4 pre-implementation docs + 22
  task/refactor/handoff commits + 1 Task 16 commit added below).

## Test status

- Full fast suite: **470 passed, 14 deselected, 1 xfailed** (the xfail is
  the pre-existing tokenizer/encode boundary marker; not a sub-D
  regression).
- Sub-D focused fast suite: **93 passed, 4 deselected** (the 4 deselected
  are the slow Layer-3 integration tests).
- Slow Layer-3 integration suite: **4 passed in ~50 s** against the
  cached `data/processed/sub_c/2026-04-15.0/singapore/` extraction (494
  tiles). End-to-end pipeline runs, validator passes, byte-determinism
  holds across two re-runs.

## Task commits (chronological)

| Task | Commit | Subject |
|---|---|---|
| T1 | `5a15fa3` | `refactor(data): extract shared determinism helpers` — Gate 1 closed |
| T2 | `e9345d4` | `feat(sub_d): add version namespace helper` |
| T3 | `5fd543e` | `feat(sub_d): add fixed macro lattice utilities` |
| T4 | `d127f59` | `feat(sub_d): add sub-C sidecar reader` |
| T5 | `2d9573d` | `feat(sub_d): add derivation evidence primitives` |
| T6 | `a2162f3` | `feat(sub_d): add macro frequency analysis artifacts` |
| T7 | `8a8f129` | `expt(sub_d): generate macro vocab proposal artifacts` |
| (notes) | `f10fa5e` | `docs(sub_d): record Phase B tension flags during Gate 2B halt` |
| T7→T8 | `12b1cdf` | `refactor(sub_d): restructure proposal artifacts for Gate 2 (A1+A2+A3)` |
| T7→T8 | `dfe9a50` | `refactor(sub_d): drop top-level candidate_strategies mirror; pin tile-pop evidence contract (F1+F2)` |
| **T8** | **`6e411cf`** | **`data(sub_d): lock macro plan vocab v1`** — **Gate 2B closed.** Byte-identity-modulo-status-marker verified on real Singapore: 392309-byte proposal → 392307-byte locked (delta = -2, exact `'proposal' → 'locked'` substitution). |
| (notes) | `db28f5d` | `docs(known_issues): file 3 sub-D follow-ups from Gate 2B review` |
| T9 | `3f89599` | `feat(sub_d): add macro core artifact writers` |
| T9-fix | `ba9e0f6` | `fix(sub_d): reject SlotKind.TILE rows in macro_core writer` |
| T10 | `e12dae5` | `feat(sub_d): add effective conditioning overlay` |
| (handoff) | `5baf0bc` | `docs(handoff): mid-sub-D handoff for Task 11 start` |
| T11 | `474808d` | `feat(sub_d): add per-tile provenance` |
| T12 | `52d2adc` | `feat(sub_d): add region manifest` |
| T13 | `361ffea` | `feat(sub_d): add sidecar validators` |
| T14 | `6c2d49a` | `feat(sub_d): add sidecar derivation pipeline` |
| T15 | `7ae392e` | `test(sub_d): add cached Singapore integration coverage` |
| T16 | (this commit) | `docs(handoff): end of sub-D macro plan derivation` |

## Locked artifacts (verbatim paths)

Phase A→B handoff (committed at T8 `6e411cf`):

- `configs/macro_plan/v1/macro_plan_vocab.yaml`

Reports — the namespace files + index that the locked vocab was promoted
from (A2: 4 namespace files + 1 index; reviewer edits the index only,
namespace files are content-pinned by sha256 in the index's
`namespace_files`):

- `reports/phase-1-sub-D/macro_vocab_proposal.yaml`
- `reports/phase-1-sub-D/zoning_analysis.yaml`
- `reports/phase-1-sub-D/cell_density_analysis.yaml`
- `reports/phase-1-sub-D/tile_population_density_analysis.yaml`
- `reports/phase-1-sub-D/road_skeleton_analysis.yaml`

Golden copies for frequency-analysis byte-identity tests:

- `tests/golden/sub_d/frequency_analysis/zoning_analysis.yaml`
- `tests/golden/sub_d/frequency_analysis/cell_density_analysis.yaml`
- `tests/golden/sub_d/frequency_analysis/tile_population_density_analysis.yaml`
- `tests/golden/sub_d/frequency_analysis/road_skeleton_analysis.yaml`

## Reviewer-confirmed design decisions during implementation

Source of truth:
`docs/superpowers/notes/2026-05-19-sub-D-phase-B-tension-flags.md`.

- **A1** — per-namespace derivation versions
  (`ZONING_DERIVATION_VERSION`, `CELL_DENSITY_DERIVATION_VERSION`,
  `TILE_POPULATION_DENSITY_DERIVATION_VERSION`,
  `ROAD_SKELETON_DERIVATION_VERSION`). Analysis dict carries
  `derivation_versions` (plural, dict). No single global
  `derivation_version` anywhere.
- **A2** — 4 namespace files + 1 index file Gate 2 layout. Reviewer
  edits the index only; namespace files are content-pinned by sha256 in
  the index's `namespace_files`. Byte-identity-modulo-status-marker test
  applies to the **index file only**.
- **A3** — `tile_population_density` evidence is in Tasks 5/6/7
  (Layer 1 emits all four candidate proxies as separate metric rows;
  reviewer picks one at Gate 2). Pinned in `src/cfm/data/sub_d/evidence.py`
  docstring.
- **D1** — Layer-3 cached-Singapore tile IDs read from the locked vocab
  artifact's `selected_layer3_tiles` field (committed at T8). Task 15
  consumes them verbatim.

Phase B implementation tension flags (B1–B7) also confirmed during
Tasks 9–14; same notes doc enumerates each with citations to the commit
where it was applied.

## Gate 2B reviewer hand-edits — LOCKED, do NOT re-litigate

- `zoning.locked_buckets`: **top_4_categories (default kept)**. Reviewer
  overrode agent recommendation to cut `base` — argument: locking vocab
  from one region's data for a globally-generalizing model. Singapore
  `base` rarity does not extrapolate globally. Token order: building→0,
  road→1, poi→2, base→3.
- `cell_density.locked_buckets`: `[0.0, 0.05, 0.15, 0.35) + [0.35, ∞)`.
- `road_skeleton.locked_buckets`: `[0, 1, 4, 9) + [9, ∞)`.
- `tile_population_density.locked_proxy`: `p75_building_footprint_ratio`.
- `tile_population_density.locked_buckets`:
  `[0.0, 0.02, 0.15, 0.31) + [0.31, ∞)`.

Sub-E and later sub-projects consume these as fixed inputs.

## Residual risks

Explicitly named so they do not lurk in commit history alone.

- **Cross-environment determinism (darwin/aarch64 vs linux/x86_64) —
  unverified.** Byte-determinism on rerun is pinned by Task 15's slow
  test, but only on the local development platform. Sub-D output has not
  yet been derived on Leonardo, so the contract has not been observed
  cross-platform. When sub-D runs on Leonardo for the first time, the
  hash of the full Singapore output must be compared against the local
  platform's hash; any difference would surface as a digest mismatch in
  downstream sub-E / training-time validators. Sentinel test:
  `tests/data/sub_d/test_singapore_integration.py::test_cross_environment_determinism_gap_is_documented_if_not_run`.
- **known_issue #9** — `cell_density` ratio > 1.0 invariant violation in
  real Singapore data. Absorbed by the locked vocab's `[0.35, ∞)` top
  bucket; root cause (sub-C cell-area vs building-footprint
  computation) is unresolved. The validator and pipeline do NOT clamp
  or reject — pinned by `test_cell_density_top_bucket_absorbs_ratio_above_one`
  with explicit tail values (0.95, 1.0, 1.5, 7.42). Investigate before
  asserting any ratio ≤ 1.0 elsewhere.
- **known_issue #10** — bucket-merge marginal-cost-of-cut metric
  degenerate (coverage stays at 1.0 across all bucket-merge
  strategies). The metric needs replacement before the next region's
  Phase A vocab proposal, otherwise the cost-of-cut elbow heuristic
  produces no signal.
- **known_issue #11** — Layer-3 sparse-side dimension scoring negates
  positive values, causing 3 of 13 dimensions to never pick a tile.
  Fix before re-running Layer-3 subset selection on a new region.

See `docs/known_issues.md` for full text on each.

## Pointers

Read these directly; do not paraphrase.

- Spec: `docs/superpowers/specs/2026-05-19-phase-1-sub-D-macro-plan-derivation-design.md`
- Plan: `docs/superpowers/plans/2026-05-19-phase-1-sub-D-macro-plan-derivation.md`
- Tension flags (A1–A3 + B1–B7 + C1 + D1): `docs/superpowers/notes/2026-05-19-sub-D-phase-B-tension-flags.md`
- Previous handoff (start of implementation): `docs/handoffs/2026-05-19-start-of-sub-D-implementation.md`
- Mid-implementation handoff (Task 11 entry point): `docs/handoffs/2026-05-19-mid-sub-D-task-11-start.md`
- Known issues: `docs/known_issues.md`

## Merge note

The branch is ready for merge review. **The merge decision is the
reviewer's** — do not merge to `main` automatically under any
circumstances. The agent halts here.

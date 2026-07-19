# Session handoff — end of Phase 1 sub-E (2026-05-20)

> **For the reviewer:** the branch is ready for merge review. The merge
> decision is yours — this doc describes the state of the branch, not a
> request to merge. Do NOT merge to main without explicit approval.

## Branch state

- Branch: `phase-1-sub-E-boundary-contracts`
- Final code commit before this handoff: `214ee75` (Task 14,
  `test(sub_e): add cached Singapore integration and empirical gate`).
- Working tree: clean (no staged or unstaged changes outside `.claude/`
  and this handoff file).
- Diverges from `main` by 41 commits (15 task commits + 22 plan / design
  / fixup commits + 2 real-data integration fix commits + 1 lint-cosmetic
  cleanup commit + this handoff commit).

## Test status

- Full fast suite: **552 passed, 21 deselected, 1 xfailed** (the xfail
  is the pre-existing Phase 0 tokenizer entry-marker, not a sub-E
  regression).
- Slow sub-E Singapore integration suite
  (`tests/data/sub_e/test_singapore_integration.py`):
  **7 passed in ~2 s** against the cached
  `data/processed/sub_d/2026-04-15.0/singapore/` extraction. End-to-end
  pipeline runs, inline + cross-tile validators pass, byte-determinism
  holds across two re-runs, empirical gate distribution matches the
  golden YAML, writer regression-guard sees both MAJOR_ROAD and
  MINOR_ROAD round-trip, lever-3 collapse path verified against real
  data.

## Sub-E scope (one-paragraph summary)

Sub-E adds **boundary contracts**: a 4-slot per-cell view (N/E/S/W) of
each cell's edges, persisted as a per-edge parquet (`boundary_contract.parquet`,
144 rows = 112 internal + 32 external), with a 4-token vocab
(`BOUNDARY_NOT_APPLICABLE=0`, `NONE=1`, `MAJOR_ROAD=2`, `MINOR_ROAD=3`).
The vocab is locked v1.0; sentinel 0 is dataloader-side only and never
on-disk. Sub-E reads sub-D's macro_core (scope marker per edge) + sub-C's
crossings+features (raw road class strings), derives `boundary_class_enum`
per active internal edge via a class-grouping map (MAJOR_ROAD ←
{primary, trunk, secondary}; MINOR_ROAD ← {tertiary, residential, service,
unclassified, footway, steps, cycleway}; default bucket → MINOR_ROAD for
null/unknown), writes per-edge storage with read-time rotation into per-cell
views, and ships an inline + cross-tile validator pair enforcing 9 + 5
invariants respectively.

## Locked artifacts (verbatim paths)

- `configs/macro_plan/v1/boundary_vocab.yaml` — 4-token vocab, locked at
  commit `633f7fc`. Sha pinned in spec §16.2:
  `0b2d9eb4c1253b12f2fe50b32ec459e8fc25cdeafa05f4dda0a47240c0c9a1fd`
  (sub-D's macro_plan_vocab.yaml — sub-E inherits scope tokens from this
  artifact unchanged, does NOT modify the locked sub-D file).

## Version constants (three-axis versioning)

Per spec §9.1, sub-E carries three independent version namespaces tracked
by `compare_version`:

| Namespace | Constant | Value | Sub-D mapping |
|---|---|---|---|
| DATA_SHAPE | `SUB_E_SCHEMA_VERSION` | `"1.0"` | governs on-disk parquet schema + YAML structure |
| VOCAB | `BOUNDARY_VOCAB_VERSION` | `"1.0"` | governs `boundary_vocab.yaml` token domain |
| DERIVATION | `BOUNDARY_DERIVATION_VERSION` | `"1.0"` | governs class-grouping + multi-crossing tie-break |

Exported from `src/cfm/data/sub_e/versions.py`. All three pinned at v1.0
for the Phase 1 de-risk run.

## Output paths

- Sub-E region output (per spec §13): `data/processed/sub_e/<release>/<region>/`
  - `manifest.yaml` — region manifest (tiles sorted by `(tile_i, tile_j)`,
    self-integrity via `manifest_sha256`)
  - `_SUCCESS` — empty marker file; written ONLY after cross-tile
    validator passes (sub-D precedent at
    `src/cfm/data/sub_d/pipeline.py:254-255`)
  - `tile=EPSG3414_i{ti}_j{tj}/` — per-tile directory
    - `boundary_contract.parquet` — 144 rows, sorted `(slot_kind, slot_index)`
    - `provenance.yaml` — per-tile sha-chain anchor with
      `extracted_utc` excluded from `provenance_sha256` per
      `SUB_E_EXCLUDED_FROM_SHA`

For the Phase 1 de-risk run: `data/processed/sub_e/2026-04-15.0/singapore/`

## Sub-D dependency contract

Sub-E reads sub-D's macro_core + sub-C's crossings + features:

- **Sub-D `_SUCCESS` gate:** sub-E's pipeline orchestrator
  (`src/cfm/data/sub_e/pipeline.py:derive_region`) calls
  `require_sub_d_success_marker(cfg.sub_d_region_dir)` as its **first**
  operation. Raises `FileNotFoundError` on absence; no partial sub-E
  output produced.
- **Lattice inheritance (spec §4.1):** sub-E inherits sub-D's lattice
  verbatim. The 8×8 cell grid + 112-internal + 32-external edge enumeration
  is sub-D's authority. Sub-E does NOT synthesize edges via rotation
  ahead of sub-D; the rotation function (`src/cfm/data/sub_e/rotation.py`)
  is the per-cell view at read-time, not an alternative lattice
  generator.
- **Scope token inheritance (spec §6.1):** sub-E references scope tokens
  `{active=0, fully_masked=1, scope_boundary=2, external_deferred=3}` by
  reference to sub-D's locked vocab. The sub-D vocab sha
  `0b2d9eb4c1253b12f2fe50b32ec459e8fc25cdeafa05f4dda0a47240c0c9a1fd` is
  pinned at spec §16.2.

## Cross-environment determinism

Sub-E inherits the cross-environment determinism residual from sub-D
(`§15 #7` deferral). Same-process byte-identity verified locally on
darwin/aarch64 + on real Singapore at Task 14 close. Cross-environment
verification (darwin/aarch64 vs Leonardo linux/x86_64) is the same
residual sub-D inherited; sub-E does NOT introduce a new determinism
posture. The first sub-E run on Leonardo is the verification trigger.

## Task commits (chronological)

| Task | Commit | Subject |
|---|---|---|
| (design) | `9ae86fa` | `docs(sub_e): phase 1 sub-E boundary contracts design` |
| (design fixup) | `5f4be16` | `docs(sub_e): fix illustrative tile_count + verify pointer paths` |
| (plan) | `a699b87` | `docs(sub_e): phase 1 sub-E boundary contracts implementation plan` |
| (plan fixup) | `f76355c` | `docs(sub_e): plumb lever-3 collapse through inline validator` |
| (plan fixup) | `6269673` | `docs(sub_e): add Layer-3 lever-3 regression test to Task 14` |
| **T1** | `633f7fc` | `data(sub_e): lock boundary vocab v1` |
| (plan fixup) | `e457d4d` | `docs(sub_e): rewrite Task 2 against sub-D compare_version actual API` |
| **T2** | `fd83e61` | `feat(sub_e): add package skeleton and version constants` |
| **T3** | `6b8db6b` | `feat(sub_e): add per-cell rotation function` |
| **T4** | `469737b` | `feat(sub_e): add class-precedence derivation function` |
| (plan fixup) | `b8d4847` | `docs(sub_e): fix parents[3] → parents[4] in src/cfm/data/sub_e/ code` |
| (plan fixup) | `2c1eece` | `feat(sub_e): pin strict-equality class_raw matching convention` |
| **T5** | `695b6b4` | `feat(sub_e): add sub-C and sub-D input readers` |
| (plan fixup) | `fabce8b` | `docs(sub_e): strengthen Task 6 writer test set and pin pa.schema` |
| **T6** | `711be4f` | `feat(sub_e): add boundary contract parquet writer` |
| (plan fixup) | `0708a2b` | `docs(sub_e): land Task-6 carry-forwards into spec + Task 7 + Task 14` |
| (plan fixup) | `d9e0fbb` | `docs(sub_e): fix two Task 7 plan defects (validator loop order + #8 I/O)` |
| **T7** | `4cad630` | `feat(sub_e): add inline validator (membership-before-semantic + required provenance kwarg)` |
| (plan fixup) | `8920bd4` | `docs(sub_e): standardize zip(strict=True) in plan code blocks for Tasks 8-14` |
| (plan fixup) | `4ffa101` | `docs(sub_e): atomic fixup for Tasks 8/9/10 — canonicalize_yaml idiom + EXCLUDED_FROM_SHA mechanism` |
| **T8** | `af22f19` | `feat(sub_e): add manifest and provenance writers with EXCLUDED_FROM_SHA` |
| (plan fixup) | `60d50fb` | `docs(sub_e): fix Task 8 Step 5 expected count off-by-one (12 → 11)` |
| (plan fixup) | `3ed0554` | `docs(sub_e): atomic fixup for Task 9 — invariant #5 rotation-aware + compare_version` |
| (plan fixup) | `a521763` | `docs(sub_e): atomic fixup for Task 9 test fixtures (three defects)` |
| **T9** | `ff6f67f` | `feat(sub_e): add cross-tile validator with rotation-aware invariant #5` |
| (plan fixup) | `fd53fdd` | `docs(sub_e): atomic fixup for Task 10 — _SUCCESS ordering + provenance round-trip` |
| (plan fixup) | `d95ace7` | `docs(sub_e): atomic fixup for Task 10 fixture (three more defects)` |
| **T10** | `9f40434` | `feat(sub_e): add pipeline orchestrator with halt-on-validator-fail` |
| (plan fixup) | `aa0004a` | `docs(sub_e): retroactive plan-fixup closing Task 10 plan-vs-code drift` |
| **T11** | `c167a2c` | `feat(sub_e): add derive and validate CLI scripts` |
| (plan fixup) | `590f468` | `docs(sub_e): clarify Task 11 smoke scope (--help only, not fixture invocation)` |
| **T12** | `f24f18e` | `feat(eval): add shuffle strategies for perplexity gap` |
| (plan fixup) | `2d19052` | `docs(sub_e): ratify scalar-NLL interface; remove dead-code ternary in plan Task 13` |
| **T13** | `1dcc7e9` | `feat(eval): add perplexity gap shell with sign test` |
| (plan fixup) | `4a31e77` | `docs(sub_e): fix Task 14 fail-loud + determinism + naming + diagnostic` |
| (spec) | `bd654d4` | `docs(sub_e): expand §15 #1 with v2 vocab candidates from Task 14 audit` |
| **T14** | `214ee75` | `test(sub_e): add cached Singapore integration and empirical gate` |
| (Task 3 real-data fix) | `8e90869` | `fix(sub_e): correct rotation axis convention to match sub-D lattice (Task 3 defect surfaced by Task 14)` |
| (Task 5 real-data fix) | `2e4f1a8` | `fix(sub_e): correct feature_class int8 type contract with sub-C (Task 5 defect surfaced by Task 14 writer-regression-guard)` |
| (lint cleanup) | `9e95e0d` | `style(sub_e): clear ruff RUF002 + I001 in rotation.py and versions.py` |
| **T15** | (this commit) | `docs(handoff): end of sub-E boundary contracts` |

## Decisions log (pointers)

- **Spec:** `docs/superpowers/specs/2026-05-20-phase-1-sub-E-boundary-contracts-design.md`
- **Plan:** `docs/superpowers/plans/2026-05-20-phase-1-sub-E-boundary-contracts.md`
- **Brainstorm topics 1–10:** captured inline in spec sections 1–12.

## Reviewer-confirmed design decisions during implementation

The following decisions were ratified during implementation as plan-fixup
commits or audit findings, not original spec decisions:

1. **Strict byte-equality class_raw matching** (`2c1eece` + spec §5.1).
   Case-variants and whitespace variants fall to MINOR_ROAD via default
   bucket; no normalization. Pinned for cross-environment determinism.
2. **SlotKind cross-enum byte-equivalence** (`0708a2b`, validator §10.1
   #9 carry-forward). Sub-D and sub-E maintain two separate `SlotKind`
   IntEnums sharing wire values `INTERNAL_EDGE=1, EXTERNAL_EDGE=2`;
   validator asserts equivalence at runtime.
3. **Inline validator loop order: membership-before-semantic**
   (`d9e0fbb`). Per-row checks fire `#5 scope` → `#6 slot_index` →
   `#7 axis` before `#3 non-null-iff` → `#4 active class membership`.
4. **`provenance_derivation_version` is required, not defaulted**
   (`d9e0fbb`). Inline invariant #8 requires the caller to thread the
   actual value from disk; constants-on-both-sides would make the check
   vacuous.
5. **`canonicalize_yaml` idiom + `SUB_E_EXCLUDED_FROM_SHA`** (`4ffa101`).
   No `write_yaml_canonical` helper exists; sub-E mirrors sub-C/sub-D's
   `path.write_text(canonicalize_yaml(data), encoding="utf-8")` pattern,
   and emits exclusion-aware sha helpers (`provenance_sha256`,
   `manifest_sha256`) so live-clock reruns don't break the digest chain.
6. **Rotation-aware invariant #5 in cross-tile validator** (`3ed0554`).
   Spec §10.2 #5's "single-cell membership" check enumerates the 8×8
   grid via `cell_to_edge_ids` and asserts set-equality between
   parquet's external tuples and rotation's external set, not just
   slot_index uniqueness.
7. **`_SUCCESS` validate-then-touch ordering** (`fd53fdd`). Cross-tile
   validator runs BEFORE `_SUCCESS.touch()`; no try/except/unlink dance.
   Sub-D precedent.
8. **Per-tile `(slot_kind, lower_cell_i, lower_cell_j, axis)` 4-tuple
   keys in `_derive_tile_rows`** (`aa0004a`). Rotation can produce
   identical 3-tuples for internal and external versions of distinct
   physical edges; `slot_kind` disambiguates.
9. **Scalar-NLL `model_forward` interface for the eval shell**
   (`2d19052`). The eval module is tokenizer-free and torch-free by
   construction; callers wrap their model to return per-token NLL.

## Discipline observations — real-data integration defects (sixth-gate pattern)

Task 14's first contact with real cached sub-D / sub-C Singapore data
surfaced **two distinct integration defects**, BOTH instances of the same
"synthetic-fixture-masks-real-data" trap. Both were caught only because
Task 14 ships a writer-regression-guard against real bytes, not because
any prior layer of defense (plan review, pre-dispatch audit, implementer
test-run, halt-and-report, pre-code data-flow reasoning) flagged them.
Sub-F should start from the corrected baseline these fixes establish.

1. **Rotation axis convention** (commit `8e90869`, Task 3 defect).
   Sub-E's `cell_to_edge_ids` (`src/cfm/data/sub_e/rotation.py:50-62`)
   swapped axes — coded north/south as `AXIS_X=0` instead of `axis=1`
   (j-neighbor face) and west/east as `AXIS_Y=1` instead of `axis=0`
   (i-neighbor face) — AND used in-grid pinning (`lower_i = 0` for west,
   `lower_i = 7` for east) instead of sub-D's off-grid convention.
   Sub-D's `src/cfm/data/sub_d/lattice.py:11-14` documented the correct
   convention. Every synthetic sub-E fixture from Tasks 6–13 consulted
   the buggy `cell_to_edge_ids` to build expected values; the cross-tile
   validator consulted the same function; both agreed self-consistently.
   First contact with real cached sub-D Singapore data via Task 14
   surfaced the swap. Fixed via plan-fixup commit with explicit Task 3
   linkage in the message.

2. **`feature_class` int8 type contract** (commit `2e4f1a8`, Task 5
   defect). Sub-E declared `SubCFeatureRow.feature_class: str`, but
   sub-C's parquet schema has it as `pa.int8()` with
   `FEATURE_CLASS: {0: "road", 1: "building", ...}` (sub-C `io.py:44` +
   `enums.py:22`). The pipeline filter `f.feature_class == "road"` was a
   silent no-op against real int8 data. `features_by_id` was empty;
   every crossing's `class_raw` lookup returned None; every active edge
   with crossings classified as MINOR_ROAD via default bucket;
   MAJOR_ROAD never appeared. **The empirical gate would have PASSED
   the broken distribution** (16% NONE / 84% MINOR_ROAD / 0%
   MAJOR_ROAD — both 0.16 and 0.84 inside `[0.02, 0.90]`). Only the
   writer-regression-guard test (`test_layer3_writer_round_trips_major_and_minor`)
   caught it. Fixed via plan-fixup commit with explicit Task 5 linkage.

### The sixth-gate principle (named after the rotation defect)

The principle was named after the rotation fix and ratified by use
within hours during the `feature_class` defect:

> When introducing a new abstraction over an existing module, write at
> least one test that cross-references the new abstraction against the
> existing module's docstring/source as ground truth, without using the
> new abstraction in the assertion logic.

Captured in memory entry `feedback_external_source_of_truth_gate.md`,
which now lists this as the SIXTH gate alongside the five
internal-consistency gates (plan review + pre-dispatch audit +
pre-code data-flow reasoning + implementer test-run + halt-and-report).

### Three additional disciplines added to the memory entry on 2026-05-21

After the second instance (the `feature_class` defect), the same memory
entry was extended with three additional disciplines (markers (h), (i),
(j) in `feedback_external_source_of_truth_gate.md`):

- **(h) Reactive corollary:** when audit/verification code is found to
  have a bug against an upstream contract, audit whether the system
  under audit has the same bug. Same mental shortcut produces both
  bugs. The reviewer's prediction script for the empirical gate had the
  exact same `feature_class == "road"` bug; fixing the audit script did
  NOT trigger a production-code audit, and the production bug shipped
  through 11 tasks.
- **(i) Proactive principle (deeper):** verify upstream contracts by
  reading the upstream module's source/schema, NOT by inferring from
  semantically-named field strings. Both authors of the
  `feature_class` bug (reviewer + sub-E implementer) made the same
  mental shortcut: "a field called `feature_class` probably stores a
  class name as a string." Field names are documentation, not
  contracts. Read the schema. This principle prevents the bug at
  write-time, before it can become a reactive corollary case.
- **(j) Threshold-pairing principle:** every threshold-based verdict
  requires a paired structural-correctness check that asserts specific
  falsifiable properties the working system must satisfy. Empirical
  thresholds are DISTRIBUTION-SHAPE verdicts, not CORRECTNESS verdicts.
  Sub-E's `[max ≤ 0.90, min ≥ 0.02]` empirical gate would have
  green-lit the broken 16/84 distribution. The
  writer-regression-guard test ("both MAJOR_ROAD and MINOR_ROAD must
  appear, both classes round-trip") was the load-bearing
  structural-correctness check for sub-E.

### Implications for sub-F's brainstorm

Sub-F should start from this corrected baseline. Pre-dispatch audits
for tasks introducing a new abstraction over an existing module must
include an explicit check: **"where in the test suite does this new
abstraction get cross-referenced against the existing module's external
documentation?"** If the answer is "nowhere," halt the dispatch and
require the cross-reference test be written before code lands.

## Deferral ledger reference

See spec §15.1 (12 entries). Final status per entry at sub-E close:

- **§15 #1 (v2 vocab + derivation expansion):** **Amended during
  implementation.** Expanded from the original
  NONE_INTERIOR/NONE_NATURAL_BOUNDARY split to enumerate five
  candidate axes (NONE split, highway-tier promotion, underground
  transit, pedestrian-class, rail handling) backed by Task 14
  pre-dispatch audit's empirical `class_raw` distribution on the
  Layer-3 9-tile subset. See spec §15.1 row 1 + commit `bd654d4` for
  evidence and rationale. Entry **remains deferred** in expanded form;
  no v1 action.
- **§15 #2 (position-shuffled perplexity gap):** Remains deferred.
  Re-open trigger is post-reset structural test; sub-E ships only the
  two shuffle strategies that the v1 gap needs.
- **§15 #3 (second-order conditioning analysis: macro+boundary joint
  attribution):** Remains deferred. Post-architecture-iteration
  concern.
- **§15 #4 (full-Singapore ~494-tile contingent training run):**
  **Not yet evaluated.** Trigger condition ("GT-gap clears AND
  generated-gap fails to clear", spec §11.5) is post-scaffold; the
  decision belongs to the post-Leonardo training-scaffold sub-project,
  not sub-E. Sub-E ships the eval harness shell that produces the
  inputs to this decision.
- **§15 #5 (generated-conditioned gap, contingent on calendar
  cascade):** **Not yet evaluated.** Same post-scaffold dependency as
  #4; sub-E ships `compute_perplexity_gap` (Task 13) as the harness
  shell — the perplexity numbers themselves come after the
  training-scaffold sub-project lands.
- **§15 #6 (per-cell denormalised storage alternative shape):**
  Remains deferred. Trigger is "Layer 2 regression guard #8 ever
  fails" — guard passes on real Singapore at Task 14, so no migration
  signal.
- **§15 #7 (cross-environment determinism, darwin/aarch64 vs Leonardo
  linux/x86_64):** **Remains deferred (partial verification).**
  Same-process determinism on darwin/aarch64 verified at Task 14
  (`test_layer3_deterministic_rerun_same_process` passing in
  ~2 seconds against real cached Singapore). Cross-environment leg
  remains deferred; first sub-E run on Leonardo is the verification
  trigger, identical to sub-D's residual.
- **§15 #8 (lever-3 collapse non-collapse re-run):** **Not triggered.**
  v1 derivation produces signal (MAJOR_ROAD landed within 0.81 pp of
  prediction, all three active-class fractions appear with healthy
  counts). Lever-3 was not pulled. The lever-3 path is implemented
  end-to-end and verified against real data by
  `test_layer3_lever_3_collapse_real_data`; the trigger condition
  (day-9 calendar slip, spec §2.3) did not fire.
- **§15 #9 (exact boundary contracts per PRD §5 stage 3: crossing
  positions, width/extent class, source-feature traceability):**
  Remains deferred. v1 ships the macro-tier surface only; exact
  contracts are sub-E v2's own sub-project per spec §15.1.
- **§15 #10 (stitching test as 4th sub-bar):** Remains deferred.
  Post-reset; likely combined with #9 in a sub-E v2 sub-project.
- **§15 #11 (sub-C minor revision request, contingent):** Remains
  deferred. Trigger is empirical evidence that v1 sub-E cannot clear
  the gap on `crossings.parquet + features.parquet` alone; that
  evidence is post-scaffold. No re-open signal from sub-E
  implementation.
- **§15 #12 (cell-density-ratio > 1.0 root cause, sub-D known_issue
  #9 pointer):** Remains deferred. Sub-E does not touch cell density;
  this is a sub-C-scope concern carried as a footnote per spec §15.2.

## Discipline observations from implementation

Pre-dispatch audit + plan-fix-then-dispatch pattern caught 15+
verify-before-asserting instances across Tasks 4–14:

1. T6 (`fabce8b`): `write_parquet_deterministic` doesn't exist — actual
   helper is `write_parquet`. Plan-vs-code drift.
2. T7 (`d9e0fbb`): two defects — invariant #5 fixture unreachable
   (membership-before-semantic loop ordering); invariant #8 vacuous when
   `provenance_derivation_version` defaults.
3. T8 (`4ffa101`): `write_yaml_canonical` doesn't exist + missing
   `EXCLUDED_FROM_SHA` mechanism (spec §9.2 mandate).
4. T9 (`3ed0554`): invariant #5 weakening (spec §10.2 #5 mandates
   rotation-aware check, plan had uniqueness-only) + bare `!=` instead
   of `compare_version`.
5. T9 (`a521763`): three test-fixture defects (YAML quote-style
   mismatch, parquet mutation breaking digest chain, `corrupt_idx=0`
   hitting axis=0 instead of axis=1).
6. T10 (`fd53fdd`): `_SUCCESS` write-then-unlink + invariant #8 vacuous
   constant-vs-itself.
7. T10 (`d95ace7` + `aa0004a`): fixture modulo-arithmetic collisions
   (edge_scope dict key collapse) + 3-tuple vs 4-tuple key for sub-E
   pipeline (caught at implementation time, retroactive plan-fixup).
8. T13 (`2d19052`): dead-code ternary `mean_gap if mean_tokens == 0 else
   mean_gap`; same shape as T10's `or True` trap.
9. T14 (`4a31e77`): four issues — vacuous "a class" diagnostic, fail-
   silent `pytest.skip` on missing sub-D, self-skip determinism test,
   misleading test name (uniqueness, not rotation-equality).

## Residual risks

- **Cross-environment determinism (spec §15 #7):** untested until first
  sub-E Leonardo run. Local same-process determinism verified at T10
  + T14.
- **§15 #1 v2 vocab expansion:** five candidate axes documented in spec
  but not implemented. v1 ships against the locked class-grouping map;
  perplexity-gap signal informs v2 deliberation.
- **§15 #8 lever-3-collapse non-collapse re-run:** if lever-3 fires
  pre-deadline, the de-risk verdict from that run is provisional and
  requires a non-collapse re-run in the post-reset 5,000-hour window.

## Empirical gate outcome

**Gate PASSED on real cached Singapore at Task 14 close (commit
`214ee75`).** Distribution recorded in
`tests/golden/sub_e/empirical_gate/layer3_boundary_class_distribution.yaml`.

### Prediction vs actual

Layer-3 9-tile subset, predicted at Task 14 pre-dispatch audit
(commit `4a31e77` + `bd654d4`) vs measured at Task 14 close:

| Class | Predicted | Actual | Delta |
|---|---|---|---|
| NONE | 29.66% | 16.05% | -13.61 pp |
| MAJOR_ROAD | 18.55% | 19.36% | +0.81 pp |
| MINOR_ROAD | 51.79% | 64.60% | +12.81 pp |
| Active edges | 1008 (predicted, worst case) | 966 | -42 (95.8%) |

### The load-bearing signal: MAJOR_ROAD landed within 0.81 pp of prediction

The class-grouping map (`primary / trunk / secondary -> MAJOR_ROAD`
per spec §5.1) is the load-bearing v1 design decision in sub-E. It
survived empirical contact with real Overture data: the predicted
MAJOR_ROAD fraction of 18.55% landed within **0.81 percentage points**
of the actual 19.36%. This is the verdict that the spec §5.1 mapping is
well-calibrated against real Singapore, not merely assumed-correct.

### Diagnostic narrative on the NONE <-> MINOR_ROAD swap

The 13.6 pp redistribution from NONE into MINOR_ROAD compounds two
distinct effects:

1. **Activity-ratio effect.** The prediction assumed the worst case of
   1008 active internal edges (all 112 per tile, 9 tiles). Reality:
   966 active edges (95.8%). The remaining 42 edges are
   `scope_boundary` or `fully_masked` per sub-D's macro_core output.
   This contributes roughly 5% of the redistribution (a small
   denominator shift across all three classes).
2. **Crossing-prevalence effect.** More edges have at least one road
   crossing than the prediction model estimated. Dense urban Singapore
   subset translates to more roads, which translates to fewer
   zero-crossing (NONE) edges. This drives the bulk of the 13.6 pp
   swap.

### Teaching observation for sub-F's prediction-step protocol

Single-step predictions over multi-stage pipelines conflate multiple
assumption effects into a single number. When the verdict is "matches
to within X pp" or "diverges by X pp," divergence diagnostic work is
forced to back-attribute the delta across all stages at once — a
poorly-conditioned inverse problem.

Future predictions for similar verdicts (sub-F brainstorm,
training-scaffold validation) should **decompose predictions into
per-stage assumptions** with attributable deltas, so divergence work
attributes deltas to specific stages. Sub-E's prediction model was
single-step over the (activity-ratio × crossing-prevalence ×
class-grouping-map) compound; sub-F-class predictions should ship as
per-stage with attributable deltas baked in from the start.

### Threshold pass + buffer

- Max class fraction: 0.646 ≤ 0.90 (headroom: 0.254).
- Min class fraction: 0.160 ≥ 0.02 (headroom: 0.140).
- All three active classes (NONE, MAJOR_ROAD, MINOR_ROAD) appear with
  healthy counts.

Not a marginal pass; both spec §11.3 #1 thresholds have substantial
headroom.

### Golden artifact pointer

- `tests/golden/sub_e/empirical_gate/layer3_boundary_class_distribution.yaml`
  committed at `214ee75`; contains the exact fractions plus active-edge
  count for byte-identity regression-guarding on future re-runs.

## Layer 3 perplexity-gap eval scaffold readiness

Tasks 12 + 13 shipped the eval-harness shell at `src/cfm/eval/`
(shuffle strategies + `compute_perplexity_gap` with sign test),
**torch-free and tokenizer-free by construction.** The interface
expects a caller-provided `model_forward` callable that returns
per-token NLL (scalar) — callers wrap their model. No tokenizer
coupling, no torch dependency in the module under `src/cfm/eval/`.

Wiring needed for the training-scaffold sub-project:

- Provide `model_forward(token_ids, conditioning) -> nll_per_token`
  by wrapping the trained checkpoint.
- Provide the shuffle dataloader iterator (sub-E supplies the two
  shuffle strategies; the training scaffold supplies the actual
  ground-truth tokens).
- Run `compute_perplexity_gap` on saved checkpoints; assert spec
  §11.4 thresholds (GT-gap ≥ 0.05 nats/token, generated-gap < 0.02
  nats/token, sign-test p-values).

No sub-E rework required; the shell is sealed at this contract.

## Full-Singapore contingent run (§15 #4)

**Not yet evaluated at sub-E close.** Trigger condition (spec §11.5):
**GT-gap clears AND generated-gap fails to clear.** Both numbers come
from the perplexity-gap shell wired into the training scaffold —
neither exists yet at sub-E close. The decision is therefore
post-scaffold, not a sub-E artifact.

When the decision is faced, the entry point is spec §15 #4 + this
handoff's deferral ledger reference; the eval-harness shell at
`src/cfm/eval/` is the consumer of the two contingent measurements.

## Lever 3 collapse status

**Lever-3 was NOT triggered during sub-E.** v1 derivation produces
signal — MAJOR_ROAD landed within 0.81 pp of prediction, and all three
active classes (NONE, MAJOR_ROAD, MINOR_ROAD) appear with healthy
counts on the Layer-3 real Singapore subset. The lever-3 trigger
condition (day-9 calendar slip per spec §2.3) did not fire.

The lever-3-collapse mechanism is fully implemented and regression-
guarded against real data:

- Code path: Tasks 6, 7, 10 are lever-3-aware
  (`PipelineConfig.lever_3_collapse` flag).
- Verification: `test_layer3_lever_3_collapse_real_data` (Task 14)
  confirms the mechanism works against real cached Singapore data
  when the operator pulls the lever. No real-data path was
  rebuilt-against-fixture; the lever-3 path was exercised end-to-end
  on real bytes.

If lever-3 fires post-deadline, the non-collapse re-run is **§15 #8**
(remains deferred; same trigger maps to the deferral ledger entry).

## Pointers

- **Spec:** `docs/superpowers/specs/2026-05-20-phase-1-sub-E-boundary-contracts-design.md`
- **Plan:** `docs/superpowers/plans/2026-05-20-phase-1-sub-E-boundary-contracts.md`
- **Sub-D handoff (inherited residuals):** `docs/handoffs/2026-05-19-end-of-sub-D.md`
- **Locked sub-D vocab (scope tokens inherited):**
  `configs/macro_plan/v1/macro_plan_vocab.yaml`
- **Locked sub-E vocab:** `configs/macro_plan/v1/boundary_vocab.yaml`

## Merge note

Do NOT merge to main without explicit reviewer approval. This handoff
describes the state of the branch, not a request to merge. The Phase 1
sub-project pattern is: each sub-project ships to its own branch, merges
to main at sub-project end via reviewer-initiated merge.

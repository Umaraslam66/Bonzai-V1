# Phase 1 sub-E boundary contracts design

Date: 2026-05-20

Status: design spec for review

> **This spec is hand-maintained.** The deferral ledger (§15), the version-bump
> trigger tables (§9), the threshold tables (§11), and the composite at-a-glance
> are not regenerated from any source file. Edits land directly in this
> document and the spec self-review checklist (§16) is a manual pass.

## 1. Scope and goal

Phase 1 sub-E derives the *boundary contract* sidecar dataset for Singapore.
It consumes sub-C's immutable extraction output and sub-D's macro-plan sidecar
and writes a new sub-E dataset under its own manifest, provenance, validation
gate, and `_SUCCESS` marker.

Sub-E exists to bridge sub-D's macro plan to the micro generator's per-cell
conditioning. The micro generator at training time reads, per cell, the
tile-level conditioning vector (sub-D), the macro plan tokens (sub-D), the
*boundary contract* for that cell's four edges (sub-E), and then is trained
to predict the micro target token sequence (sub-F, separate sub-project).

### 1.1 De-risk framing

Sub-E in Phase 1 is scoped against an **architecture-feasibility de-risk training
run**, not a production-quality boundary contract. The de-risk run launches in
the pre-deadline Leonardo window (~14 days from this spec's date to early-June
deadline) and is the architecture verdict for the hierarchical autoregressive
design. Production-quality boundary contracts — crossing positions,
width/extent, source-feature traceability — are deferred to the post-reset
5000-hour window (§15).

The descope-by-default posture applies to every sub-E design decision in this
spec: choose the cheapest variant that lets the de-risk run answer its
architecture question.

### 1.2 The three de-risk sub-bars

The de-risk run's verdict is conditioned on three sub-bars, in increasing
load-bearingness:

1. **Loss decreases monotonically.** Model is learning something. Weakest
   bar; satisfied by almost any well-formed token stream.
2. **Generated macro plans look spatially coherent on held-out tiles.** Model
   is not memorising. Tests the macro stage alone; sub-E is not on this
   critical path.
3. **Macro→micro conditioning carries signal.** Samples under different
   macro-plan conditioning prefixes produce visibly different micro outputs.
   This is the hierarchical-AR thesis. **Sub-E is the load-bearing component
   for sub-bar 3:** if its per-cell boundary tokens do not vary observably with
   the macro plan, sub-bar 3 fails by construction.

Sub-E's job description follows: produce per-cell boundary tokens that (a) are
deterministic from sub-C and sub-D inputs, (b) vary observably with the macro
plan, and (c) ship in the calendar budget. Anything beyond (a–c) is post-reset
scope.

### 1.3 Scope locks

- Singapore-only implementation. No Sweden-specific execution branches.
- Sub-C and sub-D outputs are immutable evidence. Sub-E does not request a
  sub-C re-extraction (known issue #7 stays a stub).
- Sub-F (micro tokens) and the tokenizer `emit_unknown_token` fix (known
  issue #4) are out of scope. Each is its own sub-project with its own
  brainstorm-spec-test cycle. Sub-E surfaces them only as wallclock
  dependencies in §2.
- Exact boundary contracts (positions/widths/extents/source-feature
  traceability) are out of scope (§15 #9).
- Stitching as a validation sub-bar is out of scope (§15 #10).

## 2. Calendar and engineer-time budget

The binding constraint is calendar, not GPU hours. 14 days separate this
spec's date from the early-June Leonardo deadline. ~4200 GPU-hours remain
in the pre-deadline allocation; a single 4×A100 de-risk run consumes
~50–500 GPU-hours, so the budget is calendar-bound by ~10×.

### 2.1 Work breakdown for the de-risk launch

| Item | Engineer-days | Parallelisable | Notes |
|---|---|---|---|
| Sub-E (this spec) implementation | 2–3 | yes (subagent) | Thin layer over sub-D + sub-C crossings/features |
| Sub-F (micro tokens) | 2–3 | yes (subagent) | Wraps `src/cfm/tokenizer/encode.py::encode_cell` (247 LOC, commit `05b13a0`) |
| Tokenizer `emit_unknown_token` (known issue #4) | 1–2 | yes (subagent) | ~10-line core change with brainstorm-spec-test cycle |
| Training scaffold (model + dataloader + DDP + checkpoint + conditioning encoder) | **7–10** | limited | Long pole; no scaffold code exists in `src/cfm/` today |
| Eval harness (D forward-pass + within-bucket shuffle + scope-controlled gap + A renderer + sampling loop) | 3–4 | ships after launch | Blocks verdict, not run start |
| Design cycles (4 brainstorm + spec + plan rounds) | 3–4 | sequential | |
| **Raw serial total** | **18–26** | | |
| **With subagent parallelisation** | **~10–13** | | |

### 2.2 Calendar plan

- Days 1–4: sub-E spec lands (this doc); sub-F + tokenizer-fix specs land in parallel near the end.
- Days 4–10: subagent implementation of sub-E, sub-F, tokenizer-fix; training-scaffold core begins.
- Days 10–13: training-scaffold integration + smoke + Leonardo job submission.
- Day 14: launch; eval harness completes during the run.

### 2.3 Lever map

| Lever | Status | Trigger | Saves |
|---|---|---|---|
| 1. Eval harness during run, not before | **default-pulled** | always | 3–4 pre-launch days |
| 2. Skip Lightning, raw PyTorch DDP | **off** | only on hard Lightning/Leonardo incompatibility | net neutral; Lightning's DDP/optim/clip/logging eats back any savings |
| 3. Sub-E `boundary_class_enum` collapsed to `BOUNDARY_NOT_APPLICABLE` uniformly | **contingent** | **scaffold has not passed a one-batch end-to-end forward+backward smoke test on the 4-GPU node by day 9** | ~1 day on sub-E implementation; raises false-negative-verdict risk on the perplexity gap (§11) |
| 4. Skip tokenizer `emit_unknown_token` | **non-negotiable; OFF** | never | training crashes at runtime on first unknown class |

Lever 3's trigger is binary and observable. Day 9 leaves 5 days for scaffold
finish + integration + Leonardo launch. Passing the day-9 smoke is necessary
but not sufficient — the scaffold can still hit integration surprises on
days 10–13. Day-9 smoke pass means *lever 3 not triggered*, not *all clear*.

### 2.4 Worst-case cascade

Scaffold realistic estimate is 10 days; worst case (1–2 days first-launch
debug + 1–2 days integration surprises) is 12–14 days. Worst case eats the
entire 14-day budget alone. If it hits:

- Lever 1 already pulled.
- Lever 3 must pull (day-9 trigger).
- §15 #5 absorbs: generated-conditioned gap deferred to post-reset; only
  GT-conditioned gap measured pre-deadline. The hierarchical thesis becomes
  partially testable instead of fully testable; verdict status is provisional
  (§12).

This cascade is recorded now, not on day 13.

## 3. Consumer contract

The consumer is the micro generator at training time. Sub-E's contract to
that consumer is per-cell.

### 3.1 Per-cell view

For each cell `(cell_i, cell_j) ∈ [0, 8) × [0, 8)` in a tile, sub-E exposes
four boundary slots, one per cardinal direction in tile-local coordinates
(N/E/S/W). Each slot carries:

- `boundary_class_enum` — integer enum from `boundary_vocab.yaml` (§8);
  nullable on-disk, non-null iff `scope_marker == active`.
- `scope_marker` — integer enum from sub-D's locked `macro_plan_vocab.yaml`
  `scope` block (`active`, `fully_masked`, `scope_boundary`,
  `external_deferred`).
- `edge_id` — canonical tuple `(lower_cell_i, lower_cell_j, axis)` per sub-C's
  crossing convention. **Not** a sub-D lattice slot index. **Not** a per-cell-
  local index. The byte-identity invariant (§3.4) requires this canonical
  form.

### 3.2 Conditioning prefix order

The micro generator's input sequence per cell is, in order:

1. Tile conditioning vector (from sub-D `effective_conditioning.yaml`).
2. Macro plan tokens for this cell (zoning + cell_density from sub-D
   `macro_core.parquet`).
3. Boundary contract: 4 × `(boundary_class_token, scope_marker_token)` (sub-E).
4. Micro target tokens (from sub-F).

**This ordering is architectural, not cosmetic.** Boundary tokens follow macro
plan tokens because boundary is a *consequence* of the macro plan, not an
input to it. Under causal attention, this means boundary attends to macro,
macro does not attend to boundary. Sub-bar 3's perplexity gap measures whether
the micro stage uses boundary tokens that are downstream of macro — that
signal is only well-defined under this ordering. Reversing the order would
permit the macro stage to receive boundary leakage, contaminating both the
macro coherence test (sub-bar 2) and the conditioning signal test (sub-bar 3).

### 3.3 Vocab mapping and storage shape

Sub-E writes structured records with `boundary_class_enum` as an integer
referencing `boundary_vocab.yaml` (§8). It does **not** write
pre-tokenized model token IDs. The dataloader maps `boundary_class_enum`
to model token IDs at training time. This matches sub-D's vocab indirection
pattern verbatim and lets the vocab adjust (e.g., adding the §15 #1 NONE
split) without re-running sub-E.

### 3.4 Per-shared-edge byte-identity invariant

For any `edge_id = (i, j, axis)`, the value
`(boundary_class_enum, scope_marker)` returned by sub-E must be identical
regardless of which adjacent cell's view is querying. The invariant is
about the derivation function, not training consumption. Topic 9 (§11) tests
it as a regression guard; under §7's per-edge storage choice the invariant is
*structural* — there is only one record per `edge_id`, so violation is
impossible by construction.

## 4. Inputs sub-E reads and does not read

### 4.1 Reads

| Source | Field | Why |
|---|---|---|
| Sub-C `crossings.parquet` | edge_id `(lower_cell_i, lower_cell_j, axis)`, `source_feature_id` | Source of truth for internal-edge crossings; sub-D went here too |
| Sub-C `features.parquet` | `class_raw` (joined to crossings via `source_feature_id`) | Raw Overture class for class-precedence rule (§5) |
| Sub-D `manifest.yaml` | tile inventory, `_SUCCESS` gate, sub-C input pins | Inventory source of truth; gates sub-E start on sub-D completion |
| Sub-D `macro_core.parquet` | per-edge `scope`, per-cell `zoning_class` | Scope passes through to consumer; zoning informs precedence at zoning-boundary cases |

### 4.2 Does not read

| Source | Reason |
|---|---|
| Sub-D `derivation_evidence.parquet` | Would couple sub-E to sub-D's internal evidence schema (governed by `road_skeleton_derivation_version` and friends). Sub-E goes to the same primary source sub-D went to. |
| Sub-D `effective_conditioning.yaml` | Tile-level conditioning; consumer reads directly (§3.2 step 1). Not edge-level. |
| Sub-C `cells.parquet` | Per-cell evidence; sub-D already projects what sub-E needs into `macro_core.parquet`. |
| Sub-C `meta.yaml`, `provenance.yaml` | Provenance only; digests anchored via sub-D's manifest. |

### 4.3 Read rules

- `pyarrow.parquet.ParquetFile(path).read()` for per-tile parquet reads.
  Bare `pq.read_table()` on tile directories triggers Hive partition
  inference (sub-C/D lesson; project memory `feedback_pyarrow_hive_partition_inference`).
- Digest-anchor every sub-C and sub-D artifact sub-E consumes. If referenced
  input bytes change, sub-E validation fails rather than silently accepting
  drift.
- Treat sub-C geometry and sub-D macro core as immutable evidence. Sub-E
  writes no geometry and duplicates no input bytes.
- Sub-E reads sub-D's `_SUCCESS` first; sub-D's manifest is the tile inventory.
  Sub-E does not discover tiles by filesystem glob.

### 4.4 Sub-C re-extraction discipline

Sub-E does **not** trigger a sub-C re-run, does not invoke known issue #7's
`--rerun` stub, and does not depend on any sub-C field that is not in the
locked release `2026-04-15.0` extraction. If empirical sub-E results show
that `crossings.parquet + features.parquet` are insufficient for boundary
derivation quality, that is a §15 #11 deferral — a sub-C minor revision
request for the post-reset window, not a pre-deadline change.

## 5. Derivation function

Sub-E owns exactly one new derivation function: per-edge crossings +
joined feature `class_raw` + per-cell zoning + per-edge scope →
`boundary_class_enum` per active internal edge. Inactive edges (any
`scope_marker != active`) get `null` on-disk; sub-E does zero derivation for
them.

### 5.1 class_raw → boundary_class mapping (boundary_derivation_version v1.0)

`class_raw` is the raw Overture `transportation.class` string, populated by
`src/cfm/data/sub_c/pipeline.py:847` reading `table.column("class").to_pylist()`
directly with no transform. The mapping below is stated in `class_raw` values,
not in sub-B2's `R_*` token names; sub-E reads sub-C, not sub-B2.

```text
class_raw mapping → boundary_class:
  {"primary", "trunk", "secondary"}                            → MAJOR_ROAD
  {"tertiary", "residential", "service", "unclassified",
   "footway", "steps", "cycleway"}                             → MINOR_ROAD
  any other class_raw value (incl. null, Overture rare values
   such as "proposed"/"construction", or any value that
   sub-B2 would have tokenised as R_unknown)                   → MINOR_ROAD
  edge with no road crossings (any class_raw)                  → NONE
```

The default-bucket rule (everything-else → MINOR_ROAD) handles Overture rare
values conservatively — it demotes uncertain crossings rather than promoting
them under the hierarchy-wins tie-break (§5.2).

**Matching is strict byte-equality.** No case folding, whitespace stripping,
or Unicode normalisation is applied to `class_raw` before lookup. Sub-C stores
raw Overture strings unchanged at `pipeline.py:847`; whitespace or case
variants (e.g., `" primary "`, `"Primary"`) would indicate upstream data
corruption, not legitimate values, and fall through to the `MINOR_ROAD`
default bucket via the standard default-bucket rule. This convention is
locked under `boundary_derivation_version 1.0`; changing it (e.g., adding
`.casefold()` or `.strip()`) would require a derivation-version bump because
the same sub-C input would produce different `boundary_class_enum` outputs.
Strict matching is the choice that preserves cross-environment determinism
(§14) — case folding under Python's `str.casefold` is locale-independent but
some Unicode normalisations are not, and `.strip()` strips Unicode whitespace
characters whose interpretation has varied across Python releases.

Non-road crossings (water, rail) are ignored in v1. An edge with only water
or rail crossings becomes `NONE`. This is the §15 #1 deferral source:
post-reset, NONE will likely split into `NONE_INTERIOR` and
`NONE_NATURAL_BOUNDARY` (or new active classes for water/rail will appear).

### 5.2 Multi-crossing tie-break

When a single internal edge has multiple crossings (e.g., a primary road and
a residential road both cross), hierarchy-wins:

```text
MAJOR_ROAD > MINOR_ROAD > NONE
```

This matches sub-D's road-skeleton starting hypothesis and is the natural
tie-break under §8's small enum. Counts-aware variants violate §3.1's
4-slot-per-cell contract (would require multiple tokens per edge);
presence-only collapses under §8's vocab; class-group is what §8 already
does at the vocab level.

### 5.3 Function ownership

The derivation function is locked under `boundary_derivation_version` v1.0
(§9). Any change to the class-grouping map, the multi-crossing tie-break,
the default-bucket rule, scope handling, external-edge sentinel behaviour,
or non-road crossing handling requires a version bump.

## 6. External edges and scope-boundary handling

Sub-E emits five distinct cases over the lattice. Only the first row
exercises sub-E's derivation function; the other four pass sub-D scope
through unchanged.

| Edge type | Source | `boundary_class_enum` (on-disk) | `scope_marker` | Provenance |
|---|---|---|---|---|
| Active internal (active-to-active) | sub-C crossings + sub-D scope | derived per §5 (NONE / MAJOR_ROAD / MINOR_ROAD) | `active` (from sub-D) | sub-E derives boundary_class; scope passed through |
| Scope-boundary internal (active-to-masked) | sub-D scope | `null` | `scope_boundary` (from sub-D) | both fields determined by sub-D scope alone |
| Fully-masked internal (masked-to-masked) | sub-D scope | `null` | `fully_masked` (from sub-D) | both fields determined by sub-D scope alone |
| External, interior cell active | sub-D scope | `null` | `external_deferred` (from sub-D) | deterministic sentinel per sub-D §5 |
| External, interior cell masked | sub-D scope | `null` | `fully_masked` (from sub-D) | per sub-D §5 |

`boundary_class_enum` is non-null **iff** `scope_marker == active`. Sub-E does
no derivation for non-active edges. Lever 3's collapse (§2.3) is a config
flag that bypasses the active-row derivation function; under lever 3 the
entire `boundary_class_enum` column is uniformly null with no shape change.

### 6.1 Scope vocab inheritance

The four scope tokens (`active`, `fully_masked`, `scope_boundary`,
`external_deferred`) live in sub-D's already-locked
`configs/macro_plan/v1/macro_plan_vocab.yaml`
(sha256 `0b2d9eb4c1253b12f2fe50b32ec459e8fc25cdeafa05f4dda0a47240c0c9a1fd`,
committed at `6e411cf`). Sub-E references those token IDs by reference. **The
locked sub-D artifact is not modified.** Sub-E does not introduce a new scope
vocab artifact; only `boundary_vocab.yaml` (§8) is new.

### 6.2 Per-cell external-edge distribution

Drops out of sub-D's lattice; recorded for clarity, not a decision:

- Interior cells (36 of 64, `i,j ∈ [1, 6] × [1, 6]`): 4 internal slots.
- Edge cells (24 of 64, exactly one of `i` or `j` is 0 or 7): 3 internal + 1
  external slot.
- Corner cells (4 of 64, both `i` and `j` are 0 or 7): 2 internal + 2
  external slots.

Total slot count: 36 × 4 + 24 × 4 + 4 × 4 = 256 per-cell slots, of which
112 internal edges are shared (counted twice in the per-cell view; once each
on disk per §7) and 32 external edges appear once. Storage row count: 144
per tile.

All cells emit a fixed-shape 4-slot per-cell record at the consumer; corner
and edge cells have some slots that the storage layer marks as
`slot_kind == external_edge`. No special-casing in the per-cell record shape.

## 7. Storage shape

**Per-edge storage with read-time rotation into per-cell views.** 144 rows
per tile, sorted `(slot_kind, slot_index)` — matches sub-D's
`macro_core.parquet` row conventions verbatim.

### 7.1 Trade-off summary

The decisive argument is that per-edge storage makes the per-shared-edge
byte-identity invariant (§3.4) **structural rather than assertion-checked**.
Under per-edge, one record per `edge_id` means both adjacent cells' views
project from the same source; violation is impossible by construction.
Under per-cell denormalised storage, the writer must produce identical rows
for both views of every shared edge and the validator must verify it — small
writer bugs (precedence-rule path executed twice with divergent state,
non-deterministic dict iteration on tie-break inputs, etc.) could produce
divergent values caught only after the fact.

Secondary: per-edge is ~56% the on-disk size of per-cell denormalised, and
matches sub-D infrastructure (no validator pattern adapting).

### 7.2 `boundary_contract.parquet` schema

```text
slot_kind                int8     # 1=internal_edge, 2=external_edge (sub-D enum)
slot_index               int16    # canonical within slot_kind; sub-D's lattice ordering
lower_cell_i             int8     # populated for both internal and external edges
lower_cell_j             int8
axis                     int8     # sub-C AXIS enum (0=x, 1=y; cfm.data.sub_c.enums.AXIS)
scope_marker             int8     # 0=active, 1=fully_masked, 2=scope_boundary, 3=external_deferred
boundary_class_enum      int16?   # nullable; non-null iff scope_marker == 0 (active)
```

Canonical sort key: `(slot_kind, slot_index)`.

`slot_kind` wire values **must** match `cfm.data.sub_d.enums.SlotKind` byte-for-byte at `INTERNAL_EDGE=1` and `EXTERNAL_EDGE=2`. Sub-E's writer holds a local `SlotKind` IntEnum scoped to the writer module; the cross-enum byte-equivalence is enforced by inline validator invariant §10.1 #9 (Task-6 carry-forward) to catch silent drift if either side gains a member or has values reordered.

Row count invariants:

- Exactly 144 rows per tile.
- Exactly 112 with `slot_kind == 1` (internal_edge).
- Exactly 32 with `slot_kind == 2` (external_edge).
- `slot_index ∈ [0, 112)` for internal edges; `[0, 32)` for external edges.

### 7.3 Dataloader rotation

The per-cell view at training time is a deterministic 4-edge lookup keyed
on `(cell_i, cell_j)`. The rotation function maps each cell to its N/E/S/W
`edge_id`s using sub-C's AXIS enum (0=x, 1=y). The same rotation map applies
to every tile in a region, so the table is built once at dataloader init and
applied at batch time — no per-batch derivation cost.

External-edge handling drops out of the rotation: the per-cell view derives
the owning interior cell from `(lower_cell_i, lower_cell_j)` directly. No
"is this external?" conditional in the storage layer; the per-cell view
naturally produces externals on the boundary cells without special-casing.

## 8. Vocab artifacts

### 8.1 `configs/macro_plan/v1/boundary_vocab.yaml` (NEW)

Sub-E's vocab is a new sibling artifact under `configs/macro_plan/v1/`. **The
locked `macro_plan_vocab.yaml` (sub-D) is not modified.** Draft shape (`sha256`
placeholder lands at write time; vocab content below is locked):

```yaml
boundary_vocab_schema_version: "1.0"
boundary_vocab_version: "1.0"
boundary_derivation_version: "1.0"
phase: 1
generated_from:
  overture_release: "2026-04-15.0"
  regions: ["singapore"]
  empirical_gate:
    layer3_subset_sha256: "<sha256 of the locked Layer-3 subset spec>"

scope_vocab_inherited_from:
  artifact: "configs/macro_plan/v1/macro_plan_vocab.yaml"
  artifact_sha256: "0b2d9eb4c1253b12f2fe50b32ec459e8fc25cdeafa05f4dda0a47240c0c9a1fd"
  block: "scope.tokens"

tokens:
  - {id: 0, name: BOUNDARY_NOT_APPLICABLE}   # sentinel; dataloader-side
  - {id: 1, name: NONE}                       # active edge, no road crossings
  - {id: 2, name: MAJOR_ROAD}                 # active edge, MAJOR class wins
  - {id: 3, name: MINOR_ROAD}                 # active edge, MINOR class wins

append_only_within_phase: true

class_grouping_map:
  MAJOR_ROAD: ["primary", "trunk", "secondary"]
  MINOR_ROAD: ["tertiary", "residential", "service", "unclassified",
               "footway", "steps", "cycleway"]
  # default: any other class_raw → MINOR_ROAD
  # non-road crossings ignored in v1
```

### 8.2 `BOUNDARY_NOT_APPLICABLE` sentinel

On-disk `boundary_class_enum` is `null` for non-active edges. At the model
input, the dataloader maps `null → BOUNDARY_NOT_APPLICABLE` (token id 0).
The model always sees a fixed-length 4-slot boundary conditioning per cell
with a deterministic token at every position.

This split between on-disk derivation output (null where no value) and
model-side conditioning shape (fixed-length sentinel) is intentional. Future
sentinel refinements (e.g., distinguishing `MASKED` from `EXTERNAL` at the
token level) are vocab-level changes, not sub-E rederivations. Lever-3
collapse (uniformly null `boundary_class_enum`) is similarly a config flag,
not a writer-shape change.

### 8.3 Scope vocab posture

Sub-E does **not** define a scope vocab. The four scope tokens (`active=0`,
`fully_masked=1`, `scope_boundary=2`, `external_deferred=3`) come from sub-D's
locked `macro_plan_vocab.yaml::scope.tokens` and are referenced by token ID.
At the model conditioning layer, each per-cell boundary slot is the pair
`(scope_marker_token_from_sub-D, boundary_class_token_from_sub-E)`.

## 9. Determinism, versioning, provenance

Sub-E inherits sub-D's infrastructure wholesale. Shared neutral helpers
already extracted in `src/cfm/data/io.py` and `src/cfm/data/determinism.py`
(sub-D Task 1) are reused; sub-E adds no new mechanism, only sub-E-local
entries.

### 9.1 Three version axes

| Axis | Bump iff |
|---|---|
| `sub_e_schema_version` | On-disk parquet schema or YAML structure changes. Independent from vocab/derivation per sub-D known_issue #8 lesson. |
| `boundary_vocab_version` | Adding a token (append-only within phase). Splitting `NONE` into `NONE_INTERIOR + NONE_NATURAL_BOUNDARY` (§15 #1) requires a bump. |
| `boundary_derivation_version` | The same sub-C + sub-D input would produce different `boundary_class_enum` labels. Includes class-grouping map changes, multi-crossing tie-break changes, default-bucket changes, scope-handling logic changes, external-edge sentinel changes, non-road crossing handling changes. |

All three are separate namespaces. The
`compare_version(namespace, expected, actual)` helper from sub-D rejects
cross-namespace comparisons; sub-E validators (§10) must use it rather than
ad-hoc string comparisons.

### 9.2 Digest chain

```text
_SUCCESS
  → manifest.tiles[*].provenance_sha256
  → provenance.outputs.boundary_contract_parquet_sha256
  → file bytes
```

Same shape as sub-D's. The inherited `cfm.data.determinism` exclusion-table
grammar applies: final-segment `*_sha256` field matching, file-keyed
timestamp exclusions for `extracted_utc` and `started_utc/completed_utc`.
Sub-E's `EXCLUDED_FROM_SHA` entries are sub-E-artifact-specific but use the
inherited mechanism.

### 9.3 Determinism mode

Same-process byte-identity required (verifiable locally on darwin/aarch64).
Cross-environment (darwin/aarch64 vs Leonardo linux/x86_64) is the same
residual risk as sub-D; sub-E inherits it. Verification trigger: first sub-E
run on Leonardo. See §14.

### 9.4 Provenance

Per-tile `provenance.yaml` shape (concrete values below are illustrative;
shas and timestamps land at write time):

```yaml
provenance_schema_version: "1.0"
tile_i: 12
tile_j: 17

extraction:
  commit_sha: "<40-char sha>"
  extracted_utc: "2026-05-21T12:00:00Z"
  rerun_count: 0
  rerun_reason: "initial"

inputs:
  release: "2026-04-15.0"
  sub_c_manifest_sha256: "<sha>"
  sub_c_features_parquet_sha256: "<sha>"
  sub_c_crossings_parquet_sha256: "<sha>"
  sub_d_manifest_sha256: "<sha>"
  sub_d_macro_core_parquet_sha256: "<sha>"
  boundary_vocab_sha256: "<sha>"
  derivation_config_sha256: "<sha>"

versions:
  sub_e_schema_version: "1.0"
  boundary_vocab_version: "1.0"
  boundary_derivation_version: "1.0"

outputs:
  boundary_contract_parquet_sha256: "<sha>"
```

### 9.5 Region manifest

Per-region `manifest.yaml` shape (concrete values below are illustrative
except for `tile_count`, which is the actual Singapore count sub-E inherits
from sub-D's locked extraction; shas and timestamps land at write time):

```yaml
manifest_schema_version: "1.0"
sub_e_schema_version: "1.0"
release: "2026-04-15.0"
region: "singapore"
region_crs: "EPSG:3414"

inputs:
  sub_c_manifest_sha256: "<sha>"
  sub_c_region_dir: "data/processed/sub_c/2026-04-15.0/singapore"
  sub_d_manifest_sha256: "<sha>"
  sub_d_region_dir: "data/processed/sub_d/2026-04-15.0/singapore"
  boundary_vocab_sha256: "<sha>"

versions:
  boundary_vocab_version: "1.0"
  boundary_derivation_version: "1.0"

config_source: "sub_d_manifest.config"
config:
  cell_grid: [8, 8]
  internal_edge_count: 112
  external_edge_count: 32

initial_extraction:
  commit_sha: "<40-char sha>"
  started_utc: "2026-05-21T12:00:00Z"
  completed_utc: "2026-05-21T12:05:00Z"
  tile_count: 494

tiles:
  - tile_i: 12
    tile_j: 17
    provenance_sha256: "<sha>"
```

Tiles sorted by `(tile_i, tile_j)`. The `config` block is copied from sub-D's
manifest for consumer convenience; the sub-E validator asserts these values
match the referenced sub-D manifest.

## 10. Validator invariants

Sub-D's `validator_inline.py` + `validator_cross_tile.py` patterns apply with
sub-E-local entries below. No new validator infrastructure.

### 10.1 Inline (per-tile)

| # | Invariant |
|---|---|
| 1 | Exactly 144 rows: 112 with `slot_kind == 1` (internal_edge), 32 with `slot_kind == 2` (external_edge) |
| 2 | Rows sorted by `(slot_kind, slot_index)` |
| 3 | `boundary_class_enum` non-null iff `scope_marker == 0` (active) |
| 4 | When non-null, `boundary_class_enum ∈ {NONE=1, MAJOR_ROAD=2, MINOR_ROAD=3}` — sentinel `BOUNDARY_NOT_APPLICABLE=0` is dataloader-side only, never on-disk |
| 5 | `scope_marker ∈ {0, 1, 2, 3}` per sub-D's locked scope vocab |
| 6 | `slot_index` range: internal ∈ [0, 112); external ∈ [0, 32) |
| 7 | `axis ∈ {0, 1}` per `cfm.data.sub_c.enums.AXIS` |
| 8 | Provenance `boundary_derivation_version` matches manifest |
| 9 | **(Task-6 carry-forward, sub-E-local — not in original 8.)** Sub-E writer's `SlotKind` enum integer values match `cfm.data.sub_d.enums.SlotKind` byte-for-byte at `INTERNAL_EDGE=1` and `EXTERNAL_EDGE=2`. Two separate `IntEnum` classes maintain wire compatibility manually; this invariant catches silent drift if either side gains a member or reorders values. |

### 10.2 Cross-tile (per-region)

| # | Invariant |
|---|---|
| 1 | `sub_e_schema_version` consistent across tiles. Like-for-like only (sub-D known_issue #8 lesson: do not over-couple to vocab/derivation versions). |
| 2 | `boundary_vocab_version` and `boundary_derivation_version` identical across all tiles in the region |
| 3 | Digest chain `_SUCCESS → manifest.tiles[*].provenance_sha256 → provenance.outputs.boundary_contract_parquet_sha256 → file bytes` valid for every tile |
| 4 | Sub-C and sub-D input digests (referenced in each tile's provenance) match across all tiles in the region |
| 5 | External-edge consistency: each external `slot_index` appears in exactly one cell's per-cell view (the §6.2 vacuous-invariant lifted to a real-data regression check) |

### 10.3 Validator boundaries

Sub-E validators own:

- Boundary-contract schema and lattice completeness.
- Derivation-version provenance and `compare_version` namespace enforcement.
- Digest chain and sub-C + sub-D input anchors.
- Determinism on rerun (same-process byte-identity).
- Consistency against pinned sub-C and sub-D evidence.

Sub-E validators do **not** own:

- Macro-core schema or sub-D's lattice validity (sub-D's validator territory).
- Sub-C input correctness (sub-C's territory).
- Micro token decodability (sub-F).
- Tokenizer `emit_unknown_token` behaviour (known issue #4).
- Stitching consistency (post-reset; §15 #10).

If real cached Singapore data violates a planned invariant, the invariant is
not weakened to pass. Memory `feedback_test_weakening_to_pass` applies:
when data violates an assumption, the assumption failed; fix the upstream or
escalate, do not modify the assertion.

## 11. Validation strategy

Three-layer test stratification per sub-D's pattern. Total 7 + 8 + 4 = 19
tests, plus the perplexity-gap evaluation harness.

### 11.1 Layer 1 — pure unit tests (synthetic, no Singapore dependency)

| # | Test |
|---|---|
| 1 | Lattice indexing: 144 rows split as 112 internal + 32 external; `slot_index` ranges per §10.1 #1, #6 |
| 2 | Class precedence rule: hierarchy-wins on constructed multi-crossing inputs (MAJOR + MINOR → MAJOR; MINOR + NONE → MINOR; etc.) |
| 3 | Class-grouping map: each `class_raw` listed in §5.1 maps to expected `boundary_class`; default bucket catches unlisted/null values |
| 4 | Scope-class consistency rule: `boundary_class_enum` non-null iff `scope_marker == active` |
| 5 | Edge-id canonical convention: `(lower_cell_i, lower_cell_j, axis)` produced deterministically from inputs |
| 6 | Per-cell rotation function: for cell `(i, j)`, the 4 returned edge_ids match expectation across interior, edge, and corner positions |
| 7 | `compare_version(namespace, ...)` rejects cross-namespace comparisons (sub-D meta-test inherited) |

Tiny synthetic tables. No cached-Singapore dependency. Each test runs in
under one second.

### 11.2 Layer 2 — artifact + validator tests (synthetic sub-C/sub-D-like fixtures)

| # | Test |
|---|---|
| 1 | `boundary_contract.parquet` write/read schema: column names, dtypes, nullability of `boundary_class_enum` |
| 2 | Canonical sort key `(slot_kind, slot_index)` enforced |
| 3 | All 8 inline invariants exercised with controlled-violation fixtures; each invariant has at least one failing-fixture test asserting the validator rejects it |
| 4 | All 5 cross-tile invariants exercised with controlled-violation fixtures |
| 5 | Digest chain: mutating any link fails validation |
| 6 | Sub-C input digest mismatch fails validation; sub-D input digest mismatch fails validation |
| 7 | Same-process determinism: run sub-E twice on identical synthetic inputs, assert byte-identical `boundary_contract.parquet` excluding declared timestamp fields |
| 8 | Per-shared-edge byte-identity regression guard: query both adjacent cells' per-cell views for every internal edge and assert byte-identical. Structural under §7's per-edge storage; trivially satisfied; held as guard in case storage shape ever changes. |

### 11.3 Layer 3 — cached Singapore integration

| # | Test |
|---|---|
| 1 | **Empirical gate (§5):** on the Layer-3 9-tile subset, compute active-edge `boundary_class` distribution; ship iff *no active class above 90% AND no active class below 2%*. Either violation reopens §5. |
| 2 | End-to-end real sidecar run on Layer-3 subset; all Layer 2 invariants hold on real data |
| 3 | Deterministic rerun on real data: same inputs → byte-identical outputs (same-process; cross-env deferred per §14) |
| 4 | External-edge single-cell membership on real data: each external `slot_index` appears in exactly one cell's per-cell view |

The Layer-3 subset is sub-D's 9-tile curated diverse subset, consumed by sub-E
verbatim from the locked `selected_layer3_tiles` field in
`configs/macro_plan/v1/macro_plan_vocab.yaml`. The curation carries the
sub-D known_issue #11 footnote (sparse-side dimension scoring missed three
diversity dimensions in the curation); this is documented but is not a sub-E
blocker.

### 11.4 Perplexity gap evaluation harness (sub-bar 3 measurement)

Ships during the de-risk training run (lever 1 default-pulled, §2.3).

**Training corpus.** Layer-3 9-tile subset (~576 active cells after sub-D
scope filtering).

**Held-out eval pool.** Random sample of ~50 Singapore tiles from outside
the Layer-3 subset → ~3,000 active cells → ~30,000–100,000 micro tokens. The
held-out pool comes from the same region, same Overture release, same sub-C
extraction; statistically comparable to the training subset for gap
measurement.

**Eyeball sanity check (sub-bar A).** 20–30 sample cells across 3–5 distinct
macro-plan-conditioning prefixes, end-of-run only. Requires the sub-F decoder
+ a small renderer.

**Shuffle strategies (Q1a Caveat 1).**

- **Primary: within-conditioning-bucket shuffle.** Substitute macro plan from
  a tile with matching tile-level conditioning bucket (country, morphology,
  era). Controls for surface incompatibility while still varying macro-plan
  content.
- **Secondary sanity check: cross-tile shuffle.** Substitute macro plan from
  a random other tile. Higher-variance gap; expected to be larger than
  within-bucket; serves as a sanity check on the eval harness itself.
- **Deferred to post-reset: position-shuffled within the same plan.** §15 #2.

**Scope-controlled gap (Q1a Caveat 2).** Sub-D's locked scope vocab directly
affects what the gap measures for sub-E. If the shuffle pairs tiles with
different scope-marker layouts, the gap can be driven entirely by scope
marker differences (active-here vs masked-here) rather than by `boundary_class`
content. The eval harness must report the *scope-controlled* gap: shuffle
only among tiles with matching scope-marker layouts, or subtract a control
measurement that quantifies the scope-marker contribution. The reported gap
is the content signal, not the shape signal.

**Generated-conditioned vs ground-truth-conditioned gap (Q1a Caveat 3).**
Both are measured.

- GT-conditioned: NLL on held-out micro tokens conditioned on the
  ground-truth sub-D macro plan for that tile (matched) vs a shuffled macro
  plan (mismatched). Tests whether micro stage *would* use macro conditioning
  if macro produced perfect output.
- Generated-conditioned: NLL on held-out micro tokens conditioned on macro
  plans *sampled* from the trained macro stage (matched) vs shuffled. Tests
  whether macro's actual outputs are useful to micro — the full hierarchical
  thesis.

A small generated-conditioned gap is itself informative: it indicates macro
isn't producing useful conditioning at de-risk scale, suggesting "scale up
before declaring failure" rather than "architecture broken."

### 11.5 Quantitative thresholds

Pre-committed numbers below are best-effort estimates. They are not derived
from a calibration baseline (training an unconditional baseline would double
the de-risk GPU cost; not in budget). The PoC's prior on transfer or gap
magnitude, if available, overrides.

| Threshold | Pre-committed value | Statistical test |
|---|---|---|
| **GT-gap > 0** (primary signal) | gap_gt ≥ **0.05 nats/token** on held-out eval | Per-cell sign test on (NLL_shuffled − NLL_matched); fraction of cells with positive gap statistically significant at p < 0.01 |
| **Generated-gap ≈ 0** (Q1a Caveat 3 disambiguator) | gap_gen < **0.02 nats/token** OR not statistically distinguishable from zero (sign test p > 0.05) | Same per-cell sign test on generated-macro-conditioning |
| **Contingent C-run trigger** (§15 #4) | GT-gap clears threshold AND generated-gap fails to clear | Both conditions checked at end of de-risk training |

Rationale: 0.05 nats/token ≈ a ~5% per-token perplexity reduction, well above
noise on a ~3,000-cell held-out at per-cell granularity. 0.02 nats/token is
the lower bound where "no effect" becomes defensible. Both numbers are
uncalibrated to this architecture and may need adjustment after the first
non-trivial checkpoint shows what gap range is achievable.

**Live revision rule.** If the first non-trivial checkpoint shows GT-gap
consistently 10× either threshold (≥ 0.5 nats/token or ≤ 0.002 nats/token),
recalibrate before the verdict. Otherwise hold.

## 12. Lever-3 collapse path

Lever 3 (§2.3) bypasses the active-row derivation function. The sub-E
implementation must support a config flag (working name `lever_3_collapse`)
that, when set:

- Skips the `class_raw → boundary_class` mapping (§5.1) entirely.
- Writes `boundary_class_enum = null` for every row, regardless of
  `scope_marker`.
- Preserves on-disk schema, sort key, scope marker passthrough, digest
  chain, manifest, and provenance — all unchanged.

The flag is set at sub-E invocation time; sub-E does **not** auto-detect or
auto-pull. The day-9 smoke-test trigger lives in the calendar plan
(§2.3), not in sub-E's code.

### 12.1 Test variant

Layer 1 / 2 / 3 tests have a parameterised lever-3-collapse variant:

- **Layer 1:** tests 2 and 3 (class precedence + class-grouping map) skipped
  as vacuous; tests 1, 4, 5, 6, 7 still apply.
- **Layer 2:** invariants #3 (non-null iff active) and #4 (active class
  membership in `{NONE, MAJOR_ROAD, MINOR_ROAD}`) are replaced by a single
  uniform-null check (every row's `boundary_class_enum` is `null`); other
  invariants still apply, including #9 (sub-E ↔ sub-D `SlotKind` byte-equivalence —
  this carry-forward is mode-independent). The inline validator takes a
  `lever_3_collapse: bool` kwarg to switch between modes.
- **Layer 3:** test 1 (empirical gate) skipped as vacuous; tests 2, 3, 4 still
  apply.

### 12.2 Verdict interpretation

Under lever 3, sub-E's `boundary_class_enum` carries no signal. The
perplexity gap (§11.4) is expected to be near floor — a low gap does **not**
indicate "architecture broken" under lever 3 the way it would under full
sub-E. A lever-3-collapse de-risk run produces a **provisional verdict**: it
demonstrates that the training scaffold + macro stage + sub-F + tokenizer
work end-to-end and that the model trains coherently, but does not test the
hierarchical-AR thesis.

If lever 3 fires pre-deadline, §15 #8 absorbs: a non-collapse re-run in the
post-reset 5,000-hour window is required before declaring a final
architecture-feasibility verdict.

## 13. Output directory layout

```text
data/processed/sub_e/<release>/<region>/
  manifest.yaml
  _SUCCESS
  tile=EPSG3414_i<tile_i>_j<tile_j>/
    boundary_contract.parquet
    provenance.yaml
```

Tile directory naming follows sub-C and sub-D conventions for readability.
The sub-E manifest is the authoritative tile inventory; consumers do not
filesystem-glob.

**Deliberately absent vs sub-D.** Sub-E does not ship:

- `derivation_evidence.parquet` — §5's empirical gate is a one-shot frequency
  check at implementation time (§11.3 #1), not a Gate-2-style empirical-lock
  cycle. There is no per-tile evidence to ship.
- `effective_conditioning.yaml` — sub-D owns it. Sub-E does not touch
  tile-level conditioning.

The leaner output surface reflects sub-E's narrower scope. A future sub-E v2
(§15 #9) with full exact boundary contracts may reintroduce a derivation
evidence parquet for boundary-quality auditing.

## 14. Cross-environment determinism posture

Sub-E inherits sub-D's residual: same-process byte-identity is required
(verified at Layer 2 and Layer 3 on dev hardware, darwin/aarch64);
cross-environment byte-identity (darwin/aarch64 vs Leonardo linux/x86_64) is
unverified until sub-D's
`test_cross_environment_determinism_gap_is_documented_if_not_run` sentinel
test runs on Leonardo.

Verification trigger for sub-E: first sub-E run on Leonardo. The Leonardo
output's sha256 is compared against the local output's sha256 on the same
tile inputs. Mismatch logged as a new entry in `docs/known_issues.md` (with
root-cause investigation queued for the post-reset window). Match closes the
residual.

This is a §15 #7 deferral with a concrete trigger, not a vague "we should
test this someday."

## 15. Deferrals and post-reset roadmap

Sub-E opens **zero** new entries in `docs/known_issues.md` at spec-write
time. The 12 deferrals below live in this section, analogous to sub-D §15.
The project-wide `docs/known_issues.md` is reserved for accepted defects in
shipped artifacts; sub-E's deferrals are non-goals and planned future work.

If sub-E implementation surfaces a real defect (analogous to sub-D's #9
surfacing at Gate 2B), it gets filed in `docs/known_issues.md` at that time.

### 15.1 Deferral ledger

| # | Deferral | Source topic | Re-open trigger | Required cost on re-open |
|---|---|---|---|---|
| 1 | Distinguish active-no-crossings (interior NONE) from active-only-non-road-crossings (water/rail/coastline NONE). Probable v2: `NONE_INTERIOR` vs `NONE_NATURAL_BOUNDARY`, or new active tokens for water/rail. | §5 | Post-reset boundary-quality cycle, or empirical evidence that v1 NONE contaminates the perplexity gap | `boundary_vocab_version` + `boundary_derivation_version` bump; new class-grouping map; re-derive sub-E for all tiles |
| 2 | Position-shuffled perplexity gap (Q1a Caveat 1 third shuffle strategy). | §11.4 | Post-reset; structural test of macro-plan content vs alignment after v1 gap is calibrated | Eval-harness change only; no sub-E rederivation |
| 3 | Second-order conditioning analysis: joint macro+boundary attribution. Does the model use them independently or jointly? | §11.4 | Post-architecture-iteration; quality cycle | Eval-harness change only |
| 4 | Full-Singapore (~494 tile) training run, contingent. | §1.2 sub-bar 3 disambiguator | GT-gap clears AND generated-gap fails to clear (§11.5) | Sub-E + sub-F + tokenizer re-run on full Singapore (linear scale-up, ~hours not days); full-Singapore training run in post-reset 5,000-hr window |
| 5 | Generated-conditioned gap, contingent on calendar cascade. | §2.4 worst-case cascade | Scaffold worst case (12–14 days) hits; only GT-gap measured pre-deadline | Eval-harness re-run on saved checkpoints + macro sampling code path; no sub-E rederivation |
| 6 | Per-cell denormalised storage as alternative shape. | §7 | Layer 2 regression guard #8 ever fails | Schema migration + validator rewrite; significant. No current indication this is needed. |
| 7 | Cross-environment determinism verification (darwin/aarch64 vs Leonardo linux/x86_64). | §14 | First sub-E run on Leonardo | One-tile re-run on Leonardo; hash compared against local; mismatch → log new known_issue |
| 8 | Lever-3-collapse non-collapse re-run. | §12 | Lever 3 fires pre-deadline → de-risk verdict from that run is **provisional** | Full sub-E re-run with active-class derivation enabled in post-reset window; re-measure perplexity gap |
| 9 | Exact boundary contracts per PRD §5 stage 3: crossing positions, width/extent class, source-feature traceability. | §1.3 scope locks | Post-reset boundary-quality cycle; likely "sub-E v2" as its own sub-project | `boundary_derivation_version` + `sub_e_schema_version` bump; significant schema expansion; new derivation function; new evidence parquet; possibly sub-C minor revision for source-feature ID preservation |
| 10 | Stitching test as a 4th sub-bar. | §1.3 scope locks | Post-reset | Token-to-geometry decoder + geometric edge-alignment check + adjacent-cell generation protocol; same engineering as §15 #9 — likely combined sub-project |
| 11 | Sub-C minor revision request, contingent. | §4.4 | Empirical evidence post-implementation that v1 sub-E cannot clear the gap on `crossings.parquet + features.parquet` alone | Sub-C re-extraction (known issue #7 `--rerun` CLI must be implemented first); sub-E re-derive |
| 12 | Cell-density-ratio > 1.0 root cause (sub-D known_issue #9 pointer). | §11.2 (no sub-E dependency) | When any sub-project asserts ratio ≤ 1.0 elsewhere | Sub-C minor revision; out of sub-E scope |

### 15.2 Inherited residuals from sub-D

Sub-E inherits two sub-D known_issues as spec footnotes without re-opening:

- **#11 (Layer-3 sparse-side scoring):** Layer-3 subset is sub-E's training
  corpus. Curation footnote carried; not a sub-E blocker.
- **#9 (cell density ratio > 1.0):** sub-E does not touch cell density.
  Sub-D's locked top bucket `[0.35, ∞)` absorbs the anomaly. No propagation.

### 15.3 Post-reset triage (5,000-hour window)

Ordering reflects load-bearingness for declaring a final architecture
verdict, not engineering effort:

| Priority | Deferral # | Rationale |
|---|---|---|
| P0 — verdict-completing | 5, 7, 8 | Generated-conditioned gap (if cascaded), cross-env determinism check, lever-3 non-collapse re-run (if triggered). All required to convert provisional → final verdict. Cheap. |
| P1 — architecture iteration | 4 | Full-Singapore C run if generated-gap escape hatch fires. Larger but contingent. |
| P2 — boundary quality v2 | 1, 9, 10, 11 | NONE split, exact boundary contracts, stitching, sub-C minor revision request. Likely a combined "sub-E v2" sub-project. |
| P3 — validation methodology | 2, 3 | Position-shuffled and second-order conditioning analyses. Quality cycle after v2. |
| P4 — speculative | 6, 12 | Per-cell storage migration (only if Layer 2 guard fails); cell-density root cause (sub-C scope). Not on critical path. |

The 5,000-hour post-reset budget is dominated by architecture iteration
(bake-off, scaling-curve work) after the de-risk verdict, not data-pipeline
catch-up. P0/P1 consume small fractions; P2 is a real chunk if pursued;
P3/P4 are tail.

## 16. Pointers and spec self-review

### 16.1 Pointers

Read these directly; do not paraphrase.

- PRD: `PRD.md`
- Sub-D design spec: `docs/superpowers/specs/2026-05-19-phase-1-sub-D-macro-plan-derivation-design.md`
- Sub-D end-of-implementation handoff: `docs/handoffs/2026-05-19-end-of-sub-D.md`
- Sub-D tension-flag notes: `docs/superpowers/notes/2026-05-19-sub-D-phase-B-tension-flags.md`
- Sub-C design spec: `docs/superpowers/specs/2026-05-17-phase-1-sub-C-tile-extraction-design.md`
- Sub-B2 vocab spec: `docs/superpowers/specs/2026-05-16-phase-1-sub-B2-vocab-yaml-design.md`
- Known issues: `docs/known_issues.md`
- Phase 0 tokenizer encoder reference: `src/cfm/tokenizer/encode.py::encode_cell` (247 LOC, last touched commit `05b13a0`)
- Sub-C AXIS enum: `src/cfm/data/sub_c/enums.py::AXIS` (`{0: "x", 1: "y"}`)

### 16.2 Locked artifact pin

- `configs/macro_plan/v1/macro_plan_vocab.yaml`
  sha256: `0b2d9eb4c1253b12f2fe50b32ec459e8fc25cdeafa05f4dda0a47240c0c9a1fd`
  committed at: `6e411cf` (sub-D Task 8, "data(sub_d): lock macro plan vocab v1")

Sub-E reads this file. Sub-E **does not modify it**. Any sha change to this
file is a violation of sub-D's append-only-within-phase contract and must be
prevented by code review.

### 16.3 Spec self-review checklist

- **Placeholder scan:** no `TBD` or unfinished sections.
- **Version namespace consistency:** `sub_e_schema_version`,
  `boundary_vocab_version`, `boundary_derivation_version` separate;
  `compare_version` rejects cross-namespace comparisons.
- **Scope leakage check:** no sub-F, tokenizer, sub-C re-extraction, Sweden,
  or exact-boundary-contract work hidden inside sub-E.
- **Consumer read-pattern coverage:** micro generator at training time has
  an explicit read path; sub-E does not assume sub-G or downstream consumers.
- **Schema/gate consistency:** vocab-dependent schema fields are present;
  empirical gate (§11.3 #1) is one-shot at implementation time, not a Gate-2
  cycle.
- **No sub-D mutation:** sidecar-only output boundary maintained; sub-D
  artifact sha at §16.2 is pinned and must not change.
- **No geometry duplication:** sub-E writes tokenizable macro targets only,
  no geometry.
- **Lever-3 path:** every section that mentions sub-E content explicitly
  references the lever-3 collapse behaviour (§5, §6, §8, §11, §12).
- **De-risk framing:** every design decision traceable to the descope-by-
  default posture (§1.1) and the three sub-bars (§1.2).

### 16.4 Composite at-a-glance

For paste-in to implementation handoffs and plan documents:

- **De-risk goal:** 3 sub-bars (loss monotonic, macro coherent on held-out,
  macro→micro carries signal); sub-bar 3 measured via D + A (conditional
  perplexity gap + eyeball sample).
- **Calendar:** 12 + 2 days; lever-3 trigger = scaffold one-batch end-to-end
  smoke not passing by day 9; lever 1 (eval harness during run) pulled by
  default.
- **Consumer contract:** per-cell view, 4 boundary slots per cell,
  `(boundary_class_enum, scope_marker, edge_id)`; conditioning prefix order
  tile → macro → boundary → micro target.
- **Inputs:** sub-C `crossings.parquet + features.parquet` + sub-D
  `macro_core.parquet`. One new derivation function.
- **Vocab:** new `boundary_vocab.yaml`, 4 tokens
  (`BOUNDARY_NOT_APPLICABLE`, `NONE`, `MAJOR_ROAD`, `MINOR_ROAD`);
  class-grouping map locked at v1.0; hierarchy-wins multi-crossing tie-break.
- **External + scope-boundary:** scope marker passes through;
  `boundary_class_enum` null off-active.
- **Storage:** per-edge; byte-identity structural; 144 rows/tile sorted
  `(slot_kind, slot_index)`.
- **Versioning:** 3 axes (schema, vocab, derivation); `compare_version`
  enforces namespaces; sub-D infrastructure reused.
- **Tests:** 7 Layer 1 + 8 Layer 2 + 4 Layer 3; perplexity gap thresholds
  GT-gap ≥ 0.05, generated-gap < 0.02 nats/token, sign test p < 0.01 /
  p > 0.05.
- **Deferrals:** 12 items, sub-E spec §15 — no new entries in
  `docs/known_issues.md`.

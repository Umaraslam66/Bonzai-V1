# Phase 1 Sub-D Macro Plan Derivation Design

Date: 2026-05-19

Status: design spec for review

## 1. Purpose And Scope

Phase 1 sub-D derives the macro-core sidecar dataset for Singapore. It consumes
the immutable sub-C tile-extraction output and writes a new sub-D dataset under
its own manifest, provenance, validation gate, and `_SUCCESS` marker.

The macro core is a supervised training target for the macro autoregressive
planner. At inference time, that macro planner predicts coarse city structure;
micro generation and later stitching consume that structure. Therefore sub-D's
targets are designed as tokenizable prediction vocabularies, not just analytic
summaries.

Sub-D owns four obligations:

1. Derive one zoning target per active cell slot.
2. Derive one density target per active cell slot.
3. Fill the tile-level `population_density_bucket` conditioning field that
   sub-C leaves as `null` with `population_density_bucket_owner: sub-D`.
4. Derive a coarse road-skeleton target over the fixed edge lattice.

This spec is organized by concern rather than by brainstorm topic. The body
describes what sub-D consumes, computes, writes, validates, and defers. The
derivation sections precede artifact schemas because the schema fields are the
serialization of versioned derivation choices; readers should understand what
is computed before reading where it is stored. Section 2 preserves brainstorm
traceability.

Scope locks:

- Singapore-only implementation. Schema and metadata choices may include
  near-zero-cost multi-region affordances, such as region fields, but no
  Sweden-specific execution branches are in scope.
- Tokenizer `emit_unknown_token` fall-through is out of scope.
- Overture cold-fetch performance is out of scope.
- Sub-E, sub-F, and sub-G are consumers to design for, not collaborators whose
  absent specs block sub-D's contract.

## 2. Locked Topic Summary

| Topic | Locked decision | Section |
|---|---|---|
| 0. Scope lock | Singapore-only implementation; tokenizer/cold-fetch out of scope; sub-E/F/G consume sub-D's contract. | Section 1, Section 15 |
| 1. Macro plan definition | Sub-D owns macro-core training targets: zoning, per-cell density, tile-level population-density conditioning, and coarse road skeleton. | Section 1, Sections 7-10 |
| 2. Output contract | Sub-D writes an immutable sidecar dataset with effective conditioning and no sub-C mutation or geometry duplication. | Section 4, Section 11 |
| 3. Granularity | Fixed 64-cell, 112-internal-edge, and 32-external-edge lattices with explicit scope markers. | Section 5 |
| 4. Zoning | Zoning enum is frequency-derived from deterministic evidence metrics; scope is separate from zoning. | Section 7 |
| 5. Density | Per-cell density target and tile-level population-density conditioning are separate built-form proxy derivations. | Section 8 |
| 6. Road skeleton | Coarse road skeleton uses `crossings.parquet` as the crossing source of truth and joins `features.parquet` for road class evidence. | Section 9 |
| 7. Conditioning | Sub-D adds no new conditioning fields beyond `population_density_bucket`; extra metrics are provenance, evidence, or debug only. | Section 10 |
| 8. Determinism and versioning | Shared neutral determinism helpers; sub-D-local exclusion entries; version namespaces enforced by API and tests. | Section 12 |
| 9. Validation and tests | Layer 1/2/3 test pattern with deterministic diverse Singapore subset and tested frequency-analysis artifacts. | Section 13 |
| 10. Deferrals | Nearby work stays out of sub-D; conditional sub-C minor revisions are triggered only by empirical need. | Section 15 |

## 3. Inputs: Sub-C Contract Surface

Sub-D consumes the committed sub-C output contract. Schema definitions remain
authoritative in the sub-C design spec, especially Section 11.2 through
Section 11.9 of
`docs/superpowers/specs/2026-05-17-phase-1-sub-C-tile-extraction-design.md`.
Sub-D cites those schemas rather than duplicating their full definitions.

Sub-D reads a complete sub-C region directory only when `_SUCCESS` is present.
The sub-C manifest is the tile inventory source of truth; sub-D must not
discover tiles by filesystem glob.

Sub-D input artifacts:

- `manifest.yaml`: region metadata, tile inventory, sub-C config, schema
  versions, conditioning defaults, and digests.
- Per-tile `cells.parquet`: kept-cell rows. This is evidence, not sub-D's
  output shape. Cells outside admin coverage and pure-sea/no-feature cells are
  absent.
- Per-tile `features.parquet`: per cell-local sub-feature rows. Sub-D joins to
  cells by `(cell_i, cell_j)` and joins crossings by `source_feature_id`.
- Per-tile `crossings.parquet`: canonical internal edge-crossing records.
  `edge_id = (lower_cell_i, lower_cell_j, axis)`. This is the source of truth
  for coarse road-skeleton edge evidence.
- Per-tile `meta.yaml`: sub-C aggregates and `conditioning_per_tile`, including
  sub-C-owned fields and `population_density_bucket: null`.
- Per-tile `provenance.yaml`: sub-C extraction record and output digests.

Read rules:

- Use `pyarrow.parquet.ParquetFile(path).read()` for per-tile parquet reads.
  Do not use a directory-level `pq.read_table` pattern that can trigger Hive
  partition inference.
- Digest-anchor every sub-C artifact that sub-D consumes. If the referenced
  sub-C input bytes change, sub-D validation fails rather than silently
  accepting drift.
- Treat sub-C geometry as immutable evidence. Sub-D writes no geometry and does
  not duplicate geometry bytes into its own artifacts.

## 4. Output Model And Consumer Read-Patterns

Sub-D writes a standalone sidecar dataset:

```text
data/processed/sub_d/<release>/<region>/
  manifest.yaml
  _SUCCESS
  tile=<tile-id>/
    macro_core.parquet
    effective_conditioning.yaml
    provenance.yaml
```

The exact tile directory naming should follow the local sub-C naming convention
closely enough that tile IDs remain obvious and sortable. The sub-D manifest,
not the filesystem, is still the authoritative inventory.

Consumer read-patterns:

- Training reads `effective_conditioning.yaml` as macro-stage input
  conditioning and `macro_core.parquet` as macro-stage supervised targets.
- Sub-E reads sub-C `features.parquet` and `crossings.parquet` plus sub-D
  macro core to derive exact boundary contracts.
- Sub-F reads sub-D and sub-E outputs for deterministic reconstruction,
  stitching, and evaluation workflows.
- Sub-G validates sub-D against sub-C evidence and later validates
  cross-artifact consistency across sub-D, sub-E, and sub-F.
- Humans and debugging tools read derivation evidence, debug summaries, and
  provenance for traceability.

Sub-D does not mutate sub-C `meta.yaml`, `manifest.yaml`, or provenance files.
The integrity boundary is one-way:

```text
sub-C immutable outputs -> sub-D sidecar outputs
```

The complete tile-level conditioning vector lives in sub-D's
`effective_conditioning.yaml`. Sub-C's `null + owner` marker remains upstream
state; it is not patched in place.

## 5. Granularity And Lattice Shape

Sub-D uses a fixed macro lattice. Sub-C's sparse kept-cell layout is evidence,
not target shape.

Fixed target positions:

- 64 cell slots for zoning and per-cell density.
- 112 internal cell-edge slots for coarse road skeleton.
- 32 external tile-boundary edge slots for explicit perimeter handling.
- 1 tile-level effective conditioning vector.

Cell positions:

- Sub-D can enumerate missing cell positions as the complement of sub-C
  `cells.parquet` within the 64-slot lattice.
- Sub-D cannot distinguish all missing-cell reasons from sub-C alone. Sub-C
  stores pure-sea drops only as `sea_mask_drop_count` and omits outside-admin
  cells entirely.
- Each cell slot carries a cell-scope marker. Zoning and density apply only to
  active cell slots.

Internal edge positions:

- Kept-to-kept edge: active edge. The coarse road-skeleton enum is meaningful,
  including ordinary `none`.
- Not-in-scope to not-in-scope edge: fully masked edge. No road-skeleton
  prediction target.
- Kept-to-not-in-scope edge: scope-boundary edge. This is distinct from
  ordinary `none`; it means the edge abuts an unavailable or non-generated
  cell slot.

External edge positions:

- If the adjacent interior cell is active, emit a deterministic external-edge
  placeholder. Real tile-to-tile generation semantics are not Phase 1 sub-D
  scope.
- If the adjacent interior cell is not in scope, emit a fully masked external
  edge.

This separation prevents structural impossibility from contaminating ordinary
negative examples.

## 6. Open Empirical Gates Before Final Vocab/Schema Lock

Sub-D's macro vocabularies are frequency-driven. The following empirical gates
must run before the final macro vocab artifact is committed and before any
vocab-dependent schema field is treated as fully locked:

1. Zoning frequency analysis over deterministic per-cell evidence metrics.
2. Per-cell density bucket analysis over built-form intensity metrics.
3. Tile-level population-density proxy analysis over tile aggregate built-form
   evidence.
4. Coarse road-skeleton analysis over canonical crossing evidence.
5. Deterministic Singapore Layer 3 tile-subset selection.

Each frequency-analysis output is a tested artifact:

- Same sub-C input must produce byte-identical analysis output.
- Sanity invariants must hold, including valid metric ranges and non-empty
  locked buckets.
- Marginal-cost-of-cut monotonicity must hold where the cut strategy uses that
  framing.
- After vocab lock, a golden frequency-analysis artifact is committed. Drift
  triggers a vocab-version or derivation-version review, not silent
  recomputation.

Vocab-dependent schema fields in Section 11 carry an explicit "pending
Section 6 empirical lock" marker until these gates complete.

## 7. Zoning Derivation

Zoning is a compact macro-head prediction enum. It describes dominant urban
use, not density and not raw Overture class names.

Sub-D uses a B1-to-B2-style derivation pattern:

1. Compute deterministic per-cell evidence metrics from sub-C outputs.
2. Run frequency analysis over metric patterns.
3. Propose zoning candidates from observed Singapore distribution.
4. Apply marginal-cost-of-cut reasoning to decide which classes deserve their
   own token and which collapse into mixed or unknown buckets.
5. Lock the zoning enum append-only within Phase 1.

The specific zoning enum is pending Section 6 empirical lock. Any intuitive
names used during design are examples only.

Cell scope is separate from zoning:

- `cell_scope` says whether a fixed lattice slot is active or masked.
- `zoning` applies only where `cell_scope` is active.
- `not_in_scope` is not a zoning enum value.

Candidate evidence metrics include:

- Building class composition.
- POI category evidence.
- Road class composition.
- Retained base/water evidence.
- Low-evidence flags.
- Tie-break and confidence metadata.

Signal orthogonality with density is mandatory by default. Zoning should use
composition signals: what is built or active there. Density should use
intensity signals: how much is built there. Footprint-ratio intensity must not
quietly become a zoning determinant. If a future derivation needs a
density-like signal for zoning, the comparison must be empirical: zoning labels
with and without the density signal are compared, the result is recorded, and
`zoning_derivation_version` changes.

Versioned artifacts:

- `zoning_vocab_version`: enum values and append-only order.
- `zoning_derivation_version`: metric selection, metric definitions,
  thresholds, weighting, tie-break rules, and class-assignment logic.

Adding, removing, or redefining a metric is a derivation-function change. If
the same sub-C input would produce different zoning labels, the derivation
version changes.

Two-axis zoning remains a contingent path. If frequency analysis shows that one
zoning enum contaminates labels or collapses important structure, the decision
gate happens after the empirical zoning analysis and before committing the
macro vocab artifact.

## 8. Density Derivation

Sub-D derives two separate density artifacts:

1. Per-cell density target: a 64-position macro-head prediction output.
2. Tile-level `population_density_bucket`: a single conditioning input value
   in `effective_conditioning.yaml`.

These may share evidence inputs, but they do not share bucket counts or enum
values by default.

Sub-D does not have real population data in Phase 1. The defensible available
signal is built-form density, primarily building footprint and built-area
metrics from sub-C geometry. Therefore `population_density_bucket` is a Phase 1
built-form proxy for population density.

Per-cell density:

- Derived for active cell slots.
- Empirically bucketed from built-form intensity metrics.
- Bucket counts and cut points are pending Section 6 empirical lock.

Tile-level population-density proxy:

- Derived once per tile for conditioning.
- Aggregation method is not pre-committed. Candidate methods include mean,
  area-weighted mean, median, percentile-based summaries, or another
  empirically justified statistic.
- The chosen aggregation lives under `tile_population_density_derivation_version`.

Building height and floor count are valid future density signals but are not
preserved by sub-C Phase 1 output. Sub-D does not invent or infer height. If
density labels are empirically weak without vertical information, queue a
sub-C minor-revision candidate to preserve height/floor fields from Overture
buildings.

Versioned artifacts:

- `cell_density_vocab_version`.
- `cell_density_derivation_version`.
- `tile_population_density_vocab_version`.
- `tile_population_density_derivation_version`.

Changing metric selection, metric definitions, cuts, aggregation, thresholds,
tie-breaks, or missing-data behavior requires the relevant derivation-version
bump.

When real population data eventually replaces the built-form proxy, the field
name `population_density_bucket` stays for compatibility, but vocab and
derivation versions bump and provenance records the new source. That is a
Phase 2 candidate, not Phase 1 sub-D scope.

## 9. Coarse Road Skeleton Derivation

Sub-D emits a fixed per-edge coarse road-skeleton target. This target is a
macro-scale road-continuity signal, not an exact boundary contract.

Canonical input:

- Sub-D consumes sub-C `crossings.parquet` directly for internal edge crossing
  evidence.
- Sub-D joins `crossings.parquet` to `features.parquet` by `source_feature_id`
  to identify road rows and their `class_raw`.
- Sub-D does not rederive crossings from geometry.

This keeps sub-D and sub-E on the same source of truth. Sub-E later derives
exact boundary contracts from the same crossing artifact.

Scope:

- Internal edges use `crossings.parquet` plus joined road class evidence.
- External tile-boundary edges use the deterministic Section 5 placeholder or
  mask. Sub-C crossings do not define tile-to-tile semantics.

Output semantics:

- One coarse road-skeleton enum per active internal edge.
- No exact crossing position.
- No width or extent.
- No source-feature traceability as target semantics.
- No sub-E boundary-contract geometry.

The road-skeleton enum is pending Section 6 empirical lock. The derivation
input stays broad: all retained road crossing evidence may be considered. The
output enum decides empirically which road classes deserve macro tokens.

Starting hypothesis for multiple road classes on one edge:

- Highest hierarchy wins.

This is a hypothesis to validate, not an implementation discretion point.
Frequency analysis should compare whether the target works best as binary road
presence, hierarchy buckets, selected class groups, count-aware coarse buckets,
or another compact enum.

Versioned artifacts:

- `road_skeleton_vocab_version`.
- `road_skeleton_derivation_version`.

Changing input grouping, hierarchy order, crossing aggregation, precedence,
thresholds, multi-crossing handling, or external-edge placeholder behavior
requires a derivation-version bump.

## 10. Conditioning Overlay

Sub-D writes the complete consumer-facing tile conditioning vector in
`effective_conditioning.yaml`.

Copy rule:

- Copy all sub-C-owned conditioning fields at the pinned sub-C schema version.
- Fill sub-D-owned `population_density_bucket`.

The copy rule is schema-driven, not a hardcoded field list. The implementation
should derive ownership from the sub-C conditioning schema or owner markers,
for example `population_density_bucket_owner: sub-D`, rather than a static
allowlist.

Sub-D adds no other training conditioning fields. Derived summaries such as
dominant zoning, road intensity, compactness, or water coverage are not
conditioning tokens. Adding them would risk target leakage and expand the
macro prompt surface without empirical need.

Composite version surface:

- Pinned upstream sub-C conditioning schema/vocab versions for copied fields.
- `tile_population_density_vocab_version`.
- `tile_population_density_derivation_version`.
- Digest anchors to sub-C artifacts from which copied fields came.

Support data categories:

- Provenance: digest-tracked, load-bearing, validator-checked.
- Evidence metrics: derivation inputs used to produce labels and buckets;
  versioned under relevant derivation versions.
- Debug summaries: convenience outputs for inspection only; not training
  inputs, not validator gates, and not versioned with the same force.

## 11. Artifact Schemas And Layout

All schemas here are the sub-D sidecar contract. Fields marked "pending
Section 6 empirical lock" are structurally present, but their enum domains or
bucket cuts are locked only after the frequency-analysis gates complete.

### 11.1 Directory Layout

```text
data/processed/sub_d/<release>/<region>/
  manifest.yaml
  _SUCCESS
  tile=EPSG3414_i<tile_i>_j<tile_j>/
    macro_core.parquet
    effective_conditioning.yaml
    provenance.yaml
```

The directory name mirrors sub-C's tile naming convention for readability.
Consumers still iterate `manifest.yaml`, not the filesystem.

### 11.2 `macro_core.parquet`

One row per fixed macro slot. This keeps a single canonical target file while
still distinguishing cell slots, internal edge slots, and external edge slots.

Draft schema:

```text
slot_kind                         int8      # enum: 0=cell, 1=internal_edge, 2=external_edge
slot_index                        int16     # canonical within slot_kind
cell_i                            int8?     # populated for cell slots
cell_j                            int8?     # populated for cell slots
lower_cell_i                      int8?     # populated for edge slots where applicable
lower_cell_j                      int8?     # populated for edge slots where applicable
axis                              int8?     # sub-C AXIS enum for edge slots where applicable
scope                             int8      # pending Section 6 empirical lock for final enum names
zoning_class                      int16?    # pending Section 6 empirical lock; cell active slots only
cell_density_bucket               int16?    # pending Section 6 empirical lock; cell active slots only
road_skeleton_class               int16?    # pending Section 6 empirical lock; active internal edges only
evidence_ref                      string?   # stable key into evidence/debug artifact if one is written
```

Canonical sort key:

```text
(slot_kind, slot_index)
```

Validation expectations:

- Exactly 64 `slot_kind=cell` rows.
- Exactly 112 `slot_kind=internal_edge` rows.
- Exactly 32 `slot_kind=external_edge` rows.
- Inactive or masked slots do not carry normal target classes.
- Cell targets appear only on active cell rows.
- Road-skeleton targets appear only on active internal edge rows.

The concrete integer enum maps for `scope`, `zoning_class`,
`cell_density_bucket`, and `road_skeleton_class` are supplied by the macro vocab
artifact after Section 6 empirical lock.

### 11.3 `effective_conditioning.yaml`

Draft shape:

```yaml
schema_version: "1.0"
tile_i: 12
tile_j: 17

versions:
  sub_c_conditioning_schema_version: "1.1"
  tile_population_density_vocab_version: "pending Section 6 empirical lock"
  tile_population_density_derivation_version: "pending Section 6 empirical lock"

sub_c_inputs:
  manifest_sha256: "<sha>"
  tile_meta_sha256: "<sha>"
  tile_provenance_sha256: "<sha>"

conditioning:
  # schema-driven copy of all sub-C-owned conditioning fields
  country: "SG"
  climate_zone: "tropical_rainforest"
  admin_region: "Central Region"
  morphology_class: "Asian-megacity"
  era_class: "contemporary"
  coastal_inland_river: 1

  # sub-D-owned field
  population_density_bucket: 3
```

The current sub-C-owned field examples above are illustrative. The
implementation must not use this list as a static allowlist.

### 11.4 `provenance.yaml`

Draft shape:

```yaml
schema_version: "1.0"
tile_i: 12
tile_j: 17

extraction:
  commit_sha: "<40-char sha>"
  extracted_utc: "2026-05-19T12:00:00Z"
  rerun_count: 0
  rerun_reason: "initial"

inputs:
  release: "2026-04-15.0"
  sub_c_manifest_sha256: "<sha>"
  sub_c_tile_provenance_sha256: "<sha>"
  sub_c_cells_parquet_sha256: "<sha>"
  sub_c_features_parquet_sha256: "<sha>"
  sub_c_crossings_parquet_sha256: "<sha>"
  sub_c_meta_yaml_sha256: "<sha>"
  macro_vocab_sha256: "<sha>"
  derivation_config_sha256: "<sha>"

versions:
  sub_d_schema_version: "1.0"
  macro_plan_vocab_version: "pending Section 6 empirical lock"
  zoning_vocab_version: "pending Section 6 empirical lock"
  zoning_derivation_version: "pending Section 6 empirical lock"
  cell_density_vocab_version: "pending Section 6 empirical lock"
  cell_density_derivation_version: "pending Section 6 empirical lock"
  tile_population_density_vocab_version: "pending Section 6 empirical lock"
  tile_population_density_derivation_version: "pending Section 6 empirical lock"
  road_skeleton_vocab_version: "pending Section 6 empirical lock"
  road_skeleton_derivation_version: "pending Section 6 empirical lock"

outputs:
  macro_core_parquet_sha256: "<sha>"
  effective_conditioning_yaml_sha256: "<sha>"
```

### 11.5 `manifest.yaml`

Draft shape:

```yaml
schema_version: "1.0"
sub_d_schema_version: "1.0"
release: "2026-04-15.0"
region: "singapore"
region_crs: "EPSG:3414"

inputs:
  sub_c_manifest_sha256: "<sha>"
  sub_c_region_dir: "data/processed/sub_c/2026-04-15.0/singapore"

versions:
  macro_plan_vocab_version: "pending Section 6 empirical lock"
  zoning_vocab_version: "pending Section 6 empirical lock"
  zoning_derivation_version: "pending Section 6 empirical lock"
  cell_density_vocab_version: "pending Section 6 empirical lock"
  cell_density_derivation_version: "pending Section 6 empirical lock"
  tile_population_density_vocab_version: "pending Section 6 empirical lock"
  tile_population_density_derivation_version: "pending Section 6 empirical lock"
  road_skeleton_vocab_version: "pending Section 6 empirical lock"
  road_skeleton_derivation_version: "pending Section 6 empirical lock"

config:
  cell_grid: [8, 8]
  cell_size_m: 250
  tile_size_m: 2000
  internal_edge_count: 112
  external_edge_count: 32

initial_extraction:
  commit_sha: "<40-char sha>"
  started_utc: "2026-05-19T12:00:00Z"
  completed_utc: "2026-05-19T12:10:00Z"
  tile_count: 187

tiles:
  - tile_i: 12
    tile_j: 17
    provenance_sha256: "<sha>"
```

Tiles are sorted by `(tile_i, tile_j)`.

### 11.6 Macro Vocab And Derivation Config

Sub-D needs a versioned macro vocab/config artifact, likely under
`configs/macro_plan/` or inside the sub-D region output. The exact path is an
implementation-plan decision, but the artifact must include:

- `macro_plan_vocab_version`.
- Slot-kind, scope, zoning, cell-density, tile-population-density, and
  road-skeleton enum definitions.
- Append-only ordering discipline.
- Derivation versions and config values.
- Source frequency-analysis artifact digests.

The artifact is pending Section 6 empirical lock.

## 12. Versioning, Determinism, And Provenance

Sub-D uses shared neutral determinism primitives. The required sequence is:

1. Extract shared helpers into a neutral module, such as `cfm.data.io` or
   `cfm.data.determinism`.
2. Re-point sub-C imports to the neutral module.
3. Run sub-C tests and confirm zero behavior change.
4. Implement sub-D against the neutral module.

If helper extraction proves risky because existing sub-C helpers are more
coupled than expected, the fallback is explicit: duplicate the pattern in
`cfm.data.sub_d` and accept drift cost. Sub-D must not import from
`cfm.data.sub_c` internals.

Shared mechanism:

- Canonical YAML serialization.
- Pinned parquet writer kwargs and write helper.
- Digest helpers.
- Exclusion-table grammar.

Sub-D-local entries:

- The actual `EXCLUDED_FROM_SHA` table is sub-D artifact-specific.
- Sub-D inherits final-segment `*_sha256` matching and file-keyed timestamp
  exclusions as a mechanism, not sub-C's entries.

Sub-D digest chain:

```text
_SUCCESS
  -> manifest.tiles[*].provenance_sha256
  -> provenance.outputs.*_sha256
  -> file bytes
```

Sub-D writes no geometry, so no WKB byte-order contract applies.

Version namespaces:

- Artifact format versions: manifest, effective conditioning, provenance.
- Data-shape version: `sub_d_schema_version`.
- Aggregate vocab package: `macro_plan_vocab_version`.
- Individual vocab versions: zoning, cell-density, tile-population-density,
  road-skeleton.
- Derivation versions: zoning, cell-density, tile-population-density,
  road-skeleton.
- Validator/invariant-set version if validators need stable IDs.

Validators compare like-for-like only. To prevent the sub-C known issue #8
failure mode, sub-D must include a namespace-aware comparison helper, such as:

```text
compare_version(namespace, expected, actual)
```

The namespace is an enum such as `artifact_format`, `data_shape`, `vocab`,
`derivation`, or `validator`. Cross-namespace comparisons fail. Validators
must use this helper rather than direct ad hoc string comparisons.

## 13. Validation And Test Strategy

Sub-D uses the same layer structure as sub-C.

### 13.1 Layer 1: Pure Unit Tests

Layer 1 covers:

- Fixed lattice indexing: 64 cells, 112 internal edges, 32 external edges.
- Cell and edge scope derivation from sub-C kept-cell evidence.
- Zoning evidence metrics.
- Density evidence metrics.
- Crossing-to-road-skeleton aggregation from `crossings.parquet` joined to
  `features.parquet`.
- Bucket assignment and tie-break rules.
- Deterministic sorting and serialization primitives.
- `compare_version(namespace, expected, actual)` rejects cross-namespace
  comparisons.

Layer 1 uses tiny synthetic tables and no cached Singapore dependency.

### 13.2 Layer 2: Artifact And Validator Tests

Layer 2 uses synthetic sub-C-like fixtures and covers:

- Write/read schemas for `macro_core.parquet`, `effective_conditioning.yaml`,
  `provenance.yaml`, and `manifest.yaml`.
- Canonical sort keys.
- Provenance and digest-chain correctness.
- `_SUCCESS` written only after validator pass.
- Copied sub-C conditioning mismatch fails validation.
- Pinned sub-C input digest mismatch fails validation.
- Scope/value consistency: inactive cells and edges do not carry normal target
  classes.
- Road skeleton uses `crossings.parquet` plus `features.parquet`, not geometry
  rederivation.
- Namespace-version meta-test: validators use the namespace-aware comparison
  helper and cross-namespace comparisons fail.
- Same-process determinism: run twice on identical synthetic inputs and assert
  byte-identical outputs, excluding declared timestamp and digest fields.

### 13.3 Layer 3: Cached Singapore Integration

Layer 3 uses a deterministic fixed tile subset, not random or fastest-only
selection.

Subset requirements:

- Fixed tile IDs committed in test/config.
- Selected for diversity across zoning evidence, density spread,
  road-skeleton spread, coastal/inland/riverside coverage, and active/masked
  cell/edge cases.
- Each tile has a documented rationale.
- The subset is a smoke test, not proof of global correctness.

Frequency-analysis artifacts are tested:

- Same sub-C input produces byte-identical output.
- Bucket count is at least 2 where applicable.
- No locked bucket is empty.
- Evidence-metric ranges are valid.
- Marginal-cost-of-cut monotonicity holds where applicable.
- After empirical vocab lock, a golden frequency-analysis artifact is
  committed and compared in tests.

Layer 3 also validates real sidecar artifacts end to end and real-data
deterministic reruns on the fixed subset. A full-Singapore run is optional and
depends on local sub-C output availability and runtime.

### 13.4 Cross-Environment Determinism

Required Layer 2 determinism is same-process byte identity. Target Layer 3 or
CI determinism is byte identity in a different environment, such as a different
Python minor version or OS. If cross-environment determinism is infeasible in
Phase 1 CI, the spec and implementation handoff must document the residual
risk explicitly.

### 13.5 Validator Boundaries

Sub-D validators own:

- Macro-core schema and lattice completeness.
- Conditioning overlay correctness.
- Digest chain and sub-C input anchors.
- Version namespace enforcement.
- Deterministic bytes.
- Consistency against pinned sub-C evidence.

Sub-D validators do not own:

- Exact boundary-contract validation.
- Micro-token decodability.
- Tile-to-tile stitching semantics.
- Tokenizer `emit_unknown_token` behavior.

If real cached Singapore data violates a planned invariant, the invariant is
not weakened to pass. Treat it as a failed assumption, revisit the design, and
update the PRD/spec if the data disagrees.

## 14. Implementation Boundaries

Likely modules and scripts:

```text
src/cfm/data/sub_d/
  __init__.py
  lattice.py
  io.py
  manifest.py
  provenance.py
  conditioning.py
  zoning.py
  density.py
  road_skeleton.py
  frequency_analysis.py
  versions.py
  validator_inline.py
  validator_cross_tile.py
  pipeline.py

scripts/derive_macro_plan.py
scripts/analyse_macro_plan_frequencies.py
scripts/validate_macro_plan.py
```

Implementation must keep write boundaries clear:

- Read sub-C via public sub-C APIs or explicit artifact paths.
- Do not mutate sub-C artifacts.
- Do not import determinism helpers from `cfm.data.sub_c` internals.
- Do not add Sweden-specific execution logic.
- Do not solve tokenizer unknown-token behavior inside sub-D.

Implementer dispatches must preserve the project branch pattern: no new branch,
no push, no PR unless explicitly requested in that session.

## 15. Deferrals And Known Limits

Sub-D explicitly defers:

1. Tokenizer `emit_unknown_token` fall-through. Known issue #4 remains
   training-critical but outside sub-D.
2. Overture cold-fetch performance. Known issue #1 remains out of scope until
   fresh-region work or Sweden enrollment.
3. Sweden densification and multi-region enrollment. Known issue #3 remains a
   separate sub-project.
4. Sub-C `--rerun` CLI wiring. Known issue #7 remains out of scope unless
   empirical sub-D work requires a deliberate sub-C re-extraction.
5. Fixing sub-C known issue #8. Sub-D avoids repeating it but does not fix
   sub-C's validator.
6. Conditioning vocabulary tightening. Raw sub-C strings remain schema-copied;
   a future conditioning-vocabulary project owns enum tightening.
7. Building height and floors. If empirical density labels are too weak,
   preserve height/floor fields through a sub-C minor revision.
8. Dropped-cell reason distinction. If mask-reason distinction becomes
   load-bearing, preserve dropped-cell reasons through a sub-C minor revision.
9. External tile-to-tile stitching semantics. Sub-D writes external placeholders
   only; real tile-to-tile continuity belongs to future global-generation work.
10. Exact boundary contracts. Sub-E owns crossing positions, width/extent,
    source-feature traceability, and micro-cell stitching constraints.
11. Real population data. Phase 1 uses a built-form proxy; replacing it with
    real population data is a Phase 2 candidate with vocab and derivation
    version bumps.
12. Two-axis zoning. This can reopen only after empirical zoning frequency
    analysis and before committing the macro vocab artifact.

These are recorded in this spec, not immediately promoted to
`docs/known_issues.md`. The project-wide known-issues tracker is for accepted
deferred defects, not every considered non-goal.

## 16. Spec Self-Review Checklist

- Placeholder scan: no `TBD` or unfinished sections.
- Version namespace consistency: artifact format, data shape, vocab,
  derivation, and validator versions are separate.
- Scope leakage check: no tokenizer, cold-fetch, Sweden, sub-E, sub-F, or
  sub-G implementation work is hidden inside sub-D.
- Consumer read-pattern coverage: training, sub-E, sub-F, sub-G, and humans
  have explicit read paths.
- Schema/gate consistency: vocab-dependent schema fields are marked pending
  Section 6 empirical lock.
- No sub-C mutation: sidecar-only output boundary is maintained.
- No geometry duplication: sub-D writes tokenizable macro targets and metadata,
  not geometry.

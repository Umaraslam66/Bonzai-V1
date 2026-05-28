# Task 7 implementer dispatch prompt

**Status:** Draft v3; approved for dispatch.
**Target:** General-purpose subagent / Codex agent.
**Suggested model:** Sonnet-class.
**Branch:** `phase-1-sub-F-micro-tokenizer` (base includes Task 6 close commit `cf32d81`).

> The prompt below is the verbatim text to give the implementer agent. Everything between the `===` markers is the agent's prompt body.

===

Task: Sub-F Task 7 - BP7 boundary-reference vocab + sub-C feature-splitting verification, Halt 7 surface only.

You are working in `/Users/umaraslam/Projects/Bonzai-OSM` on branch `phase-1-sub-F-micro-tokenizer`. You are not alone in the codebase: do not revert edits made by others; inspect current state and work with it. Do not push. Do not create a PR. Do not proceed past Halt 7 approval. Stop at Halt 7 with a WIP commit and a halt report.

## Preconditions

- Branch: `phase-1-sub-F-micro-tokenizer`.
- Task 1, Task 2, Task 4, and Task 6 are closed.
- `configs/sub_f/semantic_vocab.yaml`, `configs/sub_f/unknown_family.yaml`, `configs/sub_f/encoding_primitives.yaml`, and BP1/BP2/BP4/dataloader portions of `configs/sub_f/sentinel_inventory.yaml` are `LOCKED`.
- BP7 is still a placeholder in `configs/sub_f/sentinel_inventory.yaml`: range `1500..1599`, expected `8` used.
- Task 6 manifest remains provisional until BP7 boundary-reference vocab source is added at sub-F close.
- Do not push. Do not PR. Do not continue to Task 8 writer work.

## Non-negotiable discipline

- Halt-on-defect: unexpected errors, missing upstream files, type mismatches, or audit contradictions -> STOP and report `BLOCKED`.
- Verify-before-lock: this task surfaces Halt 7 data for reviewer approval. Do not write final LOCKED status or a final `feat:` commit.
- Cascade resolution: if sub-C feature-splitting contradicts the plan assumption, surface a §9.6.1 cascade candidate in the Halt 7 report. Do not silently add a multi-outbound grammar case.
- Gate 6 hand-enumeration: expected BP7 token list, direction order, BoundaryClass values, and class grouping expected sets must be hand-stated in tests/reports, not derived from sub-F's own implementation.

## Pre-dispatch audits

### Audit step 1: confirm sub-E rotation return shape

Run:

```bash
uv run python -c "from cfm.data.sub_e.rotation import cell_to_edge_ids; e=cell_to_edge_ids(3, 5); print(type(e).__name__); print(e); print(sorted(e.__dataclass_fields__)); print(e.north, e.south, e.west, e.east)"
```

Expected:
- Type name is `CellEdgeIds`.
- Fields include `north`, `south`, `west`, `east`.
- This is a dataclass, not an iterable tuple.

Important: sub-E's dataclass doc says stable iteration order is `N/S/W/E`, while sub-F boundary-ref vocab order is `N/E/S/W`. The sub-F wrapper must map fields explicitly:

```python
{"N": e.north, "E": e.east, "S": e.south, "W": e.west}
```

If `cell_to_edge_ids` moved or no longer exposes these fields: STOP, report BLOCKED.

### Audit step 2: confirm sub-E BoundaryClass values

Run:

```bash
uv run python -c "from cfm.data.sub_e.derivation import BoundaryClass; print([(c.name, int(c), c.value) for c in BoundaryClass])"
```

Expected:

```text
[('BOUNDARY_NOT_APPLICABLE', 0, 0), ('NONE', 1, 1), ('MAJOR_ROAD', 2, 2), ('MINOR_ROAD', 3, 3)]
```

If values differ: STOP, report BLOCKED. This is an upstream contract mismatch.

### Audit step 3: confirm sub-E hierarchy and class grouping

Run:

```bash
uv run python -c "from cfm.data.sub_e.derivation import BoundaryClass, _HIERARCHY, load_class_grouping_map; m=load_class_grouping_map(); print([c.name for c in _HIERARCHY]); print('major', sorted(k for k,v in m.items() if v is BoundaryClass.MAJOR_ROAD)); print('minor', sorted(k for k,v in m.items() if v is BoundaryClass.MINOR_ROAD))"
```

Expected:
- Hierarchy: `['MAJOR_ROAD', 'MINOR_ROAD', 'NONE']`.
- Major set: `primary`, `secondary`, `trunk`.
- Minor set: `cycleway`, `footway`, `residential`, `service`, `steps`, `tertiary`, `unclassified`.

If the grouping differs: STOP, report BLOCKED. Do not update expected sets inline.

### Audit step 4: confirm BP7 ID namespace placeholder remains available

Run:

```bash
uv run python -c "import yaml; d=yaml.safe_load(open('configs/sub_f/sentinel_inventory.yaml')); b=d['bp7_boundary_ref_placeholder']; print(d['_status'], b['start_id'], b['end_id'], b['placeholder'], b['status'])"
```

Expected: `LOCKED 1500 1599 True PLACEHOLDER...`.

If BP7 is already locked or the range differs: STOP, report BLOCKED.

### Audit step 5: confirm cached sub-C Singapore feature data exists

Run:

```bash
uv run python -c "from pathlib import Path; root=Path('data/processed/sub_c/2026-04-15.0/singapore'); paths=sorted(root.glob('tile=*/features.parquet')); print('tile_count', len(paths)); assert paths; print(paths[0])"
```

Expected: `tile_count > 1`.

If cached sub-C data is missing: STOP, report BLOCKED.

### Audit step 6: enumerate BP1 highway values missing from sub-E grouping

Run:

```bash
uv run python -c "import yaml; from cfm.data.sub_e.derivation import load_class_grouping_map; m=load_class_grouping_map(); sv=yaml.safe_load(open('configs/sub_f/semantic_vocab.yaml')); hw=[s['tag'].split('=')[1] for s in sv['slots'] if s['tag'].startswith('highway=')]; missing=[v for v in hw if v not in m]; print('highway_values_in_vocab', sorted(hw)); print('NONE_mapped_missing_from_grouping', sorted(missing)); print('motorway_present_in_grouping', 'motorway' in m)"
```

Expected to surface `motorway` (and possibly link/long-tail highway values) in `NONE_mapped_missing_from_grouping`.

This is a §9.6.1 cascade candidate against sub-E class grouping coverage, not an implementation detail to auto-resolve in Task 7. If any load-bearing highway value in the locked BP1 vocab maps to NONE because it is absent from `load_class_grouping_map()`, keep working only through the Halt 7 surface: list every missing value and classify the coverage gap. Do not extend sub-E grouping, add a sub-F override, or document known-loss without reviewer decision.

For every missing value, also surface Singapore boundary-load evidence:
- Preferred if cheap: exact count of cached sub-C road features with that `class_raw` whose geometry crosses a 2 km tile boundary.
- Acceptable proxy: from cached sub-C Singapore parquet, count total `feature_class == 0` rows per missing `class_raw` value and count unique `source_feature_id` values for that class that appear in more than one `tile=*` file. Report both as `singapore_row_count` and `multi_tile_source_feature_count`.
- If `source_feature_id` is unavailable, report `singapore_row_count` only and state the proxy limitation.

This count surface is required so the reviewer can classify the gap as narrow expressway-family coverage vs wide systematic grouping under-coverage at Halt 7 without a continuation.

## Implementation scope

Create:
- `src/cfm/data/sub_f/rotation.py`
- `configs/sub_f/boundary_reference_vocab.yaml` with `_status: PROPOSED`
- `scripts/sub_f/verify_sub_c_feature_splitting.py`
- `tests/data/sub_f/test_rotation.py`
- `reports/2026-05-23-phase-1-sub-F-task-7-halt.md`
- `reports/sub_f_task_7_feature_splitting.yaml`

Do not modify `configs/sub_f/sentinel_inventory.yaml` yet. BP7 transitions from PLACEHOLDER to LOCKED only after Halt 7 reviewer approval.

All YAML written by this task must be byte-deterministic. Use `cfm.data.io.canonicalize_yaml(...)` for `boundary_reference_vocab.yaml` and `reports/sub_f_task_7_feature_splitting.yaml`; do not hand-format YAML with ad hoc string assembly.

## Step 1: create boundary_reference_vocab.yaml

Create `configs/sub_f/boundary_reference_vocab.yaml` as a proposed Halt 7 artifact.

Required content:

```yaml
_status: "PROPOSED - pending Halt 7 reviewer approval"
release: "2026-04-15.0"
family: "bp7_boundary_ref"
id_block:
  start_id: 1500
  end_id: 1599
  used_count: 8
  reserved_count: 92
  status: "PROPOSED - locks after Halt 7 approval"
direction_order:
  - "N"
  - "E"
  - "S"
  - "W"
class_set:
  - "MAJOR_ROAD"
  - "MINOR_ROAD"
multi_class_collapse_rule: "MAJOR_ROAD > MINOR_ROAD > NONE"
non_emitting_classes:
  - "NONE"
  - "BOUNDARY_NOT_APPLICABLE"
slots:
  - id: 1500
    local_id: 0
    tag: "<bref_N_MAJOR>"
    direction: "N"
    boundary_class: "MAJOR_ROAD"
  - id: 1501
    local_id: 1
    tag: "<bref_E_MAJOR>"
    direction: "E"
    boundary_class: "MAJOR_ROAD"
  - id: 1502
    local_id: 2
    tag: "<bref_S_MAJOR>"
    direction: "S"
    boundary_class: "MAJOR_ROAD"
  - id: 1503
    local_id: 3
    tag: "<bref_W_MAJOR>"
    direction: "W"
    boundary_class: "MAJOR_ROAD"
  - id: 1504
    local_id: 4
    tag: "<bref_N_MINOR>"
    direction: "N"
    boundary_class: "MINOR_ROAD"
  - id: 1505
    local_id: 5
    tag: "<bref_E_MINOR>"
    direction: "E"
    boundary_class: "MINOR_ROAD"
  - id: 1506
    local_id: 6
    tag: "<bref_S_MINOR>"
    direction: "S"
    boundary_class: "MINOR_ROAD"
  - id: 1507
    local_id: 7
    tag: "<bref_W_MINOR>"
    direction: "W"
    boundary_class: "MINOR_ROAD"
```

Add `source_references` with file paths/line notes for sub-E `BoundaryClass`, `_HIERARCHY`, and `cell_to_edge_ids`.

Write the file with `cfm.data.io.canonicalize_yaml(...)`.

## Step 2: create rotation wrapper

Create `src/cfm/data/sub_f/rotation.py`.

Requirements:
- Import `CellEdgeIds`, `EdgeIdTuple`, and `cell_to_edge_ids` from `cfm.data.sub_e.rotation`.
- Define `DIRECTION_ORDER = ("N", "E", "S", "W")`.
- Define `cell_edge_directions(cell_i: int, cell_j: int) -> dict[str, EdgeIdTuple]`.
- Map fields explicitly:

```python
edge_ids = cell_to_edge_ids(cell_i, cell_j)
return {
    "N": edge_ids.north,
    "E": edge_ids.east,
    "S": edge_ids.south,
    "W": edge_ids.west,
}
```

Do not zip over `CellEdgeIds`. Do not assume dataclass iteration order.

## Step 3: create feature-splitting verification script

Create `scripts/sub_f/verify_sub_c_feature_splitting.py`.

Purpose: verify whether sub-C emits branched/multi-part road features as single rows. This informs whether spec §3.7 needs a multi-outbound grammar case.

Requirements:
- Iterate all `tile=*/features.parquet` files under the supplied sub-C region dir.
- Use `pq.ParquetFile(path).read()`.
- Decode `geometry` with `shapely.wkb.loads`.
- Count all geometry types globally.
- Separately count road rows (`feature_class == 0`) with `MultiLineString` geometry, using both decoded shapely type and `geometry_type == 4` as evidence.
- Emit `reports/sub_f_task_7_feature_splitting.yaml`.
- Emit the YAML with `cfm.data.io.canonicalize_yaml(...)`.
- Report:
  - `_status: "PROPOSED - pending Halt 7 reviewer approval"`
  - `tile_count`
  - `row_count`
  - `geometry_type_counts`
  - `road_multiline_count`
  - up to 20 `road_multiline_examples` with tile, source_feature_id, geometry type, and part count.
  - `outcome`:
    - `single_row_per_branch` if `road_multiline_count == 0`
    - `branched_multi_row_present` if `road_multiline_count > 0`
  - `recommendation`:
    - no multi-outbound grammar needed if `single_row_per_branch`
    - §9.6.1 cascade candidate if `branched_multi_row_present`

If `branched_multi_row_present`, do not add a grammar case. Surface it in Halt 7 for reviewer classification.

## Step 4: tests

Create `tests/data/sub_f/test_rotation.py`.

Minimum tests:
- `DIRECTION_ORDER == ("N", "E", "S", "W")`.
- `cell_edge_directions(3, 5)` returns exactly keys `N/E/S/W`.
- The wrapper maps to explicit sub-E fields:
  - `N == cell_to_edge_ids(...).north`
  - `E == ...east`
  - `S == ...south`
  - `W == ...west`
- Boundary-reference vocab has `_status: PROPOSED`, block `1500..1599`, 8 slots, IDs `1500..1507`, and tags exactly as listed above.
- Boundary-reference vocab class/direction cross product is exactly `{N,E,S,W} x {MAJOR_ROAD, MINOR_ROAD}`.
- BoundaryClass enum values match the hand-enumerated expected mapping.
- `_HIERARCHY` is `MAJOR_ROAD > MINOR_ROAD > NONE`.
- BP1 -> sub-E class mapping standalone test:
  - expected major set = `{"primary", "trunk", "secondary"}`
  - expected minor set = `{"tertiary", "residential", "service", "unclassified", "footway", "steps", "cycleway"}`
  - assert against `load_class_grouping_map()`.
- BP1 locked-highway coverage diagnostic:
  - hand-enumerate all locked `highway=*` semantic-vocab values from `configs/sub_f/semantic_vocab.yaml`.
  - report every value absent from `load_class_grouping_map()`.
  - for each absent value, compute Singapore boundary-load evidence from cached sub-C data:
    - `singapore_row_count`: number of `feature_class == 0` rows whose `class_raw` is that value.
    - `multi_tile_source_feature_count`: number of distinct `source_feature_id` values for that class observed in more than one tile file, if `source_feature_id` exists.
  - assert the diagnostic output is present in the Halt 7 report; do not assert an empty missing list unless reviewer has already resolved the grouping gap.
- Sentinel inventory still has BP7 as PLACEHOLDER; do not lock it in this task.
- Feature-splitting report exists after script run and has one of the two allowed outcomes.

## Step 5: Halt 7 report

Create `reports/2026-05-23-phase-1-sub-F-task-7-halt.md`.

Report must include:
- Status: `DONE_WITH_CONCERNS` pending Halt 7 reviewer approval, or `BLOCKED` with classification.
- Audit outcomes 1-6.
- Boundary-reference vocab proposal:
  - 8 tokens.
  - IDs `1500..1507`.
  - BP7 reserved headroom `1508..1599`.
  - `NONE` and `BOUNDARY_NOT_APPLICABLE` non-emitting.
- Inbound/outbound token-count confirmation:
  - confirm from spec §3.2 and §3.7 that inbound case C/D reuses the same 8 directional `<bref_dir_class>` tokens by prepending position; inbound/outbound distinction is position-carried, not distinct-token-carried.
  - if the spec text implies distinct inbound tokens are needed, STOP and report BLOCKED because the BP7 vocab count becomes 16, not 8.
- Rotation wrapper result:
  - sub-E returns `CellEdgeIds`.
  - sub-F maps fields explicitly into vocab order `N/E/S/W`.
- Class-mapping evidence:
  - BoundaryClass enum values.
  - hierarchy.
  - major/minor class grouping expected sets.
- BP7 class-coverage gap:
  - list every locked BP1 `highway=*` value absent from sub-E `load_class_grouping_map()`.
  - explicitly call out `motorway` / `motorway_link` if present.
  - include per-value `singapore_row_count` and `multi_tile_source_feature_count` (or exact boundary-crossing count if implemented instead).
  - classify as a §9.6.1 cascade candidate against sub-E class grouping if any load-bearing highway value maps to NONE.
  - classify the observed shape as narrow vs wide for reviewer decision:
    - NARROW if missing load-bearing evidence concentrates in motorway/expressway-family values.
    - WIDE if many missing values have non-trivial Singapore counts or multi-tile evidence across mixed highway families.
  - state that reviewer must choose one of: (a) extend sub-E grouping, (b) add sub-F-local BP7 grouping override, or (c) document known-loss only if data shows the value is not boundary-load-bearing.
- Feature-splitting outcome:
  - paste `outcome`, counts, and example rows if any.
  - if `branched_multi_row_present`, call it a §9.6.1 cascade candidate against the Task 7 assumption and stop at halt.
- Manifest obligation note:
  - Task 6 provisional manifest must add this BP7 boundary-reference vocab source after Halt 7 approval and recompute final manifest sha at sub-F close.
- Section 10.5 telemetry.

Reviewer approves:
- BP7 vocab slot list and ID range.
- BP7 class-coverage resolution for locked highway values missing from sub-E grouping.
- Whether sub-C feature-splitting validates the v1 grammar or triggers a cascade.
- Whether BP7 placeholder can transition to LOCKED in continuation.

## Verification

Run:

```bash
uv run python scripts/sub_f/verify_sub_c_feature_splitting.py --sub-c-region-dir data/processed/sub_c/2026-04-15.0/singapore
uv run pytest tests/data/sub_f/test_rotation.py -v
uv run pytest tests/data/sub_f/test_manifest.py tests/data/sub_f/test_provenance.py -v
git diff --check
```

If `uv` cache access is blocked by sandboxing, report that exactly; do not claim tests passed.

## Commit

Commit the halt surface with a WIP message:

```text
wip(sub_f): T7 pre-halt - BP7 boundary-ref vocab + sub-C feature-splitting (Halt 7 pending)
```

Do not use a final `feat:` commit before reviewer approval.

Final status:
- `DONE_WITH_CONCERNS` if Halt 7 report and WIP commit are ready for reviewer approval.
- `BLOCKED` if any audit, upstream contract, or test outcome contradicts the prompt outside the expected feature-splitting decision surface.

Surface:
- commit SHA
- changed files
- verification results
- halt report content

===

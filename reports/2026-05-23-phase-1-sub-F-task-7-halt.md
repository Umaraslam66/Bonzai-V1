# Phase 1 Sub-F Task 7 Halt 7 Report

Status: BLOCKED.

Branch: `phase-1-sub-F-micro-tokenizer`

WIP commit: pending at report creation time.

## Scope

Implemented the Halt 7 surface only:

- Added proposed BP7 boundary-reference vocab at `configs/sub_f/boundary_reference_vocab.yaml`.
- Added sub-F rotation wrapper at `src/cfm/data/sub_f/rotation.py`.
- Added sub-C feature-splitting verification script at `scripts/sub_f/verify_sub_c_feature_splitting.py`.
- Added Task 7 tests at `tests/data/sub_f/test_rotation.py`.
- Added feature-splitting report at `reports/sub_f_task_7_feature_splitting.yaml`.

No change was made to `configs/sub_f/sentinel_inventory.yaml`. BP7 remains PLACEHOLDER. No writer/orchestrator work was added. No push or PR was performed.

Blocker classification: sub-C Singapore feature-splitting verification surfaced branched/multi-part road rows (`road_multiline_count=5605`, outcome `branched_multi_row_present`). Per Task 7 dispatch stop condition, this is a §9.6.1 cascade candidate against the Task 7 assumption; do not silently add a multi-outbound grammar case in this task.

Cascade A (sub-E grouping under-covering locked BP1 highway vocab) is resolved in this amendment by a sub-F-local BP7 drivable-continuity override. Cascade B (MultiLineString part-edge relationship) remains BLOCKED pending reviewer classification.

## Audit Step Outcomes

1. sub-E rotation return shape matched the expected contract:
   type name `CellEdgeIds`; dataclass fields `['east', 'north', 'south', 'west']`; sample `cell_to_edge_ids(3, 5)` exposed `.north`, `.south`, `.west`, `.east`.

2. sub-E `BoundaryClass` values matched the expected hand enumeration:
   `[('BOUNDARY_NOT_APPLICABLE', 0, 0), ('NONE', 1, 1), ('MAJOR_ROAD', 2, 2), ('MINOR_ROAD', 3, 3)]`.

3. sub-E hierarchy and class grouping matched the expected hand enumeration:
   hierarchy `['MAJOR_ROAD', 'MINOR_ROAD', 'NONE']`;
   major set `['primary', 'secondary', 'trunk']`;
   minor set `['cycleway', 'footway', 'residential', 'service', 'steps', 'tertiary', 'unclassified']`.

4. BP7 ID namespace placeholder remains available:
   sentinel inventory status `LOCKED`; BP7 range `1500..1599`; `placeholder=True`; status `PLACEHOLDER; final size locks at Task 7 halt`.

5. Cached sub-C Singapore data exists:
   `tile_count=494`, first observed file `data/processed/sub_c/2026-04-15.0/singapore/tile=EPSG3414_i10_j10/features.parquet`.

6. BP1 highway values missing from sub-E grouping were surfaced:
   locked BP1 highway values are `['*', 'bridleway', 'busway', 'cycleway', 'footway', 'living_street', 'motorway', 'motorway_link', 'path', 'pedestrian', 'primary', 'primary_link', 'residential', 'road', 'secondary', 'secondary_link', 'service', 'steps', 'subway', 'tertiary', 'tertiary_link', 'track', 'trunk', 'trunk_link', 'unclassified']`;
   missing from sub-E grouping are `['*', 'bridleway', 'busway', 'living_street', 'motorway', 'motorway_link', 'path', 'pedestrian', 'primary_link', 'road', 'secondary_link', 'subway', 'tertiary_link', 'track', 'trunk_link']`;
   `motorway_present_in_grouping=False`.

## Boundary-Reference Vocab Proposal

Proposed BP7 vocab:

- Family: `bp7_boundary_ref`.
- Release: `2026-04-15.0`.
- 8 emitted tokens.
- IDs `1500..1507`.
- Reserved headroom `1508..1599` (92 slots).
- Direction order: `N`, `E`, `S`, `W`.
- Class set: `MAJOR_ROAD`, `MINOR_ROAD`.
- Multi-class collapse rule: `MAJOR_ROAD > MINOR_ROAD > NONE`.
- Non-emitting classes: `NONE`, `BOUNDARY_NOT_APPLICABLE`.

Token list:

| id | local_id | tag | direction | boundary_class |
|---:|---:|---|---|---|
| 1500 | 0 | `<bref_N_MAJOR>` | N | MAJOR_ROAD |
| 1501 | 1 | `<bref_E_MAJOR>` | E | MAJOR_ROAD |
| 1502 | 2 | `<bref_S_MAJOR>` | S | MAJOR_ROAD |
| 1503 | 3 | `<bref_W_MAJOR>` | W | MAJOR_ROAD |
| 1504 | 4 | `<bref_N_MINOR>` | N | MINOR_ROAD |
| 1505 | 5 | `<bref_E_MINOR>` | E | MINOR_ROAD |
| 1506 | 6 | `<bref_S_MINOR>` | S | MINOR_ROAD |
| 1507 | 7 | `<bref_W_MINOR>` | W | MINOR_ROAD |

YAML was written with `cfm.data.io.canonicalize_yaml(...)`.

## Inbound/Outbound Token-Count Confirmation

Spec §3.2 confirms Cases C and D use `<bref_dir_class>` as an inbound marker prepended before anchor. Cases B and D use `<bref_dir_class>` as the outbound edge marker replacing the final direction+magnitude pair. The token text is the same `<bref_dir_class>` shape in both positions.

Spec §3.7 defines exactly 8 composite BP7 tokens: 4 directions × 2 active classes. It does not define separate inbound-token and outbound-token families. Inbound/outbound distinction is position-carried, not distinct-token-carried.

Result: no BP7 expansion to 16 tokens is indicated by the spec text.

## Rotation Wrapper Result

sub-E returns `CellEdgeIds`, a dataclass with fields `.north`, `.south`, `.west`, `.east`.

sub-E documentation states stable iteration order is N/S/W/E, while BP7 vocab order is N/E/S/W. The sub-F wrapper does not zip or iterate `CellEdgeIds`; it maps fields explicitly:

```python
{
    "N": edge_ids.north,
    "E": edge_ids.east,
    "S": edge_ids.south,
    "W": edge_ids.west,
}
```

## Class-Mapping Evidence

BoundaryClass enum values:

| name | int | value |
|---|---:|---:|
| BOUNDARY_NOT_APPLICABLE | 0 | 0 |
| NONE | 1 | 1 |
| MAJOR_ROAD | 2 | 2 |
| MINOR_ROAD | 3 | 3 |

Hierarchy: `MAJOR_ROAD > MINOR_ROAD > NONE`.

Major grouping set: `primary`, `secondary`, `trunk`.

Minor grouping set: `cycleway`, `footway`, `residential`, `service`, `steps`, `tertiary`, `unclassified`.

## BP7 Class-Coverage Gap

The following locked BP1 `highway=*` values are absent from sub-E `load_class_grouping_map()`. Counts are the acceptable proxy requested in the dispatch prompt: `singapore_row_count` is cached Singapore `feature_class == 0` row count for that `class_raw`; `multi_tile_source_feature_count` is the number of distinct `source_feature_id` values for that class observed in more than one `tile=*` file. `source_feature_id` is available in cached sub-C Singapore parquet.

| highway value | singapore_row_count | multi_tile_source_feature_count |
|---|---:|---:|
| `*` | singapore_row_count=0 | multi_tile_source_feature_count=0 |
| `bridleway` | singapore_row_count=0 | multi_tile_source_feature_count=0 |
| `busway` | singapore_row_count=0 | multi_tile_source_feature_count=0 |
| `living_street` | singapore_row_count=448 | multi_tile_source_feature_count=17 |
| `motorway` | singapore_row_count=4929 | multi_tile_source_feature_count=337 |
| `motorway_link` | singapore_row_count=0 | multi_tile_source_feature_count=0 |
| `path` | singapore_row_count=3491 | multi_tile_source_feature_count=204 |
| `pedestrian` | singapore_row_count=579 | multi_tile_source_feature_count=33 |
| `primary_link` | singapore_row_count=0 | multi_tile_source_feature_count=0 |
| `road` | singapore_row_count=0 | multi_tile_source_feature_count=0 |
| `secondary_link` | singapore_row_count=0 | multi_tile_source_feature_count=0 |
| `subway` | singapore_row_count=4314 | multi_tile_source_feature_count=188 |
| `tertiary_link` | singapore_row_count=0 | multi_tile_source_feature_count=0 |
| `track` | singapore_row_count=3444 | multi_tile_source_feature_count=234 |
| `trunk_link` | singapore_row_count=0 | multi_tile_source_feature_count=0 |

Classification: REAL §9.6.1 cascade #9 against upstream composition, resolved locally in sub-F. The gap was WIDE as a raw absence list, but the correct resolution is per-value drivable-network semantics rather than blanket sub-E grouping extension.

Resolution mechanism: sub-F-local BP7 override composed after sub-E grouping. Values already covered by sub-E continue to defer to sub-E:

| source | highway values | BP7 class |
|---|---|---|
| sub-E grouping | `primary`, `secondary`, `trunk` | MAJOR_ROAD |
| sub-E grouping | `cycleway`, `footway`, `residential`, `service`, `steps`, `tertiary`, `unclassified` | MINOR_ROAD |

For values absent from sub-E grouping, sub-F now resolves all locked BP1 `highway=*` values explicitly:

| highway value | BP7 class | rationale |
|---|---|---|
| `motorway` | MAJOR_ROAD | Expressway / core drivable arterial; stitch across cells. |
| `living_street` | MINOR_ROAD | Vehicular but local. |
| `subway` | NONE | Deliberately non-emitting; not drivable-road continuity for AV routing. |
| `path` | NONE | Deliberately non-emitting; non-vehicular. |
| `track` | NONE | Deliberately non-emitting; non-vehicular for v1 AV routing. |
| `pedestrian` | NONE | Deliberately non-emitting; non-vehicular. |
| `motorway_link`, `primary_link`, `secondary_link`, `tertiary_link`, `trunk_link`, `bridleway`, `busway`, `road`, `*` | NONE | Scope-of-coverage-zero in Singapore cached data at Halt 7; retained as explicit NONE rather than omission. |

Test coverage: `test_sub_f_bp7_override_resolves_every_locked_highway_value_explicitly` asserts every locked BP1 `highway=*` value resolves to exactly one `BoundaryClass`; NONE is explicit, never a fallback-by-omission.

## Feature-Splitting Outcome

Report: `reports/sub_f_task_7_feature_splitting.yaml`.

Summary:

- `_status`: `PROPOSED - pending Halt 7 reviewer approval`.
- `tile_count=494`.
- `row_count=862436`.
- `geometry_type_counts`: `LineString=303302`, `MultiLineString=5690`, `MultiPolygon=4083`, `Point=149666`, `Polygon=399695`.
- `encoded_geometry_type_counts`: `0=149666`, `1=303302`, `2=399695`, `4=5690`, `5=4083`.
- `road_multiline_decoded_count=5605`.
- `road_geometry_type_4_count=5605`.
- `road_multiline_count=5605`.
- `road_multiline_part_edge_buckets`: `same_cell_edge_multi_part=5206`, `different_cell_edges=352`, `no_multi_part_boundary_interaction=47`, `mergeable_artifact=0`.
- `outcome=branched_multi_row_present`.
- `recommendation=§9.6.1 cascade candidate; do not add multi-outbound grammar in Task 7`.

Example rows:

| tile | source_feature_id | geometry_type | encoded_geometry_type | part_count |
|---|---|---|---:|---:|
| tile=EPSG3414_i10_j10 | 920f98c1-98a6-421e-8374-02ba48b4c584 | MultiLineString | 4 | 2 |
| tile=EPSG3414_i10_j11 | a6587721-6f44-4feb-9018-e28820c993ea | MultiLineString | 4 | 2 |
| tile=EPSG3414_i10_j11 | 391e9951-e9d4-46cb-800f-0edc6ddc29af | MultiLineString | 4 | 2 |
| tile=EPSG3414_i10_j11 | e887926c-2035-45a7-9c62-b1c2d4320a45 | MultiLineString | 4 | 2 |
| tile=EPSG3414_i10_j11 | 47b08875-0726-4a99-b06d-706d8e76c406 | MultiLineString | 4 | 2 |

Because `branched_multi_row_present` contradicts the Task 7 assumption, this report is BLOCKED pending reviewer classification of the §9.6.1 cascade. No multi-outbound grammar case was added in this task.

Same-cell-edge examples from the decomposition:

| tile | cell | class_raw | source_feature_id | part_count | repeated_edges | part_edges |
|---|---|---|---|---:|---|---|
| tile=EPSG3414_i10_j10 | (3,4) | service | 920f98c1-98a6-421e-8374-02ba48b4c584 | 2 | E | `[['E'], ['E']]` |
| tile=EPSG3414_i10_j11 | (1,6) | service | a6587721-6f44-4feb-9018-e28820c993ea | 2 | E | `[['E', 'S'], ['E']]` |
| tile=EPSG3414_i10_j11 | (2,5) | service | 391e9951-e9d4-46cb-800f-0edc6ddc29af | 2 | E | `[['E', 'W'], ['E']]` |
| tile=EPSG3414_i10_j11 | (4,7) | service | e887926c-2035-45a7-9c62-b1c2d4320a45 | 2 | N | `[['N'], ['E', 'N']]` |
| tile=EPSG3414_i10_j11 | (5,5) | service | 47b08875-0726-4a99-b06d-706d8e76c406 | 2 | W | `[['S', 'W'], ['W']]` |
| tile=EPSG3414_i10_j14 | (6,6) | service | a62be704-c350-4a1b-84e1-b3f23b706f86 | 2 | W | `[['W'], ['W']]` |
| tile=EPSG3414_i10_j14 | (6,7) | service | 6e3ba62e-fee9-4f51-bbde-0ab4dedd21c2 | 2 | W | `[['S', 'W'], ['W']]` |
| tile=EPSG3414_i10_j14 | (7,5) | service | 2fd8dc54-0399-40ab-b2ff-6ec829a016a1 | 2 | E | `[['E'], ['E']]` |
| tile=EPSG3414_i10_j14 | (7,5) | service | 6cf613eb-2926-40a0-b054-18b3c61647a6 | 2 | N | `[['N'], ['N']]` |
| tile=EPSG3414_i10_j14 | (7,6) | footway | dd6d65ba-2aef-4dfc-a100-950c19a3512f | 2 | N | `[['E', 'N'], ['N']]` |

Cascade B classification surface: bucket (i) is large (`same_cell_edge_multi_part=5206`), so the single-row-per-branch assumption is materially false for cached Singapore. Reviewer classification is required before BP7 can lock.

## Manifest Obligation Note

Task 6 provisional manifest remains provisional. After Halt 7 approval, the sub-F manifest must add this BP7 boundary-reference vocab source and recompute the final manifest SHA at sub-F close.

## Verification

Commands run:

```text
uv run python scripts/sub_f/verify_sub_c_feature_splitting.py --sub-c-region-dir data/processed/sub_c/2026-04-15.0/singapore
```

Result: passed, rewrote `reports/sub_f_task_7_feature_splitting.yaml`.

```text
uv run pytest tests/data/sub_f/test_rotation.py -v
```

Result: `11 passed` before amendment; `16 passed` after adding cascade A override and cascade B bucket tests.

```text
uv run pytest tests/data/sub_f/test_manifest.py tests/data/sub_f/test_provenance.py -v
```

Result: `14 passed`.

```text
git diff --check
```

Result: passed with no output.

Note: sandboxed `uv` runs failed with:
`failed to open file /Users/umaraslam/.cache/uv/sdists-v9/.git: Operation not permitted`.
Verification therefore requires approved access to the external uv cache.

## Reviewer Ratification Checklist

- Approve or reject BP7 vocab slot list and ID range.
- Ratify cascade #9 sub-F-local BP7 override for locked highway values missing from sub-E grouping.
- Classify whether sub-C `branched_multi_row_present` requires a §3.7 multi-outbound grammar cascade.
- Decide whether BP7 placeholder can transition to LOCKED in a continuation.

## Section 10.5 Telemetry

- implementer-time-to-data-surface: same-session implementation and verification on 2026-05-28; no separate wall-clock timer was instrumented.
- reviewer-time-to-approval: pending.
- reviewer-time-to-rejection-or-cascade: pending.

## Halt Decision

Status: BLOCKED.

Blocking issues:

- Feature-splitting verification surfaced `branched_multi_row_present` with `road_multiline_count=5605` and `same_cell_edge_multi_part=5206`, which is a §9.6.1 cascade candidate against the Task 7 single-row-per-branch assumption and must be reviewer-classified before BP7 lock continuation.
- BP7 class coverage cascade #9 has been resolved locally in sub-F via explicit drivable-continuity override, pending reviewer ratification.

Do not proceed past Halt 7 until reviewer approval and cascade resolution.

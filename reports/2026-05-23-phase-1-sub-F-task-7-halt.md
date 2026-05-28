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

Cascade A (sub-E grouping under-covering locked BP1 highway vocab) was provisionally explored with a sub-F-local BP7 override, but that resolution is NOT ratified. A later reviewer correction pulled back the unverified drivable-only premise. The BP7 purpose and derivation architecture must be classified before any class map locks. Cascade B (MultiLineString part-edge relationship) remains BLOCKED pending reviewer classification after the class-purpose decision.

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

Classification: REAL §9.6.1 cascade #9 against upstream composition. The gap was WIDE as a raw absence list. The sub-F-local drivable-network override below is PROVISIONAL ONLY and not ratified; it depends on the unresolved BP7 purpose question.

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

Test coverage currently includes `test_sub_f_bp7_override_resolves_every_locked_highway_value_explicitly`, but that test is provisional with the override. If BP7 purpose or derivation architecture changes, this test must be revised before BP7 lock.

## Addendum: BP7 Purpose + Derivation Contract Reclassification Surface

Reviewer correction: the prior drivable-only classification was reviewer-supplied input and was not grounded in the pre-existing §3.7 text before implementation. Cascade #9 is therefore provisional pending BP7 purpose classification.

### Purpose text from spec

Pre-existing spec language before the cascade #9 insertion says:

- §1.1 lines 32-35: cells generate independently and "the boundary contracts ensure they fit together"; cross-cell coherence is via boundary-reference tokens to sub-E pre-derived contracts, not sequence concatenation.
- §1.2 line 41: sub-E boundary contracts are a locked upstream input.
- §1.4 lines 57-59: sub-F-v1 consumes sub-E `boundary_contract.parquet` as authoritative; token layer represents roads only for cross-cell references.
- §3.7 lines 279-298: BP7 defines 8 composite road-crossing tokens and class set `{MAJOR_ROAD, MINOR_ROAD}`; NONE is non-emitting.
- §8.1 line 786: BP7 structural check requires every emitted `<bref>` to match sub-E parquet for that cell edge.

Interpretation surface:

- The spec does clearly say BP7 is for roads / road-crossing tokens, not buildings or POIs.
- The spec does NOT, before the cascade #9 insertion, explicitly say "drivable AV route continuity only" versus "geometric continuity for all Overture transportation road-class rows."
- The spec DOES say sub-E boundary contracts are authoritative and emitted `<bref>` tokens must match sub-E parquet.

### Derivation contract from code

Current upstream code path:

- `src/cfm/data/sub_e/pipeline.py:304-326` filters sub-C features to `feature_class == road`, groups crossing `class_raw` values per edge, and calls `derive_boundary_class(class_raws)`.
- `src/cfm/data/sub_e/derivation.py:81-85` loads sub-E grouping and maps any class_raw absent from the grouping to `BoundaryClass.MINOR_ROAD`, not NONE.
- `src/cfm/data/sub_f/rotation.py:12-57` currently contains the provisional sub-F-local override from cascade #9, but no Task 8 writer exists yet, and no emitted `<bref>` code path currently calls it.

Architecture implication:

- The existing spec points to architecture (b): sub-F consumes sub-E's per-edge `BoundaryClass` from `boundary_contract.parquet` and tokenizes that class. Under architecture (b), a sub-F local `motorway -> MAJOR` override is incoherent unless sub-E also derives the corresponding edge as MAJOR; otherwise the Task 10 cross-reference check would fail.
- If reviewer chooses architecture (a), sub-F's map must become the authoritative complete map over all locked highway values, and §8.1 cross-reference semantics must be revised accordingly.

Important correction to earlier premise: missing values in sub-E grouping do NOT fall to NONE in sub-E. They fall to MINOR_ROAD by default when they appear in crossing `class_raws`. NONE is returned only when there are no road crossings on the edge.

### Subway attribute dump

Five cached Singapore rows with `feature_class=0` and `class_raw="subway"`:

| row | tile | source_feature_id | cell | subtype_raw | categories_primary | geometry_type | length_m | bounds | coords_first_last |
|---:|---|---|---|---|---|---|---:|---|---|
| 1 | tile=EPSG3414_i10_j16 | 9c5215ef-3228-4196-9577-783e415b2d95 | (0,5) | `None` | `None` | LineString | 144.296 | (169.107, 130.511, 250.0, 250.0) | [(169.107, 250.0), (250.0, 130.511)] |
| 2 | tile=EPSG3414_i10_j16 | ea6fbb6a-c591-42ea-8dd8-80f9db194174 | (0,5) | `None` | `None` | LineString | 160.138 | (160.243, 117.38, 250.0, 250.0) | [(250.0, 117.38), (160.243, 250.0)] |
| 3 | tile=EPSG3414_i10_j16 | 9c5215ef-3228-4196-9577-783e415b2d95 | (0,6) | `None` | `None` | LineString | 300.53 | (2.469, 0.0, 169.107, 250.0) | [(2.469, 250.0), (169.107, 0.0)] |
| 4 | tile=EPSG3414_i10_j16 | ea6fbb6a-c591-42ea-8dd8-80f9db194174 | (0,6) | `None` | `None` | LineString | 290.249 | (0.0, 0.0, 160.243, 241.946) | [(160.243, 0.0), (0.0, 241.946)] |
| 5 | tile=EPSG3414_i10_j16 | 9c5215ef-3228-4196-9577-783e415b2d95 | (0,7) | `None` | `None` | LineString | 4.956 | (0.0, 0.0, 2.469, 4.297) | [(0.0, 4.297), (2.469, 0.0)] |

These rows are transportation `class_raw="subway"` with no subtype/category refinement in cached sub-C output. The cache alone does not prove pedestrian underpass versus rail/road tunnel; it only proves they are linear transportation features split across cells.

### Cascade B gate

Cascade B remains real as a raw feature-shape mismatch (`same_cell_edge_multi_part=5206`), but its v1 severity depends on the corrected BP7 class architecture/purpose. BoundaryClass breakdown of those 5206 rows is intentionally deferred until the purpose and class-source decision is ratified.

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
- Classify BP7 purpose: drivable-routing continuity vs geometric continuity for all road-class rows.
- Ratify derivation architecture: sub-F-local class map vs sub-E `boundary_contract.parquet` authoritative class.
- Decide subway after the attribute dump.
- Then ratify or replace the provisional cascade #9 override.
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
- BP7 class coverage cascade #9 is provisional. Purpose, derivation architecture, and subway semantics must be classified before any class map locks.

Do not proceed past Halt 7 until reviewer approval and cascade resolution.

# Phase 1 Sub-F Task 7 Halt 7 Report

Status: DONE.

Branch: `phase-1-sub-F-micro-tokenizer`

Close commit: Task 7 close continuation commit.

## Scope

Implemented the Halt 7 surface and close continuation:

- Added proposed BP7 boundary-reference vocab at `configs/sub_f/boundary_reference_vocab.yaml`.
- Added sub-F rotation wrapper at `src/cfm/data/sub_f/rotation.py`.
- Added sub-C feature-splitting verification script at `scripts/sub_f/verify_sub_c_feature_splitting.py`.
- Added Task 7 tests at `tests/data/sub_f/test_rotation.py`.
- Added feature-splitting report at `reports/sub_f_task_7_feature_splitting.yaml`.
- Locked `configs/sub_f/boundary_reference_vocab.yaml` at 8 BP7 tokens, IDs 1500..1507.
- Locked `configs/sub_f/sentinel_inventory.yaml` BP7 block at 1500..1599 with 8 used and 92 reserved.

No writer/orchestrator work was added. No push or PR was performed.

Halt 7 approval classification: sub-C Singapore feature-splitting verification surfaced branched/multi-part road rows (`road_multiline_count=5605`, outcome `branched_multi_row_present`), but under architecture (b), sub-F never sees parts; it tokenizes sub-E's per-edge `BoundaryClass` result. No sub-F §3.7 multi-outbound grammar change ships in v1.

Cascade A (sub-E grouping under-covering locked BP1 highway vocab) was initially explored with a sub-F-local BP7 override. That resolution is now DISCARDED, not revised: the architecture check confirms sub-F consumes sub-E `boundary_contract.parquet` as authoritative and has no local class-override authority. The real cascade #9 is sub-E's MINOR-default behavior: omitted-but-present values such as `motorway`, `path`, `pedestrian`, `track`, and `subway` emit as MINOR_ROAD under sub-E, not NONE. This is accepted for sub-F-v1 as an inherited sub-E limitation and tracked as a sub-E-v2 candidate.

## Audit Step Outcomes

1. sub-E rotation return shape matched the expected contract:
   type name `CellEdgeIds`; dataclass fields `['east', 'north', 'south', 'west']`; sample `cell_to_edge_ids(3, 5)` exposed `.north`, `.south`, `.west`, `.east`.

2. sub-E `BoundaryClass` values matched the expected hand enumeration:
   `[('BOUNDARY_NOT_APPLICABLE', 0, 0), ('NONE', 1, 1), ('MAJOR_ROAD', 2, 2), ('MINOR_ROAD', 3, 3)]`.

3. sub-E hierarchy and class grouping matched the expected hand enumeration:
   hierarchy `['MAJOR_ROAD', 'MINOR_ROAD', 'NONE']`;
   major set `['primary', 'secondary', 'trunk']`;
   minor set `['cycleway', 'footway', 'residential', 'service', 'steps', 'tertiary', 'unclassified']`.

4. BP7 ID namespace is locked:
   sentinel inventory status `LOCKED`; BP7 range `1500..1599`; `placeholder=False`; `used_count=8`; `reserved_count=92`; status `LOCKED at Halt 7 approval`.

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

Classification: REAL §9.6.1 cascade #9 against upstream composition, but the earlier sub-F-local override resolution is DISCARDED. Architecture (b) applies: sub-F consumes sub-E `boundary_contract.parquet` BoundaryClass values verbatim. sub-F must not re-derive or override boundary class from `highway=*`. Halt 7 decision: ACCEPT sub-E contract for sub-F-v1; BP7 locks.

Values already covered by sub-E grouping:

| source | highway values | BP7 class |
|---|---|---|
| sub-E grouping | `primary`, `secondary`, `trunk` | MAJOR_ROAD |
| sub-E grouping | `cycleway`, `footway`, `residential`, `service`, `steps`, `tertiary`, `unclassified` | MINOR_ROAD |

For values absent from sub-E grouping, sub-E's derivation does not fall to NONE. `src/cfm/data/sub_e/derivation.py:81-85` maps any present but unmapped `class_raw` to `BoundaryClass.MINOR_ROAD`; NONE is returned only when the edge has no road crossings.

| highway value group | sub-E v1 behavior | consequence for sub-F-v1 |
|---|---|---|
| `motorway` | MINOR_ROAD default | Expressway connectivity is emitted but under-tiered as MINOR, not lost. |
| `living_street` | MINOR_ROAD default | Emits as MINOR; likely harmless if accepted for v1. |
| `subway`, `path`, `track`, `pedestrian` | MINOR_ROAD default | Non-vehicular or ambiguous ways over-emit as MINOR under sub-E's road-class contract. |
| `motorway_link`, `primary_link`, `secondary_link`, `tertiary_link`, `trunk_link`, `bridleway`, `busway`, `road`, `*` | MINOR_ROAD default if present | Scope-zero in cached Singapore at Halt 7, so no v1 Singapore emission observed. |

Final disposition: sub-E's MINOR-default is accepted for sub-F-v1 as a known upstream limitation. `motorway` can be under-tiered as MINOR when crossing alone, and non-vehicular or ambiguous ways can over-emit as MINOR. sub-F remains correct by faithfully tokenizing sub-E's authoritative class. The BP7 vocab shape is locked: 8 tokens, IDs 1500..1507, inbound position-carried.

## Addendum: BP7 Purpose + Derivation Contract Reclassification Surface

Reviewer correction: the prior drivable-only classification was reviewer-supplied input and was not grounded in the pre-existing §3.7 text before implementation. A second correction found the prior NONE-default premise was false: sub-E defaults omitted but present values to MINOR_ROAD. Cascade #9 is therefore a sub-E authority/derivation cascade, not a sub-F local-map cascade.

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
- `src/cfm/data/sub_f/rotation.py` now contains only the N/E/S/W wrapper over sub-E edge ids. The discarded sub-F-local override was removed.

Architecture implication:

- The existing spec points to architecture (b): sub-F consumes sub-E's per-edge `BoundaryClass` from `boundary_contract.parquet` and tokenizes that class. Under architecture (b), a sub-F local `motorway -> MAJOR` override is incoherent unless sub-E also derives the corresponding edge as MAJOR; otherwise the Task 10 cross-reference check would fail.
- Architecture (a) is discarded for v1 unless the spec is deliberately revised to make sub-F own boundary class derivation. No such revision is currently authorized.

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

### Sub-E boundary_contract surface

Local cache availability check:

- `data/processed/sub_e` is absent in this workspace.
- No `boundary_contract.parquet` files were found under the repository at Halt 7 reclassification time.
- Therefore no real parquet sample for a motorway edge or a same-edge MultiLineString edge can be pasted without first restoring or regenerating sub-E output. The emission behavior below is code-inferred, not data-verified.

Schema from code (`src/cfm/data/sub_e/writer.py:34-45`):

| column | type | nullable |
|---|---|---|
| `slot_kind` | int8 | no |
| `slot_index` | int16 | no |
| `lower_cell_i` | int8 | no |
| `lower_cell_j` | int8 | no |
| `axis` | int8 | no |
| `scope_marker` | int8 | no |
| `boundary_class_enum` | int16 | yes |

Writer invariant: exactly 144 rows per tile, 112 internal + 32 external, sorted by `(slot_kind, slot_index)`.

Code-inferred emission behavior (`src/cfm/data/sub_e/pipeline.py:304-326`):

- sub-E filters cached sub-C rows to `feature_class == road`.
- sub-E groups all crossing `class_raw` values by edge.
- each active internal edge gets one `boundary_class_enum`, derived by `derive_boundary_class(class_raws)`.
- a motorway-only edge therefore emits `MINOR_ROAD` under the current sub-E default, unless the same edge also has a grouped MAJOR value (`primary`, `secondary`, `trunk`).
- same-edge MultiLineString parts collapse into the edge's single class derivation surface; the code does not emit one boundary row per crossing part.

### Cascade B gate

Cascade B final disposition: sub-E emits one row per edge, so same-edge MultiLineString parts collapse to the edge's single class in sub-E's contract. sub-F tokenizes one `<bref>` per edge and does not need a §3.7 multi-outbound grammar change in v1. Any multi-part fidelity loss is inherited from sub-E's per-edge contract and is tracked as a sub-E-v2 candidate. Real parquet sampling remains verification debt until sub-E output is restored or regenerated.

## Feature-Splitting Outcome

Report: `reports/sub_f_task_7_feature_splitting.yaml`.

Summary:

- `_status`: `LOCKED - Halt 7 approved`.
- `tile_count=494`.
- `row_count=862436`.
- `geometry_type_counts`: `LineString=303302`, `MultiLineString=5690`, `MultiPolygon=4083`, `Point=149666`, `Polygon=399695`.
- `encoded_geometry_type_counts`: `0=149666`, `1=303302`, `2=399695`, `4=5690`, `5=4083`.
- `road_multiline_decoded_count=5605`.
- `road_geometry_type_4_count=5605`.
- `road_multiline_count=5605`.
- `road_multiline_part_edge_buckets`: `same_cell_edge_multi_part=5206`, `different_cell_edges=352`, `no_multi_part_boundary_interaction=47`, `mergeable_artifact=0`.
- `outcome=branched_multi_row_present`.
- `recommendation=Accepted v1 as sub-E-inherited per-edge collapse; no sub-F multi-outbound grammar change`.

Example rows:

| tile | source_feature_id | geometry_type | encoded_geometry_type | part_count |
|---|---|---|---:|---:|
| tile=EPSG3414_i10_j10 | 920f98c1-98a6-421e-8374-02ba48b4c584 | MultiLineString | 4 | 2 |
| tile=EPSG3414_i10_j11 | a6587721-6f44-4feb-9018-e28820c993ea | MultiLineString | 4 | 2 |
| tile=EPSG3414_i10_j11 | 391e9951-e9d4-46cb-800f-0edc6ddc29af | MultiLineString | 4 | 2 |
| tile=EPSG3414_i10_j11 | e887926c-2035-45a7-9c62-b1c2d4320a45 | MultiLineString | 4 | 2 |
| tile=EPSG3414_i10_j11 | 47b08875-0726-4a99-b06d-706d8e76c406 | MultiLineString | 4 | 2 |

Because sub-F consumes sub-E's per-edge contract, `branched_multi_row_present` does not require a sub-F multi-outbound grammar case in v1. No multi-outbound grammar case was added in this task.

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

Cascade B classification surface: bucket (i) is large (`same_cell_edge_multi_part=5206`), so the single-row-per-branch assumption is materially false for cached Singapore. Halt 7 accepts this as sub-E-inherited per-edge collapse; no sub-F grammar revision blocks BP7.

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

Result: `11 passed` before amendment; `16 passed` after adding the discarded cascade A override and cascade B bucket tests; current override-removed surface: `16 passed`.

```text
uv run pytest tests/data/sub_f/test_manifest.py tests/data/sub_f/test_provenance.py -v
```

Result: `14 passed` before and after the override-removal surface.

```text
./.venv/bin/python -m pytest tests/data/sub_f/test_vocab.py tests/data/sub_f/test_encoder.py -q
```

Result: `63 passed` after BP7 lock status updates.

```text
./.venv/bin/ruff check src/cfm/data/sub_f/rotation.py scripts/sub_f/verify_sub_c_feature_splitting.py tests/data/sub_f/test_rotation.py tests/data/sub_f/test_manifest.py tests/data/sub_f/test_vocab.py tests/data/sub_f/test_encoder.py
```

Result: passed with `All checks passed!`.

```text
git diff --check
```

Result: passed with no output.

Note: sandboxed `uv` runs failed with:
`failed to open file /Users/umaraslam/.cache/uv/sdists-v9/.git: Operation not permitted`.
Verification therefore requires approved access to the external uv cache.

## Reviewer Ratification Checklist

- BP7 vocab slot list and ID range: APPROVED, 8 tokens at 1500..1507.
- Architecture (b): RATIFIED; sub-F consumes sub-E `boundary_contract.parquet` authoritative class verbatim.
- sub-E MINOR-default behavior: ACCEPTED for sub-F-v1; documented as sub-E-v2 candidate.
- Absent local sub-E cache: ACCEPTED for this halt surface; code-inferred caveat documented.
- Same-edge MultiLineString per-edge collapse: ACCEPTED for sub-F-v1 as sub-E-inherited; documented as sub-E-v2 candidate.
- BP7 block transition: LOCKED in `configs/sub_f/sentinel_inventory.yaml`.

## Section 10.5 Telemetry

- implementer-time-to-data-surface: same-session implementation and verification on 2026-05-28; no separate wall-clock timer was instrumented.
- reviewer-time-to-approval: approved after reclassification surface on 2026-05-28.
- reviewer-time-to-rejection-or-cascade: Halt 7 surfaced cascade #9 + cascade B; both accepted as sub-E-inherited v1 limitations.

## Halt Decision

Status: DONE.

Final decisions:

- Feature-splitting verification surfaced `branched_multi_row_present` with `road_multiline_count=5605` and `same_cell_edge_multi_part=5206`; accepted for sub-F-v1 as sub-E-inherited per-edge collapse. No sub-F multi-outbound grammar change.
- BP7 class coverage cascade #9 is a sub-E MINOR-default/tiering limitation, not sub-F-resolvable under architecture (b); accepted for sub-F-v1 with sub-E-v2 follow-up.
- Local sub-E `boundary_contract.parquet` cache is absent; real parquet samples for motorway and same-edge MultiLineString edges remain verification debt when sub-E output is regenerated/restored.

Task 7 is closed. Downstream manifest completion at sub-F close must add BP7 vocab source and recompute final manifest SHA.

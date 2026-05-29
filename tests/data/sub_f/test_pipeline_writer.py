"""Sub-F T8.8 per-tile encode_tile orchestrator integration tests.

Covers:
  - encode_tile produces 64 rows; only populated cells have non-zero feature_count.
  - Case A polygon (building) round-trip: encode_tile -> cells.parquet ->
    decode_feature -> L_inf <= 4.8m.
  - Case B structural bref: road endpoint on active East edge -> <bref_E_MAJOR>
    token present in cell token_sequence at correct position.
  - Case C structural bref: road entry from active South edge -> <bref_S_MINOR>
    token present.
  - Real sub-E integration test is @pytest.mark.skip pending sub-E cache
    regeneration (BP7 verification debt; see close-checklist).

SYNTHETIC FIXTURE DISCIPLINE:
  The sub-E fixture uses _make_full_tile_rows() from test_boundary_contract.py
  (reused inline as _make_full_tile_contract_rows below to keep this module
  self-contained). All 144 rows, all non-active by default; overrides activate
  specific edges per test. Rationale: load_boundary_contract enforces strict
  source-derived 7-column schema + 144-row count; the 3-column fixture in the
  original plan would raise SubEContractViolation (CORRECTION 1 in T8.8 spec).

  Sub-C features.parquet is synthetic: minimum required columns per sub-C
  _FEATURES_SCHEMA (src/cfm/data/sub_c/io.py:40-58). Full 15-column schema
  needed because pipeline_writer reads with pq.ParquetFile().read() which
  returns all columns. We include all required columns with nulls where
  pipeline_writer doesn't consume them.

BREF TOKEN IDs (verified against vocab_tag_to_id() at T8.8 write time):
  <bref_E_MAJOR> = 1501
  <bref_S_MINOR> = 1506  (verified below; see _BREF_S_MINOR_ID)
"""

from __future__ import annotations

import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely.geometry import LineString, Polygon
from shapely.wkb import dumps as wkb_dumps

# ------ constants ------------------------------------------------------------

_L_INF_THRESHOLD_M = 4.8  # BP2 position lock; do NOT relax without re-running Halt 2.

# Structural sentinels — must match encoder._FEATURE_TOKEN_ID / _FEATURE_END_TOKEN_ID.
_FEATURE_TOKEN_ID = 509
_FEATURE_END_TOKEN_ID = 510

# BP7 bref token IDs — verified against vocab_tag_to_id() for T8.8.
# <bref_E_MAJOR> id verified inline below in test assertions.
# <bref_S_MINOR> id verified inline below in test assertions.
# These are NOT asserted as constants here; tests resolve via vocab_tag_to_id()
# so a future vocab relocation surfaces as a test failure, not a silent pass.

# ------ sub-E 144-row synthetic fixture helpers ------------------------------

# Sub-E parquet schema — SOURCE-DERIVED (bit-for-bit match against
# src/cfm/data/sub_e/writer.py:38-48 _BOUNDARY_CONTRACT_SCHEMA).
# Replicating from test_boundary_contract.py (_SUB_E_SCHEMA) for self-contained
# test module. If sub-E schema drifts, both files will need updating.
_SUB_E_SCHEMA = pa.schema(
    [
        pa.field("slot_kind", pa.int8(), nullable=False),
        pa.field("slot_index", pa.int16(), nullable=False),
        pa.field("lower_cell_i", pa.int8(), nullable=False),
        pa.field("lower_cell_j", pa.int8(), nullable=False),
        pa.field("axis", pa.int8(), nullable=False),
        pa.field("scope_marker", pa.int8(), nullable=False),
        pa.field("boundary_class_enum", pa.int16(), nullable=True),
    ]
)


def _make_full_tile_contract_rows(
    overrides: dict[tuple[int, int, int, int], dict] | None = None,
) -> list[dict]:
    """Build 144 well-formed sub-E contract rows for one tile (112 INTERNAL + 32 EXTERNAL).

    All rows default to non-active (scope_marker=1, boundary_class_enum=None).
    Override specific edges via (slot_kind, lower_cell_i, lower_cell_j, axis)
    -> field-update dict.

    Source: _make_full_tile_rows() in tests/data/sub_f/test_boundary_contract.py.
    Replicated here for self-contained test module.
    """
    from cfm.data.sub_e.rotation import EdgeKind, cell_to_edge_ids

    seen_internal: set[tuple[int, int, int]] = set()
    rows: list[dict] = []
    internal_slot_idx = 0
    external_slot_idx = 0
    overrides = overrides or {}

    for cell_i in range(8):
        for cell_j in range(8):
            edges = cell_to_edge_ids(cell_i, cell_j)
            for edge in (edges.north, edges.south, edges.west, edges.east):
                lower_i, lower_j, axis, kind = edge
                if kind is EdgeKind.INTERNAL:
                    key = (lower_i, lower_j, axis)
                    if key in seen_internal:
                        continue
                    seen_internal.add(key)
                    slot_kind = 1
                    slot_index = internal_slot_idx
                    internal_slot_idx += 1
                else:
                    slot_kind = 2
                    slot_index = external_slot_idx
                    external_slot_idx += 1
                row = {
                    "slot_kind": slot_kind,
                    "slot_index": slot_index,
                    "lower_cell_i": lower_i,
                    "lower_cell_j": lower_j,
                    "axis": axis,
                    "scope_marker": 1,  # non-active default
                    "boundary_class_enum": None,
                }
                ov = overrides.get((slot_kind, lower_i, lower_j, axis))
                if ov:
                    row.update(ov)
                rows.append(row)
    return rows


def _write_sub_e_parquet(path: Path, overrides: dict | None = None) -> None:
    """Write a synthetic 144-row sub-E boundary_contract.parquet."""
    rows = _make_full_tile_contract_rows(overrides)
    table = pa.Table.from_pylist(rows, schema=_SUB_E_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


# ------ sub-C features.parquet synthetic fixture helpers --------------------

# Sub-C features.parquet schema — from src/cfm/data/sub_c/io.py:40-58
# _FEATURES_SCHEMA. We replicate it here to build synthetic fixtures.
_SUB_C_FEATURES_SCHEMA = pa.schema(
    [
        pa.field("cell_i", pa.int8()),
        pa.field("cell_j", pa.int8()),
        pa.field("feature_class", pa.int8()),
        pa.field("source_feature_id", pa.string()),
        pa.field("geometry", pa.binary()),
        pa.field("geometry_type", pa.int8()),
        pa.field("bbox_min_x", pa.float64()),
        pa.field("bbox_min_y", pa.float64()),
        pa.field("bbox_max_x", pa.float64()),
        pa.field("bbox_max_y", pa.float64()),
        pa.field("class_raw", pa.string()),
        pa.field("subtype_raw", pa.string()),
        pa.field("categories_primary", pa.string()),
        pa.field("categories_alternate", pa.list_(pa.string())),
        pa.field("sea_overlap_fraction", pa.float64()),
    ]
)


def _feature_row(
    cell_i: int,
    cell_j: int,
    feature_class: int,
    geometry,  # shapely geometry
    class_raw: str | None,
    source_feature_id: str = "test-0",
) -> dict:
    """Build one sub-C features.parquet row with minimal required fields."""
    geom_type_map = {"Point": 0, "LineString": 1, "Polygon": 2}
    gt = geom_type_map.get(geometry.geom_type, 0)
    wkb = wkb_dumps(geometry, include_srid=False)
    bounds = geometry.bounds  # (minx, miny, maxx, maxy)
    return {
        "cell_i": cell_i,
        "cell_j": cell_j,
        "feature_class": feature_class,
        "source_feature_id": source_feature_id,
        "geometry": wkb,
        "geometry_type": gt,
        "bbox_min_x": bounds[0],
        "bbox_min_y": bounds[1],
        "bbox_max_x": bounds[2],
        "bbox_max_y": bounds[3],
        "class_raw": class_raw,
        "subtype_raw": None,
        "categories_primary": None,
        "categories_alternate": None,
        "sea_overlap_fraction": 0.0,
    }


def _write_sub_c_parquet(path: Path, rows: list[dict]) -> None:
    """Write a synthetic sub-C features.parquet."""
    table = pa.Table.from_pylist(rows, schema=_SUB_C_FEATURES_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


# ------ round-trip L_inf helper (vertex-count-aware, mirrors test_decoder.py) ---


def _source_vertex_l_inf(
    source_coords: list[tuple[float, float]], decoded_coords: list[tuple[float, float]]
) -> float:
    """Vertex-count-aware round-trip L_inf (same pattern as test_decoder.py).

    Maps each SOURCE vertex to its decoded counterpart via cumulative
    chunked_segment_pairs (spec §3.5 chunking may insert collinear vertices
    on long segments; those are admitted per §3.8 and skipped here).
    """
    from cfm.data.sub_f.token_cost import chunked_segment_pairs

    cum = 0
    max_linf = max(
        abs(source_coords[0][0] - decoded_coords[0][0]),
        abs(source_coords[0][1] - decoded_coords[0][1]),
    )
    for k in range(1, len(source_coords)):
        seg = math.hypot(
            source_coords[k][0] - source_coords[k - 1][0],
            source_coords[k][1] - source_coords[k - 1][1],
        )
        cum += chunked_segment_pairs(seg)
        d = decoded_coords[cum]
        max_linf = max(max_linf, abs(source_coords[k][0] - d[0]), abs(source_coords[k][1] - d[1]))
    return max_linf


# ------ helper: split token_sequence into per-feature chunks -----------------


def _split_into_feature_chunks(token_sequence: list[int]) -> list[list[int]]:
    """Split a cell token sequence into per-feature chunks.

    Each chunk includes its <feature>(509) ... <feature_end>(510) markers.
    """
    chunks: list[list[int]] = []
    i = 0
    while i < len(token_sequence):
        if token_sequence[i] == _FEATURE_TOKEN_ID:
            # Find matching <feature_end>
            j = i + 1
            while j < len(token_sequence) and token_sequence[j] != _FEATURE_END_TOKEN_ID:
                j += 1
            if j < len(token_sequence):
                chunks.append(token_sequence[i : j + 1])
                i = j + 1
            else:
                break  # malformed: no matching end
        else:
            i += 1
    return chunks


# ============================================================================
# Tests
# ============================================================================


def test_encode_tile_produces_64_rows(tmp_path: Path):
    """encode_tile must write exactly 64 rows (8x8 cell grid)."""
    from cfm.data.sub_f.pipeline_writer import encode_tile

    sub_c = tmp_path / "features.parquet"
    sub_e = tmp_path / "boundary_contract.parquet"
    out = tmp_path / "cells.parquet"

    _write_sub_c_parquet(sub_c, [])  # no features — all cells empty
    _write_sub_e_parquet(sub_e)

    result = encode_tile(sub_c, sub_e, out)
    assert result == out
    assert out.exists()

    table = pq.ParquetFile(out).read()
    assert table.num_rows == 64, f"expected 64 rows, got {table.num_rows}"


def test_encode_tile_empty_cells_have_zero_feature_count(tmp_path: Path):
    """Cells with no features must have feature_count=0 and token_sequence=[]."""
    from cfm.data.sub_f.pipeline_writer import encode_tile

    sub_c = tmp_path / "features.parquet"
    sub_e = tmp_path / "boundary_contract.parquet"
    out = tmp_path / "cells.parquet"

    # One feature in cell (2, 3); all other cells empty.
    building = Polygon([(50, 50), (50, 80), (80, 80), (80, 50), (50, 50)])
    rows = [_feature_row(2, 3, 1, building, "residential", "bldg-0")]
    _write_sub_c_parquet(sub_c, rows)
    _write_sub_e_parquet(sub_e)

    encode_tile(sub_c, sub_e, out)
    table = pq.ParquetFile(out).read()
    pyrows = table.to_pylist()

    populated = [r for r in pyrows if r["cell_i"] == 2 and r["cell_j"] == 3]
    assert len(populated) == 1
    assert populated[0]["feature_count"] == 1, (
        f"cell (2,3) should have feature_count=1, got {populated[0]['feature_count']}"
    )

    empty_rows = [r for r in pyrows if not (r["cell_i"] == 2 and r["cell_j"] == 3)]
    bad = [r for r in empty_rows if r["feature_count"] != 0]
    assert not bad, (
        f"{len(bad)} empty cells have non-zero feature_count: "
        f"{[(r['cell_i'], r['cell_j'], r['feature_count']) for r in bad[:5]]}"
    )
    bad_tokens = [r for r in empty_rows if r["token_sequence"] != []]
    assert not bad_tokens, f"{len(bad_tokens)} empty cells have non-empty token_sequence"


def test_encode_tile_case_a_polygon_round_trip(tmp_path: Path):
    """Case A: polygon (building=residential) encodes and decodes within L_inf threshold.

    Full geometric round-trip through encode_tile -> cells.parquet ->
    read back token_sequence -> decode_feature. Compares decoded vertices
    to CANONICAL coords (polygon is canonicalized inside encode_cell).

    Uses the vertex-count-aware L_inf metric (same as test_decoder.py) to
    handle spec §3.5 chunking on long segments.
    """
    from cfm.data.sub_f.decoder import decode_feature
    from cfm.data.sub_f.encoder import canonicalize_geometry
    from cfm.data.sub_f.pipeline_writer import encode_tile

    # Building polygon — fully inside cell (0,0), feature_class=1 -> "building=residential".
    source = Polygon([(30, 40), (30, 120), (100, 120), (100, 40), (30, 40)])
    canonical = canonicalize_geometry(source)
    canonical_coords = list(canonical.exterior.coords)

    sub_c = tmp_path / "features.parquet"
    sub_e = tmp_path / "boundary_contract.parquet"
    out = tmp_path / "cells.parquet"

    rows = [_feature_row(0, 0, 1, source, "residential", "bldg-1")]
    _write_sub_c_parquet(sub_c, rows)
    _write_sub_e_parquet(sub_e)  # all-NONE contract -> Case A

    encode_tile(sub_c, sub_e, out)

    # Read back the token_sequence for cell (0,0).
    table = pq.ParquetFile(out).read()
    pyrows = table.to_pylist()
    cell_00 = next(r for r in pyrows if r["cell_i"] == 0 and r["cell_j"] == 0)
    assert cell_00["feature_count"] == 1
    token_seq = list(cell_00["token_sequence"])
    assert len(token_seq) > 0

    # Split the sequence into per-feature chunks and decode the first one.
    chunks = _split_into_feature_chunks(token_seq)
    assert len(chunks) == 1, f"expected 1 feature chunk, got {len(chunks)}"

    decoded = decode_feature(chunks[0])
    assert "coordinates" in decoded
    decoded_coords = [tuple(p) for p in decoded["coordinates"]]

    # decoder returns LineString shape even for closed polygons (see decoder.py:144-155):
    # closed coord sequence with no bref tokens -> {"type": "LineString", ...}.
    # Caller reconstructs Polygon from coords when needed; test anchors this contract.
    assert decoded["type"] == "LineString", (
        f"decode_feature should return type='LineString' for a closed polygon "
        f"(no bref tokens); got {decoded['type']!r}"
    )
    # Compare against canonical exterior coords.
    l_inf = _source_vertex_l_inf(canonical_coords, decoded_coords)
    assert l_inf <= _L_INF_THRESHOLD_M, (
        f"Case A polygon round-trip L_inf {l_inf:.4f}m exceeds "
        f"threshold {_L_INF_THRESHOLD_M}m — BP2 position lock violated"
    )


def test_encode_tile_case_a_road_round_trip(tmp_path: Path):
    """Case A: open road (highway=residential) fully inside cell round-trips cleanly."""
    from cfm.data.sub_f.decoder import decode_feature
    from cfm.data.sub_f.encoder import canonicalize_geometry
    from cfm.data.sub_f.pipeline_writer import encode_tile

    # Road — stays well inside cell (1, 1) bounds (0..250), feature_class=0.
    source = LineString([(20.0, 30.0), (80.0, 90.0), (130.0, 150.0)])
    canonical = canonicalize_geometry(source)
    canonical_coords = list(canonical.coords)

    sub_c = tmp_path / "features.parquet"
    sub_e = tmp_path / "boundary_contract.parquet"
    out = tmp_path / "cells.parquet"

    rows = [_feature_row(1, 1, 0, source, "residential", "road-0")]
    _write_sub_c_parquet(sub_c, rows)
    _write_sub_e_parquet(sub_e)  # all-NONE -> Case A

    encode_tile(sub_c, sub_e, out)

    table = pq.ParquetFile(out).read()
    pyrows = table.to_pylist()
    cell_11 = next(r for r in pyrows if r["cell_i"] == 1 and r["cell_j"] == 1)
    assert cell_11["feature_count"] == 1
    token_seq = list(cell_11["token_sequence"])

    chunks = _split_into_feature_chunks(token_seq)
    assert len(chunks) == 1
    decoded = decode_feature(chunks[0])
    decoded_coords = [tuple(p) for p in decoded["coordinates"]]

    l_inf = _source_vertex_l_inf(canonical_coords, decoded_coords)
    assert l_inf <= _L_INF_THRESHOLD_M, (
        f"Case A road round-trip L_inf {l_inf:.4f}m exceeds {_L_INF_THRESHOLD_M}m"
    )


def test_encode_tile_case_b_east_edge_bref_token_present(tmp_path: Path):
    """Case B: road endpoint touching East edge of cell (0,0) on an active
    MAJOR_ROAD East edge -> <bref_E_MAJOR> token present in the cell's
    token_sequence at the outbound position (end of feature, before <feature_end>).

    Position of the bref vertex is NOT round-trip asserted per spec §1.4
    scope lock #1 + §13.1 v2-deferral. Structural-only assertion.

    Sub-E override: cell (0,0) East edge is (lower_i=0, lower_j=0, axis=0,
    INTERNAL -> slot_kind=1). Verified via cell_to_edge_ids(0,0).east at
    T8.8 write time.
    """
    from cfm.data.sub_f.boundary_contract import resolve_bref_tag
    from cfm.data.sub_f.pipeline_writer import encode_tile
    from cfm.data.sub_f.vocab import vocab_tag_to_id

    tag_to_id = vocab_tag_to_id()
    bref_e_major_tag = resolve_bref_tag("E", "MAJOR_ROAD")
    assert bref_e_major_tag == "<bref_E_MAJOR>"
    bref_e_major_id = tag_to_id[bref_e_major_tag]

    # Road: starts interior, endpoint exactly on East edge (x=250).
    # feature_class=0 -> "highway=residential". class_raw="residential" resolves
    # directly in BP1 vocab.
    road = LineString([(50.0, 100.0), (200.0, 100.0), (250.0, 100.0)])

    sub_c = tmp_path / "features.parquet"
    sub_e = tmp_path / "boundary_contract.parquet"
    out = tmp_path / "cells.parquet"

    rows = [_feature_row(0, 0, 0, road, "residential", "road-b")]
    _write_sub_c_parquet(sub_c, rows)

    # Activate East edge of cell (0,0): slot_kind=1, lower_i=0, lower_j=0, axis=0
    # -> MAJOR_ROAD (boundary_class_enum=2, scope_marker=0).
    _write_sub_e_parquet(
        sub_e,
        overrides={(1, 0, 0, 0): {"scope_marker": 0, "boundary_class_enum": 2}},
    )

    encode_tile(sub_c, sub_e, out)

    table = pq.ParquetFile(out).read()
    pyrows = table.to_pylist()
    cell_00 = next(r for r in pyrows if r["cell_i"] == 0 and r["cell_j"] == 0)
    token_seq = list(cell_00["token_sequence"])
    assert cell_00["feature_count"] == 1

    # The <bref_E_MAJOR> token must be present in the sequence.
    assert bref_e_major_id in token_seq, (
        f"<bref_E_MAJOR> token (id={bref_e_major_id}) not found in token_sequence. "
        f"Full sequence: {token_seq}"
    )

    # Structural position check: bref token appears immediately before <feature_end>.
    chunks = _split_into_feature_chunks(token_seq)
    assert len(chunks) == 1
    chunk = chunks[0]
    # chunk[-1] == <feature_end>, chunk[-2] == <bref_E_MAJOR> (Case B outbound bref).
    assert chunk[-1] == _FEATURE_END_TOKEN_ID
    assert chunk[-2] == bref_e_major_id, (
        f"Case B: expected <bref_E_MAJOR> (id={bref_e_major_id}) at chunk[-2], "
        f"got {chunk[-2]}. Full chunk: {chunk}"
    )


def test_encode_tile_case_c_south_entry_bref_token_present(tmp_path: Path):
    """Case C: road entry from South edge of cell (0,0) on an active
    MINOR_ROAD South edge -> <bref_S_MINOR> token present as inbound bref
    (immediately after semantic tag token, before anchor).

    The South edge of cell (0,0) is at y=0 in cell-local coords. In sub-E:
    south edge of cell (0,0) is (lower_i=0, lower_j=0, axis=1, INTERNAL ->
    slot_kind=1). Verified via cell_to_edge_ids(0,0).south at T8.8 write time.

    Position of the bref vertex is NOT round-trip asserted per spec §1.4.
    Structural-only assertion.
    """
    from cfm.data.sub_f.boundary_contract import resolve_bref_tag
    from cfm.data.sub_f.pipeline_writer import encode_tile
    from cfm.data.sub_f.vocab import vocab_tag_to_id

    tag_to_id = vocab_tag_to_id()
    bref_s_minor_tag = resolve_bref_tag("S", "MINOR_ROAD")
    assert bref_s_minor_tag == "<bref_S_MINOR>"
    bref_s_minor_id = tag_to_id[bref_s_minor_tag]

    # Road: first vertex on South edge (y=0), continues into cell interior.
    road = LineString([(100.0, 0.0), (100.0, 80.0), (150.0, 150.0)])

    sub_c = tmp_path / "features.parquet"
    sub_e = tmp_path / "boundary_contract.parquet"
    out = tmp_path / "cells.parquet"

    rows = [_feature_row(0, 0, 0, road, "residential", "road-c")]
    _write_sub_c_parquet(sub_c, rows)

    # Activate South edge of cell (0,0): slot_kind=1, lower_i=0, lower_j=0, axis=1
    # -> MINOR_ROAD (boundary_class_enum=3, scope_marker=0).
    _write_sub_e_parquet(
        sub_e,
        overrides={(1, 0, 0, 1): {"scope_marker": 0, "boundary_class_enum": 3}},
    )

    encode_tile(sub_c, sub_e, out)

    table = pq.ParquetFile(out).read()
    pyrows = table.to_pylist()
    cell_00 = next(r for r in pyrows if r["cell_i"] == 0 and r["cell_j"] == 0)
    token_seq = list(cell_00["token_sequence"])
    assert cell_00["feature_count"] == 1

    # The <bref_S_MINOR> token must be present.
    assert bref_s_minor_id in token_seq, (
        f"<bref_S_MINOR> token (id={bref_s_minor_id}) not found in token_sequence. "
        f"Full sequence: {token_seq}"
    )

    # Structural position check: inbound bref appears at chunk[2] (after <feature>=509
    # and semantic_tag_id, before anchor).
    chunks = _split_into_feature_chunks(token_seq)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk[0] == _FEATURE_TOKEN_ID
    # chunk[1] = semantic_tag_id, chunk[2] = inbound bref (Case C).
    assert chunk[2] == bref_s_minor_id, (
        f"Case C: expected <bref_S_MINOR> (id={bref_s_minor_id}) at chunk[2], "
        f"got {chunk[2]}. Full chunk: {chunk}"
    )


def test_encode_tile_case_d_inbound_outbound_bref(tmp_path: Path):
    """Case D: road whose entry vertex touches one active edge AND exit vertex
    touches a different active edge -> encoder emits BOTH an inbound and an
    outbound bref token in the same feature chunk.

    Cell (1,1) is used. Its West edge and East edge are both INTERNAL:
      West: lower_i=0, lower_j=1, axis=0 -> override key (1, 0, 1, 0)
      East: lower_i=1, lower_j=1, axis=0 -> override key (1, 1, 1, 0)
    Both activated as MAJOR_ROAD (boundary_class_enum=2) so the encoder
    resolves inbound=<bref_W_MAJOR> and outbound=<bref_E_MAJOR>.

    Road: coords[0] on West edge (x=0, cell-local), coords[-1] on East edge
    (x=250, cell-local) — genuine two-boundary-touching feature.

    Sub-E override keys verified via cell_to_edge_ids(1,1) at T8.8 write time.

    Position of bref vertices NOT round-trip asserted per spec §1.4 scope
    lock #1 + §13.1 v2-deferral. Structural-only assertions.
    """
    from cfm.data.sub_f.boundary_contract import resolve_bref_tag
    from cfm.data.sub_f.pipeline_writer import encode_tile
    from cfm.data.sub_f.vocab import vocab_tag_to_id

    tag_to_id = vocab_tag_to_id()

    bref_w_major_tag = resolve_bref_tag("W", "MAJOR_ROAD")
    assert bref_w_major_tag == "<bref_W_MAJOR>"
    bref_w_major_id = tag_to_id[bref_w_major_tag]

    bref_e_major_tag = resolve_bref_tag("E", "MAJOR_ROAD")
    assert bref_e_major_tag == "<bref_E_MAJOR>"
    bref_e_major_id = tag_to_id[bref_e_major_tag]

    # Road: entry at West edge (x=0), exit at East edge (x=250), feature_class=0.
    road = LineString([(0.0, 100.0), (125.0, 100.0), (250.0, 100.0)])

    sub_c = tmp_path / "features.parquet"
    sub_e = tmp_path / "boundary_contract.parquet"
    out = tmp_path / "cells.parquet"

    rows = [_feature_row(1, 1, 0, road, "residential", "road-d")]
    _write_sub_c_parquet(sub_c, rows)

    # Activate West edge of cell (1,1): lower_i=0, lower_j=1, axis=0 -> MAJOR_ROAD.
    # Activate East edge of cell (1,1): lower_i=1, lower_j=1, axis=0 -> MAJOR_ROAD.
    _write_sub_e_parquet(
        sub_e,
        overrides={
            (1, 0, 1, 0): {"scope_marker": 0, "boundary_class_enum": 2},  # West
            (1, 1, 1, 0): {"scope_marker": 0, "boundary_class_enum": 2},  # East
        },
    )

    encode_tile(sub_c, sub_e, out)

    table = pq.ParquetFile(out).read()
    pyrows = table.to_pylist()
    cell_11 = next(r for r in pyrows if r["cell_i"] == 1 and r["cell_j"] == 1)
    token_seq = list(cell_11["token_sequence"])

    assert cell_11["feature_count"] == 1, (
        f"Case D: expected feature_count=1, got {cell_11['feature_count']}"
    )

    # Both bref tokens must be present.
    assert bref_w_major_id in token_seq, (
        f"<bref_W_MAJOR> (id={bref_w_major_id}) not found in token_sequence: {token_seq}"
    )
    assert bref_e_major_id in token_seq, (
        f"<bref_E_MAJOR> (id={bref_e_major_id}) not found in token_sequence: {token_seq}"
    )

    chunks = _split_into_feature_chunks(token_seq)
    assert len(chunks) == 1, f"Case D: expected 1 feature chunk, got {len(chunks)}"
    chunk = chunks[0]

    # Layout: [509, semantic, <bref_W_MAJOR>(inbound), anchor..., pairs...,
    #          <bref_E_MAJOR>(outbound), 510]
    assert chunk[0] == _FEATURE_TOKEN_ID
    assert chunk[-1] == _FEATURE_END_TOKEN_ID

    # chunk[2] = inbound bref (immediately after <feature> and semantic_tag_id).
    assert chunk[2] == bref_w_major_id, (
        f"Case D: expected <bref_W_MAJOR> (id={bref_w_major_id}) at chunk[2], "
        f"got {chunk[2]}. Full chunk: {chunk}"
    )
    # chunk[-2] = outbound bref (immediately before <feature_end>).
    assert chunk[-2] == bref_e_major_id, (
        f"Case D: expected <bref_E_MAJOR> (id={bref_e_major_id}) at chunk[-2], "
        f"got {chunk[-2]}. Full chunk: {chunk}"
    )

    # Exactly two bref-range ids (1500-1507) confirms both substitutions occurred.
    bref_ids_in_chunk = [t for t in chunk if 1500 <= t <= 1507]
    assert len(bref_ids_in_chunk) == 2, (
        f"Case D: expected exactly 2 bref tokens (1500-1507), "
        f"got {len(bref_ids_in_chunk)}: {bref_ids_in_chunk}. Full chunk: {chunk}"
    )


def test_encode_tile_cells_parquet_schema_correct(tmp_path: Path):
    """cells.parquet written by encode_tile has the CELLS_SCHEMA columns."""
    from cfm.data.sub_f.io import CELLS_SCHEMA
    from cfm.data.sub_f.pipeline_writer import encode_tile

    sub_c = tmp_path / "features.parquet"
    sub_e = tmp_path / "boundary_contract.parquet"
    out = tmp_path / "cells.parquet"

    _write_sub_c_parquet(sub_c, [])
    _write_sub_e_parquet(sub_e)

    encode_tile(sub_c, sub_e, out)

    table = pq.ParquetFile(out).read()
    assert table.schema == CELLS_SCHEMA, (
        f"cells.parquet schema mismatch.\n  expected: {CELLS_SCHEMA}\n  got:      {table.schema}"
    )


def test_encode_tile_returns_out_path(tmp_path: Path):
    """encode_tile must return the out_cells_parquet path (same as write_cells_parquet)."""
    from cfm.data.sub_f.pipeline_writer import encode_tile

    sub_c = tmp_path / "features.parquet"
    sub_e = tmp_path / "boundary_contract.parquet"
    out = tmp_path / "cells.parquet"

    _write_sub_c_parquet(sub_c, [])
    _write_sub_e_parquet(sub_e)

    result = encode_tile(sub_c, sub_e, out)
    assert result == out


def test_semantic_tag_from_row_unknown_feature_class_raises():
    """_semantic_tag_from_row must raise ValueError with a clear message for
    any feature_class outside {0, 1, 2, 3}.  Validates the structural boundary
    guard added in T8.8 follow-up; prevents opaque KeyError on bad sub-C input.
    """
    from cfm.data.sub_f.pipeline_writer import _semantic_tag_from_row

    with pytest.raises(ValueError, match="unknown feature_class"):
        _semantic_tag_from_row({"feature_class": 4, "class_raw": None})


def test_encode_tile_case_b_non_road_no_bref(tmp_path: Path):
    """Non-road geometries (building polygon) do NOT emit bref tokens even when
    the adjacent edge is active — per spec §1.4 (buildings clipped at geometry
    layer, not token layer) and _classify_feature_for_bref LineString-only guard.
    """
    from cfm.data.sub_f.pipeline_writer import encode_tile

    # Building touching East edge — should still be Case A (no bref).
    building = Polygon([(50, 50), (50, 100), (250, 100), (250, 50), (50, 50)])

    sub_c = tmp_path / "features.parquet"
    sub_e = tmp_path / "boundary_contract.parquet"
    out = tmp_path / "cells.parquet"

    rows = [_feature_row(0, 0, 1, building, "residential", "bldg-edge")]
    _write_sub_c_parquet(sub_c, rows)

    # East edge of cell (0,0) active MAJOR_ROAD — should NOT affect building encoding.
    _write_sub_e_parquet(
        sub_e,
        overrides={(1, 0, 0, 0): {"scope_marker": 0, "boundary_class_enum": 2}},
    )

    encode_tile(sub_c, sub_e, out)

    table = pq.ParquetFile(out).read()
    pyrows = table.to_pylist()
    cell_00 = next(r for r in pyrows if r["cell_i"] == 0 and r["cell_j"] == 0)
    token_seq = list(cell_00["token_sequence"])

    # No bref tokens (1500..1507) should appear for a building.
    bref_ids_in_seq = [t for t in token_seq if 1500 <= t <= 1507]
    assert not bref_ids_in_seq, f"building encoded with unexpected bref tokens: {bref_ids_in_seq}"


@pytest.mark.skip(
    reason=(
        "awaiting sub-E cache regeneration — BP7 verification debt; see close-checklist. "
        "Un-skip when sub-E cache regenerates. Assert encoder output matches real sub-E "
        "parquet on grammar cases B/C/D edge scenarios. Verify BP7 four-test composite "
        "per spec §8.1 (cross-reference, symmetry, non-road non-emission, coverage). "
        "This is the T8 layer of the BP7 verification debt inherited from T7."
    )
)
def test_encode_tile_against_real_sub_e_singapore(tmp_path: Path):  # type: ignore[empty-body]
    """Integration test against real sub-E Singapore boundary_contract.parquet.

    Requires: real sub-E cache at data/processed/sub_e/2024-04-16-beta.3/singapore/
    This test is the T8 layer of the BP7 verification debt inherited from T7.
    See reports/2026-05-23-phase-1-sub-F-close-checklist.md.
    """
    ...

"""BP5 per-axis determinism suite — Task T5b.

Tests cover:
  A. §5.2 per-axis discipline (one test per locked axis):
     - coordinate quantization (banker's tie-break via real quantize_coord_m)
     - Python round() banker's documentation + binding via quantize_coord_m
     - direction_bin tie-break at the locked 360-count
     - sub-C feature sort key (Gate-6 content-anchored)
     - sub-E slot sort key (Gate-6 content-anchored)

  B. §5.6 canonicalization adversarial tests (5 DOFs + closed-LineString):
     - 5a open LineString preserves source direction (forward & reverse)
     - 5f closed LineString (roundabout) preserves direction (NOT rotated/reversed)
     - 5b Polygon ring rotates to lex-min start (winding held constant — CCW input)
     - 5c Polygon exterior winding corrected to CCW (lex-min start held constant — CW input)
     - 5d Multi-part order sorted by first/lex-min vertex
     - 5e Multi-part tiebreak (shared first vertex → next vertex, then vertex count)

  C. Determinism integration (Risk 2 discipline — non-vacuous):
     - Same-process encode_tile twice → byte-identical token sequences
     - Canonical-equivalence: permuted source vertex order → identical tokens via encode_feature
     - Fresh-process (cold subprocess) determinism on same synthetic tile → byte-identical
     - Vocab order across >=2 cold subprocesses with different PYTHONHASHSEED
       -> identical + YAML order

  D. Real-cached-Singapore §5.5 integration — SKIP STUB (cache absent).

Risk 2 non-vacuity notes (confirmed at write time):
  - same-process-twice: encodes fresh objects; if encode_cell were non-deterministic
    (e.g. dict-iteration order in cell_edges), the two token sequences would differ.
  - canonical-equivalence: input polygons have different exterior vertex orders; if
    canonicalize_geometry were missing/broken, the encoder would produce different
    anchor tokens for the two inputs, breaking equality.
  - fresh-process: subprocess uses a separate Python interpreter (cold lru_cache);
    if any module-level state or hash-seed affected token emission, sequences differ.
  - cross-seed vocab: two subprocesses run with PYTHONHASHSEED=0 and PYTHONHASHSEED=1;
    if vocab order depended on dict hash iteration, the sequences of (token_id, tag) pairs
    would differ between seeds. load_sub_f_vocab() sorts by token_id, so it would need to
    accidentally produce the same incorrect order at both seeds to give a false pass.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.wkb import dumps as wkb_dumps

# ============================================================================
# A. §5.2 per-axis discipline
# ============================================================================


# ---- A1: Coordinate quantization via quantize_coord_m ----------------------


def test_quantize_coord_m_ties_round_to_even():
    """quantize_coord_m uses Python banker's round() (PEP 3141).

    0.25 / 0.5 = 0.5 → rounds to even → 0.
    0.75 / 0.5 = 1.5 → rounds to even → 2.
    1.0  / 0.5 = 2.0 → exact integer   → 2.
    """
    from cfm.data.sub_f.encoder import quantize_coord_m

    assert quantize_coord_m(0.25, 0.5) == 0, (
        "0.25/0.5=0.5: banker's round-half-to-even must yield 0, not 1"
    )
    assert quantize_coord_m(0.75, 0.5) == 2, (
        "0.75/0.5=1.5: banker's round-half-to-even must yield 2, not 1"
    )
    assert quantize_coord_m(1.0, 0.5) == 2, "1.0/0.5=2.0: exact integer — must yield 2"


def test_quantize_coord_m_non_tie_values():
    """quantize_coord_m rounds non-tie values in the expected direction."""
    from cfm.data.sub_f.encoder import quantize_coord_m

    # Non-tie values: normal rounding applies
    assert quantize_coord_m(0.3, 0.5) == 1, "0.3/0.5=0.6 → 1"
    assert quantize_coord_m(0.7, 0.5) == 1, "0.7/0.5=1.4 → 1"
    assert quantize_coord_m(0.0, 0.5) == 0, "0.0 → 0"


# ---- A2: Python round() banker's documentation (bound to encoder) -----------


def test_python_round_is_bankers_via_quantize_coord_m():
    """Python round() is banker's (PEP 3141): round-half-to-even.

    This test documents the contract AND binds it to quantize_coord_m so that
    a future change to the quantize_coord_m implementation (e.g. math.floor or
    int(x + 0.5)) surfaces here, not silently.
    """
    from cfm.data.sub_f.encoder import quantize_coord_m

    # Direct Python round() checks (contract documentation)
    assert round(0.5) == 0, "0.5 rounds to even (0)"
    assert round(1.5) == 2, "1.5 rounds to even (2)"
    assert round(2.5) == 2, "2.5 rounds to even (2)"
    assert round(3.5) == 4, "3.5 rounds to even (4)"

    # Bound via the real encoder function with quantum=1.0 so coord_m/quantum = coord_m
    assert quantize_coord_m(0.5, 1.0) == 0, "quantize_coord_m(0.5, 1.0): banker's → 0"
    assert quantize_coord_m(1.5, 1.0) == 2, "quantize_coord_m(1.5, 1.0): banker's → 2"
    assert quantize_coord_m(2.5, 1.0) == 2, "quantize_coord_m(2.5, 1.0): banker's → 2"
    assert quantize_coord_m(3.5, 1.0) == 4, "quantize_coord_m(3.5, 1.0): banker's → 4"


# ---- A3: direction_bin tie-break at locked 360-count -----------------------


def test_direction_bin_locked_360_basic():
    """direction_bin at the locked 360-count (1° bins).

    Bin width = 360/360 = 1°. Tie-break to LOWER bin at exact boundaries.
    """
    from cfm.data.sub_f.encoder import direction_bin

    # Exact boundaries and mid-bin values
    assert direction_bin(0.0, 360) == 0, "0° → bin 0"
    assert direction_bin(1.0, 360) == 1, "1° → bin 1 (exact lower boundary)"
    assert direction_bin(359.999, 360) == 359, "359.999° → bin 359"
    assert direction_bin(180.0, 360) == 180, "180° → bin 180"


def test_direction_bin_locked_360_tie_break_to_lower():
    """At an exact bin boundary, direction_bin ties to the LOWER bin.

    0.5° is the exact mid-point between bin 0 (0°-1°) and bin 1 (1°-2°).
    With floor-division, 0.5 // 1.0 = 0, so direction_bin(0.5, 360) == 0.

    Non-vacuity: if the implementation used round() instead of floor-div,
    direction_bin(1.5, 360) would differ — floor-div gives 1 (1.5 // 1.0 = 1),
    whereas round() would give 2 (round(1.5) = 2 via banker's round-half-to-even).
    The 1.5° case is the canonical discriminator between floor-div and round().
    """
    from cfm.data.sub_f.encoder import direction_bin

    # Exact half-bin boundary: tie-break to lower
    assert direction_bin(0.5, 360) == 0, (
        "0.5° is the exact half-bin boundary between dir_0 and dir_1; "
        "floor-div ties to LOWER → bin 0"
    )

    # 1.5° is mid-bin between 1° and 2° → lower bin = 1
    assert direction_bin(1.5, 360) == 1, (
        "1.5° is the exact half-bin boundary between dir_1 and dir_2; "
        "floor-div ties to LOWER → bin 1"
    )


def test_direction_bin_locked_360_wraps_modulo():
    """Angles outside [0, 360) are normalised via modulo."""
    from cfm.data.sub_f.encoder import direction_bin

    assert direction_bin(360.0, 360) == 0, "360° wraps to bin 0"
    assert direction_bin(361.0, 360) == 1, "361° wraps to bin 1"
    assert direction_bin(-1.0, 360) == 359, "-1° wraps to bin 359"


# ---- A4: sub-C feature sort key (Gate-6 content-anchored) ------------------


def test_sub_c_feature_sort_key_is_cell_i_j_class_id():
    """Sub-C feature iteration order is determined by the sort key in sub_c/io.py.

    Gate-6 content-anchored: assert the exact substring
    ``(r.cell_i, r.cell_j, r.feature_class, r.source_feature_id)``
    appears in the source, without using any abstraction from sub_c.io.

    Failure message: a §9.6.1 sub-F cascade is needed if sub-C changed
    this sort key, because sub-F encoder receives features in sub-C row
    order and relies on it being deterministic.
    """
    sub_c_io_path = Path(__file__).resolve().parents[3] / "src" / "cfm" / "data" / "sub_c" / "io.py"
    source = sub_c_io_path.read_text(encoding="utf-8")
    expected_key_fragment = "(r.cell_i, r.cell_j, r.feature_class, r.source_feature_id)"
    assert expected_key_fragment in source, (
        f"sub_c/io.py sort key fragment not found: {expected_key_fragment!r}\n"
        "A §9.6.1 sub-F cascade is needed if sub-C changed this sort key — "
        "sub-F encoder iteration order depends on it."
    )


# ---- A8: sub-E slot sort key (Gate-6 content-anchored) ---------------------


def test_sub_e_slot_sort_key_is_slot_kind_slot_index():
    """Sub-E boundary contract slot order is determined by the sort key in sub_e/writer.py.

    Gate-6 content-anchored: assert the exact substring
    ``(int(r.slot_kind), r.slot_index)``
    appears in the source, without using any abstraction from sub_e.writer.

    Failure: sub-E slot ordering changed; sub-F boundary_contract.py reader
    may be consuming slots in a different order than it was designed for.
    """
    sub_e_writer_path = (
        Path(__file__).resolve().parents[3] / "src" / "cfm" / "data" / "sub_e" / "writer.py"
    )
    source = sub_e_writer_path.read_text(encoding="utf-8")
    expected_key_fragment = "(int(r.slot_kind), r.slot_index)"
    assert expected_key_fragment in source, (
        f"sub_e/writer.py sort key fragment not found: {expected_key_fragment!r}\n"
        "Sub-F boundary_contract.py reader may be consuming slots in the wrong order — "
        "review load_boundary_contract() if sub-E changed this key."
    )


# ============================================================================
# B. §5.6 canonicalization adversarial tests (5 DOFs + closed LineString)
#    Each test holds all other DOFs constant to isolate the one being tested.
# ============================================================================


# ---- 5a: open LineString preserves source direction ------------------------


def test_5a_open_linestring_forward_direction_preserved():
    """§5.6: open LineString source direction is PRESERVED (forward)."""
    from cfm.data.sub_f.encoder import canonicalize_geometry

    # Non-lex-min start (5,5) → (1,1): lex-min would be (1,1) but direction preserved
    fwd = LineString([(5, 5), (1, 1)])
    result = canonicalize_geometry(fwd)
    assert list(result.coords) == [(5, 5), (1, 1)], (
        "Forward LineString: direction must be preserved, NOT reversed to lex-min start"
    )


def test_5a_open_linestring_reverse_direction_preserved():
    """§5.6: open LineString source direction is PRESERVED — adversarial version.

    The previous version used (1,1)→(5,5), which already starts at the lex-min
    vertex; a buggy implementation that rotated open LineStrings to lex-min start
    would produce the same output, making the test vacuous.

    This version is adversarial on two axes:
    - ``forward`` starts at the NON-lex-min vertex (5,5). A lex-min-rotation bug
      would change it to (1,1)→(5,5), so the first assertion catches that bug.
    - The inequality assertion catches any direction-collapse bug: an encoder that
      canonicalises all traversals of the same segment to one orientation would
      make forward and reverse map to the same coord list, failing the !=  check.

    DOFs held constant: open LineString (no ring canonicalizer), 2-vertex segment.
    """
    from cfm.data.sub_f.encoder import canonicalize_geometry

    forward = LineString([(5, 5), (1, 1)])  # starts at non-lex-min (5,5)
    reverse = LineString([(1, 1), (5, 5)])  # opposite traversal of same segment

    # Each direction preserved exactly
    assert list(canonicalize_geometry(forward).coords) == [(5, 5), (1, 1)], (
        "Forward LineString starting at non-lex-min (5,5): direction must be "
        "PRESERVED, NOT rotated to lex-min start (1,1)"
    )
    assert list(canonicalize_geometry(reverse).coords) == [(1, 1), (5, 5)], (
        "Reverse LineString starting at (1,1): direction must be preserved"
    )

    # The two opposite traversals must remain DISTINCT after canonicalization
    assert list(canonicalize_geometry(forward).coords) != list(
        canonicalize_geometry(reverse).coords
    ), (
        "Forward and reverse traversals of the same segment must stay DISTINCT — "
        "a direction-collapse bug would canonicalise both to the same orientation"
    )


# ---- 5f: closed LineString preserves direction (roundabout) ----------------


def test_5f_closed_linestring_roundabout_direction_preserved():
    """§5.6: closed LineString (roundabout) routes to LineString-preserve, NOT ring
    canonicalizer. If dispatch used geom.is_ring instead of geom.geom_type, this
    closed LineString would be rotated to lex-min start (0,0) and possibly reversed,
    destroying oneway/flow semantics.

    DOFs held constant: winding (CW), starting vertex (non-lex-min (2,0)).
    """
    from cfm.data.sub_f.encoder import canonicalize_geometry

    # CW roundabout starting at non-lex-min vertex (2,0).
    # Lex-min vertex is (0,0); a ring canonicalizer would rotate to start there.
    roundabout = LineString([(2, 0), (2, 2), (0, 2), (0, 0), (2, 0)])
    result = canonicalize_geometry(roundabout)
    expected = [(2, 0), (2, 2), (0, 2), (0, 0), (2, 0)]
    assert list(result.coords) == expected, (
        "Closed LineString (roundabout): source vertex order must be PRESERVED. "
        "If this fails, the encoder is routing closed LineStrings to the ring "
        "canonicalizer (dispatch-on-is_ring bug per Assertion 2)."
    )


# ---- 5b: Polygon ring rotates to lex-min start (winding held constant) -----


def test_5b_polygon_ring_rotated_to_lex_min_start():
    """§5.6 rule (i): Polygon ring rotates to start at lex-min vertex.

    DOF isolated: start rotation only.
    Winding held constant: input already CCW (positive signed area).
    """
    from cfm.data.sub_f.encoder import canonicalize_geometry

    # CCW polygon (positive area), lex-min vertex is (1,1) at index 1 in source.
    # Source: [(5,5),(1,1),(3,1)] — area = +8 (CCW). No winding correction needed.
    poly = Polygon([(5, 5), (1, 1), (3, 1), (5, 5)])
    canon = canonicalize_geometry(poly)
    coords = list(canon.exterior.coords)

    assert coords[0] == (1, 1), (
        f"Polygon ring must start at lex-min vertex (1,1); got {coords[0]}. "
        "If this fails, the ring rotation is broken."
    )
    assert coords[0] == coords[-1], "Ring must remain closed (first == last)"
    # Verify all three unique vertices are present (no vertex lost)
    unique = set(coords[:-1])
    assert unique == {(1, 1), (3, 1), (5, 5)}, f"Unexpected vertex set: {unique}"


# ---- 5c: Polygon exterior winding corrected to CCW -------------------------


def test_5c_polygon_cw_winding_corrected_to_ccw():
    """§5.6 rule (ii): CW Polygon exterior is reversed to CCW (RFC 7946).

    DOF isolated: winding correction only.
    Lex-min start held constant: input starts at lex-min vertex (1,1).
    """
    from cfm.data.sub_f.encoder import canonicalize_geometry

    # CW polygon starting at lex-min vertex (1,1); area = -8 (CW).
    # After winding correction: (1,1) → (3,1) → (5,5) → (1,1).
    poly = Polygon([(1, 1), (5, 5), (3, 1), (1, 1)])
    canon = canonicalize_geometry(poly)
    coords = list(canon.exterior.coords)

    assert coords == [(1, 1), (3, 1), (5, 5), (1, 1)], (
        f"CW polygon must be reversed to CCW; got {coords}. "
        "If this fails, winding correction is broken."
    )


# ---- 5d: Multi-part order sorted by first/lex-min vertex -------------------


def test_5d_multilinestring_parts_sorted_by_first_vertex():
    """§5.6 rule (iv): MultiLineString parts sorted by first vertex (lex order).

    DOF isolated: part order only.
    Within-part direction held constant (LineStrings: preserve).
    """
    from cfm.data.sub_f.encoder import canonicalize_geometry

    # Parts are supplied in high-first order; canonical output must be low-first.
    part_high = LineString([(3, 3), (7, 7)])
    part_low = LineString([(1, 1), (5, 5)])
    multi_in = MultiLineString([part_high, part_low])  # high before low

    canon = canonicalize_geometry(multi_in)
    parts = list(canon.geoms)

    first_vertices = [next(iter(p.coords)) for p in parts]
    assert first_vertices[0] == (1, 1), (
        f"First part must start at lex-min vertex (1,1); got {first_vertices[0]}. "
        "Multi-part sort order is broken."
    )
    assert first_vertices[1] == (3, 3), f"Second part must start at (3,3); got {first_vertices[1]}"


def test_5d_multipolygon_parts_sorted_by_lex_min_vertex():
    """§5.6 rule (iv): MultiPolygon parts sorted after per-part canonicalization."""
    from cfm.data.sub_f.encoder import canonicalize_geometry

    # Two CCW polygons (already canonical winding); supplied in reverse lex order.
    poly_high = Polygon([(10, 10), (12, 10), (12, 12), (10, 12), (10, 10)])
    poly_low = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    multi_in = MultiPolygon([poly_high, poly_low])  # high before low

    canon = canonicalize_geometry(multi_in)
    parts = list(canon.geoms)

    # After per-part canonicalization, lex-min vertex of poly_low's ring is (0,0).
    first_vertices = [next(iter(p.exterior.coords)) for p in parts]
    assert first_vertices[0] == (0, 0), (
        f"MultiPolygon: first part must have lex-min start (0,0); got {first_vertices[0]}"
    )
    assert first_vertices[1] == (10, 10), (
        f"MultiPolygon: second part must start at (10,10); got {first_vertices[1]}"
    )


# ---- 5e: Multi-part tiebreak (shared first vertex) -------------------------


def test_5e_multilinestring_tiebreak_on_next_vertex():
    """§5.6 rule (iv) tiebreak: when two LineString parts share the same first vertex,
    order is determined by the next vertex (full coord-chain comparison).

    DOF isolated: tiebreak within shared-first-vertex group.
    """
    from cfm.data.sub_f.encoder import canonicalize_geometry

    # Both parts start at (1, 1). Part A continues to (5, 5); Part B continues to (3, 3).
    # Lexicographic sort on full coord tuples: ((1,1),(3,3)) < ((1,1),(5,5)).
    # So Part B (→(3,3)) must come first.
    part_a = LineString([(1, 1), (5, 5)])
    part_b = LineString([(1, 1), (3, 3)])
    multi_in = MultiLineString([part_a, part_b])  # A before B

    canon = canonicalize_geometry(multi_in)
    parts = list(canon.geoms)
    second_vertices = [list(p.coords)[1] for p in parts]

    assert second_vertices[0] == (3, 3), (
        f"Tiebreak: part with smaller next vertex (3,3) must come first; "
        f"got {second_vertices[0]}. If this fails, the tiebreak key is broken."
    )
    assert second_vertices[1] == (5, 5), (
        f"Tiebreak: part with larger next vertex (5,5) must come second; got {second_vertices[1]}"
    )


def test_5e_multilinestring_tiebreak_on_vertex_count():
    """§5.6 rule (iv) tiebreak: when two parts share ALL vertex coords (same geometry
    with different vertex counts is degenerate, but the tiebreak falls to vertex count
    — smaller count first).

    This tests the third tiebreak level: coord chain equal → smaller count first.
    We construct two parts sharing first vertex and equal second vertex; the third
    vertex differs only by existence (one part is a 2-vertex segment, the other a
    3-vertex path sharing the first two vertices and adding a third at a higher coord).

    In practice this tiebreak fires when a 2-vertex part is a prefix of a 3-vertex part.
    """
    from cfm.data.sub_f.encoder import canonicalize_geometry

    # Part A: (1,1) → (2,2)          — 2 vertices
    # Part B: (1,1) → (2,2) → (3,3)  — 3 vertices
    # Coord-chain sort: A = ((1,1),(2,2)), B = ((1,1),(2,2),(3,3)).
    # Tuple comparison: A < B because A is a prefix and tuples compare element-by-element;
    # len(A) < len(B) and A exhausts first → A is "less than" B in Python tuple comparison.
    part_a = LineString([(1, 1), (2, 2)])
    part_b = LineString([(1, 1), (2, 2), (3, 3)])
    multi_in = MultiLineString([part_b, part_a])  # B (longer) before A (shorter)

    canon = canonicalize_geometry(multi_in)
    parts = list(canon.geoms)
    coord_counts = [len(list(p.coords)) for p in parts]

    assert coord_counts[0] == 2, (
        f"Shorter part (2 vertices) must come first; got {coord_counts[0]}. "
        f"Vertex-count tiebreak is broken."
    )
    assert coord_counts[1] == 3, f"Longer part (3 vertices) must come second; got {coord_counts[1]}"


# ============================================================================
# C. Determinism integration (Risk 2 discipline)
#    Helpers reused from test_pipeline_writer.py (replicated here for
#    self-contained test module; see close-checklist for dedup obligation).
# ============================================================================

# Sub-E parquet schema (source-derived, mirrors test_pipeline_writer.py._SUB_E_SCHEMA)
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

# Sub-C features.parquet schema (source-derived, mirrors test_pipeline_writer.py)
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


def _make_full_tile_contract_rows(
    overrides: dict[tuple[int, int, int, int], dict] | None = None,
) -> list[dict]:
    """Build 144 well-formed sub-E contract rows (112 INTERNAL + 32 EXTERNAL).

    Replicated from test_pipeline_writer.py._make_full_tile_contract_rows.
    See close-checklist: extract to conftest.py to avoid dual maintenance.
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
                    "scope_marker": 1,
                    "boundary_class_enum": None,
                }
                ov = overrides.get((slot_kind, lower_i, lower_j, axis))
                if ov:
                    row.update(ov)
                rows.append(row)
    return rows


def _write_sub_e_parquet(path: Path) -> None:
    rows = _make_full_tile_contract_rows()
    table = pa.Table.from_pylist(rows, schema=_SUB_E_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _feature_row(
    cell_i: int,
    cell_j: int,
    feature_class: int,
    geometry,
    class_raw: str | None,
    source_feature_id: str = "test-0",
) -> dict:
    geom_type_map = {"Point": 0, "LineString": 1, "Polygon": 2}
    gt = geom_type_map.get(geometry.geom_type, 0)
    wkb = wkb_dumps(geometry, include_srid=False)
    bounds = geometry.bounds
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
    table = pa.Table.from_pylist(rows, schema=_SUB_C_FEATURES_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _encode_tile_and_read_tokens(sub_c: Path, sub_e: Path, out: Path) -> list[list[int]]:
    """Encode a tile and return the list of per-cell token_sequence lists in
    (cell_i, cell_j) order so two calls can be compared byte-identically."""
    from cfm.data.sub_f.pipeline_writer import encode_tile

    encode_tile(sub_c, sub_e, out)
    table = pq.ParquetFile(out).read()
    rows = sorted(table.to_pylist(), key=lambda r: (r["cell_i"], r["cell_j"]))
    return [list(r["token_sequence"]) for r in rows]


# ---- C1: Same-process encode_tile twice → byte-identical -------------------


def test_same_process_encode_tile_twice_byte_identical(tmp_path: Path):
    """Encoding the same tile twice in the same process produces identical token
    sequences for every cell.

    Non-vacuity: each call to encode_tile reads from fresh parquet files and
    builds new Python objects (no caching across calls). If cell_edges dict
    iteration order or any other stateful component were non-deterministic,
    the two sequences would differ for cells with active boundary edges.
    We use all-NONE edges (no bref) so the test isolates the coordinate
    quantization and direction_bin paths specifically.
    """
    building = Polygon([(30, 40), (30, 120), (100, 120), (100, 40), (30, 40)])
    road = LineString([(20.0, 30.0), (80.0, 90.0), (130.0, 150.0)])

    sub_c = tmp_path / "features.parquet"
    sub_e = tmp_path / "boundary_contract.parquet"

    rows = [
        _feature_row(0, 0, 1, building, "residential", "bldg-det-0"),
        _feature_row(1, 1, 0, road, "residential", "road-det-0"),
    ]
    _write_sub_c_parquet(sub_c, rows)
    _write_sub_e_parquet(sub_e)

    out1 = tmp_path / "cells_run1.parquet"
    out2 = tmp_path / "cells_run2.parquet"

    tokens1 = _encode_tile_and_read_tokens(sub_c, sub_e, out1)
    tokens2 = _encode_tile_and_read_tokens(sub_c, sub_e, out2)

    assert tokens1 == tokens2, (
        "Same-process encode_tile called twice on identical inputs produced "
        "different token sequences — encoder is non-deterministic."
    )


# ---- C2: Canonical-equivalence — permuted vertex order → identical tokens --


def test_encode_feature_canonical_equivalence_polygon_rotation():
    """Two polygon representations with different source vertex orders (ring rotated)
    must produce IDENTICAL token sequences after canonicalize_geometry is applied
    inside encode_cell / encode_feature.

    Non-vacuity: the two inputs have DIFFERENT exterior.coords lists, so if
    canonicalize_geometry were absent or broken (not rotating to lex-min start),
    the anchor tokens would differ (different first vertex → different quantized
    anchor coords) and the assertion would FAIL.
    """
    from cfm.data.sub_f.encoder import canonicalize_geometry, encode_feature

    # A square polygon — same shape, different ring rotations (4 representations).
    coords_v0 = [(0, 0), (0, 100), (100, 100), (100, 0), (0, 0)]  # starts at lex-min (0,0)
    coords_v1 = [(0, 100), (100, 100), (100, 0), (0, 0), (0, 100)]  # starts at (0,100)
    coords_v2 = [(100, 100), (100, 0), (0, 0), (0, 100), (100, 100)]  # starts at (100,100)
    coords_v3 = [(100, 0), (0, 0), (0, 100), (100, 100), (100, 0)]  # starts at (100,0)

    polys = [Polygon(c) for c in [coords_v0, coords_v1, coords_v2, coords_v3]]

    # Verify that these are genuinely different before canonicalization
    raw_starts = [next(iter(p.exterior.coords)) for p in polys]
    assert len(set(raw_starts)) > 1, (
        "Test setup error: all polygon rotations have the same start vertex — the test is vacuous"
    )

    # Canonicalize and encode each representation
    semantic_tag = "building=residential"
    token_seqs = []
    for poly in polys:
        canon = canonicalize_geometry(poly)
        ef = encode_feature(canon, semantic_tag=semantic_tag)
        token_seqs.append(ef.tokens)

    # All must be identical
    for i in range(1, len(token_seqs)):
        assert token_seqs[0] == token_seqs[i], (
            f"Canonical-equivalence FAILED: polygon rotation {i} produced different "
            f"tokens than rotation 0.\n"
            f"  rotation 0 tokens: {token_seqs[0]}\n"
            f"  rotation {i} tokens: {token_seqs[i]}\n"
            "canonicalize_geometry is not removing the source rotation DOF."
        )


def test_encode_feature_canonical_equivalence_multilinestring_part_order():
    """Two MultiLineString representations with different part order produce
    IDENTICAL token sequences after canonicalization.

    Non-vacuity: the two inputs have DIFFERENT part orders, so if the multi-part
    sort in canonicalize_geometry were absent, encode_cell would produce tokens
    in part-a-first vs part-b-first order and the assertion would FAIL.
    """
    from cfm.data.sub_f.encoder import canonicalize_geometry, encode_cell

    part_a = LineString([(10, 10), (50, 50)])
    part_b = LineString([(1, 1), (20, 20)])

    # Two orderings of the same MultiLineString
    multi_ab = MultiLineString([part_a, part_b])  # part_a first (high first vertex)
    multi_ba = MultiLineString([part_b, part_a])  # part_b first (low first vertex)

    # Verify genuinely different part order before canonicalization
    first_ab = next(iter(next(iter(multi_ab.geoms)).coords))
    first_ba = next(iter(next(iter(multi_ba.geoms)).coords))
    assert first_ab != first_ba, (
        "Test setup error: both MultiLineStrings have same first part — vacuous test"
    )

    semantic_tag = "highway=residential"
    cell_edges: dict[str, str] = {}

    canon_ab = canonicalize_geometry(multi_ab)
    ec_ab = encode_cell([(canon_ab, semantic_tag)], cell_edges)

    canon_ba = canonicalize_geometry(multi_ba)
    ec_ba = encode_cell([(canon_ba, semantic_tag)], cell_edges)

    assert ec_ab.tokens == ec_ba.tokens, (
        "Canonical-equivalence FAILED: MultiLineString with different part orders "
        "produced different tokens after canonicalization.\n"
        f"  AB tokens: {ec_ab.tokens}\n"
        f"  BA tokens: {ec_ba.tokens}"
    )


# ---- C3: Fresh-process (cold subprocess) determinism -----------------------

# Script run in the subprocess. Encodes a fixed synthetic geometry and prints
# the token list as JSON to stdout. The exact geometry is chosen to exercise
# both the coordinate-quantization and direction-bin paths.
_SUBPROCESS_ENCODE_SCRIPT = """
import sys, json
sys.path.insert(0, {src_path!r})
from shapely.geometry import Polygon, LineString
from cfm.data.sub_f.encoder import canonicalize_geometry, encode_feature

building = Polygon([(30, 40), (30, 120), (100, 120), (100, 40), (30, 40)])
road = LineString([(20.0, 30.0), (80.0, 90.0), (130.0, 150.0)])

results = []
for geom, tag in [
    (building, "building=residential"),
    (road, "highway=residential"),
]:
    canon = canonicalize_geometry(geom)
    ef = encode_feature(canon, semantic_tag=tag)
    results.append(ef.tokens)

print(json.dumps(results))
"""


def _run_encode_subprocess(src_path: str, env_extra: dict[str, str] | None = None) -> list:
    """Run _SUBPROCESS_ENCODE_SCRIPT in a cold subprocess and return the decoded token lists."""
    import os

    script = _SUBPROCESS_ENCODE_SCRIPT.format(src_path=src_path)
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"Subprocess encode script failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return json.loads(result.stdout.strip())


def test_fresh_process_encode_byte_identical_to_in_process():
    """Token sequences produced by a cold subprocess match in-process results.

    Non-vacuity: the subprocess starts a fresh Python interpreter with an
    empty lru_cache. If any module-level state (e.g. global dict iteration
    order, random seeding) affected token emission, the subprocess result
    would differ. PYTHONHASHSEED is deliberately NOT fixed, so the test
    probes across the default seed environment.
    """
    from shapely.geometry import LineString, Polygon

    from cfm.data.sub_f.encoder import canonicalize_geometry, encode_feature

    building = Polygon([(30, 40), (30, 120), (100, 120), (100, 40), (30, 40)])
    road = LineString([(20.0, 30.0), (80.0, 90.0), (130.0, 150.0)])

    in_process = []
    for geom, tag in [
        (building, "building=residential"),
        (road, "highway=residential"),
    ]:
        canon = canonicalize_geometry(geom)
        ef = encode_feature(canon, semantic_tag=tag)
        in_process.append(ef.tokens)

    src_path = str(Path(__file__).resolve().parents[3] / "src")
    subprocess_result = _run_encode_subprocess(src_path)

    assert in_process == subprocess_result, (
        "Fresh-process determinism FAILED: cold subprocess produced different "
        "token sequences than in-process encoding.\n"
        f"  In-process:  {in_process}\n"
        f"  Subprocess:  {subprocess_result}"
    )


def test_fresh_process_encode_two_cold_subprocesses_agree():
    """Two independent cold subprocesses produce identical token sequences.

    This catches sources of non-determinism that happen to match the in-process
    result (e.g. a side effect that stabilises after one call) but vary between
    cold starts.
    """
    src_path = str(Path(__file__).resolve().parents[3] / "src")
    result_1 = _run_encode_subprocess(src_path)
    result_2 = _run_encode_subprocess(src_path)

    assert result_1 == result_2, (
        "Two cold subprocesses produced different token sequences — "
        "encoder is non-deterministic across cold starts."
    )


# ---- C4: Vocab order under PYTHONHASHSEED=0 vs PYTHONHASHSEED=1 ------------

# Script run in each subprocess: prints the vocab (token_id, tag) pairs as JSON.
_SUBPROCESS_VOCAB_SCRIPT = """
import sys, json
sys.path.insert(0, {src_path!r})
from cfm.data.sub_f.vocab import load_sub_f_vocab
slots = [(s.token_id, s.tag) for s in load_sub_f_vocab()]
print(json.dumps(slots))
"""


def _run_vocab_subprocess(src_path: str, hashseed: str) -> list[list]:
    """Run _SUBPROCESS_VOCAB_SCRIPT in a cold subprocess with PYTHONHASHSEED=hashseed."""
    import os

    script = _SUBPROCESS_VOCAB_SCRIPT.format(src_path=src_path)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = hashseed
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"Vocab subprocess (PYTHONHASHSEED={hashseed}) failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return json.loads(result.stdout.strip())


def test_vocab_order_invariant_across_pythonhashseed():
    """load_sub_f_vocab() returns the same (token_id, tag) sequence regardless
    of PYTHONHASHSEED.

    Two cold subprocesses run with PYTHONHASHSEED=0 and PYTHONHASHSEED=1 must
    produce byte-identical (token_id, tag) orderings.

    Non-vacuity: if load_sub_f_vocab() were iterating over a plain dict
    (Python 3.6 pre-insertion-order era) or any set in a hash-dependent way
    during slot assembly, the emitted order would differ between seeds.
    The current implementation sorts by token_id, so a bug would have to
    produce the same incorrect ordering at BOTH seeds to give a false pass.
    Using two different seeds breaks that coincidence.
    """
    src_path = str(Path(__file__).resolve().parents[3] / "src")

    vocab_seed_0 = _run_vocab_subprocess(src_path, "0")
    vocab_seed_1 = _run_vocab_subprocess(src_path, "1")

    assert vocab_seed_0 == vocab_seed_1, (
        "Vocab order differs between PYTHONHASHSEED=0 and PYTHONHASHSEED=1. "
        "load_sub_f_vocab() is not hash-seed-stable — it may be iterating "
        "over a set or unordered dict somewhere in the loader chain."
    )


def test_vocab_order_is_ascending_token_id():
    """load_sub_f_vocab() returns slots in strictly ascending token_id order.

    This is the YAML-intended / spec-mandated order. We test it in-process
    (the value is deterministic per the cross-seed test above, so in-process
    is sufficient for the ordering assertion itself).
    """
    from cfm.data.sub_f.vocab import load_sub_f_vocab

    slots = load_sub_f_vocab()
    ids = [s.token_id for s in slots]
    assert ids == sorted(ids), (
        f"load_sub_f_vocab() is not in ascending token_id order. "
        f"First out-of-order pair: "
        f"{next((ids[i], ids[i + 1]) for i in range(len(ids) - 1) if ids[i] >= ids[i + 1])}"
    )
    # Also check no duplicates
    assert len(ids) == len(set(ids)), (
        f"load_sub_f_vocab() contains duplicate token_ids: {[x for x in ids if ids.count(x) > 1]}"
    )


def test_vocab_cross_seed_matches_in_process_order():
    """Vocab order from PYTHONHASHSEED=0 subprocess matches the in-process result.

    Bridges the cross-seed test (subprocess vs subprocess) with the in-process
    test (in-process vs YAML-ascending). Three-way agreement: in-process,
    seed=0 subprocess, seed=1 subprocess all agree.
    """
    from cfm.data.sub_f.vocab import load_sub_f_vocab

    # JSON round-trips tuples as lists; normalise to list-of-list for comparison.
    in_process = [[s.token_id, s.tag] for s in load_sub_f_vocab()]

    src_path = str(Path(__file__).resolve().parents[3] / "src")
    vocab_seed_0 = _run_vocab_subprocess(src_path, "0")

    assert in_process == vocab_seed_0, (
        "In-process vocab order differs from PYTHONHASHSEED=0 subprocess. "
        "This indicates load_sub_f_vocab() behaviour depends on the current "
        "process's hash seed (likely via lru_cache warming the in-process result)."
    )


# §5.5 real-cached-Singapore determinism is now in the consolidated
# tests/data/sub_f/test_singapore_integration.py (T13), gated fail-loud on the
# sub-E cache. See reports/2026-05-23-phase-1-sub-F-close-checklist.md.

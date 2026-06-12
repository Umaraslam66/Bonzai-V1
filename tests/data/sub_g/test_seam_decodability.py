from __future__ import annotations

from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.wkb import dumps as wkb_dumps

from cfm.data.sub_f.encoder import encode_cell, encode_feature
from cfm.data.sub_g.seam_decodability import (
    _feature_accuracy,
    check_decodability,
    has_outbound_bref,
    split_cell_into_features,
)


def _wkb(geom):
    return wkb_dumps(geom, byte_order=1)


def test_split_cell_into_features():
    seq = [509, 7, 300, 301, 302, 303, 510, 509, 7, 300, 301, 302, 303, 510]
    feats = split_cell_into_features(seq)
    assert len(feats) == 2
    assert feats[0][0] == 509 and feats[0][-1] == 510
    assert feats[1][0] == 509 and feats[1][-1] == 510


def test_split_ignores_trailing_incomplete_feature():
    seq = [509, 7, 300, 510, 509, 7, 300]  # second feature never closes
    feats = split_cell_into_features(seq)
    assert len(feats) == 1


def test_check_decodability_passes_on_valid_roundtrip():
    # Real encoder fixture: an open road LineString, no brefs (Case A).
    ef = encode_feature(
        LineString([(10.0, 10.0), (30.0, 10.0)]), semantic_tag="highway=residential"
    )
    diags, _errors, _ = check_decodability(
        tile_id="tile=i0_j0", cell=(0, 0), token_sequence=ef.tokens, sub_c_features=[]
    )
    assert diags == []  # decodes to a valid LineString within the cell bound


def test_check_decodability_measures_core_position_error_against_canonical_original():
    line = LineString([(10.0, 10.0), (30.0, 10.0)])
    ef = encode_feature(line, semantic_tag="highway=residential")
    sub_c = [{"feature_class": 0, "geometry": _wkb(line), "source_feature_id": "r"}]
    diags, errors, _ = check_decodability(
        tile_id="tile=i0_j0", cell=(0, 0), token_sequence=ef.tokens, sub_c_features=sub_c
    )
    assert diags == []
    assert len(errors) == 1
    # 0.5m magnitude-quantum round-trip on axis-aligned integers -> sub-metre error.
    assert errors[0]["position_core_m"] < 1.0
    assert errors[0]["position_full_m"] < 1.0  # Case A: no bref, core == full regime
    assert errors[0]["angle_core_deg"] is not None
    assert errors[0]["angle_core_deg"] < 5.0


# ---- construction-identity bref detection (encoder.py:438-456 + decoder.py:104-134) ----


def testhas_outbound_bref_true_for_case_b():
    # Case B: outbound crossing -> last body token is a bref (1500..1507).
    block = [509, 7, 300, 301, 302, 303, 444, 1502, 510]
    assert has_outbound_bref(block) is True


def testhas_outbound_bref_false_for_inbound_only_case_c():
    # Case C: inbound bref right after the semantic tag; body does NOT end in a bref.
    block = [509, 7, 1500, 300, 301, 302, 303, 444, 445, 510]
    assert has_outbound_bref(block) is False


def test_has_outbound_bref_false_for_case_a():
    block = [509, 7, 300, 301, 302, 303, 444, 445, 510]
    assert has_outbound_bref(block) is False


# ---- core/full split + the reviewer-mandated floor-still-fires GUARD ----


def test_feature_accuracy_core_excludes_outbound_placeholder():
    # Canonical original road A->B->C; C is the (outbound) exit-edge crossing.
    canon = LineString([(10.0, 10.0), (20.0, 10.0), (30.0, 10.0)])
    # Decoder for Case B reproduces A,B then appends a placeholder = duplicate of B.
    decoded = {"type": "LineString", "coordinates": [[10.0, 10.0], [20.0, 10.0], [20.0, 10.0]]}
    rec = _feature_accuracy([decoded], [True], [canon])
    # full sees the true exit vertex C far from the placeholder -> large.
    assert rec["position_full_m"] > 5.0
    # core drops the v1-unencoded placeholder AND the true exit vertex -> faithful.
    assert rec["position_core_m"] < 1e-6


def test_feature_accuracy_core_FIRES_on_displaced_non_bref_vertex():
    """GUARD (reviewer 2026-06-01): excluding the bref vertex must NOT blind the
    floor. Displace an ENCODED (non-bref) interior vertex on a crossing road ->
    the CORE metric must fire, proving the exclusion is by identity, not magnitude."""
    canon = LineString([(10.0, 10.0), (20.0, 10.0), (30.0, 10.0)])
    # The encoded interior vertex B decodes wildly wrong (99,99); placeholder dups it.
    decoded = {"type": "LineString", "coordinates": [[10.0, 10.0], [99.0, 99.0], [99.0, 99.0]]}
    rec = _feature_accuracy([decoded], [True], [canon])
    assert rec["position_core_m"] > 50.0  # core catches the real (non-bref) decode error


# ---- multi-part feature pairing (encode_cell splits Multi* into one block/part) ----


def test_check_decodability_pairs_multipart_feature_blocks():
    """A MultiLineString sub-C row becomes >1 decoded block; sub-G must advance the
    decoded pointer by #parts so later features are not compared to a wrong part."""
    a = LineString([(10.0, 10.0), (40.0, 10.0)])
    m = MultiLineString([[(10.0, 60.0), (40.0, 60.0)], [(10.0, 90.0), (40.0, 90.0)]])
    p = Point((100.0, 100.0))
    feats = [(a, "highway=residential"), (m, "highway=service"), (p, "place=poi")]
    enc = encode_cell(feats, cell_edges={})  # empty edges -> Case A throughout
    sub_c = [
        {"feature_class": 0, "geometry": _wkb(a), "source_feature_id": "a"},
        {"feature_class": 0, "geometry": _wkb(m), "source_feature_id": "m"},
        {"feature_class": 2, "geometry": _wkb(p), "source_feature_id": "p"},
    ]
    _diags, errors, _ = check_decodability(
        tile_id="tile=i0_j0", cell=(0, 0), token_sequence=enc.tokens, sub_c_features=sub_c
    )
    assert len(errors) == 3  # one accuracy record per sub-C feature, not per block
    assert all(e["position_core_m"] < 1.0 for e in errors)  # all correctly paired


def test_check_decodability_subdivided_long_road_is_geometry_faithful():
    """A long straight road is chunked into many collinear (dir,mag) steps, so the
    decoded vertex COUNT exceeds the original. A geometry-aware (Hausdorff) metric
    stays sub-metre; the old index-positional metric would report ~half the length."""
    line = LineString([(10.0, 10.0), (200.0, 10.0)])  # 190m single segment, chunked
    ef = encode_feature(line, semantic_tag="highway=primary")
    sub_c = [{"feature_class": 0, "geometry": _wkb(line), "source_feature_id": "r"}]
    _diags, errors, _ = check_decodability(
        tile_id="tile=i0_j0", cell=(0, 0), token_sequence=ef.tokens, sub_c_features=sub_c
    )
    assert len(errors) == 1
    assert errors[0]["position_core_m"] < 1.5


def test_check_decodability_angle_robust_to_polygon_canonicalization():
    """A polygon's ring is canonicalized (lex-min rotation + CCW winding) before
    encode, so decoded vertex ORDER differs from the raw original. Comparing angle
    against the CANONICAL original keeps the bearing error tiny (the old raw-order
    positional compare produced ~180deg)."""
    # Small CW square (won't be chunked); canonicalize flips winding to CCW.
    sq = Polygon([(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0), (10.0, 10.0)])
    enc = encode_cell([(sq, "building=yes")], cell_edges={})
    sub_c = [{"feature_class": 1, "geometry": _wkb(sq), "source_feature_id": "b"}]
    _diags, errors, _ = check_decodability(
        tile_id="tile=i0_j0", cell=(0, 0), token_sequence=enc.tokens, sub_c_features=sub_c
    )
    assert len(errors) == 1
    assert errors[0]["angle_core_deg"] is not None
    assert errors[0]["angle_core_deg"] < 5.0


def test_structural_bound_flags_far_vertex():
    # A decoded geometry with a vertex far outside the cell is a gate failure.
    from cfm.data.sub_g.seam_decodability import _VERTEX_BOUND_M, _max_abs_coord

    far = {"type": "LineString", "coordinates": [[0.0, 0.0], [10000.0, 0.0]]}
    assert _max_abs_coord(far) > _VERTEX_BOUND_M


# ---- H3 (2026-06-01): OGC-validity gate excludes the v1 outbound-bref placeholder
# collapse by CONSTRUCTION IDENTITY, reported (not gated), with a guard that genuine
# degeneracy still fires. sub-G T11 H3; consistent with H1's report-not-gate call. ----

# Real Singapore tile=i10_j10 cell (0,0) block: <feature> tag anchor x4
# <outbound_bref 1506> <feature_end>. A 2-vertex crossing road with NO interior
# vertex -> decode appends the v1-unencoded exit-vertex placeholder onto the anchor
# -> [(0.0,195.5),(0.0,195.5)], an OGC-invalid (zero-length) LineString.
_BREF_COLLAPSE_BLOCK = [509, 41, 300, 323, 363, 369, 1506, 510]
# SAME anchor, but a SYNTHETIC magnitude-0 inner pair (m=443=_MAGNITUDE_BASE-1, dir
# token 511=bin 0) and NO outbound bref -> decodes to the IDENTICAL zero-length
# [(0.0,195.5),(0.0,195.5)]. Construction identity differs (no bref) -> the gate
# MUST still quarantine it. (Unreachable from a real encoder; only corrupt tokens.)
_DEGENERATE_NO_BREF_BLOCK = [509, 41, 300, 323, 363, 369, 511, 443, 510]
_OGC_SIG = "decoded geometry not OGC-valid"


def test_check_decodability_excludes_bref_placeholder_collapse_from_gate():
    diags, _errors, n_bref_collapse = check_decodability(
        tile_id="tile=i0_j0",
        cell=(0, 0),
        token_sequence=_BREF_COLLAPSE_BLOCK,
        sub_c_features=[],
    )
    # Excluded from the blocking gate by construction identity (NOT magnitude)...
    assert [d for d in diags if d.signature == _OGC_SIG] == []
    # ...but counted, so the reported baseline still surfaces the v1-bref family.
    assert n_bref_collapse == 1


def test_check_decodability_GATE_FIRES_on_degenerate_without_outbound_bref():
    """GUARD (reviewer 2026-06-01): the bref exclusion must NOT blind the gate to
    genuine degeneracy. An IDENTICAL zero-length [anchor,anchor] LineString that did
    NOT come from an outbound bref MUST still quarantine -> proves the exclusion is
    by CONSTRUCTION IDENTITY, never a bare zero-length / magnitude test."""
    diags, _errors, n_bref_collapse = check_decodability(
        tile_id="tile=i0_j0",
        cell=(0, 0),
        token_sequence=_DEGENERATE_NO_BREF_BLOCK,
        sub_c_features=[],
    )
    assert len([d for d in diags if d.signature == _OGC_SIG]) == 1  # gate still fires
    assert n_bref_collapse == 0


def test_is_bref_placeholder_collapse_predicate_is_construction_identity():
    from cfm.data.sub_g.seam_decodability import _is_bref_placeholder_collapse

    zero_len = {"type": "LineString", "coordinates": [[0.0, 195.5], [0.0, 195.5]]}
    # bref + zero-length collapse -> excluded.
    assert _is_bref_placeholder_collapse(_BREF_COLLAPSE_BLOCK, zero_len) is True
    # SAME degenerate geom, NO outbound bref -> NOT excluded (gate would still fire).
    assert _is_bref_placeholder_collapse(_DEGENERATE_NO_BREF_BLOCK, zero_len) is False
    # bref present but geom is non-degenerate (>=2 distinct) -> NOT excluded (tightness).
    two_distinct = {"type": "LineString", "coordinates": [[0.0, 195.5], [10.0, 195.5]]}
    assert _is_bref_placeholder_collapse(_BREF_COLLAPSE_BLOCK, two_distinct) is False


def test_has_outbound_bref_is_public_and_private_name_is_gone():
    """W7: four sanctioned-private import sites earned the public re-export;
    the private name is fully migrated (no half-public split surface)."""
    import cfm.data.sub_g.seam_decodability as sd

    assert callable(sd.has_outbound_bref)
    assert not hasattr(sd, "_" + "has_outbound_bref")

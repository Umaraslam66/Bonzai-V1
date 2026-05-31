from __future__ import annotations

from shapely.geometry import LineString
from shapely.wkb import dumps as wkb_dumps

from cfm.data.sub_f.encoder import encode_feature
from cfm.data.sub_g.seam_decodability import (
    check_decodability,
    split_cell_into_features,
)


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
    diags, _errors = check_decodability(
        tile_id="tile=i0_j0", cell=(0, 0), token_sequence=ef.tokens, sub_c_features=[]
    )
    assert diags == []  # decodes to a valid LineString within the cell bound


def test_check_decodability_measures_position_error_against_original():
    line = LineString([(10.0, 10.0), (30.0, 10.0)])
    ef = encode_feature(line, semantic_tag="highway=residential")
    sub_c = [
        {"feature_class": 0, "geometry": wkb_dumps(line, byte_order=1), "source_feature_id": "r"}
    ]
    diags, errors = check_decodability(
        tile_id="tile=i0_j0", cell=(0, 0), token_sequence=ef.tokens, sub_c_features=sub_c
    )
    assert diags == []
    assert len(errors) == 1
    # 0.5m magnitude-quantum round-trip on axis-aligned integers -> sub-metre error.
    assert errors[0]["position_err_m"] < 1.0
    assert errors[0]["angle_err_deg"] < 5.0


def test_structural_bound_flags_far_vertex():
    # A decoded geometry with a vertex far outside the cell is a gate failure.
    from cfm.data.sub_g.seam_decodability import _VERTEX_BOUND_M, _max_abs_coord

    far = {"type": "LineString", "coordinates": [[0.0, 0.0], [10000.0, 0.0]]}
    assert _max_abs_coord(far) > _VERTEX_BOUND_M

from __future__ import annotations

from shapely.geometry import LineString, Polygon
from shapely.wkb import dumps as wkb_dumps

from cfm.data.sub_g.seam_macro_geometry import (
    check_density,
    check_road_skeleton,
    recompute_density_ratio,
    recompute_road_crossing_count,
)


def _square_wkb(side: float) -> bytes:
    return wkb_dumps(Polygon([(0, 0), (side, 0), (side, side), (0, side), (0, 0)]), byte_order=1)


def _line_wkb() -> bytes:
    return wkb_dumps(LineString([(0, 0), (10, 0)]), byte_order=1)


def test_recompute_density_ratio_matches_formula():
    # cell area 1000 m^2; two building squares of area 100 each -> ratio 0.2.
    features = [
        {"feature_class": 1, "geometry": _square_wkb(10.0), "source_feature_id": "b1"},
        {"feature_class": 1, "geometry": _square_wkb(10.0), "source_feature_id": "b2"},
        {"feature_class": 0, "geometry": _line_wkb(), "source_feature_id": "r1"},  # road: ignored
    ]
    assert abs(recompute_density_ratio(features, cell_area_m2=1000.0) - 0.2) < 1e-9


def test_check_density_flags_mismatch_with_signature():
    # one building square area 100 / cell 1000 -> ratio 0.1 -> bucket 1 (edges 0,0.05,0.15,0.35).
    # sub-D stored bucket 3 -> a 2-step mismatch.
    diags = check_density(
        tile_id="tile=i0_j0",
        per_cell_features={
            (0, 0): [{"feature_class": 1, "geometry": _square_wkb(10.0), "source_feature_id": "b"}]
        },
        per_cell_area={(0, 0): 1000.0},
        sub_d_density_by_cell={(0, 0): 3},
    )
    assert len(diags) == 1
    assert diags[0].invariant_name == "density_bucket_matches_footprint"
    assert "bucket" in diags[0].signature


def test_check_density_passes_on_agreement():
    # ratio 0.1 -> bucket 1; sub-D agrees -> no diagnostic.
    diags = check_density(
        tile_id="tile=i0_j0",
        per_cell_features={
            (0, 0): [{"feature_class": 1, "geometry": _square_wkb(10.0), "source_feature_id": "b"}]
        },
        per_cell_area={(0, 0): 1000.0},
        sub_d_density_by_cell={(0, 0): 1},
    )
    assert diags == []


def test_check_density_skips_inactive_cell_slot():
    # sub-D stores None on an inactive cell slot -> not checked.
    diags = check_density(
        tile_id="tile=i0_j0",
        per_cell_features={(0, 0): []},
        per_cell_area={(0, 0): 1000.0},
        sub_d_density_by_cell={(0, 0): None},
    )
    assert diags == []


def test_recompute_road_crossing_count_filters_non_road():
    features = [
        {"feature_class": 0, "source_feature_id": "r1"},
        {"feature_class": 1, "source_feature_id": "b1"},
    ]
    crossings = [
        {"source_feature_id": "r1", "lower_cell_i": 0, "lower_cell_j": 0, "axis": 0},
        {"source_feature_id": "b1", "lower_cell_i": 0, "lower_cell_j": 0, "axis": 0},
    ]
    counts = recompute_road_crossing_count(features, crossings)
    assert counts[(0, 0, 0)] == 1  # only the road crossing counted


def test_check_road_skeleton_flags_mismatch():
    # two road crossings on edge (0,0,0) -> count 2 -> bucket 1 (edges 0,1,4,9).
    # sub-D stored bucket 3 -> mismatch.
    features = [{"feature_class": 0, "source_feature_id": "r1"}]
    crossings = [
        {"source_feature_id": "r1", "lower_cell_i": 0, "lower_cell_j": 0, "axis": 0},
        {"source_feature_id": "r1", "lower_cell_i": 0, "lower_cell_j": 0, "axis": 0},
    ]
    diags = check_road_skeleton(
        tile_id="tile=i0_j0",
        features=features,
        crossings=crossings,
        sub_d_skeleton_by_edge={(0, 0, 0): 3},
    )
    assert len(diags) == 1
    assert diags[0].invariant_name == "road_skeleton_bucket_matches_crossings"

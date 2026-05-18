"""Tests for cell partitioning, split-at-boundaries, and crossing-record derivation.

Per spec §8.1 (split-at-boundaries), §8.2 (8-column schema + canonical sort key),
§8.3 (7 edge cases), and §11.5 (sliver-drop with strict β user-threshold).
"""

from __future__ import annotations

from shapely.geometry import LineString, Polygon
from shapely.geometry import box as shapely_box

from cfm.data.sub_c.coords import TILE_SIZE_M
from cfm.data.sub_c.enums import AXIS, EVENT_TYPE, encode_enum
from cfm.data.sub_c.geom import (
    CellSubFeature,
    apply_sliver_drop,
    partition_into_cells,
)


def _tile_box(tile_i: int = 0, tile_j: int = 0):
    return shapely_box(
        tile_i * TILE_SIZE_M,
        tile_j * TILE_SIZE_M,
        (tile_i + 1) * TILE_SIZE_M,
        (tile_j + 1) * TILE_SIZE_M,
    )


def test_split_at_boundaries_single_cell_feature_emits_one_subfeature_zero_crossings():
    # A small road entirely inside cell (0, 0) of tile (0, 0)
    feature = LineString([(50, 50), (200, 200)])
    subfeatures, crossings = partition_into_cells(
        features=[(feature, "road_001", "road")],
        tile_i=0,
        tile_j=0,
    )
    assert len(subfeatures) == 1
    assert subfeatures[0].cell_i == 0
    assert subfeatures[0].cell_j == 0
    assert subfeatures[0].source_feature_id == "road_001"
    assert len(crossings) == 0


def test_split_at_boundaries_multi_cell_road_emits_n_subfeatures_n_minus_one_crossings():
    # Road crossing 3 cells: (0,0) → (1,0) → (2,0) along y=125 from x=100 to x=600
    feature = LineString([(100, 125), (600, 125)])
    subfeatures, crossings = partition_into_cells(
        features=[(feature, "road_002", "road")],
        tile_i=0,
        tile_j=0,
    )
    assert len(subfeatures) == 3
    assert {s.cell_i for s in subfeatures} == {0, 1, 2}
    assert all(s.cell_j == 0 for s in subfeatures)
    assert len(crossings) == 2
    # All share source_feature_id
    assert all(c.source_feature_id == "road_002" for c in crossings)
    # Both are axis=x crossings (vertical edges between (i,j) and (i+1,j))
    assert all(c.axis == encode_enum(AXIS, "x") for c in crossings)


def test_corner_crossing_emits_two_records_one_per_axis():
    # Road passes through exact corner (250, 250) — cell-boundary corner
    feature = LineString([(125, 125), (375, 375)])
    subfeatures, crossings = partition_into_cells(
        features=[(feature, "road_003", "road")],
        tile_i=0,
        tile_j=0,
    )
    # Two sub-features: (0,0) and (1,1)
    assert len(subfeatures) == 2
    # Two crossing records: one x-axis edge, one y-axis edge
    axis_codes = sorted(c.axis for c in crossings)
    assert axis_codes == [encode_enum(AXIS, "x"), encode_enum(AXIS, "y")]
    # Both share source_feature_id
    assert all(c.source_feature_id == "road_003" for c in crossings)


def test_polygon_interior_ring_emits_multiple_records_per_source_feature():
    # Polygon with a hole crossing a cell boundary
    shell = Polygon(
        [(100, 100), (700, 100), (700, 700), (100, 700)],
        holes=[[(200, 200), (600, 200), (600, 600), (200, 600)]],
    )
    _subfeatures, crossings = partition_into_cells(
        features=[(shell, "building_004", "building")],
        tile_i=0,
        tile_j=0,
    )
    # Many crossings expected; at least one should have ring_index >= 1 (interior ring)
    assert any(c.ring_index >= 1 for c in crossings)


def test_co_linear_entirety_emits_zero_records_attaches_to_higher_ij():
    # Road lying exactly on cell-boundary y=250
    feature = LineString([(100, 250), (200, 250)])
    subfeatures, crossings = partition_into_cells(
        features=[(feature, "road_005", "road")],
        tile_i=0,
        tile_j=0,
    )
    assert len(subfeatures) == 1
    # Half-open: y=250 attaches to cell j=1 (the higher-j side)
    assert subfeatures[0].cell_j == 1
    assert len(crossings) == 0


def test_touch_but_not_cross_emits_zero_records():
    # Road ending exactly at boundary x=250
    feature = LineString([(100, 100), (250, 100)])
    subfeatures, crossings = partition_into_cells(
        features=[(feature, "road_006", "road")],
        tile_i=0,
        tile_j=0,
    )
    # Per spec §8.3: touch-but-not-cross means feature wholly in one cell, no crossing record
    assert len(subfeatures) == 1
    assert len(crossings) == 0


def test_partial_co_linearity_emits_interval_event_with_extent():
    # Polygon with one shell segment lying along a cell boundary;
    # body spans both adjacent cells
    poly = Polygon([(100, 100), (400, 100), (400, 400), (100, 400)])
    # Cell boundary at x=250 cuts the polygon
    _subfeatures, crossings = partition_into_cells(
        features=[(poly, "building_007", "building")],
        tile_i=0,
        tile_j=0,
    )
    # Crossings include intervals on the x=250 edge
    interval_crossings = [
        c for c in crossings if c.event_type == encode_enum(EVENT_TYPE, "interval")
    ]
    assert len(interval_crossings) >= 1
    assert all(c.edge_extent_length_m > 0 for c in interval_crossings)


def test_multi_crossing_same_edge_emits_alternating_enter_exit_sorted_by_position():
    # Zigzag road crossing x=250 three times
    feature = LineString(
        [
            (100, 100),
            (350, 200),
            (200, 300),
            (350, 400),
        ]
    )
    _subfeatures, crossings = partition_into_cells(
        features=[(feature, "road_008", "road")],
        tile_i=0,
        tile_j=0,
    )
    # All crossings on edge between cells (0, *) and (1, *) at axis=x
    x_edge_crossings = [
        c for c in crossings if c.lower_cell_i == 0 and c.axis == encode_enum(AXIS, "x")
    ]
    # Sort by edge_position_m
    sorted_crossings = sorted(x_edge_crossings, key=lambda c: c.edge_position_m)
    # event_types should alternate enter / exit
    types = [c.event_type for c in sorted_crossings]
    assert len(types) >= 3
    for i in range(1, len(types)):
        assert types[i] != types[i - 1], "enter/exit must alternate"


def test_crossings_sort_key_canonical():
    # Mix of axes, source_feature_ids, ring_indices, event_types, positions
    poly = Polygon([(100, 100), (400, 100), (400, 400), (100, 400)])
    road = LineString([(100, 100), (300, 300)])
    _subfeatures, crossings = partition_into_cells(
        features=[(poly, "polygon_009", "building"), (road, "road_010", "road")],
        tile_i=0,
        tile_j=0,
    )
    # Per spec §8.2: sort key (lower_cell_i, lower_cell_j, axis,
    # source_feature_id, ring_index, edge_position_m, event_type)
    sort_keys = [
        (
            c.lower_cell_i,
            c.lower_cell_j,
            c.axis,
            c.source_feature_id,
            c.ring_index,
            c.edge_position_m,
            c.event_type,
        )
        for c in crossings
    ]
    assert sort_keys == sorted(sort_keys)


def test_apply_sliver_drop_removes_below_threshold_features():
    # A normal feature + a tiny sliver
    normal = LineString([(0, 0), (100, 0)])
    sliver_line = LineString([(0, 0), (0.005, 0)])  # 5 mm
    normal_sub = CellSubFeature(
        cell_i=0,
        cell_j=0,
        source_feature_id="n",
        feature_class="road",
        geometry=normal,
        geometry_type="LineString",
    )
    sliver_sub = CellSubFeature(
        cell_i=0,
        cell_j=0,
        source_feature_id="s",
        feature_class="road",
        geometry=sliver_line,
        geometry_type="LineString",
    )
    kept = apply_sliver_drop(
        [normal_sub, sliver_sub],
        area_threshold_m2=0.01,
        length_threshold_m=0.01,
    )
    assert len(kept) == 1
    assert kept[0].source_feature_id == "n"

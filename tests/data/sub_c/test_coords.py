from __future__ import annotations

import pytest
from shapely.geometry import LineString, Point, Polygon

from cfm.data.sub_c.coords import (
    CELL_SIZE_M,
    SVY21_EPSG_CODE,
    TILE_SIZE_M,
    cell_id_within_tile,
    clip_to_admin_polygon,
    densify_polygon,
    partition_into_tiles,
    reproject_geometry_to_svy21,
    reproject_lonlat_to_svy21,
    tile_id_from_svy21,
)


def test_svy21_epsg_code_is_3414():
    assert SVY21_EPSG_CODE == 3414


def test_reproject_lonlat_to_svy21_byte_deterministic():
    # Marina Bay, Singapore: ~103.8587°E, 1.2839°N → SVY21 ~30000, 29000 m
    lon, lat = 103.8587, 1.2839
    x1, y1 = reproject_lonlat_to_svy21(lon, lat)
    x2, y2 = reproject_lonlat_to_svy21(lon, lat)
    assert x1 == x2 and y1 == y2  # bit-identical
    assert 25000 < x1 < 35000
    assert 25000 < y1 < 35000


def test_tile_id_from_svy21_point_basic():
    # SVY21 (3000, 9000) → tile (1, 4) under 2km grid
    assert tile_id_from_svy21(3000.0, 9000.0) == (1, 4)


def test_tile_id_half_open_boundary_at_exact_x_equals_2000():
    # x = 2000.0 belongs to tile i=1 (NOT i=0) per half-open [i*2000, (i+1)*2000)
    assert tile_id_from_svy21(2000.0, 5000.0) == (1, 2)
    # x just below 2000.0 belongs to tile i=0
    assert tile_id_from_svy21(1999.999999, 5000.0) == (0, 2)


def test_tile_id_half_open_boundary_at_exact_y_equals_2000():
    assert tile_id_from_svy21(5000.0, 2000.0) == (2, 1)
    assert tile_id_from_svy21(5000.0, 1999.999999) == (2, 0)


def test_co_linear_feature_attaches_to_higher_ij_cell():
    # A point on the boundary x=4000.0 lives in tile i=2, not i=1
    assert tile_id_from_svy21(4000.0, 4000.0) == (2, 2)


def test_tile_size_constant():
    assert TILE_SIZE_M == 2000


def test_reproject_geometry_to_svy21_point_in_marina_bay_range():
    """Geometry-level reprojection wraps the scalar reprojection;
    verifies the shapely.ops.transform pathway works."""
    p = Point(103.8587, 1.2839)  # Marina Bay (lon, lat)
    out = reproject_geometry_to_svy21(p)
    assert isinstance(out, Point)
    assert 25000 < out.x < 35000
    assert 25000 < out.y < 35000


# --- Task 3 tests: cell partitioning, densification, clipping, tile inventory ---


def test_cell_size_constant():
    assert CELL_SIZE_M == 250


def test_cell_id_within_tile_half_open_at_exact_x_equals_250():
    # x_in_tile=250 → cell ci=1 (NOT 0)
    assert cell_id_within_tile(250.0, 500.0) == (1, 2)
    assert cell_id_within_tile(249.999999, 500.0) == (0, 2)


def test_cell_id_within_tile_half_open_at_exact_y_equals_250():
    assert cell_id_within_tile(500.0, 250.0) == (2, 1)
    assert cell_id_within_tile(500.0, 249.999999) == (2, 0)


def test_densify_polygon_with_none_returns_unchanged():
    poly = Polygon([(0, 0), (10000, 0), (10000, 10000), (0, 10000)])
    out = densify_polygon(poly, max_edge_length_m=None)
    assert out.equals(poly)
    # Same vertex count
    assert len(list(out.exterior.coords)) == len(list(poly.exterior.coords))


def test_densify_polygon_with_real_threshold_inserts_vertices_on_long_edges():
    # 4-vertex square 10km on each side; with 1000m threshold should densify
    poly = Polygon([(0, 0), (10000, 0), (10000, 10000), (0, 10000)])
    out = densify_polygon(poly, max_edge_length_m=1000.0)
    out_n = len(list(out.exterior.coords))
    assert out_n > len(list(poly.exterior.coords))
    # Every edge now <= 1000m
    import itertools

    coords = list(out.exterior.coords)
    for a, b in itertools.pairwise(coords):
        edge_len = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        assert edge_len <= 1000.0 + 1e-6


def test_clip_to_admin_polygon_clips_in_svy21():
    # admin polygon = unit-square 1km x 1km at origin in SVY21
    admin = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
    # a feature linestring extending past admin
    feature = LineString([(500, 500), (1500, 500)])
    clipped = clip_to_admin_polygon([feature], admin)
    assert len(clipped) == 1
    assert clipped[0].length == pytest.approx(500.0)  # clipped at x=1000


def test_partition_into_tiles_emits_inventory_sorted_by_ij():
    # admin polygon covering tiles (0,0), (0,1), (1,0), (1,1)
    admin = Polygon([(0, 0), (4000, 0), (4000, 4000), (0, 4000)])
    inventory = partition_into_tiles(admin)
    keys = list(inventory.keys())
    assert keys == sorted(keys)  # lexicographic sort by (i, j)
    assert (0, 0) in keys
    assert (1, 1) in keys

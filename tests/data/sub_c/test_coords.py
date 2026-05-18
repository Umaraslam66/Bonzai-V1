from __future__ import annotations

from shapely.geometry import Point

from cfm.data.sub_c.coords import (
    SVY21_EPSG_CODE,
    TILE_SIZE_M,
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

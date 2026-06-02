from __future__ import annotations

import itertools

import pytest
from shapely.geometry import LineString, Point, Polygon

from cfm.data.sub_c.coords import (
    CELL_SIZE_M,
    SVY21_EPSG_CODE,
    TILE_SIZE_M,
    cell_id_within_tile,
    clip_to_admin_polygon,
    densify_polygon,
    epsg_label_from_crs,
    partition_into_tiles,
    reproject_geometry_to_svy21,
    reproject_lonlat_to_svy21,
    tile_id_from_svy21,
    utm_epsg_for_centroid,
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


# --- Multi-region (Q1): centroid -> ETRS89/UTM zone selection ---------------
# Policy (PI lock 2026-06-02): one conformal UTM zone PER CITY, chosen from the
# city centroid, applied to all that city's tiles. Conformal (angle-preserving)
# is required for a shape model scored on rectilinearity. ETRS89/UTM-North codes
# are EPSG:258zz for European zones 28-38 (e.g. zone 33 -> EPSG:25833).


def test_utm_epsg_for_centroid_berlin_is_utm33n():
    # Berlin centroid ~13.40°E, 52.52°N -> UTM zone 33 -> ETRS89/UTM33N
    assert utm_epsg_for_centroid(13.40, 52.52) == "EPSG:25833"


def test_utm_epsg_for_centroid_madrid_is_utm30n():
    # Madrid centroid ~-3.70°E, 40.42°N -> UTM zone 30 -> ETRS89/UTM30N
    assert utm_epsg_for_centroid(-3.70, 40.42) == "EPSG:25830"


def test_utm_epsg_for_centroid_zone_boundary_is_lower_zone_at_exact_multiple():
    # Zone 33 covers [12.0, 18.0); half-open lower bound like the tile grid.
    # lon exactly 12.0 -> zone 33; just below -> zone 32.
    assert utm_epsg_for_centroid(12.0, 50.0) == "EPSG:25833"
    assert utm_epsg_for_centroid(11.999999, 50.0) == "EPSG:25832"


def test_utm_epsg_for_centroid_is_deterministic():
    a = utm_epsg_for_centroid(13.40, 52.52)
    b = utm_epsg_for_centroid(13.40, 52.52)
    assert a == b


def test_utm_epsg_for_centroid_rejects_non_european_longitude():
    # Tokyo (~139.7°E) is outside the ETRS89/UTM European zone range (28-38).
    # Refuse rather than silently emit a wrong/undefined ETRS89 code.
    with pytest.raises(ValueError):
        utm_epsg_for_centroid(139.69, 35.69)


def test_utm_epsg_for_centroid_rejects_southern_hemisphere():
    # ETRS89/UTM-North policy is for Europe (northern hemisphere).
    with pytest.raises(ValueError):
        utm_epsg_for_centroid(13.40, -33.92)


# --- Multi-region (Q1): tile-dir label derived from region CRS --------------
# The EPSG prefix in tile=EPSG{code}_iN_jM disambiguates tile indices across
# cities in different UTM zones (Berlin i5_j3 != Madrid i5_j3). Load-bearing,
# not cosmetic. ONE source derives the label so no module re-hardcodes it.


def test_epsg_label_from_crs_strips_colon_for_utm():
    assert epsg_label_from_crs("EPSG:25833") == "EPSG25833"


def test_epsg_label_from_crs_singapore_backcompat_is_exactly_EPSG3414():
    # The locked Singapore tile-dir label is "EPSG3414"; must not drift.
    assert epsg_label_from_crs("EPSG:3414") == "EPSG3414"


def test_epsg_label_from_crs_rejects_missing_authority():
    with pytest.raises(ValueError):
        epsg_label_from_crs("25833")


def test_epsg_label_from_crs_rejects_non_numeric_code():
    with pytest.raises(ValueError):
        epsg_label_from_crs("EPSG:abc")

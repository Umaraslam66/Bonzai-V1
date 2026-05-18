"""Tests for cfm.data.sub_c.sea_mask.

11 named tests covering:
- derive_sea_polygons: class filter, subtype filter, union, pipeline-order guard
- apply_sea_mask: drop rule, coastal keep, inland keep, admin-clipped denominator
- compute_sea_overlap_fraction: intersects-for-points, fast-path None, cache arg
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from shapely import wkb
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

from cfm.data.sub_c.epsilon import EPS_RATIO
from cfm.data.sub_c.sea_mask import (
    apply_sea_mask,
    compute_sea_overlap_fraction,
    derive_sea_polygons,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wkb(geom: BaseGeometry) -> bytes:
    return wkb.dumps(geom)


def _make_base_table(rows: list[dict]) -> pa.Table:
    """Build a minimal pyarrow Table with columns: class, subtype, geometry."""
    classes = pa.array([r["class"] for r in rows], type=pa.string())
    subtypes = pa.array([r["subtype"] for r in rows], type=pa.string())
    geometries = pa.array([_wkb(r["geometry"]) for r in rows], type=pa.binary())
    return pa.table({"class": classes, "subtype": subtypes, "geometry": geometries})


# Synthetic geometry helpers (tile-local coords; no real CRS needed)
_OCEAN_POLY = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
_STRAIT_POLY = Polygon([(200, 0), (300, 0), (300, 50), (200, 50)])
_BAY_POLY = Polygon([(400, 0), (500, 0), (500, 80), (400, 80)])
_LAKE_POLY = Polygon([(600, 0), (700, 0), (700, 60), (600, 60)])
_RIVER_LINE = LineString([(800, 0), (900, 50)])
_SUBTYPE_OCEAN_POLY = Polygon([(1000, 0), (1100, 0), (1100, 90), (1000, 90)])


# ---------------------------------------------------------------------------
# Test 1: derive_sea_polygons filters class IN {ocean, strait, bay}
# ---------------------------------------------------------------------------


def test_derive_sea_polygons_filters_class_in_ocean_strait_bay():
    rows = [
        {"class": "ocean", "subtype": "ocean", "geometry": _OCEAN_POLY},
        {"class": "strait", "subtype": "other", "geometry": _STRAIT_POLY},
        {"class": "bay", "subtype": "other", "geometry": _BAY_POLY},
        {"class": "lake", "subtype": "lake", "geometry": _LAKE_POLY},
        {"class": "river", "subtype": "river", "geometry": _RIVER_LINE},
    ]
    table = _make_base_table(rows)
    result = derive_sea_polygons(table)

    # Should contain points inside the sea polygons
    assert result.contains(Point(50, 50))  # inside _OCEAN_POLY
    assert result.contains(Point(250, 25))  # inside _STRAIT_POLY
    assert result.contains(Point(450, 40))  # inside _BAY_POLY

    # Should NOT contain points inside the non-sea geometries
    assert not result.contains(Point(650, 30))  # inside _LAKE_POLY
    assert not result.intersects(Point(850, 25))  # on _RIVER_LINE — not sea


# ---------------------------------------------------------------------------
# Test 2: derive_sea_polygons filters subtype = ocean (regardless of class)
# ---------------------------------------------------------------------------


def test_derive_sea_polygons_filters_subtype_ocean():
    rows = [
        # class is NOT in SEA_CLASS_VALUES, but subtype = ocean → should be included
        {"class": "water", "subtype": "ocean", "geometry": _SUBTYPE_OCEAN_POLY},
        {"class": "lake", "subtype": "lake", "geometry": _LAKE_POLY},
    ]
    table = _make_base_table(rows)
    result = derive_sea_polygons(table)

    assert result.contains(Point(1050, 45))  # inside _SUBTYPE_OCEAN_POLY
    assert not result.contains(Point(650, 30))  # inside _LAKE_POLY — not sea


# ---------------------------------------------------------------------------
# Test 3: derive_sea_polygons returns a unioned multipolygon
# ---------------------------------------------------------------------------


def test_derive_sea_polygons_returns_multipolygon_union():
    rows = [
        {"class": "ocean", "subtype": "ocean", "geometry": _OCEAN_POLY},
        {"class": "strait", "subtype": "other", "geometry": _STRAIT_POLY},
    ]
    table = _make_base_table(rows)
    result = derive_sea_polygons(table)

    # The union covers both polygons
    assert result.area == pytest.approx(_OCEAN_POLY.area + _STRAIT_POLY.area)

    # Both representative points are inside the union
    assert result.contains(Point(50, 50))
    assert result.contains(Point(250, 25))


# ---------------------------------------------------------------------------
# Test 4: pipeline-ordering guard — raw vs "policied" (sea rows removed)
# ---------------------------------------------------------------------------


def test_derive_sea_polygons_runs_against_raw_base_not_policied_themes():
    """Verify that derive_sea_polygons must see raw themes (with sea rows).

    Simulate apply_missing_value_policy removing sea-defining rows (not-in-vocab
    drop_row) by constructing a "policied" table that only contains non-sea rows.
    derive_sea_polygons on raw → non-empty result.
    derive_sea_polygons on policied → empty MultiPolygon (no sea rows left).
    """
    sea_row = {"class": "ocean", "subtype": "ocean", "geometry": _OCEAN_POLY}
    non_sea_row = {"class": "lake", "subtype": "lake", "geometry": _LAKE_POLY}

    raw_table = _make_base_table([sea_row, non_sea_row])
    # "policied" table: simulate drop_row removing sea-defining rows
    policied_table = _make_base_table([non_sea_row])

    raw_result = derive_sea_polygons(raw_table)
    policied_result = derive_sea_polygons(policied_table)

    # Raw: contains the sea polygon
    assert not raw_result.is_empty, "raw table must yield non-empty sea polygons"
    assert raw_result.area == pytest.approx(_OCEAN_POLY.area)

    # Policied: no sea rows → empty MultiPolygon
    assert policied_result.is_empty, "policied table must yield empty sea result"
    assert isinstance(policied_result, MultiPolygon)


# ---------------------------------------------------------------------------
# Test 5: apply_sea_mask drops pure-sea cell with zero non-sea features
# ---------------------------------------------------------------------------


def test_apply_sea_mask_drops_pure_sea_cell_with_zero_non_sea_features():
    # Cell box entirely covered by ocean
    cell_box = Polygon([(0, 0), (250, 0), (250, 250), (0, 250)])
    sea_poly = Polygon([(0, 0), (300, 0), (300, 300), (0, 300)])  # covers cell entirely

    sea_water_fraction, _water_fraction, drop_flag = apply_sea_mask(
        cell_box_admin_clipped=cell_box,
        cell_features=[],
        sea_polygons_svy21=sea_poly,
    )

    assert sea_water_fraction == pytest.approx(1.0)
    assert drop_flag is True


# ---------------------------------------------------------------------------
# Test 6: apply_sea_mask keeps coastal cell with bridge (non-sea feature present)
# ---------------------------------------------------------------------------


def test_apply_sea_mask_keeps_coastal_cell_with_bridge():
    # Cell mostly ocean, but there is a non-sea feature (bridge/road)
    cell_box = Polygon([(0, 0), (250, 0), (250, 250), (0, 250)])
    sea_poly = Polygon([(0, 0), (300, 0), (300, 300), (0, 300)])  # covers cell entirely

    # Simulate one non-sea feature in the cell (e.g., a bridge segment)
    bridge_feature = object()  # any non-sea stand-in

    sea_water_fraction, _water_fraction, drop_flag = apply_sea_mask(
        cell_box_admin_clipped=cell_box,
        cell_features=[bridge_feature],
        sea_polygons_svy21=sea_poly,
    )

    assert sea_water_fraction == pytest.approx(1.0)
    # High sea fraction BUT non-sea features present → keep
    assert drop_flag is False


# ---------------------------------------------------------------------------
# Test 7: apply_sea_mask keeps inland water (MacRitchie-like) cell
# ---------------------------------------------------------------------------


def test_apply_sea_mask_keeps_inland_water_macritchie_like_cell():
    # Cell with inland lake — sea_polygons do not overlap at all
    cell_box = Polygon([(0, 0), (250, 0), (250, 250), (0, 250)])
    # Sea polygon is far away — no overlap with cell
    sea_poly = Polygon([(5000, 5000), (6000, 5000), (6000, 6000), (5000, 6000)])

    # inland water features are non-sea features
    inland_lake_feature = object()

    sea_water_fraction, _water_fraction, drop_flag = apply_sea_mask(
        cell_box_admin_clipped=cell_box,
        cell_features=[inland_lake_feature],
        sea_polygons_svy21=sea_poly,
    )

    assert sea_water_fraction == pytest.approx(0.0)
    assert drop_flag is False


# ---------------------------------------------------------------------------
# Test 8: apply_sea_mask uses admin-clipped denominator (not raw cell box)
# ---------------------------------------------------------------------------


def test_apply_sea_mask_uses_admin_clipped_denominator():
    # Raw cell: 250m x 250m = 62500 m2
    # Admin-clipped cell: 125m x 250m = 31250 m2 (half the raw cell)
    admin_clipped_box = Polygon([(0, 0), (125, 0), (125, 250), (0, 250)])

    # Sea polygon covers the right half of the raw cell — but admin only covers left half
    # So sea overlap with admin-clipped box = 0 (sea is to the right, admin is left)
    sea_poly = Polygon([(125, 0), (250, 0), (250, 250), (125, 250)])

    sea_water_fraction, _, _ = apply_sea_mask(
        cell_box_admin_clipped=admin_clipped_box,
        cell_features=[],
        sea_polygons_svy21=sea_poly,
    )

    # Sea doesn't overlap admin-clipped box at all → fraction = 0
    assert sea_water_fraction == pytest.approx(0.0)

    # Now test with sea covering the admin-clipped area entirely
    sea_poly_full = Polygon([(0, 0), (200, 0), (200, 300), (0, 300)])
    sea_water_fraction_full, _, drop_flag_full = apply_sea_mask(
        cell_box_admin_clipped=admin_clipped_box,
        cell_features=[],
        sea_polygons_svy21=sea_poly_full,
    )

    # Admin-clipped area = 31250; sea covers all of it → fraction = 1.0
    assert sea_water_fraction_full == pytest.approx(1.0)
    assert drop_flag_full is True


# ---------------------------------------------------------------------------
# Test 9: compute_sea_overlap_fraction uses intersects predicate for points
# ---------------------------------------------------------------------------


def test_sea_overlap_fraction_uses_intersects_predicate_for_points():
    """A point on the boundary of a sea polygon intersects it → overlap = 1.0.

    Spec §9.3 precision item 1: coastline POIs count as sea-adjacent via
    INTERSECTS predicate (not contains). A point on the boundary returns
    intersects=True even though contains=False.
    """
    # Sea polygon
    sea_poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])

    # Point exactly on the boundary of the sea polygon
    point_on_boundary = Point(50, 0)  # on the bottom edge of sea_poly

    # Sanity check our test setup: intersects=True, contains=False
    assert sea_poly.intersects(point_on_boundary)
    assert not sea_poly.contains(point_on_boundary)

    overlap = compute_sea_overlap_fraction(
        feature_geom=point_on_boundary,
        feature_type="Point",
        cell_local_sea_geometry=sea_poly,
    )
    assert overlap == pytest.approx(1.0)

    # Point fully inside the sea polygon → also 1.0
    point_inside = Point(50, 50)
    overlap_inside = compute_sea_overlap_fraction(
        feature_geom=point_inside,
        feature_type="Point",
        cell_local_sea_geometry=sea_poly,
    )
    assert overlap_inside == pytest.approx(1.0)

    # Point clearly outside → 0.0
    point_outside = Point(200, 200)
    overlap_outside = compute_sea_overlap_fraction(
        feature_geom=point_outside,
        feature_type="Point",
        cell_local_sea_geometry=sea_poly,
    )
    assert overlap_outside == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 10: fast-path — cell_local_sea_geometry=None → 0.0 immediately
# ---------------------------------------------------------------------------


def test_sea_overlap_fraction_zero_when_cell_sea_water_fraction_zero():
    """Fast-path: None cell_local_sea_geometry → 0.0 without any geometry ops.

    Represents a cell whose sea_water_fraction is 0 (no sea overlap); the
    caller passes None to skip per-feature computation.
    """
    point_geom = Point(50, 50)

    result = compute_sea_overlap_fraction(
        feature_geom=point_geom,
        feature_type="Point",
        cell_local_sea_geometry=None,
    )
    assert result == pytest.approx(0.0)

    # Also applies to LineString and Polygon types
    line_geom = LineString([(0, 0), (100, 100)])
    result_line = compute_sea_overlap_fraction(
        feature_geom=line_geom,
        feature_type="LineString",
        cell_local_sea_geometry=None,
    )
    assert result_line == pytest.approx(0.0)

    poly_geom = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])
    result_poly = compute_sea_overlap_fraction(
        feature_geom=poly_geom,
        feature_type="Polygon",
        cell_local_sea_geometry=None,
    )
    assert result_poly == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 11: compute_sea_overlap_fraction uses passed cell_local_sea_geometry directly
# ---------------------------------------------------------------------------


def test_sea_overlap_fraction_caches_cell_local_sea_geometry():
    """The function uses the passed cell_local_sea_geometry directly — no recompute.

    The caller is responsible for computing cell_local_sea_geometry once per cell
    and passing it to each feature call. This test verifies the function honors
    the passed argument: when a known polygon is passed, the computed ratio matches
    the expected geometric truth.
    """
    # Sea polygon: right half of a 200x200 area
    sea_poly = Polygon([(100, 0), (200, 0), (200, 200), (100, 200)])

    # LineString: goes from x=0 to x=200 (50% overlaps sea, 50% does not)
    line_geom = LineString([(0, 100), (200, 100)])  # total length = 200
    # Overlap with sea_poly: from x=100 to x=200 = length 100 → ratio = 0.5

    result = compute_sea_overlap_fraction(
        feature_geom=line_geom,
        feature_type="LineString",
        cell_local_sea_geometry=sea_poly,
    )
    assert result == pytest.approx(0.5)

    # Polygon: left half of the 200x200 area (no sea overlap)
    poly_no_overlap = Polygon([(0, 0), (100, 0), (100, 200), (0, 200)])
    result_no_overlap = compute_sea_overlap_fraction(
        feature_geom=poly_no_overlap,
        feature_type="Polygon",
        cell_local_sea_geometry=sea_poly,
    )
    # The boundary at x=100 is shared — intersection area is 0 (boundaries touch only)
    assert result_no_overlap == pytest.approx(0.0, abs=EPS_RATIO)

    # Polygon: right half — fully in sea
    poly_full_overlap = Polygon([(100, 0), (200, 0), (200, 200), (100, 200)])
    result_full = compute_sea_overlap_fraction(
        feature_geom=poly_full_overlap,
        feature_type="Polygon",
        cell_local_sea_geometry=sea_poly,
    )
    assert result_full == pytest.approx(1.0)

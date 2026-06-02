"""Tests for cfm.data.sub_c.sea_mask.

Tests covering:
- derive_sea_polygons: class filter, subtype filter, union, pipeline-order guard
- derive_inland_water_polygons: class filter, geometry union, empty result
- apply_sea_mask: drop rule, coastal keep, inland keep, admin-clipped denominator,
  epsilon boundary, inland-water combined water_fraction
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
    INLAND_WATER_CLASSES,
    apply_sea_mask,
    compute_sea_overlap_fraction,
    derive_inland_water_polygons,
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
# Test 9 (new): apply_sea_mask epsilon boundary at exactly 1.0 - EPS_RATIO
# ---------------------------------------------------------------------------


def test_sea_water_fraction_epsilon_boundary_at_exactly_1_minus_eps():
    """At sea_water_fraction = 1.0 - EPS_RATIO exactly (zero features), drop_flag is True.
    Just below (1.0 - 2*EPS_RATIO), drop_flag is False.

    Verifies the alpha structural-boundary EPSILON application at 1.0 per spec §9.2 + §14.4.
    """
    # Build a cell box with admin-clipped area = 1.0 m² exactly for arithmetic clarity
    cell_box = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    # Construct sea_polygons whose intersection with cell_box has area = 1.0 - EPS_RATIO
    boundary_sea = Polygon([(0, 0), (1, 0), (1, 1 - EPS_RATIO), (0, 1 - EPS_RATIO)])
    swf, _, drop_at = apply_sea_mask(
        cell_box_admin_clipped=cell_box,
        cell_features=[],  # zero non-sea features
        sea_polygons_svy21=boundary_sea,
    )
    assert abs(swf - (1.0 - EPS_RATIO)) < 1e-12  # FP-equality check
    assert drop_at is True, "exactly at 1.0 - EPS_RATIO must drop (>= boundary)"

    # Just below boundary: area = 1.0 - 2*EPS_RATIO; must NOT drop
    below_sea = Polygon([(0, 0), (1, 0), (1, 1 - 2 * EPS_RATIO), (0, 1 - 2 * EPS_RATIO)])
    swf2, _, drop_below = apply_sea_mask(
        cell_box_admin_clipped=cell_box,
        cell_features=[],
        sea_polygons_svy21=below_sea,
    )
    assert swf2 < 1.0 - EPS_RATIO
    assert drop_below is False, "below 1.0 - EPS_RATIO must NOT drop"


# ---------------------------------------------------------------------------
# Test 11: compute_sea_overlap_fraction uses intersects predicate for points
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
# Test 12: fast-path — cell_local_sea_geometry=None → 0.0 immediately
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
# Test 13: compute_sea_overlap_fraction uses passed cell_local_sea_geometry directly
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


# ---------------------------------------------------------------------------
# Fix #1: derive_inland_water_polygons tests
# ---------------------------------------------------------------------------


def _make_base_table_with_geom(rows: list[dict]) -> pa.Table:
    """Build a minimal pyarrow Table with columns: class, subtype, geometry.
    (Same schema as _make_base_table above but local to this section.)
    """
    return pa.table(
        {
            "class": pa.array([r["class"] for r in rows], type=pa.string()),
            "subtype": pa.array([r.get("subtype", "") for r in rows], type=pa.string()),
            "geometry": pa.array([wkb.dumps(r["geometry"]) for r in rows], type=pa.binary()),
        }
    )


# Shared synthetic geometries for inland-water tests
_RESERVOIR_POLY = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
_RIVER_LINE = LineString([(200, 50), (400, 50)])  # LineString river (zero area)
_CANAL_POLY = Polygon([(500, 0), (600, 0), (600, 80), (500, 80)])
_LAKE_POLY = Polygon([(700, 0), (800, 0), (800, 60), (700, 60)])
_OCEAN_POLY2 = Polygon([(900, 0), (1000, 0), (1000, 100), (900, 100)])  # sea, NOT inland


def test_derive_inland_water_polygons_filters_inland_classes():
    """derive_inland_water_polygons returns a union of inland-water rows only.

    Rows with class IN INLAND_WATER_CLASSES appear in the output.
    Rows with sea-defining classes (ocean) do NOT appear.

    This verifies Fix #1: the function correctly separates inland water from sea.
    """
    rows = [
        {"class": "reservoir", "geometry": _RESERVOIR_POLY},
        {"class": "river", "geometry": _RIVER_LINE},
        {"class": "canal", "geometry": _CANAL_POLY},
        {"class": "ocean", "geometry": _OCEAN_POLY2},  # sea — must be excluded
    ]
    table = _make_base_table_with_geom(rows)
    result = derive_inland_water_polygons(table)

    # Reservoir and canal polygons contribute area to the union.
    assert result.contains(Point(50, 50))  # inside _RESERVOIR_POLY
    assert result.contains(Point(550, 40))  # inside _CANAL_POLY

    # The ocean polygon must NOT appear in the inland-water union.
    assert not result.contains(Point(950, 50))  # inside _OCEAN_POLY2


def test_derive_inland_water_polygons_returns_empty_when_no_inland_rows():
    """Returns an empty MultiPolygon when no rows match INLAND_WATER_CLASSES."""
    rows = [
        {"class": "ocean", "geometry": _OCEAN_POLY2},
        {"class": "bay", "geometry": Polygon([(1100, 0), (1200, 0), (1200, 50), (1100, 50)])},
    ]
    table = _make_base_table_with_geom(rows)
    result = derive_inland_water_polygons(table)
    assert result.is_empty
    assert isinstance(result, MultiPolygon)


def test_derive_inland_water_polygons_includes_linestring_rivers_no_crash():
    """derive_inland_water_polygons handles LineString rivers without crashing.

    unary_union of a mixed Polygon + LineString set returns a GeometryCollection.
    The .area of a GeometryCollection sums only Polygon components, so LineString
    rivers contribute zero to water_fraction area computations — which is correct.
    """
    rows = [
        {"class": "river", "geometry": _RIVER_LINE},  # LineString — zero area
        {"class": "lake", "geometry": _LAKE_POLY},  # Polygon — contributes area
    ]
    table = _make_base_table_with_geom(rows)
    result = derive_inland_water_polygons(table)

    # Must not raise; result area should be dominated by the lake polygon.
    assert result.area == pytest.approx(_LAKE_POLY.area)
    # The line is geometrically inside the union (intersects it).
    assert result.intersects(_RIVER_LINE)


def test_derive_inland_water_polygons_covers_all_spec_classes():
    """Every class in INLAND_WATER_CLASSES appears in the frozenset."""
    expected = frozenset(
        {
            "river",
            "stream",
            "reservoir",
            "lake",
            "pond",
            "swimming_pool",
            "canal",
            "drain",
        }
    )
    assert INLAND_WATER_CLASSES == expected


# ---------------------------------------------------------------------------
# Fix #1: apply_sea_mask with inland_water_polygons_svy21 tests
# ---------------------------------------------------------------------------


def test_apply_sea_mask_with_inland_water_returns_combined_water_fraction():
    """water_fraction = sea_water_fraction + inland_fraction when inland polygons passed.

    Construct a cell box with:
      - 25% covered by sea polygon (sea_water_fraction = 0.25)
      - 25% covered by inland lake polygon (inland_fraction = 0.25, non-overlapping)
    Expected: water_fraction = 0.50; sea_water_fraction = 0.25; drop_flag = False.

    This is the core Fix #1 correctness assertion.
    """
    # Cell box: 100x100 = 10000 m²
    cell_box = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])

    # Sea polygon covers left 25 columns: 25 x 100 = 2500 m²  → fraction = 0.25
    sea_poly = Polygon([(0, 0), (25, 0), (25, 100), (0, 100)])

    # Inland lake covers right 25 columns: 25 x 100 = 2500 m² → fraction = 0.25
    inland_poly = Polygon([(75, 0), (100, 0), (100, 100), (75, 100)])

    # One feature in the cell (prevents drop_flag)
    dummy_feature = object()

    sea_wf, wf, drop_flag = apply_sea_mask(
        cell_box_admin_clipped=cell_box,
        cell_features=[dummy_feature],
        sea_polygons_svy21=sea_poly,
        inland_water_polygons_svy21=inland_poly,
    )

    assert sea_wf == pytest.approx(0.25)
    assert wf == pytest.approx(0.50)
    assert drop_flag is False


def test_apply_sea_mask_without_inland_water_behaves_as_before():
    """When inland_water_polygons_svy21 is None (default), water_fraction == sea_water_fraction.

    Backward-compatibility assertion: existing callers that don't pass
    inland_water_polygons_svy21 get the same result as before Fix #1.
    """
    cell_box = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    sea_poly = Polygon([(0, 0), (50, 0), (50, 100), (0, 100)])  # 50%

    sea_wf, wf, drop_flag = apply_sea_mask(
        cell_box_admin_clipped=cell_box,
        cell_features=[],
        sea_polygons_svy21=sea_poly,
        # inland_water_polygons_svy21 not passed → defaults to None
    )

    assert sea_wf == pytest.approx(0.50)
    assert wf == pytest.approx(sea_wf)  # water_fraction == sea_water_fraction
    # 0.5 < 1 - EPS_RATIO so drop_flag = False (no features but below threshold)
    assert drop_flag is False


def test_apply_sea_mask_water_fraction_clamped_to_1_when_fp_overflow():
    """water_fraction is clamped to 1.0 when sea + inland FP sum exceeds 1.0.

    Sea and inland don't geometrically overlap, but FP arithmetic can push the
    sum slightly above 1.0.  The min(1.0, ...) cap handles this.

    We test the cap by using large near-full sea and near-full inland polygons
    whose precise areas may sum to > 1.0 due to floating-point geometry ops.
    """
    # Cell box: 1 x 1 exact.
    cell_box = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    # sea = entire cell (sea_fraction = 1.0)
    sea_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    # inland = entire cell (inland_fraction = 1.0) — unrealistic but tests cap
    inland_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    # One feature prevents drop
    dummy_feature = object()

    _, wf, _ = apply_sea_mask(
        cell_box_admin_clipped=cell_box,
        cell_features=[dummy_feature],
        sea_polygons_svy21=sea_poly,
        inland_water_polygons_svy21=inland_poly,
    )

    assert wf <= 1.0, f"water_fraction must be clamped to 1.0, got {wf}"


def test_apply_sea_mask_inland_water_empty_geometry_is_noop():
    """Passing an empty inland geometry produces water_fraction == sea_water_fraction."""
    cell_box = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    sea_poly = Polygon([(0, 0), (30, 0), (30, 100), (0, 100)])  # 30%
    empty_inland = MultiPolygon()  # empty

    sea_wf, wf, _ = apply_sea_mask(
        cell_box_admin_clipped=cell_box,
        cell_features=[],
        sea_polygons_svy21=sea_poly,
        inland_water_polygons_svy21=empty_inland,
    )

    assert sea_wf == pytest.approx(0.30)
    assert wf == pytest.approx(sea_wf)  # empty inland adds nothing


# ---------------------------------------------------------------------------
# Fix #2: _derive_region_lookup_svy21 and _lookup_admin_region tests
# ---------------------------------------------------------------------------

# Import the helpers — they are module-private but tested directly.
from cfm.data.sub_c.coords import region_coords  # noqa: E402
from cfm.data.sub_c.pipeline import (  # noqa: E402
    _derive_region_lookup_svy21,
    _lookup_admin_region,
)

# Singapore-region coords for the admin-region lookup tests (was an implicit
# module-level SVY21 transformer; now an explicit region-bound RegionCoords).
_SG_COORDS = region_coords("EPSG:3414")


def _make_divisions_table(rows: list[dict]) -> pa.Table:
    """Build a minimal divisions theme table for testing region lookup."""
    names_type = pa.struct([pa.field("primary", pa.string())])
    return pa.table(
        {
            "subtype": pa.array([r["subtype"] for r in rows], type=pa.string()),
            "country": pa.array([r["country"] for r in rows], type=pa.string()),
            "names": pa.array(
                [{"primary": r["name"]} for r in rows],
                type=names_type,
            ),
            "geometry": pa.array([wkb.dumps(r["geometry"]) for r in rows], type=pa.binary()),
        }
    )


def test_derive_region_lookup_filters_subtype_region_and_country():
    """_derive_region_lookup_svy21 returns only rows where subtype='region' AND country='SG'."""
    # Two SG regions + one MY region + one SG locality
    rows = [
        {
            "subtype": "region",
            "country": "SG",
            "name": "Central Region",
            "geometry": Polygon([(103.8, 1.3), (103.9, 1.3), (103.9, 1.4), (103.8, 1.4)]),
        },
        {
            "subtype": "region",
            "country": "SG",
            "name": "West Region",
            "geometry": Polygon([(103.6, 1.2), (103.8, 1.2), (103.8, 1.4), (103.6, 1.4)]),
        },
        {
            "subtype": "region",
            "country": "MY",
            "name": "Johor",
            "geometry": Polygon([(103.5, 1.1), (104.5, 1.1), (104.5, 1.8), (103.5, 1.8)]),
        },
        {
            "subtype": "locality",  # wrong subtype — should be excluded
            "country": "SG",
            "name": "Some Locality",
            "geometry": Polygon([(103.81, 1.31), (103.82, 1.31), (103.82, 1.32), (103.81, 1.32)]),
        },
    ]
    table = _make_divisions_table(rows)
    lookup = _derive_region_lookup_svy21(table, _SG_COORDS, country_code="SG")

    # Should have exactly the 2 SG regions (sorted alphabetically by name).
    assert len(lookup) == 2
    names = [p[0] for p in lookup]
    assert names == ["Central Region", "West Region"]  # alphabetical order


def test_derive_region_lookup_returns_empty_when_no_region_rows():
    """Returns empty list when no rows match subtype='region' for the given country."""
    rows = [
        {
            "subtype": "locality",
            "country": "SG",
            "name": "Some Locality",
            "geometry": Polygon([(103.8, 1.3), (103.9, 1.3), (103.9, 1.4), (103.8, 1.4)]),
        },
    ]
    table = _make_divisions_table(rows)
    lookup = _derive_region_lookup_svy21(table, _SG_COORDS, country_code="SG")
    assert lookup == []


def test_lookup_admin_region_centroid_in_region():
    """_lookup_admin_region returns the matching region name for a point inside a polygon."""
    # Build a synthetic lookup with two non-overlapping regions in SVY21 coords.
    # Region A: x ∈ [0, 500], y ∈ [0, 500]
    # Region B: x ∈ [500, 1000], y ∈ [0, 500]
    region_a = Polygon([(0, 0), (500, 0), (500, 500), (0, 500)])
    region_b = Polygon([(500, 0), (1000, 0), (1000, 500), (500, 500)])
    lookup = [("Alpha Region", region_a), ("Beta Region", region_b)]

    centroid_in_a = Point(250, 250)
    centroid_in_b = Point(750, 250)

    assert _lookup_admin_region(centroid_in_a, lookup) == "Alpha Region"
    assert _lookup_admin_region(centroid_in_b, lookup) == "Beta Region"


def test_lookup_admin_region_centroid_outside_all_regions():
    """_lookup_admin_region returns None when the point doesn't fall inside any region."""
    region = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    lookup = [("Some Region", region)]

    outside_point = Point(500, 500)
    assert _lookup_admin_region(outside_point, lookup) is None


def test_lookup_admin_region_empty_lookup_returns_none():
    """_lookup_admin_region returns None when region_lookup_svy21 is empty."""
    centroid = Point(100, 100)
    assert _lookup_admin_region(centroid, []) is None


def test_derive_region_lookup_sorted_alphabetically_for_determinism():
    """Lookup list is sorted alphabetically by region name (byte-determinism)."""
    rows = [
        {
            "subtype": "region",
            "country": "SG",
            "name": "West Region",
            "geometry": Polygon([(103.6, 1.2), (103.8, 1.2), (103.8, 1.4), (103.6, 1.4)]),
        },
        {
            "subtype": "region",
            "country": "SG",
            "name": "East Region",
            "geometry": Polygon([(103.8, 1.3), (104.0, 1.3), (104.0, 1.4), (103.8, 1.4)]),
        },
        {
            "subtype": "region",
            "country": "SG",
            "name": "Central Region",
            "geometry": Polygon([(103.7, 1.25), (103.85, 1.25), (103.85, 1.35), (103.7, 1.35)]),
        },
    ]
    table = _make_divisions_table(rows)
    lookup = _derive_region_lookup_svy21(table, _SG_COORDS, country_code="SG")
    names = [p[0] for p in lookup]
    assert names == sorted(names), f"Region lookup not alphabetically sorted: {names}"

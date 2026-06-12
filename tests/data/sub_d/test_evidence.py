"""Tests for sub-D Layer-1 derivation evidence primitives.

Evidence is *raw, deterministic counts and ratios* over sub-C tables. No
threshold lookups, no class assignment, no vocab dependencies. Frequency
analysis (Layer 3) consumes these metrics to propose vocab cuts; the locked
vocab (Phase A->B handoff) consumes the cuts to assign final classes. Layer 1
must stay purely empirical.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from shapely import wkb as shapely_wkb
from shapely.geometry import Polygon

from cfm.data.sub_d.enums import FeatureClass, MetricNamespace, SlotKind
from cfm.data.sub_d.evidence import (
    CELL_DENSITY_DERIVATION_VERSION,
    ROAD_SKELETON_DERIVATION_VERSION,
    TILE_POPULATION_DENSITY_DERIVATION_VERSION,
    ZONING_DERIVATION_VERSION,
    EvidenceMetric,
    derive_cell_scope_metrics,
    derive_density_evidence,
    derive_road_skeleton_evidence,
    derive_tile_population_density_evidence,
    derive_zoning_evidence,
)


def _wkb(polygon: Polygon) -> bytes:
    """Little-endian WKB so the test bytes match sub-C's NDR convention."""
    return shapely_wkb.dumps(polygon, hex=False, byte_order=1)


def test_cell_scope_from_sub_c_cells_marks_complement_masked():
    # Cells (0,0), (1,1), (7,7) are active in sub-C. The other 61 are masked.
    cells = pa.table(
        {
            "cell_i": [0, 1, 7],
            "cell_j": [0, 1, 7],
            "cell_area_admin_clipped_m2": [100.0, 100.0, 100.0],
        }
    )
    scope = derive_cell_scope_metrics(cells)

    assert len(scope) == 64
    assert scope[(0, 0)] is True
    assert scope[(1, 1)] is True
    assert scope[(7, 7)] is True
    # Spot-check the masked complement.
    assert scope[(0, 1)] is False
    assert scope[(3, 4)] is False
    assert scope[(7, 6)] is False
    # Total counts match.
    assert sum(1 for v in scope.values() if v) == 3
    assert sum(1 for v in scope.values() if not v) == 61


def test_zoning_evidence_counts_class_composition_without_density_thresholds():
    # Two active cells. Cell (0,0) has 2 roads, 3 buildings, 1 poi, 0 base.
    # Cell (1,1) is active but has no features at all (empty cell).
    cells = pa.table(
        {
            "cell_i": [0, 1],
            "cell_j": [0, 1],
            "cell_area_admin_clipped_m2": [100.0, 200.0],
        }
    )
    features = pa.table(
        {
            "cell_i": [0, 0, 0, 0, 0, 0],
            "cell_j": [0, 0, 0, 0, 0, 0],
            "feature_class": [0, 0, 1, 1, 1, 2],  # 2 roads, 3 buildings, 1 poi
            "source_feature_id": ["r1", "r2", "b1", "b2", "b3", "p1"],
            "geometry": [b"", b"", b"", b"", b"", b""],
        }
    )

    metrics = derive_zoning_evidence(features, cells)

    # 4 class counts per active cell, 2 active cells -> 8 metrics.
    assert len(metrics) == 8
    # All zoning evidence sits in the ZONING metric namespace.
    assert all(m.metric_namespace == MetricNamespace.ZONING for m in metrics)
    assert all(m.slot_kind == SlotKind.CELL for m in metrics)
    # All metric names are class-composition counts; no density-ratio metric
    # appears in this primitive layer. This pins "zoning evidence is raw
    # class counts only — never footprint-ratio intensity."
    assert all(m.metric_name.startswith("feature_count_") for m in metrics)
    assert not any("footprint_ratio" in m.metric_name for m in metrics)
    assert not any("density" in m.metric_name for m in metrics)

    # Cell (0,0): slot_index 0. Verify each class count.
    by_name_00 = {m.metric_name: m for m in metrics if m.slot_index == 0}
    assert by_name_00["feature_count_road"].value == 2
    assert by_name_00["feature_count_building"].value == 3
    assert by_name_00["feature_count_poi"].value == 1
    assert by_name_00["feature_count_base"].value == 0

    # Cell (1,1): slot_index 9. Empty cell -> all-zero counts emitted.
    by_name_11 = {m.metric_name: m for m in metrics if m.slot_index == 9}
    assert by_name_11["feature_count_road"].value == 0
    assert by_name_11["feature_count_building"].value == 0
    assert by_name_11["feature_count_poi"].value == 0
    assert by_name_11["feature_count_base"].value == 0


def test_density_evidence_uses_building_footprint_ratio_only():
    # Cell (0,0): area 100 m^2, one 25 m^2 building -> ratio 0.25.
    # Cell (1,1): area 100 m^2, no buildings -> ratio 0.0.
    # Both cells active. Non-building features (roads, POIs) must not affect
    # the ratio — density is *building* footprint ratio only.
    cells = pa.table(
        {
            "cell_i": [0, 1],
            "cell_j": [0, 1],
            "cell_area_admin_clipped_m2": [100.0, 100.0],
        }
    )
    building_geom = Polygon([(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)])
    features = pa.table(
        {
            "cell_i": [0, 0, 0, 1],
            "cell_j": [0, 0, 0, 1],
            "feature_class": [0, 1, 2, 2],  # road, building, poi (cell 0,0); poi (cell 1,1)
            "source_feature_id": ["r1", "b1", "p1", "p2"],
            "geometry": [b"", _wkb(building_geom), b"", b""],
        }
    )

    metrics = derive_density_evidence(features, cells)

    # Exactly 1 metric per active cell, both in CELL_DENSITY namespace,
    # always named building_footprint_ratio. No class counts, no POI mixing in.
    assert len(metrics) == 2
    assert all(m.metric_namespace == MetricNamespace.CELL_DENSITY for m in metrics)
    assert all(m.slot_kind == SlotKind.CELL for m in metrics)
    assert all(m.metric_name == "building_footprint_ratio" for m in metrics)
    # Density layer does NOT count POIs or use class composition.
    assert not any("feature_count" in m.metric_name for m in metrics)
    assert not any("poi" in m.metric_name.lower() for m in metrics)

    by_slot = {m.slot_index: m for m in metrics}
    assert by_slot[0].value == pytest.approx(0.25)
    assert by_slot[9].value == 0.0


def test_road_evidence_joins_crossings_to_features_by_source_feature_id():
    # Two roads (r1, r2). Three crossings, all on r1 or r2. No buildings.
    features = pa.table(
        {
            "cell_i": [0, 0],
            "cell_j": [0, 1],
            "feature_class": [0, 0],
            "source_feature_id": ["r1", "r2"],
            "geometry": [b"", b""],
        }
    )
    crossings = pa.table(
        {
            "source_feature_id": ["r1", "r2", "r1"],
            "lower_cell_i": [0, 0, 1],
            "lower_cell_j": [0, 1, 1],
            "axis": [0, 1, 0],
        }
    )

    metrics = derive_road_skeleton_evidence(crossings, features)

    # One metric per internal edge slot (112), all road_crossing_count.
    assert len(metrics) == 112
    assert all(m.metric_namespace == MetricNamespace.ROAD_SKELETON for m in metrics)
    assert all(m.slot_kind == SlotKind.INTERNAL_EDGE for m in metrics)
    assert all(m.metric_name == "road_crossing_count" for m in metrics)
    assert sorted(m.slot_index for m in metrics) == list(range(112))

    # Verify the join produced the right per-edge counts using the documented
    # internal-edge slot ordering: axis=0 occupies slots 0-55 row-major over
    # (lower_i, lower_j) with lower_i in [0,6], lower_j in [0,7]; axis=1
    # occupies slots 56-111 row-major over lower_i in [0,7], lower_j in [0,6].
    by_slot = {m.slot_index: m for m in metrics}
    # (lower=(0,0), axis=0) -> slot 0*8 + 0 = 0; r1 contributes 1.
    assert by_slot[0].value == 1
    # (lower=(1,1), axis=0) -> slot 1*8 + 1 = 9; r1 contributes 1.
    assert by_slot[9].value == 1
    # (lower=(0,1), axis=1) -> slot 56 + 0*7 + 1 = 57; r2 contributes 1.
    assert by_slot[57].value == 1
    # An untouched edge stays at 0.
    assert by_slot[1].value == 0


def test_road_evidence_ignores_non_road_crossings():
    # b1 is a building, r1 is a road. Crossings come from BOTH source ids;
    # only the road-source ones should appear in road evidence.
    features = pa.table(
        {
            "cell_i": [0, 0],
            "cell_j": [0, 1],
            "feature_class": [0, 1],  # r1 road, b1 building
            "source_feature_id": ["r1", "b1"],
            "geometry": [b"", b""],
        }
    )
    crossings = pa.table(
        {
            "source_feature_id": ["r1", "b1", "b1", "r1"],
            "lower_cell_i": [0, 0, 1, 2],
            "lower_cell_j": [0, 1, 1, 2],
            "axis": [0, 0, 1, 1],
        }
    )

    metrics = derive_road_skeleton_evidence(crossings, features)
    total = sum(int(m.value) for m in metrics)
    # Only the two r1 crossings contribute; the two b1 crossings are filtered out.
    assert total == 2

    # EvidenceMetric dataclass is frozen and records derivation_version.
    sample = metrics[0]
    assert isinstance(sample, EvidenceMetric)
    # Per-namespace derivation versions: road metrics carry the
    # ROAD_SKELETON version, never the zoning/density/tile-pop versions.
    assert sample.derivation_version == ROAD_SKELETON_DERIVATION_VERSION
    # FeatureClass enum mirrors sub-C encoding without importing from cfm.data.sub_c.
    assert FeatureClass.ROAD == 0
    assert FeatureClass.BUILDING == 1


def test_per_namespace_derivation_versions_are_stamped_independently():
    # One synthetic tile with mixed features so every derive_* function emits
    # at least one metric.
    cells = pa.table(
        {
            "cell_i": [0],
            "cell_j": [0],
            "cell_area_admin_clipped_m2": [100.0],
        }
    )
    building = Polygon([(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)])
    features = pa.table(
        {
            "cell_i": [0, 0],
            "cell_j": [0, 0],
            "feature_class": [0, 1],
            "source_feature_id": ["r1", "b1"],
            "geometry": [b"", _wkb(building)],
        }
    )
    crossings = pa.table(
        {
            "source_feature_id": ["r1"],
            "lower_cell_i": [0],
            "lower_cell_j": [0],
            "axis": [0],
        }
    )

    zoning = derive_zoning_evidence(features, cells)
    density = derive_density_evidence(features, cells)
    roads = derive_road_skeleton_evidence(crossings, features)
    tile_pop = derive_tile_population_density_evidence(cells, features)

    assert zoning and all(m.derivation_version == ZONING_DERIVATION_VERSION for m in zoning)
    assert density and all(m.derivation_version == CELL_DENSITY_DERIVATION_VERSION for m in density)
    assert roads and all(m.derivation_version == ROAD_SKELETON_DERIVATION_VERSION for m in roads)
    assert tile_pop and all(
        m.derivation_version == TILE_POPULATION_DENSITY_DERIVATION_VERSION for m in tile_pop
    )

    # All four namespaces' tile_pop metrics live at slot_kind=TILE, slot_index=0.
    assert all(m.slot_kind == SlotKind.TILE for m in tile_pop)
    assert all(m.slot_index == 0 for m in tile_pop)
    assert all(m.metric_namespace == MetricNamespace.TILE_POPULATION_DENSITY for m in tile_pop)
    # Multiple candidate proxies emitted as distinct metric_names so the
    # reviewer picks one at Gate 2. Layer 1 does not pre-commit to a formula.
    proxy_names = {m.metric_name for m in tile_pop}
    expected = {
        "mean_building_footprint_ratio",
        "area_weighted_building_density",
        "median_building_footprint_ratio",
        "p75_building_footprint_ratio",
    }
    assert proxy_names == expected
    # All proxy values are floats; values are consistent for a single-cell
    # tile with one 16 m^2 building in a 100 m^2 cell (ratio == 0.16).
    by_name = {m.metric_name: float(m.value) for m in tile_pop}
    assert by_name["mean_building_footprint_ratio"] == pytest.approx(0.16)
    assert by_name["area_weighted_building_density"] == pytest.approx(0.16)
    assert by_name["median_building_footprint_ratio"] == pytest.approx(0.16)
    assert by_name["p75_building_footprint_ratio"] == pytest.approx(0.16)


# ---------------------------------------------------------------------------
# Direct unit tests for derive_tile_population_density_evidence (F2)
# ---------------------------------------------------------------------------


_EXPECTED_TILE_POPULATION_DENSITY_PROXIES: set[str] = {
    "mean_building_footprint_ratio",
    "area_weighted_building_density",
    "median_building_footprint_ratio",
    "p75_building_footprint_ratio",
}


def test_tile_population_density_emits_exactly_four_proxies_with_documented_shape():
    """Cardinality + slot-kind + namespace + bounded-value contract."""
    cells = pa.table(
        {
            "cell_i": [0, 1],
            "cell_j": [0, 1],
            "cell_area_admin_clipped_m2": [100.0, 100.0],
        }
    )
    # Two buildings of different sizes in two active cells, so mean/median/
    # area-weighted differ enough to be distinguishable.
    big = Polygon([(0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0)])  # 64 m^2
    small = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])  # 4 m^2
    features = pa.table(
        {
            "cell_i": [0, 1],
            "cell_j": [0, 1],
            "feature_class": [1, 1],
            "source_feature_id": ["b_big", "b_small"],
            "geometry": [_wkb(big), _wkb(small)],
        }
    )

    metrics = derive_tile_population_density_evidence(cells, features)

    # F2 cardinality lock: exactly four rows per tile, one per proxy.
    assert len(metrics) == 4

    # F2 proxy-name lock: exact set, no extra, no missing.
    proxy_names = {m.metric_name for m in metrics}
    assert proxy_names == _EXPECTED_TILE_POPULATION_DENSITY_PROXIES

    # F2 schema lock: every row is slot_kind=TILE, slot_index=0,
    # metric_namespace=TILE_POPULATION_DENSITY, derivation_version stamped.
    for m in metrics:
        assert m.slot_kind == SlotKind.TILE
        assert m.slot_index == 0
        assert m.metric_namespace == MetricNamespace.TILE_POPULATION_DENSITY
        assert m.derivation_version == TILE_POPULATION_DENSITY_DERIVATION_VERSION

    # F2 value-bound lock: every proxy value is a float in [0.0, 1.0] (a
    # ratio of building area to cell area is bounded above by 1.0; any value
    # outside that range is a derivation bug).
    by_name = {m.metric_name: float(m.value) for m in metrics}
    for name, value in by_name.items():
        assert isinstance(value, float), f"{name} value not float"
        assert 0.0 <= value <= 1.0, f"{name}={value} outside [0, 1]"

    # The four proxies disagree on this fixture, which is the whole point of
    # emitting all of them: mean and area-weighted are (0.64+0.04)/2 = 0.34
    # (equal cell areas make these match); median sits between the two
    # ratios; p75 sits at the higher value.
    assert by_name["mean_building_footprint_ratio"] == pytest.approx(0.34)
    assert by_name["area_weighted_building_density"] == pytest.approx(0.34)
    # median of [0.04, 0.64] depends on the implementation; my _percentile_of
    # returns the value at the rounded index, which lands at the higher entry.
    assert by_name["median_building_footprint_ratio"] in (
        pytest.approx(0.04),
        pytest.approx(0.64),
    )
    assert by_name["p75_building_footprint_ratio"] == pytest.approx(0.64)


def test_tile_population_density_empty_tile_returns_zero_for_all_four_proxies():
    """Pinned empty-tile convention: all four proxies return 0.0, never NaN.

    See evidence.py docstring for the rationale. The reviewer's instruction
    is to lock this convention in a test so it cannot drift silently.
    """
    cells = pa.table(
        {
            "cell_i": [],
            "cell_j": [],
            "cell_area_admin_clipped_m2": [],
        },
        schema=pa.schema(
            [
                pa.field("cell_i", pa.int8()),
                pa.field("cell_j", pa.int8()),
                pa.field("cell_area_admin_clipped_m2", pa.float64()),
            ]
        ),
    )
    features = pa.table(
        {
            "cell_i": [],
            "cell_j": [],
            "feature_class": [],
            "source_feature_id": [],
            "geometry": [],
        },
        schema=pa.schema(
            [
                pa.field("cell_i", pa.int8()),
                pa.field("cell_j", pa.int8()),
                pa.field("feature_class", pa.int8()),
                pa.field("source_feature_id", pa.string()),
                pa.field("geometry", pa.binary()),
            ]
        ),
    )

    metrics = derive_tile_population_density_evidence(cells, features)

    # Still emit all four — dense metric stream regardless of tile content.
    assert len(metrics) == 4
    assert {m.metric_name for m in metrics} == _EXPECTED_TILE_POPULATION_DENSITY_PROXIES
    # All four exactly 0.0. NOT NaN, NOT None.
    import math

    for m in metrics:
        value = float(m.value)
        assert value == 0.0, f"empty-tile convention requires {m.metric_name}=0.0, got {value}"
        assert not math.isnan(value), (
            f"empty-tile convention forbids NaN; {m.metric_name} returned NaN"
        )


def test_tile_population_density_proxies_disagree_on_skewed_distributions():
    """When per-cell ratios are skewed, mean and area_weighted/median/p75
    diverge — which is exactly the point of emitting all four candidates.
    The Gate 2 reviewer picks the proxy with the most discriminative power.

    Three active cells of equal area, one with a large building and two
    with no buildings. The mean and area-weighted density are both 1/3 of
    the large ratio (because equal cell areas), but the median is 0.0
    (two zero cells dominate the middle of the sorted list).
    """
    cells = pa.table(
        {
            "cell_i": [0, 1, 2],
            "cell_j": [0, 0, 0],
            "cell_area_admin_clipped_m2": [100.0, 100.0, 100.0],
        }
    )
    big = Polygon([(0.0, 0.0), (9.0, 0.0), (9.0, 9.0), (0.0, 9.0)])  # 81 m^2
    features = pa.table(
        {
            "cell_i": [0],
            "cell_j": [0],
            "feature_class": [1],
            "source_feature_id": ["b1"],
            "geometry": [_wkb(big)],
        }
    )

    metrics = derive_tile_population_density_evidence(cells, features)
    by_name = {m.metric_name: float(m.value) for m in metrics}

    # mean and area-weighted both = 0.81/3 = 0.27.
    assert by_name["mean_building_footprint_ratio"] == pytest.approx(0.27)
    assert by_name["area_weighted_building_density"] == pytest.approx(0.27)
    # median of [0.0, 0.0, 0.81] sits at index round(0.5 * 2) = 1, i.e. 0.0.
    assert by_name["median_building_footprint_ratio"] == pytest.approx(0.0)
    # p75 of [0.0, 0.0, 0.81] sits at index round(0.75 * 2) = 2, i.e. 0.81.
    assert by_name["p75_building_footprint_ratio"] == pytest.approx(0.81)

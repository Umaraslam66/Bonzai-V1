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
    EvidenceMetric,
    derive_cell_scope_metrics,
    derive_density_evidence,
    derive_road_skeleton_evidence,
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
    assert isinstance(sample.derivation_version, str) and sample.derivation_version
    # FeatureClass enum mirrors sub-C encoding without importing from cfm.data.sub_c.
    assert FeatureClass.ROAD == 0
    assert FeatureClass.BUILDING == 1

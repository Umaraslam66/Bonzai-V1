"""Layer-1 derivation evidence primitives (sub-D spec section 11.3).

Each function returns deterministic empirical metrics over sub-C tables.
No thresholds, no class assignment, no vocab lookups. Frequency analysis
(Task 6) consumes these metrics to propose vocab cuts; the locked macro
vocab (Phase A->B handoff at Gate 2) consumes the cuts to assign final
classes. Keeping Layer 1 free of thresholds is what lets Gate 2 review the
*data*, not the algorithm.

Density of the metric stream is deliberately asymmetric across slot kinds:

- **Cell-keyed metrics (zoning, density): sparse over non-active cells.**
  These metrics are semantically conditioned on the cell being ``ACTIVE``.
  Emitting ``feature_count_road=0`` for a ``FULLY_MASKED`` cell would
  conflate "we looked and found nothing" with "we don't look here at all,"
  which would mislead frequency analysis when it computes distributions
  over kept-cell evidence.

- **Edge-keyed metrics (road skeleton): dense over all 112 internal edge
  slots.** These metrics count *physical crossings*, which exist
  independent of cell scope. Uniform file shape across tiles also keeps
  Task 6's frequency-analysis joins cheap and lets future analyses
  reconsider how SCOPE_BOUNDARY edges feed into vocab derivation without
  re-extracting.

The rule: cell-keyed metrics depend on cell semantics, edge-keyed metrics
depend on physical feature presence. Task 6 is responsible for re-applying
the scope filter (joining edge metrics to ``derive_internal_edge_scope`` and
restricting to ``ACTIVE`` when proposing vocab cuts) — Layer 1 does not
silently drop edge rows by scope.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa
from shapely import wkb as shapely_wkb

from cfm.data.sub_d.enums import FeatureClass, MetricNamespace, SlotKind
from cfm.data.sub_d.lattice import CELL_GRID_SIZE, iter_internal_edge_slots

#: Bumped whenever the derivation algorithm changes. Recorded on every
#: EvidenceMetric so the validator (Task 13) can detect drift between
#: pinned-vocab assumptions and the algorithm that produced the metrics.
DERIVATION_VERSION: str = "1.0"


@dataclass(frozen=True)
class EvidenceMetric:
    """One row of derivation_evidence.parquet (spec section 11.3).

    The on-disk schema splits ``value`` into typed columns (value_float,
    value_int, value_string, value_bool) plus a value_type tag. In memory
    we keep a single ``value`` field as a union and let the parquet writer
    (Task 9) dispatch on Python type.
    """

    slot_kind: SlotKind
    slot_index: int
    metric_namespace: MetricNamespace
    metric_name: str
    value: float | int | str | bool
    derivation_version: str


# ---------------------------------------------------------------------------
# Cell scope
# ---------------------------------------------------------------------------


def derive_cell_scope_metrics(cells: pa.Table) -> dict[tuple[int, int], bool]:
    """Return ``{(cell_i, cell_j): active}`` for every cell in the 8x8 lattice.

    ``active=True`` iff the (i, j) pair appears in ``cells["cell_i"]`` and
    ``cells["cell_j"]``. Sub-C only writes cells it keeps, so the complement
    is implicitly masked.
    """
    present: set[tuple[int, int]] = set()
    cell_i = cells["cell_i"].to_pylist()
    cell_j = cells["cell_j"].to_pylist()
    for i, j in zip(cell_i, cell_j):
        present.add((int(i), int(j)))
    return {
        (i, j): (i, j) in present
        for i in range(CELL_GRID_SIZE)
        for j in range(CELL_GRID_SIZE)
    }


# ---------------------------------------------------------------------------
# Zoning evidence (class composition counts)
# ---------------------------------------------------------------------------


def derive_zoning_evidence(features: pa.Table, cells: pa.Table) -> list[EvidenceMetric]:
    """Emit feature-count metrics per (active cell, FeatureClass).

    Layer 1 zoning evidence is intentionally limited to raw counts of each
    feature class within each active cell. Footprint-ratio or density-based
    intensity belongs to the cell-density namespace (Layer 1
    ``derive_density_evidence``) so the two evidence streams can be analysed
    independently at Layer 3 before any vocab cut is proposed.
    """
    scope = derive_cell_scope_metrics(cells)

    counts: dict[tuple[int, int, int], int] = {}
    feat_i = features["cell_i"].to_pylist()
    feat_j = features["cell_j"].to_pylist()
    feat_class = features["feature_class"].to_pylist()
    for i, j, c in zip(feat_i, feat_j, feat_class):
        key = (int(i), int(j), int(c))
        counts[key] = counts.get(key, 0) + 1

    metrics: list[EvidenceMetric] = []
    for (cell_i, cell_j), active in scope.items():
        if not active:
            continue
        slot_index = cell_i * CELL_GRID_SIZE + cell_j
        for fc in (FeatureClass.ROAD, FeatureClass.BUILDING, FeatureClass.POI, FeatureClass.BASE):
            metrics.append(
                EvidenceMetric(
                    slot_kind=SlotKind.CELL,
                    slot_index=slot_index,
                    metric_namespace=MetricNamespace.ZONING,
                    metric_name=f"feature_count_{fc.name.lower()}",
                    value=counts.get((cell_i, cell_j, int(fc)), 0),
                    derivation_version=DERIVATION_VERSION,
                )
            )
    return metrics


# ---------------------------------------------------------------------------
# Density evidence (building footprint ratio)
# ---------------------------------------------------------------------------


def derive_density_evidence(features: pa.Table, cells: pa.Table) -> list[EvidenceMetric]:
    """Emit ``building_footprint_ratio`` per active cell.

    ratio = sum(area of building polygons whose feature_class==BUILDING) /
            cell_area_admin_clipped_m2.

    Only building geometries are parsed; road and POI rows are skipped before
    any WKB deserialisation. ``ratio`` is 0.0 for active cells with no
    buildings (so the metric stream is dense over all active cells).
    """
    scope = derive_cell_scope_metrics(cells)

    cell_area: dict[tuple[int, int], float] = {}
    ci = cells["cell_i"].to_pylist()
    cj = cells["cell_j"].to_pylist()
    ca = cells["cell_area_admin_clipped_m2"].to_pylist()
    for i, j, area in zip(ci, cj, ca):
        cell_area[(int(i), int(j))] = float(area)

    building_area: dict[tuple[int, int], float] = {}
    fi = features["cell_i"].to_pylist()
    fj = features["cell_j"].to_pylist()
    fc = features["feature_class"].to_pylist()
    fg = features["geometry"].to_pylist()
    for i, j, c, g in zip(fi, fj, fc, fg):
        if int(c) != int(FeatureClass.BUILDING):
            continue
        geom = shapely_wkb.loads(bytes(g))
        key = (int(i), int(j))
        building_area[key] = building_area.get(key, 0.0) + geom.area

    metrics: list[EvidenceMetric] = []
    for (cell_i, cell_j), active in scope.items():
        if not active:
            continue
        area = cell_area.get((cell_i, cell_j), 0.0)
        bldg = building_area.get((cell_i, cell_j), 0.0)
        ratio = bldg / area if area > 0.0 else 0.0
        metrics.append(
            EvidenceMetric(
                slot_kind=SlotKind.CELL,
                slot_index=cell_i * CELL_GRID_SIZE + cell_j,
                metric_namespace=MetricNamespace.CELL_DENSITY,
                metric_name="building_footprint_ratio",
                value=ratio,
                derivation_version=DERIVATION_VERSION,
            )
        )
    return metrics


# ---------------------------------------------------------------------------
# Road-skeleton evidence (per internal edge slot)
# ---------------------------------------------------------------------------


def derive_road_skeleton_evidence(
    crossings: pa.Table, features: pa.Table
) -> list[EvidenceMetric]:
    """Emit ``road_crossing_count`` per internal edge slot (all 112 slots).

    Crossings are joined to features by ``source_feature_id``; only crossings
    whose source feature has ``feature_class==ROAD`` are counted. This filters
    out building-ring crossings that sub-C also emits to ``crossings.parquet``
    so they do not contaminate road-skeleton evidence.

    Every internal-edge slot gets a metric even when count==0 so the metric
    stream is dense across the 112-slot lattice; the consumer doesn't have
    to reason about missing keys.
    """
    feat_class = features["feature_class"].to_pylist()
    feat_sid = features["source_feature_id"].to_pylist()
    road_ids: set[str] = {
        str(sid) for sid, c in zip(feat_sid, feat_class) if int(c) == int(FeatureClass.ROAD)
    }

    counts: dict[tuple[int, int, int], int] = {}
    csid = crossings["source_feature_id"].to_pylist()
    cli = crossings["lower_cell_i"].to_pylist()
    clj = crossings["lower_cell_j"].to_pylist()
    cax = crossings["axis"].to_pylist()
    for sid, li, lj, ax in zip(csid, cli, clj, cax):
        if str(sid) not in road_ids:
            continue
        key = (int(li), int(lj), int(ax))
        counts[key] = counts.get(key, 0) + 1

    metrics: list[EvidenceMetric] = []
    for edge in iter_internal_edge_slots():
        key = (edge.lower_cell_i, edge.lower_cell_j, edge.axis)
        metrics.append(
            EvidenceMetric(
                slot_kind=SlotKind.INTERNAL_EDGE,
                slot_index=edge.slot_index,
                metric_namespace=MetricNamespace.ROAD_SKELETON,
                metric_name="road_crossing_count",
                value=counts.get(key, 0),
                derivation_version=DERIVATION_VERSION,
            )
        )
    return metrics

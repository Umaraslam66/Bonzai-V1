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

#: Per-namespace derivation versions. Each ``derive_*_evidence`` function
#: stamps the appropriate constant on every emitted ``EvidenceMetric``.
#: Bumping one of these is the signal that *this namespace's algorithm*
#: changed; bumps are independent so a zoning fix doesn't invalidate cached
#: density artifacts. Recorded in every metric, in the locked macro vocab
#: artifact (spec §11.7), and in per-tile provenance.yaml (spec §11.5).
ZONING_DERIVATION_VERSION: str = "1.0"
CELL_DENSITY_DERIVATION_VERSION: str = "1.0"
TILE_POPULATION_DENSITY_DERIVATION_VERSION: str = "1.0"
ROAD_SKELETON_DERIVATION_VERSION: str = "1.0"


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
    for i, j in zip(cell_i, cell_j, strict=True):
        present.add((int(i), int(j)))
    return {(i, j): (i, j) in present for i in range(CELL_GRID_SIZE) for j in range(CELL_GRID_SIZE)}


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
    for i, j, c in zip(feat_i, feat_j, feat_class, strict=True):
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
                    derivation_version=ZONING_DERIVATION_VERSION,
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
    for i, j, area in zip(ci, cj, ca, strict=True):
        cell_area[(int(i), int(j))] = float(area)

    building_area: dict[tuple[int, int], float] = {}
    fi = features["cell_i"].to_pylist()
    fj = features["cell_j"].to_pylist()
    fc = features["feature_class"].to_pylist()
    fg = features["geometry"].to_pylist()
    for i, j, c, g in zip(fi, fj, fc, fg, strict=True):
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
                derivation_version=CELL_DENSITY_DERIVATION_VERSION,
            )
        )
    return metrics


# ---------------------------------------------------------------------------
# Road-skeleton evidence (per internal edge slot)
# ---------------------------------------------------------------------------


def derive_road_skeleton_evidence(crossings: pa.Table, features: pa.Table) -> list[EvidenceMetric]:
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
        str(sid)
        for sid, c in zip(feat_sid, feat_class, strict=True)
        if int(c) == int(FeatureClass.ROAD)
    }

    counts: dict[tuple[int, int, int], int] = {}
    csid = crossings["source_feature_id"].to_pylist()
    cli = crossings["lower_cell_i"].to_pylist()
    clj = crossings["lower_cell_j"].to_pylist()
    cax = crossings["axis"].to_pylist()
    for sid, li, lj, ax in zip(csid, cli, clj, cax, strict=True):
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
                derivation_version=ROAD_SKELETON_DERIVATION_VERSION,
            )
        )
    return metrics


# ---------------------------------------------------------------------------
# Tile-level population-density proxy evidence (one row per candidate proxy)
# ---------------------------------------------------------------------------


#: Candidate proxies for tile-level population density (spec §8). Layer-1
#: emits *all* of these so the Gate 2 reviewer can compare distributions
#: across candidate aggregations and pick one at lock time. Do not pre-commit
#: to one formula here; the choice belongs in the reviewer-locked vocab.
_POPULATION_DENSITY_PROXY_NAMES: tuple[str, ...] = (
    "mean_building_footprint_ratio",
    "area_weighted_building_density",
    "median_building_footprint_ratio",
    "p75_building_footprint_ratio",
)


def derive_tile_population_density_evidence(
    cells: pa.Table, features: pa.Table
) -> list[EvidenceMetric]:
    """Emit one ``EvidenceMetric`` per candidate population-density proxy.

    All proxies are computed from per-active-cell building footprint ratios
    plus the cells' ``cell_area_admin_clipped_m2`` (for area-weighting). Each
    proxy summarises the per-cell distribution differently:

    - ``mean_building_footprint_ratio`` — arithmetic mean of per-cell ratios.
    - ``area_weighted_building_density`` — sum(building_area) / sum(cell_area)
      across all active cells.
    - ``median_building_footprint_ratio`` — p50 of per-cell ratios.
    - ``p75_building_footprint_ratio`` — p75 of per-cell ratios.

    Rows have ``slot_kind=TILE`` and ``slot_index=0`` (tile-level rows do
    not index into the 64/112/32 lattices). The four metric names above are
    the locked-for-Phase-1 proxy set; the Gate 2 reviewer picks one as
    ``locked_proxy`` at vocab-lock time.

    Empty-tile convention (pinned)
    ------------------------------

    When a tile has zero active cells, **all four proxies emit
    ``value=0.0``** rather than NaN or null. Rationale:

    1. *Consistency across proxies.* Ratio-based proxies (mean,
       area-weighted) are naturally 0.0 on an empty tile — there are no
       buildings, so the proportion is zero. We extend the same convention
       to median/percentile proxies so downstream bucket-assignment logic
       does not have to special-case NaN inputs.
    2. *Semantic match.* An empty tile is genuinely "zero density," not
       "missing data." If the meaning ever shifts to "missing data" (e.g.
       a tile rejected mid-extraction), the validator gates that case
       upstream — sub-D only sees tiles sub-C kept.
    3. *Downstream simplicity.* Frequency analysis, bucket-cut proposal,
       and the eventual ``cell_density_bucket`` lookup all treat 0.0 as
       the floor of the distribution. NaN would propagate through
       percentile calculations and require explicit guards.

    Bumping ``TILE_POPULATION_DENSITY_DERIVATION_VERSION`` is required if
    this convention changes.
    """
    cell_density_metrics = derive_density_evidence(features, cells)
    per_cell_ratios = [float(m.value) for m in cell_density_metrics]

    cell_area_by_cell: dict[tuple[int, int], float] = {}
    ci = cells["cell_i"].to_pylist()
    cj = cells["cell_j"].to_pylist()
    ca = cells["cell_area_admin_clipped_m2"].to_pylist()
    for i, j, area in zip(ci, cj, ca, strict=True):
        cell_area_by_cell[(int(i), int(j))] = float(area)

    # Reconstruct per-cell building area (ratio * cell_area) so the
    # area-weighted aggregation has the numerator/denominator it needs
    # without re-running shapely on the WKB bytes.
    total_building_area = 0.0
    total_cell_area = 0.0
    for m in cell_density_metrics:
        ci_, cj_ = divmod(m.slot_index, CELL_GRID_SIZE)
        area = cell_area_by_cell.get((ci_, cj_), 0.0)
        total_building_area += float(m.value) * area
        total_cell_area += area

    proxy_values: dict[str, float] = {
        "mean_building_footprint_ratio": (
            sum(per_cell_ratios) / len(per_cell_ratios) if per_cell_ratios else 0.0
        ),
        "area_weighted_building_density": (
            total_building_area / total_cell_area if total_cell_area > 0.0 else 0.0
        ),
        "median_building_footprint_ratio": _percentile_of(per_cell_ratios, 0.50),
        "p75_building_footprint_ratio": _percentile_of(per_cell_ratios, 0.75),
    }

    metrics: list[EvidenceMetric] = []
    for proxy_name in _POPULATION_DENSITY_PROXY_NAMES:
        metrics.append(
            EvidenceMetric(
                slot_kind=SlotKind.TILE,
                slot_index=0,
                metric_namespace=MetricNamespace.TILE_POPULATION_DENSITY,
                metric_name=proxy_name,
                value=float(proxy_values[proxy_name]),
                derivation_version=TILE_POPULATION_DENSITY_DERIVATION_VERSION,
            )
        )
    return metrics


def _percentile_of(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = round(q * (len(ordered) - 1))
    idx = max(0, min(idx, len(ordered) - 1))
    return float(ordered[idx])

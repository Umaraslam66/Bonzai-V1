"""Sea masking: derivation of sea polygons (pre-policy view), cell-level drop
rule, feature-level sea_overlap_fraction.

CRITICAL ORDERING (spec §6 + §9.1 + §5 cross-decision dependency):
derive_sea_polygons MUST run against RAW base theme BEFORE
apply_missing_value_policy. The base.class not-in-vocab drop_row policy
(spec §10.2) would otherwise eliminate ocean/strait/bay rows (35 SG rows
below Strict-300 floor), leaving sea-mask with no polygons to work with.
Sea polygons are masks, not features — features.parquet does NOT contain
sea polygons; the policied themes correctly drop them.

derive_inland_water_polygons, by contrast, runs against the POLICIED base
theme — inland-water classes (river, stream, reservoir, …) are in the
Phase 1 BASE_ vocab and survive the policy step, so the policied table is
the correct source.  Per spec §9.2 + §11.3.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
from shapely import wkb
from shapely.geometry import MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from cfm.data.sub_c.epsilon import EPS_RATIO

# Module-level frozensets — immutable, fast membership test
SEA_CLASS_VALUES = frozenset({"ocean", "strait", "bay"})
SEA_SUBTYPE_VALUES = frozenset({"ocean"})

# Inland-water classes per spec §9.2 canonical list.  These classes are all in
# the Phase 1 BASE_ vocab, so they survive apply_missing_value_policy and are
# present in the policied base theme.
#
# DECISION: chose this exact set from spec §9.2 text over dynamically deriving
# from vocab YAML at runtime.  Spec text is the authority; the vocab YAML is
# the downstream consumer.  If a new water class is added to the vocab, it
# should be added here too — that linkage is documented in the spec, not
# auto-wired.  Revisit if inland-water classification expands in Phase 2.
INLAND_WATER_CLASSES: frozenset[str] = frozenset(
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


def derive_inland_water_polygons(base_theme: pa.Table) -> BaseGeometry:
    """Extract inland-water polygons from the POLICIED base theme.

    Filter: class IN INLAND_WATER_CLASSES (river, stream, reservoir, lake,
    pond, swimming_pool, canal, drain — spec §9.2 canonical set).

    Unlike derive_sea_polygons, this MUST be called on the POLICIED base theme
    (after apply_missing_value_policy), because inland-water classes are in the
    Phase 1 BASE_ vocab and survive the policy step.  Calling it on the raw
    theme would include rows that policy subsequently drops for other reasons,
    but in practice only the sea rows (ocean, strait, bay) are policy-dropped,
    so the ordering difference is cosmetic for the current vocab.

    Returns: unary_union of all matching geometries as a single (Multi)Polygon
    or GeometryCollection.  Returns empty MultiPolygon if no matching rows.

    DECISION: LineString rivers/streams have zero area; their contribution to
    water_fraction (an area ratio) is mathematically zero regardless.  We
    include them in the union anyway — the call is cheap, the union is correct
    (unary_union of mixed Polygon + LineString returns a GeometryCollection
    whose .area sums only the polygon components), and it avoids a filtering
    step that could silently exclude future polygon-typed rivers from sources
    that store some rivers as LineStrings.  If profiling shows the mixed-type
    union is slow, filter to Polygon-typed rows here.  Per spec §11.3.
    """
    class_col = base_theme.column("class")
    mask = pc.is_in(class_col, value_set=pa.array(list(INLAND_WATER_CLASSES)))
    rows = base_theme.filter(mask)
    if len(rows) == 0:
        return MultiPolygon()
    geometries = [wkb.loads(g) for g in rows.column("geometry").to_pylist()]
    return unary_union(geometries)


def derive_sea_polygons(base_theme: pa.Table) -> BaseGeometry:
    """Extract sea-defining polygons from the RAW base theme.

    Filter: class IN {ocean, strait, bay} OR subtype = ocean.
    Returns: unioned MultiPolygon (or empty MultiPolygon) for efficient
    per-cell intersection downstream. Sha256 of WKB bytes is recorded in
    manifest.sea_polygons_sha256 per spec §11.7.

    Per spec §9.1: this MUST run on raw themes BEFORE apply_missing_value_policy;
    sea polygons are masks (not features) so policy-drop correctly removes them
    from feature emission.
    """
    class_col = base_theme.column("class")
    subtype_col = base_theme.column("subtype")
    in_sea_class = pc.is_in(class_col, value_set=pa.array(list(SEA_CLASS_VALUES)))
    in_sea_subtype = pc.is_in(subtype_col, value_set=pa.array(list(SEA_SUBTYPE_VALUES)))
    mask = pc.or_(in_sea_class, in_sea_subtype)
    sea_rows = base_theme.filter(mask)
    if len(sea_rows) == 0:
        return MultiPolygon()
    geometries = [wkb.loads(g) for g in sea_rows.column("geometry").to_pylist()]
    return unary_union(geometries)


def apply_sea_mask(
    *,
    cell_box_admin_clipped: BaseGeometry,
    cell_features: list,  # list[CellSubFeature]; circular-import avoided via untyped list
    sea_polygons_svy21: BaseGeometry,
    inland_water_polygons_svy21: BaseGeometry | None = None,
) -> tuple[float, float, bool]:
    """For a single cell, compute (sea_water_fraction, water_fraction, drop_flag).

    sea_water_fraction = area(cell ∩ admin ∩ sea_polygons) / area(cell ∩ admin)
    inland_fraction    = area(cell ∩ admin ∩ inland_water_polygons) / area(cell ∩ admin)
                         (0.0 if inland_water_polygons_svy21 is None or empty)
    water_fraction     = min(1.0, sea_water_fraction + inland_fraction)
    drop_flag = (sea_water_fraction >= 1.0 - EPS_RATIO) AND (zero non-sea features)

    Per spec §9.2 + §4.3 alpha structural-boundary EPSILON.  Per spec §11.3
    water_fraction = "all-water (sea + inland)" coverage of the cell.

    The min(1.0, ...) cap handles FP arithmetic: sea and inland polygons do
    not overlap by definition, but floating-point area sums can produce
    1.0 + epsilon.  The cap satisfies inline-validator invariant #5
    (water_fraction <= 1 + EPS_RATIO) strictly.

    inland_water_polygons_svy21 defaults to None for backward compatibility
    with callers that pre-date Fix #1.  Passing None yields
    water_fraction == sea_water_fraction (the previous behaviour).
    """
    cell_admin_area = cell_box_admin_clipped.area
    if cell_admin_area <= 0:
        # Degenerate cell — should have been filtered upstream; treat as drop
        return 0.0, 0.0, True

    sea_overlap = cell_box_admin_clipped.intersection(sea_polygons_svy21)
    sea_water_fraction = sea_overlap.area / cell_admin_area

    # Inland-water contribution (spec §11.3).  Only computed when the caller
    # passes the derived inland geometry (Fix #1); defaults to 0.0 otherwise.
    if inland_water_polygons_svy21 is not None and not inland_water_polygons_svy21.is_empty:
        inland_overlap = cell_box_admin_clipped.intersection(inland_water_polygons_svy21)
        inland_fraction = inland_overlap.area / cell_admin_area
    else:
        inland_fraction = 0.0

    water_fraction = min(1.0, sea_water_fraction + inland_fraction)

    # "Zero non-sea features": under the pipeline order (sea polygons are removed
    # from policied themes before feature extraction), every feature in cell_features
    # is a non-sea feature. No secondary filtering needed here.
    non_sea_count = len(cell_features)
    # DECISION: >= (1.0 - EPS_RATIO) not > (1.0 - EPS_RATIO); structural boundary
    # at exactly 1.0 must be caught. Per spec §9.2 + §14.4 alpha EPSILON.
    drop_flag = sea_water_fraction >= (1.0 - EPS_RATIO) and non_sea_count == 0
    return sea_water_fraction, water_fraction, drop_flag


def compute_sea_overlap_fraction(
    *,
    feature_geom: BaseGeometry,
    feature_type: str,
    cell_local_sea_geometry: BaseGeometry | None,
) -> float:
    """Per spec §9.3: sea_overlap_fraction for a single feature.

    The caller computes cell_local_sea_geometry once per cell
    (cell_box_admin_clipped.intersection(sea_polygons_svy21)) and passes it to
    every feature in the cell — this is the "cache" the function name refers to.
    The function itself is stateless; the caller manages caching.

    Fast-path: if cell_local_sea_geometry is None (no sea in cell), return 0.0
    immediately — avoids per-feature geometry ops for inland cells.

    For Points: INTERSECTS predicate (NOT contains). Coastline POIs sitting on
    the sea polygon boundary are considered sea-adjacent (spec §9.3 precision 1).

    For LineStrings: length(feature ∩ sea) / length(feature).
    For Polygons: area(feature ∩ sea) / area(feature).

    over_sea is DERIVED at read-time by the consumer as sea_overlap_fraction > EPS_RATIO;
    it is NOT stored and NOT computed here.
    """
    if cell_local_sea_geometry is None:
        return 0.0

    if feature_type == "Point":
        return 1.0 if feature_geom.intersects(cell_local_sea_geometry) else 0.0

    if feature_type == "LineString":
        total = feature_geom.length
        if total <= 0:
            return 0.0
        return float(feature_geom.intersection(cell_local_sea_geometry).length / total)

    if feature_type == "Polygon":
        total = feature_geom.area
        if total <= 0:
            return 0.0
        return float(feature_geom.intersection(cell_local_sea_geometry).area / total)

    # DECISION: unknown geometry types return 0.0 (safe default; no crash).
    # Revisit if MultiLineString or MultiPolygon sub-features appear in Task 4+.
    return 0.0

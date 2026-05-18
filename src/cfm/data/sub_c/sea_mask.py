"""Sea masking: derivation of sea polygons (pre-policy view), cell-level drop
rule, feature-level sea_overlap_fraction.

CRITICAL ORDERING (spec §6 + §9.1 + §5 cross-decision dependency):
derive_sea_polygons MUST run against RAW base theme BEFORE
apply_missing_value_policy. The base.class not-in-vocab drop_row policy
(spec §10.2) would otherwise eliminate ocean/strait/bay rows (35 SG rows
below Strict-300 floor), leaving sea-mask with no polygons to work with.
Sea polygons are masks, not features — features.parquet does NOT contain
sea polygons; the policied themes correctly drop them.
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
) -> tuple[float, float, bool]:
    """For a single cell, compute (sea_water_fraction, water_fraction, drop_flag).

    sea_water_fraction = area(cell ∩ admin ∩ sea_polygons) / area(cell ∩ admin)
    drop_flag = (sea_water_fraction >= 1.0 - EPS_RATIO) AND (zero non-sea features)

    Per spec §9.2 + §4.3 alpha structural-boundary EPSILON.

    The water_fraction returned here is a placeholder equal to sea_water_fraction.
    The pipeline orchestrator (Task 12) combines sea + inland water into the
    final water_fraction written to cells.parquet.
    """
    cell_admin_area = cell_box_admin_clipped.area
    if cell_admin_area <= 0:
        # Degenerate cell — should have been filtered upstream; treat as drop
        return 0.0, 0.0, True

    sea_overlap = cell_box_admin_clipped.intersection(sea_polygons_svy21)
    sea_water_fraction = sea_overlap.area / cell_admin_area

    # water_fraction: placeholder — orchestrator overrides with combined value
    water_fraction = sea_water_fraction

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

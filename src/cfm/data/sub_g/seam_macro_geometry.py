"""Seam 1: macro plan (sub-D) <-> geometry (sub-C). Structural invariants.

Independence (design rule 3 / spec Decision 3a): each invariant recomputes the
METRIC from sub-C (formula anchored in a spec clause OUTSIDE sub-D), buckets via
the LOCKED vocab cut-points (a contract input), and compares to sub-D's stored
class. It catches metric-computation and bucketing-application bugs; it does NOT
judge whether the cut-points are right (Gate-2 reviewer call; deferred).

  SI-1 density:       PRD §5 line 65 "binned building footprint area".
  SI-2 road skeleton: sub-D design §9 (crossings.parquet source of truth).
  SI-3 zoning:        DEFERRED after Step-0 read (2026-05-31). sub-D assigns
                      zoning_class = "dominant feature class (by raw count) ->
                      vocab token_id" (sub_d/pipeline.py:310, _zoning_token_id
                      :346). PRD §5 anchors only the vague CONCEPT ("dominant
                      land use"); the precise argmax-of-raw-count + tie-break
                      rule is sub-D-internal, so an independent invariant has no
                      external truth-statement (circular-by-provenance, rule 3).
                      Trigger to re-add as a seam-1 check (sub-G design §8): an
                      external spec clause locks the feature-count -> zoning-
                      token mapping.
"""

from __future__ import annotations

from shapely.wkb import loads as wkb_loads

from cfm.data.sub_g.buckets import bucket_of, load_density_edges, load_road_skeleton_edges
from cfm.data.sub_g.diagnostics import Diagnostic

_EPS_RATIO = 1e-9  # structural-boundary epsilon for bucket ties


def recompute_density_ratio(features: list[dict], cell_area_m2: float) -> float:
    """Sum building (feature_class==1) polygon areas / cell area. PRD §5 line 65."""
    total = 0.0
    for f in features:
        if int(f["feature_class"]) == 1:
            total += wkb_loads(bytes(f["geometry"])).area
    return total / cell_area_m2 if cell_area_m2 > 0 else 0.0


def _buckets_for(value: float, edges: list[float]) -> set[int]:
    return {
        bucket_of(value, edges),
        bucket_of(value + _EPS_RATIO, edges),
        bucket_of(value - _EPS_RATIO, edges),
    }


def _bucket_signature(name: str, recomputed: int, stored: int) -> str:
    direction = "too-high" if stored > recomputed else "too-low"
    return f"{name} bucket {abs(stored - recomputed)}-step {direction} vs recomputed metric"


def check_density(
    tile_id: str,
    per_cell_features: dict[tuple[int, int], list[dict]],
    per_cell_area: dict[tuple[int, int], float],
    sub_d_density_by_cell: dict[tuple[int, int], int | None],
) -> list[Diagnostic]:
    edges = load_density_edges()
    diags: list[Diagnostic] = []
    for cell, expected_bucket in sub_d_density_by_cell.items():
        if expected_bucket is None:
            continue  # inactive cell slot
        ratio = recompute_density_ratio(
            per_cell_features.get(cell, []), per_cell_area.get(cell, 0.0)
        )
        if expected_bucket not in _buckets_for(
            ratio, edges
        ):  # fails only if no eps neighbour agrees
            recomputed = bucket_of(ratio, edges)
            diags.append(
                Diagnostic(
                    tile_id=tile_id,
                    invariant_name="density_bucket_matches_footprint",
                    artifact_left="sub_c.building_footprint_ratio",
                    observed_left=round(ratio, 6),
                    artifact_right="sub_d.cell_density_bucket",
                    observed_right=expected_bucket,
                    expected_relationship=f"ratio in {edges} => bucket {recomputed}",
                    spec_clause_citation="PRD §5 line 65 + macro_plan_vocab.yaml:3472-3488",
                    signature=_bucket_signature("density", recomputed, expected_bucket),
                )
            )
    return diags


def recompute_road_crossing_count(
    features: list[dict], crossings: list[dict]
) -> dict[tuple[int, int, int], int]:
    """Count crossings whose source feature is feature_class==0 (ROAD), per (li,lj,axis).

    sub-D design §9: crossings.parquet is the source of truth; join to features by
    source_feature_id; ROAD only.
    """
    road_ids = {str(f["source_feature_id"]) for f in features if int(f["feature_class"]) == 0}
    counts: dict[tuple[int, int, int], int] = {}
    for c in crossings:
        if str(c["source_feature_id"]) in road_ids:
            key = (int(c["lower_cell_i"]), int(c["lower_cell_j"]), int(c["axis"]))
            counts[key] = counts.get(key, 0) + 1
    return counts


def check_road_skeleton(
    tile_id: str,
    features: list[dict],
    crossings: list[dict],
    sub_d_skeleton_by_edge: dict[tuple[int, int, int], int | None],
) -> list[Diagnostic]:
    edges = load_road_skeleton_edges()
    counts = recompute_road_crossing_count(features, crossings)
    diags: list[Diagnostic] = []
    for edge_key, expected_bucket in sub_d_skeleton_by_edge.items():
        if expected_bucket is None:
            continue
        n = counts.get(edge_key, 0)
        recomputed = bucket_of(n, edges)
        if recomputed != expected_bucket:  # integer counts -> no FP epsilon needed
            diags.append(
                Diagnostic(
                    tile_id=tile_id,
                    invariant_name="road_skeleton_bucket_matches_crossings",
                    artifact_left="sub_c.road_crossing_count",
                    observed_left=n,
                    artifact_right="sub_d.road_skeleton_class",
                    observed_right=expected_bucket,
                    expected_relationship=f"count in {edges} => bucket {recomputed}",
                    spec_clause_citation="sub-D design §9 + macro_plan_vocab.yaml:3489-3505",
                    signature=_bucket_signature("road_skeleton", recomputed, expected_bucket),
                )
            )
    return diags

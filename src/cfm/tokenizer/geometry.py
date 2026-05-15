from __future__ import annotations

from collections import defaultdict

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

GeoJSON = dict


def geometric_equal(a: GeoJSON, b: GeoJSON, *, tol_m: float = 0.5) -> bool:
    """Return True iff two GeoJSON FeatureCollections are geometrically equivalent.

    Equivalence rule:
      - Features are grouped by `properties.class`. Class counts must match.
      - Within each class, features in `a` are greedily paired with the
        unmatched feature in `b` of minimum geometric distance.
      - Every pair must be within `tol_m`:
            * Points: Euclidean distance.
            * Lines/Polygons: symmetric Hausdorff distance.
    """
    grouped_a = _group_by_class(a)
    grouped_b = _group_by_class(b)
    if set(grouped_a) != set(grouped_b):
        return False
    for cls, geoms_a in grouped_a.items():
        geoms_b = list(grouped_b[cls])
        if len(geoms_a) != len(geoms_b):
            return False
        for ga in geoms_a:
            best_idx = None
            best_dist = float("inf")
            for i, gb in enumerate(geoms_b):
                d = _distance(ga, gb)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx is None or best_dist > tol_m:
                return False
            geoms_b.pop(best_idx)
    return True


def _group_by_class(fc: GeoJSON) -> dict[str, list[BaseGeometry]]:
    out: dict[str, list[BaseGeometry]] = defaultdict(list)
    for feat in fc["features"]:
        cls = feat["properties"]["class"]
        out[cls].append(shape(feat["geometry"]))
    return out


def _distance(a: BaseGeometry, b: BaseGeometry) -> float:
    if a.geom_type == "Point" and b.geom_type == "Point":
        return a.distance(b)
    if a.geom_type != b.geom_type:
        return float("inf")
    return max(a.hausdorff_distance(b), b.hausdorff_distance(a))

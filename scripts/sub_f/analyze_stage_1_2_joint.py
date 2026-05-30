"""Compute joint P(feature_count, vertex_count | cell, feature_type) per type.

Per spec §7.3: joint distribution NOT separate marginals — dense areas
correlate with simpler geometry per feature. Treating as independent
inflates tail prediction.

Output: configs/sub_f/stage_1_2_joint.yaml as intermediate input to Task 3b/3c.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, quantiles

import pyarrow.parquet as pq
import yaml
from shapely.wkb import loads as wkb_loads

ROOT = Path(__file__).resolve().parents[2]


def _vertex_count(geom) -> int:
    """Count vertices for the geometry types sub-C emits.

    Per `src/cfm/data/sub_c/enums.py:GEOMETRY_TYPE`:
      0 Point, 1 LineString, 2 Polygon, 3 MultiPoint,
      4 MultiLineString, 5 MultiPolygon.
    Multi* geometries: sum component vertex counts (matches encoder cost basis,
    each part emits its own feature primitive in sub-F encoder grammar).
    """
    gt = geom.geom_type
    if gt == "LineString":
        return len(geom.coords)
    if gt == "Polygon":
        return len(geom.exterior.coords)
    if gt == "Point":
        return 1
    if gt == "MultiPoint":
        return sum(1 for _ in geom.geoms)
    if gt == "MultiLineString":
        return sum(len(part.coords) for part in geom.geoms)
    if gt == "MultiPolygon":
        return sum(len(part.exterior.coords) for part in geom.geoms)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub-c-region-dir", required=True, type=Path)
    args = parser.parse_args()

    # Joint: { (tile_i, tile_j, cell_i, cell_j): { feature_type: [vertex_counts] } }
    per_cell: dict[tuple[int, int, int, int], dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )

    tile_features = sorted(args.sub_c_region_dir.glob("tile=*/features.parquet"))
    print(f"[analyze stage 1+2] {len(tile_features)} tiles", flush=True)
    tile_keys: set[tuple[int, int]] = set()
    for path in tile_features:
        tile_name = path.parent.name  # tile=EPSG3414_iN_jM
        parts = tile_name.replace("tile=", "").split("_")
        tile_i = int(parts[1].lstrip("i"))
        tile_j = int(parts[2].lstrip("j"))
        tile_keys.add((tile_i, tile_j))
        table = pq.ParquetFile(path).read()
        records = table.to_pylist()
        for r in records:
            geom = wkb_loads(r["geometry"])
            v = _vertex_count(geom)
            cell_key = (tile_i, tile_j, int(r["cell_i"]), int(r["cell_j"]))
            per_cell[cell_key][int(r["feature_class"])].append(v)

    # Cell denominator includes ALL grid cells (8x8) in every tile (even those
    # with empty features.parquet), not only cells with at least one feature.
    # Empty cells are part of the joint distribution (§7.8 empty-cell-as-floor).
    for ti, tj in tile_keys:
        for ci in range(8):
            for cj in range(8):
                _ = per_cell[(ti, tj, ci, cj)]  # defaultdict creates empty dict if missing

    n_cells_total = len(per_cell)
    n_empty_cells = sum(1 for fc_map in per_cell.values() if not fc_map)

    feature_classes = {fc for cell in per_cell.values() for fc in cell}

    output: dict = {"per_feature_type": {}, "empty_cell_fraction": 0.0}

    for fc in sorted(feature_classes):
        fc_rows: list[dict] = []
        cells_containing_fc = 0
        for _cell_key, fc_map in per_cell.items():
            if fc in fc_map:
                cells_containing_fc += 1
                fc_count_in_cell = len(fc_map[fc])
                for vc in fc_map[fc]:
                    fc_rows.append({"feature_count_in_cell": fc_count_in_cell, "vertex_count": vc})

        feat_counts = [r["feature_count_in_cell"] for r in fc_rows]
        vert_counts = [r["vertex_count"] for r in fc_rows]

        output["per_feature_type"][int(fc)] = {
            "n_observations": len(fc_rows),
            "n_cells_with_type": cells_containing_fc,
            "feature_count_mean": float(mean(feat_counts)) if feat_counts else 0.0,
            "vertex_count_mean": float(mean(vert_counts)) if vert_counts else 0.0,
            "vertex_count_p95": (
                float(quantiles(vert_counts, n=20)[18]) if len(vert_counts) >= 20 else None
            ),
            "vertex_count_p99": (
                float(quantiles(vert_counts, n=100)[98]) if len(vert_counts) >= 100 else None
            ),
            "vertex_count_max": int(max(vert_counts)) if vert_counts else 0,
            "feature_count_p95": (
                float(quantiles(feat_counts, n=20)[18]) if len(feat_counts) >= 20 else None
            ),
            "feature_count_p99": (
                float(quantiles(feat_counts, n=100)[98]) if len(feat_counts) >= 100 else None
            ),
            "feature_count_max": int(max(feat_counts)) if feat_counts else 0,
        }

    output["empty_cell_fraction"] = n_empty_cells / n_cells_total if n_cells_total else 0.0
    output["n_cells_total"] = n_cells_total
    output["n_empty_cells"] = n_empty_cells
    output["n_tiles_total"] = len(tile_keys)
    output["sub_c_region_dir"] = str(args.sub_c_region_dir)
    output["_status"] = "INTERMEDIATE — feeds Task 3b/3c."

    out = ROOT / "configs" / "sub_f" / "stage_1_2_joint.yaml"
    out.write_text(yaml.safe_dump(output, sort_keys=True), encoding="utf-8")
    print(f"[analyze stage 1+2] wrote {out}")
    print(
        f"[analyze stage 1+2] {n_cells_total} cells "
        f"({n_empty_cells} empty, {len(tile_keys)} tiles), "
        f"feature_classes_present={sorted(feature_classes)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

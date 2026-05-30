#!/usr/bin/env python
"""Verify whether sub-C emits branched road geometries as multi-part rows."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from shapely import wkb
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge

from cfm.data.io import canonicalize_yaml
from cfm.data.sub_c.coords import CELL_SIZE_M
from cfm.data.sub_c.epsilon import EPS_COORD_M

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "reports" / "sub_f_task_7_feature_splitting.yaml"


def _geometry_name(geometry_bytes: bytes | None) -> tuple[str, int | None]:
    if geometry_bytes is None:
        return "NULL", None
    geom = wkb.loads(geometry_bytes)
    part_count = len(geom.geoms) if hasattr(geom, "geoms") else 1
    return geom.geom_type, part_count


def _line_touched_edges(line: LineString) -> list[str]:
    edges: set[str] = set()
    for x, y in line.coords:
        if abs(x) <= EPS_COORD_M:
            edges.add("W")
        if abs(x - CELL_SIZE_M) <= EPS_COORD_M:
            edges.add("E")
        if abs(y) <= EPS_COORD_M:
            edges.add("S")
        if abs(y - CELL_SIZE_M) <= EPS_COORD_M:
            edges.add("N")
    min_x, min_y, max_x, max_y = line.bounds
    if abs(min_x) <= EPS_COORD_M and abs(max_x) <= EPS_COORD_M:
        edges.add("W")
    if abs(min_x - CELL_SIZE_M) <= EPS_COORD_M and abs(max_x - CELL_SIZE_M) <= EPS_COORD_M:
        edges.add("E")
    if abs(min_y) <= EPS_COORD_M and abs(max_y) <= EPS_COORD_M:
        edges.add("S")
    if abs(min_y - CELL_SIZE_M) <= EPS_COORD_M and abs(max_y - CELL_SIZE_M) <= EPS_COORD_M:
        edges.add("N")
    return sorted(edges)


def classify_multiline_part_edge_relationship(geom: MultiLineString) -> dict[str, Any]:
    """Classify a cell-local road MultiLineString's BP7 edge interaction."""
    merged = linemerge(geom)
    part_edges = [_line_touched_edges(part) for part in geom.geoms if isinstance(part, LineString)]

    if isinstance(merged, LineString):
        bucket = "mergeable_artifact"
    else:
        edge_counts: Counter[str] = Counter(edge for edges in part_edges for edge in edges)
        repeated_edges = sorted(edge for edge, count in edge_counts.items() if count >= 2)
        distinct_edges = sorted(edge_counts)
        if repeated_edges:
            bucket = "same_cell_edge_multi_part"
        elif len(distinct_edges) >= 2:
            bucket = "different_cell_edges"
        else:
            bucket = "no_multi_part_boundary_interaction"
        return {
            "bucket": bucket,
            "part_edges": part_edges,
            "repeated_edges": repeated_edges,
            "distinct_edges": distinct_edges,
        }

    edge_counts = Counter(edge for edges in part_edges for edge in edges)
    return {
        "bucket": bucket,
        "part_edges": part_edges,
        "repeated_edges": sorted(edge for edge, count in edge_counts.items() if count >= 2),
        "distinct_edges": sorted(edge_counts),
    }


def build_feature_splitting_report(sub_c_region_dir: Path) -> dict[str, Any]:
    paths = sorted(sub_c_region_dir.glob("tile=*/features.parquet"))
    if not paths:
        raise FileNotFoundError(f"no tile=*/features.parquet files under {sub_c_region_dir}")

    geometry_type_counts: Counter[str] = Counter()
    encoded_geometry_type_counts: Counter[str] = Counter()
    road_multiline_decoded_count = 0
    road_geometry_type_4_count = 0
    road_multiline_examples: list[dict[str, Any]] = []
    road_multiline_part_edge_buckets: Counter[str] = Counter()
    road_multiline_same_edge_examples: list[dict[str, Any]] = []
    row_count = 0

    for path in paths:
        table = pq.ParquetFile(path).read()
        tile = path.parent.name
        for row in table.to_pylist():
            row_count += 1
            geometry_name, part_count = _geometry_name(row.get("geometry"))
            encoded_type = row.get("geometry_type")
            geometry_type_counts[geometry_name] += 1
            encoded_geometry_type_counts[str(encoded_type)] += 1

            is_road = row.get("feature_class") == 0
            is_decoded_multiline = geometry_name == "MultiLineString"
            is_encoded_multiline = encoded_type == 4
            if is_road and is_decoded_multiline:
                road_multiline_decoded_count += 1
            if is_road and is_encoded_multiline:
                road_geometry_type_4_count += 1

            if is_road and (is_decoded_multiline or is_encoded_multiline):
                if geometry_name == "MultiLineString":
                    geom = wkb.loads(row["geometry"])
                    relationship = classify_multiline_part_edge_relationship(geom)
                    road_multiline_part_edge_buckets[relationship["bucket"]] += 1
                    if (
                        relationship["bucket"] == "same_cell_edge_multi_part"
                        and len(road_multiline_same_edge_examples) < 10
                    ):
                        road_multiline_same_edge_examples.append(
                            {
                                "tile": tile,
                                "source_feature_id": row.get("source_feature_id"),
                                "class_raw": row.get("class_raw"),
                                "cell_i": row.get("cell_i"),
                                "cell_j": row.get("cell_j"),
                                "part_count": part_count,
                                "part_edges": relationship["part_edges"],
                                "repeated_edges": relationship["repeated_edges"],
                            }
                        )
                if len(road_multiline_examples) < 20:
                    road_multiline_examples.append(
                        {
                            "tile": tile,
                            "source_feature_id": row.get("source_feature_id"),
                            "geometry_type": geometry_name,
                            "encoded_geometry_type": encoded_type,
                            "part_count": part_count,
                        }
                    )

    road_multiline_count = max(road_multiline_decoded_count, road_geometry_type_4_count)
    if road_multiline_count == 0:
        outcome = "single_row_per_branch"
        recommendation = "no multi-outbound grammar needed"
    else:
        outcome = "branched_multi_row_present"
        recommendation = "§9.6.1 cascade candidate; do not add multi-outbound grammar in Task 7"

    part_edge_buckets = {
        "different_cell_edges": road_multiline_part_edge_buckets["different_cell_edges"],
        "mergeable_artifact": road_multiline_part_edge_buckets["mergeable_artifact"],
        "no_multi_part_boundary_interaction": road_multiline_part_edge_buckets[
            "no_multi_part_boundary_interaction"
        ],
        "same_cell_edge_multi_part": road_multiline_part_edge_buckets[
            "same_cell_edge_multi_part"
        ],
    }

    return {
        "_status": "PROPOSED - pending Halt 7 reviewer approval",
        "sub_c_region_dir": str(sub_c_region_dir),
        "tile_count": len(paths),
        "row_count": row_count,
        "geometry_type_counts": dict(sorted(geometry_type_counts.items())),
        "encoded_geometry_type_counts": dict(sorted(encoded_geometry_type_counts.items())),
        "road_multiline_count": road_multiline_count,
        "road_multiline_decoded_count": road_multiline_decoded_count,
        "road_geometry_type_4_count": road_geometry_type_4_count,
        "road_multiline_examples": road_multiline_examples,
        "road_multiline_part_edge_buckets": part_edge_buckets,
        "road_multiline_same_edge_examples": road_multiline_same_edge_examples,
        "outcome": outcome,
        "recommendation": recommendation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub-c-region-dir", required=True, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, type=Path)
    args = parser.parse_args()

    report = build_feature_splitting_report(args.sub_c_region_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonicalize_yaml(report), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

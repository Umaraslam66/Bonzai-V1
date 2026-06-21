"""READ-ONLY binding count: distinct conditionable held-out cells per 4-tuple stratum.

Mirrors extract_features_by_city_stratum_metric's loop (conditioning_discrimination.py:427-466)
EXACTLY — same keying functions (holdout_manifest_for_region tiles, read_tile_labels,
_cell_density_by_cell, read_sub_f_cells) — but counts DISTINCT NON-EMPTY CELLS per
(zoning, skeleton, density, coastal) stratum instead of binning features. Writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import yaml

from cfm.data.sub_g.readers import read_sub_f_cells
from cfm.eval.holdout.labels import read_tile_labels
from cfm.eval.holdout.paths import (
    epsg_label_for_region,
    holdout_manifest_for_region,
    sub_d_region_dir,
    sub_f_region_dir,
    tile_dirname,
)
from cfm.eval.holdout.pipeline import _cell_density_by_cell
from cfm.eval.lane_s_sampler import SampledCell, write_cell_census

RELEASE = "2026-04-15.0"
HELD = ["glasgow", "eisenhuttenstadt", "munich", "krakow"]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="held-out per-stratum cell census")
    ap.add_argument(
        "--emit",
        default=None,
        metavar="PATH",
        help="write the per-cell census parquet to this path (default: no emit)",
    )
    return ap.parse_args()


def _dist(label: str, sizes: list[int]) -> None:
    sizes = sorted(sizes)
    print(f"--- {label} ---")
    print(f"  strata: {len(sizes)}   total distinct cells (COUNTED): {sum(sizes)}")
    if sizes:
        print(f"  stratum size: min={sizes[0]} median={sizes[len(sizes) // 2]} max={sizes[-1]}")
        for thr in (50, 75, 100):
            print(f"  strata < {thr} cells: {sum(1 for n in sizes if n < thr)} / {len(sizes)}")


def main() -> int:
    args = _parse_args()
    by_stratum: dict[tuple, int] = defaultdict(int)  # global 4-tuple -> distinct non-empty cells
    by_city_stratum: dict[tuple, int] = defaultdict(int)  # (city, 4-tuple) -> distinct cells
    total_tiles = nonempty = empty = missing = 0
    per_city_tiles: dict[str, int] = {}
    # census accumulators — only populated when --emit is requested
    cell_rows: list[SampledCell] = []
    tile_strata: dict[tuple[str, int, int], tuple] = {}

    for city in HELD:
        manifest = yaml.safe_load(holdout_manifest_for_region(RELEASE, city).read_text())
        tiles = manifest["regions"][city]["tiles"]
        per_city_tiles[city] = len(tiles)
        epsg = epsg_label_for_region(city)
        sub_d = sub_d_region_dir(RELEASE, city)
        sub_f = sub_f_region_dir(RELEASE, city)
        for tile in tiles:
            ti, tj = int(tile["tile_i"]), int(tile["tile_j"])
            total_tiles += 1
            dirname = tile_dirname(ti, tj, epsg)
            cells_path = sub_f / dirname / "cells.parquet"
            if not cells_path.exists():
                missing += 1
                continue
            labels = read_tile_labels(sub_d / dirname, tile_i=ti, tile_j=tj)
            zoning = labels.morphology_stratum.dominant_zoning_class
            skeleton = labels.morphology_stratum.modal_road_skeleton_class
            coastal = labels.coastal_inland_river
            cdbc = _cell_density_by_cell(sub_d / dirname)
            tokens = read_sub_f_cells(cells_path)
            # Record the tile's stratum triple once (for census emit)
            if args.emit:
                tile_strata[(city, ti, tj)] = (zoning, skeleton, coastal)
            for (ci, cj), toks in tokens.items():
                if not toks:
                    empty += 1
                    continue
                nonempty += 1
                density = cdbc.get((ci, cj), -1)
                if density is None:
                    density = -1
                stratum = (zoning, skeleton, density, coastal)
                by_stratum[stratum] += 1
                by_city_stratum[(city, stratum)] += 1
                # Collect cell row for census emit
                if args.emit:
                    cell_rows.append(SampledCell(city, ti, tj, ci, cj, int(density)))

    print("==================== HELD-OUT DISTINCT-CELL COUNT ====================")
    print(f"cities: {HELD}")
    print(f"tiles: {total_tiles}  per-city: {per_city_tiles}  missing-parquet: {missing}")
    print(f"non-empty (conditionable) cells: {nonempty}   empty cells: {empty}")
    print()
    _dist("GLOBAL 4-tuple stratum (merged across cities)", list(by_stratum.values()))
    print()
    _dist(
        "(city, 4-tuple) — the eval's per-city generation+floor unit",
        list(by_city_stratum.values()),
    )
    print()
    print("--- per-city (city,stratum) thin distribution ---")
    for city in HELD:
        rows = sorted(v for (c, s), v in by_city_stratum.items() if c == city)
        if not rows:
            print(f"  {city}: none")
            continue
        print(
            f"  {city:18s}: strata={len(rows):3d} cells={sum(rows):6d} min={rows[0]:4d} "
            f"median={rows[len(rows) // 2]:5d}  <50:{sum(1 for n in rows if n < 50)} "
            f"<75:{sum(1 for n in rows if n < 75)} <100:{sum(1 for n in rows if n < 100)}"
        )
    print("=====================================================================")

    if args.emit:
        write_cell_census(cell_rows, tile_strata, args.emit)
        print(f"census parquet emitted: {args.emit}  rows={len(cell_rows)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

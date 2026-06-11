#!/usr/bin/env python3
"""PI-ordered investigation (2026-06-11): are over-length cells GENUINE COMPLEXITY
(more real features) or SOURCE VERTEX-RICHNESS (same features, more coordinate
points — the NL cadastral signature)? READ-ONLY; changes nothing; decides nothing.

Anchor: per city, the recomputed frac_over_budget must match the step-15.5 report
(reports/2026-06-11-cell-token-lengths-38cities.yaml values passed in _ANCHORS)
or the script ABORTS — analysis trusted only while it reproduces the fired gate.

Per city (6 fired + 2 clean controls):
  A. TOKEN DECOMPOSITION on over-length cells (>5760) vs a 1-in-50 sample of
     normal cells: features per cell, buildings/roads per cell, tokens per
     building, tokens per road (sub-F blocks via the sealed splitter/decoder +
     ring promotion — the established classifier).
  B. SOURCE VERTEX ANALYSIS from sub-C features.parquet: building exterior-ring
     vertices per building (raw), and after shapely.simplify(0.5 m,
     preserve_topology=True) — sub-meter, shape-preserving. Reduction ratio.
  C. PROJECTION (stated assumption: cell token count is ~proportional to total
     source vertices — checked empirically via the per-cell tokens/vertex ratio
     reported beside it): projected cell length under simplified vertices ->
     projected frac_over_budget per city.

Run on Leonardo (serial partition):
    .venv/bin/python scripts/investigate_token_lengths.py
"""

from __future__ import annotations

import logging
import statistics
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import pyarrow.parquet as pq  # noqa: E402
from shapely import wkb as shapely_wkb  # noqa: E402

from cfm.data.io import canonicalize_yaml  # noqa: E402
from cfm.data.sub_d.enums import FeatureClass  # noqa: E402
from cfm.data.sub_f.decoder import decode_feature  # noqa: E402
from cfm.data.sub_g.readers import read_sub_f_cells  # noqa: E402
from cfm.data.sub_g.seam_decodability import split_cell_into_features  # noqa: E402
from cfm.data.training.build_shards import _validated_inventory  # noqa: E402
from cfm.eval.geometry import promote_building_rings  # noqa: E402
from cfm.eval.holdout.paths import (  # noqa: E402
    epsg_label_for_region,
    sub_c_region_dir,
    sub_f_region_dir,
    tile_dirname,
)

logger = logging.getLogger(__name__)

_BUDGET = 5760
_SIMPLIFY_TOL_M = 0.5
_NORMAL_SAMPLE_EVERY = 50
_RELEASE = "2026-04-15.0"
#: step-15.5 report values (the gate this analysis must reproduce before reporting).
_ANCHORS = {
    "barcelona": 0.079932,
    "rotterdam": 0.062616,
    "amsterdam": 0.057943,
    "almere": 0.044592,
    "tilburg": 0.024088,
    "eindhoven": 0.023096,
    "budapest": 0.000000,
    "warsaw": 0.000353,
}


def _cell_profile(tokens: list[int]) -> dict:
    """Token decomposition of one cell via the sealed splitter + classifier."""
    blocks = split_cell_into_features(tokens)
    geoms = [decode_feature(b) for b in blocks]
    promoted = promote_building_rings(blocks, geoms)
    n_b = n_r = t_b = t_r = 0
    for block, geom in zip(blocks, promoted, strict=True):
        gt = geom.get("type", "")
        if gt in ("Polygon", "MultiPolygon"):
            n_b += 1
            t_b += len(block)
        elif gt in ("LineString", "MultiLineString"):
            n_r += 1
            t_r += len(block)
    return {
        "len": len(tokens),
        "n_features": len(blocks),
        "n_buildings": n_b,
        "n_roads": n_r,
        "tok_buildings": t_b,
        "tok_roads": t_r,
    }


def _agg(profiles: list[dict]) -> dict:
    """City-level aggregates over a list of cell profiles."""
    if not profiles:
        return {"n_cells": 0}
    tpb = [p["tok_buildings"] / p["n_buildings"] for p in profiles if p["n_buildings"]]
    tpr = [p["tok_roads"] / p["n_roads"] for p in profiles if p["n_roads"]]
    return {
        "n_cells": len(profiles),
        "median_len": statistics.median(p["len"] for p in profiles),
        "median_features_per_cell": statistics.median(p["n_features"] for p in profiles),
        "median_buildings_per_cell": statistics.median(p["n_buildings"] for p in profiles),
        "median_roads_per_cell": statistics.median(p["n_roads"] for p in profiles),
        "median_tokens_per_building": statistics.median(tpb) if tpb else None,
        "median_tokens_per_road": statistics.median(tpr) if tpr else None,
        "building_token_share": (
            sum(p["tok_buildings"] for p in profiles) / sum(p["len"] for p in profiles)
        ),
    }


def _vertex_stats(city: str, cells_of_interest: set[tuple[int, int, int, int]]) -> dict:
    """Source building-vertex stats per cell-of-interest (tile_i, tile_j, ci, cj):
    raw exterior vertices and post-simplify(0.5 m) vertices, per cell.
    Returns {cell_key: (n_buildings, raw_vertices, simplified_vertices)}."""
    epsg = epsg_label_for_region(city)
    base = sub_c_region_dir(_RELEASE, city)
    tiles = {(k[0], k[1]) for k in cells_of_interest}
    out: dict = {}
    for ti, tj in sorted(tiles):
        table = pq.ParquetFile(base / tile_dirname(ti, tj, epsg) / "features.parquet").read(
            columns=["cell_i", "cell_j", "feature_class", "geometry"]
        )
        for ci, cj, fc, g in zip(
            table["cell_i"].to_pylist(),
            table["cell_j"].to_pylist(),
            table["feature_class"].to_pylist(),
            table["geometry"].to_pylist(),
            strict=True,
        ):
            key = (ti, tj, int(ci), int(cj))
            if key not in cells_of_interest or int(fc) != int(FeatureClass.BUILDING):
                continue
            geom = shapely_wkb.loads(bytes(g))
            polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            raw = sum(len(p.exterior.coords) for p in polys)
            simp = geom.simplify(_SIMPLIFY_TOL_M, preserve_topology=True)
            spolys = list(simp.geoms) if simp.geom_type == "MultiPolygon" else [simp]
            srav = sum(len(p.exterior.coords) for p in spolys)
            nb, rv, sv = out.get(key, (0, 0, 0))
            out[key] = (nb + 1, rv + raw, sv + srav)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report: dict = {"budget": _BUDGET, "simplify_tolerance_m": _SIMPLIFY_TOL_M, "cities": {}}

    for city, anchor in _ANCHORS.items():
        logger.info("scanning %s", city)
        epsg = epsg_label_for_region(city)
        sub_f = sub_f_region_dir(_RELEASE, city)
        over: list[dict] = []
        normal: list[dict] = []
        over_keys: set = set()
        lengths_by_key: dict = {}
        n_nonempty = i = 0
        for t in _validated_inventory(_RELEASE, city):
            ti, tj = int(t["tile_i"]), int(t["tile_j"])
            for (ci, cj), toks in sorted(
                read_sub_f_cells(sub_f / tile_dirname(ti, tj, epsg) / "cells.parquet").items()
            ):
                if not toks:
                    continue
                n_nonempty += 1
                if len(toks) > _BUDGET:
                    over.append(_cell_profile(list(toks)))
                    over_keys.add((ti, tj, ci, cj))
                    lengths_by_key[(ti, tj, ci, cj)] = len(toks)
                else:
                    i += 1
                    if i % _NORMAL_SAMPLE_EVERY == 0:
                        normal.append(_cell_profile(list(toks)))
        frac = len(over_keys) / n_nonempty if n_nonempty else 0.0
        if abs(frac - anchor) > 5e-4:
            raise RuntimeError(
                f"ANCHOR MISMATCH {city}: recomputed frac {frac:.6f} != report {anchor} — abort."
            )

        vx = _vertex_stats(city, over_keys) if over_keys else {}
        proj_over = 0
        ratios, tok_per_vertex = [], []
        for key, (_nb, rv, sv) in vx.items():
            if rv:
                ratios.append(sv / rv)
                tok_per_vertex.append(lengths_by_key[key] / rv)
                if lengths_by_key[key] * (sv / rv) > _BUDGET:
                    proj_over += 1
        report["cities"][city] = {
            "frac_over_anchor_ok": True,
            "n_nonempty_cells": n_nonempty,
            "n_over": len(over_keys),
            "frac_over": frac,
            "over_cells": _agg(over),
            "normal_cells_sampled": _agg(normal),
            "vertex_analysis_over_cells": {
                "n_cells_analyzed": len(vx),
                "median_vertices_per_building": (
                    statistics.median(rv / nb for nb, rv, _ in vx.values() if nb) if vx else None
                ),
                "median_simplified_vertices_per_building": (
                    statistics.median(sv / nb for nb, _, sv in vx.values() if nb) if vx else None
                ),
                "median_vertex_reduction_ratio": statistics.median(ratios) if ratios else None,
                "median_cell_tokens_per_source_vertex": (
                    statistics.median(tok_per_vertex) if tok_per_vertex else None
                ),
                "projected_n_over_after_simplify": proj_over,
                "projected_frac_over_after_simplify": (
                    proj_over / n_nonempty if n_nonempty else 0.0
                ),
            },
        }

    out = _REPO / "reports" / "2026-06-11-token-length-investigation.yaml"
    out.write_text(canonicalize_yaml(report), encoding="utf-8")
    print(f"investigation report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the static visualizer's data bundle from the eyeball-probe artifacts.

READ-ONLY over ``reports/_eyeball_probe/`` (immutable). Re-decodes each generated
cell's tokens through the SEALED sub-F decoder and classifies every feature by
CONSTRUCTION IDENTITY (the grammar's building-class tokens, via
``cfm.eval.geometry``) -- NOT by geometry shape. This is the whole point: a
building footprint whose ring does not close to *exact* float equality stays a
``LineString`` in the probe's GeoJSON and is therefore indistinguishable from a
road there ("near-closed buildings misread as roads", SUMMARY.md). Keying on the
building token recovers the true class, so the viz can show:

    building_sealed   -- building block, ring closes exactly  -> filled polygon
    building_unsealed -- building block, ring does NOT close   -> open footprint
    road              -- non-building line
    road_node         -- non-building point

Emits ``viz/data.js`` (``window.PROBE_DATA = {...}``) so the site opens by
double-clicking ``viz/index.html`` -- a ``<script src>`` tag works over
``file://`` where ``fetch()`` does not. Nothing under ``reports/`` is written.

Run:  uv run python viz/build_data.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from cfm.data.sub_g.seam_decodability import split_cell_into_features
from cfm.eval.geometry import _is_building_block, _is_closed_ring
from cfm.inference.generate import try_decode_block

# ---------------------------------------------------------------------------
# Human-readable semantics for the conditioning dimensions, transcribed from
# configs/macro_plan/v1/macro_plan_vocab.yaml (locked_buckets) and
# src/cfm/eval/holdout/labels.py. Each stratum tuple is
# (zoning, road_skeleton, density, coastal); see GROUND_TRUTH.md s3.
# ---------------------------------------------------------------------------
DIMENSIONS: dict[str, Any] = {
    "zoning": {
        "label": "Dominant zoning class",
        "kind": "categorical",
        "values": {0: "building", 1: "road", 2: "poi", 3: "base"},
        "note": "Most common per-cell zoning class. Fixed at 1 across the probe contexts.",
    },
    "road_skeleton": {
        "label": "Road-skeleton density",
        "kind": "ordinal",
        "buckets": [
            {"id": 0, "range": "0-1 edges", "gloss": "almost no internal road grid"},
            {"id": 1, "range": "1-4 edges", "gloss": "a few internal roads"},
            {"id": 2, "range": "4-9 edges", "gloss": "grid-like internal road network"},
            {"id": 3, "range": ">=9 edges", "gloss": "dense internal road grid"},
        ],
        "note": "Modal internal-edge road-skeleton bucket.",
    },
    "density": {
        "label": "Built-up density (population_density_bucket)",
        "kind": "ordinal",
        "buckets": [
            {"id": 0, "range": "0-2% footprint", "gloss": "very sparse / suburban"},
            {"id": 1, "range": "2-15% footprint", "gloss": "low density"},
            {"id": 2, "range": "15-31% footprint", "gloss": "medium / mixed"},
            {"id": 3, "range": ">=31% footprint", "gloss": "very built-up / dense urban"},
        ],
        "note": "p75 of per-cell building-footprint ratio, bucketed (tile-level density label).",
    },
    "coastal": {
        "label": "Coastal / inland / river",
        "kind": "categorical",
        "values": {0: "class 0", 1: "class 1", 2: "class 2", 3: "class 3"},
        "note": "Unscored near-constant dimension; fixed at 2 across the probe contexts.",
    },
}

# The 7 character-stats channels (build_shards.character_stats_for_cell). These
# are the per-cell continuous vector handed to the model at generation time --
# the "echo" PROJECT_FOCUS.md flags: channels 0-3 literally describe building
# size/count. We decode them back to physical units for an honest readout.
CHAR_CHANNELS = [
    {"name": "median building area", "unit": "m2", "decode": "pow10"},
    {"name": "building-size IQR", "unit": "m2", "decode": "pow10_minus1"},
    {"name": "p90/p50 building-size ratio", "unit": "x", "decode": "pow10"},
    {"name": "building count", "unit": "", "decode": "pow10_minus1_round"},
    {"name": "median road-segment length", "unit": "m", "decode": "pow10_minus1"},
    {"name": "buildings present", "unit": "flag", "decode": "flag"},
    {"name": "roads present", "unit": "flag", "decode": "flag"},
]

CONTEXT_LABELS = {
    "dense_urban": "Dense urban",
    "medium_mixed": "Medium / mixed",
    "sparse_suburban": "Sparse suburban",
}


def _decode_char(value: float, how: str) -> float:
    if how == "flag":
        return float(value)
    if how == "pow10":
        return 10.0**value
    if how == "pow10_minus1":
        return max(0.0, 10.0**value - 1.0)
    if how == "pow10_minus1_round":
        return round(max(0.0, 10.0**value - 1.0))
    raise ValueError(how)


def _round_coords(obj: Any, nd: int = 3) -> Any:
    if isinstance(obj, (int, float)):
        return round(float(obj), nd)
    return [_round_coords(x, nd) for x in obj]


def _closure_gap(coords: list[list[float]]) -> float | None:
    """Ring closure gap = |first - last| / bbox-diagonal (the SUMMARY's metric)."""
    if not coords or len(coords) < 4:
        return None
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    (x0, y0), (x1, y1) = coords[0], coords[-1]
    return math.hypot(x1 - x0, y1 - y0) / diag


def _bbox_of(features: list[dict]) -> list[float]:
    xs: list[float] = []
    ys: list[float] = []

    def walk(c: Any) -> None:
        if c and isinstance(c[0], (int, float)):
            xs.append(c[0])
            ys.append(c[1])
            return
        for x in c:
            walk(x)

    for f in features:
        walk(f["coords"])
    if not xs:
        return [0.0, 0.0, 1.0, 1.0]
    return [min(xs), min(ys), max(xs), max(ys)]


def _classify_cell(tokens: list[int]) -> tuple[list[dict], dict[str, int]]:
    """tokens -> (display features, class counts), faithful to construction identity."""
    features: list[dict] = []
    counts = {
        "building_sealed": 0,
        "building_unsealed": 0,
        "road": 0,
        "road_node": 0,
        "undecodable": 0,
    }
    for block in split_cell_into_features(tokens):
        geom = try_decode_block(block)
        if geom is None:
            counts["undecodable"] += 1
            continue
        gtype = geom["type"]
        coords = _round_coords(geom["coordinates"])
        if _is_building_block(block):
            if gtype == "LineString" and _is_closed_ring(geom):
                # closed building ring -> promote to filled polygon (the one
                # promotion authority, cfm.eval.geometry.promote_building_rings).
                counts["building_sealed"] += 1
                features.append({"cls": "building_sealed", "type": "Polygon", "coords": [coords]})
            elif gtype in ("Polygon", "MultiPolygon"):
                counts["building_sealed"] += 1
                features.append({"cls": "building_sealed", "type": gtype, "coords": coords})
            else:
                # building block whose ring does NOT close exactly -- the
                # "near-closed building misread as a road" case.
                counts["building_unsealed"] += 1
                feat = {"cls": "building_unsealed", "type": gtype, "coords": coords}
                gap = _closure_gap(coords) if gtype == "LineString" else None
                if gap is not None:
                    feat["gap"] = round(gap, 4)
                features.append(feat)
        else:
            if gtype == "Point":
                counts["road_node"] += 1
                features.append({"cls": "road_node", "type": "Point", "coords": coords})
            else:
                counts["road"] += 1
                features.append({"cls": "road", "type": gtype, "coords": coords})
    return features, counts


def build_probe_data(gen_tokens_path: Path) -> dict[str, Any]:
    """Build the full data bundle from the probe's gen_tokens.json (read-only)."""
    raw = json.loads(gen_tokens_path.read_text())
    meta = dict(raw["meta"])
    meta["max_new"] = raw["max_new"]

    cells: list[dict] = []
    for rec in raw["records"]:
        features, counts = _classify_cell(rec["tokens"])
        n_blocks = sum(counts.values())
        decodable = n_blocks - counts["undecodable"]
        char = rec["char_stats"]
        char_decoded = [
            {
                "name": ch["name"],
                "unit": ch["unit"],
                "raw": round(char[i], 4),
                "value": round(_decode_char(char[i], ch["decode"]), 2),
            }
            for i, ch in enumerate(CHAR_CHANNELS)
        ]
        cells.append(
            {
                "context": rec["context"],
                "cell_index": rec["cell_index"],
                "stratum": rec["stratum"],
                "density": rec["pop_density"],
                "gen_seed": rec["gen_seed"],
                "n_tokens": rec["n_tokens"],
                "self_terminated": rec.get("self_terminated"),
                "hit_cap": rec.get("hit_cap"),
                "decodability": round(decodable / n_blocks, 4) if n_blocks else 0.0,
                "counts": counts,
                "char_decoded": char_decoded,
                "bbox": [round(v, 3) for v in _bbox_of(features)],
                "features": features,
            }
        )

    contexts = list(dict.fromkeys(c["context"] for c in cells))
    summary = []
    for ctx in contexts:
        cc = [c for c in cells if c["context"] == ctx]
        n_buildings = [
            c["counts"]["building_sealed"] + c["counts"]["building_unsealed"] for c in cc
        ]
        summary.append(
            {
                "context": ctx,
                "n_cells": len(cc),
                "med_tokens": int(statistics.median([c["n_tokens"] for c in cc])),
                "med_buildings": round(statistics.median(n_buildings), 1),
                "med_roads": round(statistics.median([c["counts"]["road"] for c in cc]), 1),
            }
        )

    return {
        "meta": meta,
        "dimensions": DIMENSIONS,
        "char_channels": CHAR_CHANNELS,
        "contexts": contexts,
        "context_labels": CONTEXT_LABELS,
        "summary": summary,
        "cells": cells,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument(
        "--in",
        dest="inp",
        default=str(repo / "reports" / "_eyeball_probe" / "gen_tokens.json"),
    )
    ap.add_argument("--out", default=str(repo / "viz" / "data.js"))
    args = ap.parse_args()

    bundle = build_probe_data(Path(args.inp))
    payload = json.dumps(bundle, separators=(",", ":"))
    header = (
        "// AUTO-GENERATED by viz/build_data.py from reports/_eyeball_probe/ "
        "(immutable). Do not edit by hand; re-run the builder.\n"
    )
    Path(args.out).write_text(header + "window.PROBE_DATA = " + payload + ";\n")

    n_cells = len(bundle["cells"])
    totals: dict[str, int] = {}
    for c in bundle["cells"]:
        for k, v in c["counts"].items():
            totals[k] = totals.get(k, 0) + v
    print(f"[build_data] wrote {args.out}")
    print(f"[build_data] {n_cells} cells, class totals: {totals}")
    for s in bundle["summary"]:
        print(
            f"  {s['context']:16s} n={s['n_cells']} "
            f"med_tokens={s['med_tokens']:>5} med_buildings={s['med_buildings']:>5} "
            f"med_roads={s['med_roads']:>5}"
        )


if __name__ == "__main__":
    main()

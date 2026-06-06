#!/usr/bin/env python
"""Prevalence of decode-inflation (the 0.5m-quantum-floor bloat of over-densified
polygons) across the VALIDATED corpus — the decision-maker for Path A (de-densify +
full re-derive) vs Path B (add cities + document the limitation).

For a representative sample of cities x morphologies, decode every feature with the
validator's OWN authoritative pairing (read_sub_c_features_by_cell + parts-walk),
compute inflation = decoded_path_length / sub_c_source_path_length, and bucket by
ratio as a fraction of features AND of TOKEN SHARE (an inflated building dominating
a tile matters more than a rounding speck). Reported overall, per feature_class, and
per morphology. Read-only over data/.
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

import pyarrow.parquet as pq  # noqa: E402
from shapely.wkb import loads as wkb_loads  # noqa: E402

from cfm.data.sub_f.decoder import decode_feature  # noqa: E402
from cfm.data.sub_f.encoder import canonicalize_geometry  # noqa: E402
from cfm.data.sub_g.readers import read_sub_c_features_by_cell  # noqa: E402
from cfm.data.sub_g.seam_decodability import _canon_parts, split_cell_into_features  # noqa: E402

R = "2026-04-15.0"
PROC = _REPO / "data" / "processed"
TILES_PER_CITY = 30  # evenly-spaced sample

# representative validated cities -> morphology (axis label, NOT the sub_c constant)
CITIES = {
    "prague": "medieval-organic",
    "bologna": "medieval-organic",
    "toledo": "medieval-organic",
    "turin": "planned-grid",
    "karlsruhe": "planned-grid",
    "umea": "planned-grid",
    "cergy": "modernist-sprawl",
    "tychy": "modernist-sprawl",
    "espoo": "modernist-sprawl",
    "copenhagen": "mixed",
    "budapest": "mixed",
    "malmo": "mixed",
}
FC = {0: "road", 1: "building", 2: "poi", 3: "base"}
BUCKETS = [(0.0, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 4.0), (4.0, math.inf)]


def _bucket(r: float) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= r < hi:
            return i
    return len(BUCKETS) - 1


def _dpathlen(geom: dict) -> float:
    c = geom["coordinates"]
    t = geom["type"]
    if t == "Point":
        return 0.0
    pts = [p for ring in c for p in ring] if t == "Polygon" else c
    return sum(math.dist(pts[i - 1], pts[i]) for i in range(1, len(pts)))


def main() -> int:
    # (scope, class) -> per-bucket [feat_count, token_sum]; scope in {"ALL", morph}
    counts: dict = defaultdict(lambda: [[0, 0] for _ in BUCKETS])
    n_feat = 0
    for city, morph in CITIES.items():
        region = PROC / "sub_f" / R / city
        tiles = sorted(region.glob("tile=*"))
        if not tiles:
            continue
        step = max(1, len(tiles) // TILES_PER_CITY)
        for tdir in tiles[::step][:TILES_PER_CITY]:
            tile = tdir.name
            cpath = PROC / "sub_c" / R / city / tile / "features.parquet"
            if not cpath.exists():
                continue
            fbc = read_sub_c_features_by_cell(cpath)
            for row in pq.ParquetFile(tdir / "cells.parquet").read().to_pylist():
                cell = (row["cell_i"], row["cell_j"])
                blocks = split_cell_into_features(list(row["token_sequence"]))
                decoded = []
                for b in blocks:
                    try:
                        decoded.append(decode_feature(b))
                    except Exception:
                        decoded.append(None)
                di = 0
                for f in fbc.get(cell, []):
                    canon = canonicalize_geometry(wkb_loads(bytes(f["geometry"])))
                    parts = _canon_parts(canon)
                    grp_d = decoded[di : di + len(parts)]
                    grp_b = blocks[di : di + len(parts)]
                    di += len(parts)
                    cls = FC.get(f["feature_class"], str(f["feature_class"]))
                    for dg, sp, blk in zip(grp_d, parts, grp_b, strict=False):
                        if dg is None or sp.length <= 0:
                            continue
                        ratio = _dpathlen(dg) / sp.length
                        bi = _bucket(ratio)
                        ntok = len(blk)
                        for scope in ("ALL", morph):
                            for key in ((scope, cls), (scope, "ALL")):
                                counts[key][bi][0] += 1
                                counts[key][bi][1] += ntok
                        n_feat += 1

    labels = ["<1.0", "1.0-1.5", "1.5-2", "2-4", ">=4"]
    print(f"features measured: {n_feat}  (cities={len(CITIES)}, ~{TILES_PER_CITY} tiles each)")

    def emit(scope: str, cls: str) -> None:
        rows = counts.get((scope, cls))
        if not rows:
            return
        tot_c = sum(r[0] for r in rows) or 1
        tot_t = sum(r[1] for r in rows) or 1
        cfrac = "  ".join(f"{labels[i]}={100 * rows[i][0] / tot_c:.2f}%" for i in range(len(rows)))
        tfrac = "  ".join(f"{labels[i]}={100 * rows[i][1] / tot_t:.2f}%" for i in range(len(rows)))
        ge2_t = 100 * (rows[3][1] + rows[4][1]) / tot_t
        ge4_t = 100 * rows[4][1] / tot_t
        print(f"\n[{scope} / {cls}]  n={tot_c}")
        print("  count%:  " + cfrac)
        print("  token%:  " + tfrac)
        print(f"  >=2x token-share = {ge2_t:.2f}%   >=4x token-share = {ge4_t:.2f}%")

    print("\n========== OVERALL ==========")
    emit("ALL", "ALL")
    emit("ALL", "building")
    emit("ALL", "road")
    print("\n========== PER-MORPHOLOGY (building) ==========")
    for m in sorted({v for v in CITIES.values()}):
        emit(m, "building")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

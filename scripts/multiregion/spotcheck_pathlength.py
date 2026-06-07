#!/usr/bin/env python3
"""Path-length undistortion spot-check for the worst-degraded cities (rotterdam/warsaw).

Validator-clean (within-bound + OGC-valid) is NECESSARY but NOT SUFFICIENT to re-admit
our two worst-degraded cities: a building can pass the 300 m bound while still being,
say, 1.5x distorted. This confirms they are actually UNdistorted by measuring, per
BUILDING (Polygon), the ratio:

    path_length(decoded post-fix ring) / path_length(sub_c source ring)

Path-length is vertex-count-invariant (immune to §3.5 chunking inserts and to
de-densify's vertex removal — both preserve the shape), so a ratio != 1.0 means the
0.5 m quantum INFLATED the perimeter (the #19 distortion), not a vertex-count artifact.
The sub_c source is the authoritative third party: it is the encoder's input, derived
directly from Overture and UNCHANGED by the sub_f-only fix, and it is the same authority
that proved the ~13x inflation originally. For an interior (unclipped) building the
sub_c ring equals the unclipped Overture footprint, so this is the decoded-vs-Overture
trace the PI asked for; the worst building's source_feature_id (GERS) is printed so the
raw-Overture leg can confirm it directly.

Reports per city: max ratio, p99, count > 1.10x, and the single WORST building
(source_feature_id, tile/cell, ratio, decoded vs source perimeter, interior?).
Re-admit only if max ~ 1.0x; if either city's worst is still > 1.0x -> exclude.
"""

from __future__ import annotations

import glob
import math
import statistics
import sys
from pathlib import Path

from shapely.wkb import loads as wkb_loads

from cfm.data.sub_f.decoder import decode_feature
from cfm.data.sub_f.encoder import canonicalize_geometry
from cfm.data.sub_g.readers import read_sub_c_features_by_cell, read_sub_f_cells
from cfm.data.sub_g.seam_decodability import (
    _canon_parts,
    _decoded_coords,
    _part_coords,
    split_cell_into_features,
)

RELEASE = "2026-04-15.0"
PROC = Path("data/processed")
CELL_EXTENT_M = 250.0
EDGE_EPS = 0.5  # interior = bbox at least this far from every cell edge (clip-free)


def _pathlen(coords: list[tuple[float, float]]) -> float:
    return sum(
        math.hypot(coords[i][0] - coords[i - 1][0], coords[i][1] - coords[i - 1][1])
        for i in range(1, len(coords))
    )


def _is_interior(coords: list[tuple[float, float]]) -> bool:
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return (
        min(xs) > EDGE_EPS
        and min(ys) > EDGE_EPS
        and max(xs) < CELL_EXTENT_M - EDGE_EPS
        and max(ys) < CELL_EXTENT_M - EDGE_EPS
    )


def main() -> int:
    city = sys.argv[1]
    ratios: list[float] = []
    worst = None  # (ratio, sfid, tile, cell, dec_len, src_len, interior)
    n_buildings = 0
    for tile_dir in sorted(glob.glob(str(PROC / "sub_f" / RELEASE / city / "tile=*"))):
        tile = Path(tile_dir).name
        cells = read_sub_f_cells(Path(tile_dir) / "cells.parquet")
        feats = read_sub_c_features_by_cell(
            PROC / "sub_c" / RELEASE / city / tile / "features.parquet"
        )
        for cell, flist in feats.items():
            seq = cells.get(cell)
            if not seq:
                continue
            blocks = split_cell_into_features(seq)
            decoded: list[dict | None] = []
            for b in blocks:
                try:
                    decoded.append(decode_feature(b))
                except Exception:
                    decoded.append(None)
            di = 0
            for f in flist:
                canon = canonicalize_geometry(wkb_loads(bytes(f["geometry"])))
                parts = _canon_parts(canon)
                grp = decoded[di : di + len(parts)]
                di += len(parts)
                if len(grp) < len(parts) or any(g is None for g in grp):
                    continue
                for part, dec in zip(parts, grp, strict=True):
                    if part.geom_type != "Polygon":  # buildings only (Case A, no bref)
                        continue
                    src = _part_coords(part)
                    dco = _decoded_coords(dec)
                    src_len = _pathlen(src)
                    if src_len <= 0:
                        continue
                    n_buildings += 1
                    r = _pathlen(dco) / src_len
                    ratios.append(r)
                    if worst is None or r > worst[0]:
                        worst = (
                            r,
                            f["source_feature_id"],
                            tile,
                            cell,
                            _pathlen(dco),
                            src_len,
                            _is_interior(src),
                        )

    if not ratios:
        print(f"{city}: NO building features paired (unexpected)")
        return 1
    ratios.sort()
    p99 = ratios[min(len(ratios) - 1, int(0.99 * len(ratios)))]
    n_over = sum(1 for r in ratios if r > 1.10)
    print(f"=== {city} building path-length undistortion (decoded / sub_c-source) ===")
    print(f"  buildings paired: {n_buildings}")
    print(f"  ratio: max={ratios[-1]:.3f}x  p99={p99:.3f}x  mean={statistics.mean(ratios):.3f}x")
    print(f"  buildings > 1.10x: {n_over} ({100 * n_over / n_buildings:.3f}%)")
    r, sfid, tile, cell, dlen, slen, interior = worst
    print("  WORST building:")
    print(f"    ratio={r:.3f}x  source_feature_id={sfid}")
    print(f"    {tile} cell={cell}  decoded_perim={dlen:.2f}m  source_perim={slen:.2f}m")
    print(f"    interior(clip-free)={interior}  <- if True, sub_c ring == unclipped Overture")
    verdict = (
        "UNDISTORTED (max ~1.0x)" if ratios[-1] <= 1.10 else "STILL DISTORTED (>1.10x) -> EXCLUDE"
    )
    print(f"  VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

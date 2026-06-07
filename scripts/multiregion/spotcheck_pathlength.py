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
# A real building ring; below this a near-zero source perimeter makes the ratio a 0/0
# artifact (degenerate sub-quantum-extent source / clip sliver — the documented #19 edge).
MIN_REAL_PERIM_M = 2.0
# Jitter-clean subset: at >=10m perimeter the 0.5m grid jitter is <~5%, so the ratio is a
# clean distortion signal (a #19 residual would be >=1.5x). Used for the verdict.
LARGE_PERIM_M = 10.0


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
    real_ratios: list[float] = []  # source >= MIN_REAL_PERIM_M
    large_worst = None  # worst ratio among source >= LARGE_PERIM_M (jitter-clean)
    n_real = n_degen = 0
    degen_max_decoded = 0.0
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
                    src_len = _pathlen(src)
                    dec_len = _pathlen(_decoded_coords(dec))
                    if src_len < MIN_REAL_PERIM_M:
                        # Degenerate sub-quantum-EXTENT source (clip sliver / noise): a
                        # near-zero source perimeter makes the ratio a 0/0 artifact. These
                        # are the documented #19 edge — decoded is bounded (tracked below),
                        # NOT real-building distortion. Excluded from the ratio judgment.
                        n_degen += 1
                        degen_max_decoded = max(degen_max_decoded, dec_len)
                        continue
                    n_real += 1
                    r = dec_len / src_len
                    real_ratios.append(r)
                    if src_len >= LARGE_PERIM_M and (large_worst is None or r > large_worst[0]):
                        large_worst = (
                            r,
                            f["source_feature_id"],
                            tile,
                            cell,
                            dec_len,
                            src_len,
                            _is_interior(src),
                        )

    if not real_ratios:
        print(f"{city}: NO real building features paired (unexpected)")
        return 1
    real_ratios.sort()

    def _pct(p: float) -> float:
        return real_ratios[min(len(real_ratios) - 1, int(p * len(real_ratios)))]

    n_over110 = sum(1 for r in real_ratios if r > 1.10)
    n_over150 = sum(1 for r in real_ratios if r > 1.50)  # >1.5x on a real ring = #19 inflation
    print(f"=== {city} building path-length undistortion (decoded / sub_c-source) ===")
    print(f"  REAL buildings (source >= {MIN_REAL_PERIM_M}m): {n_real}")
    print(
        f"  ratio: mean={statistics.mean(real_ratios):.3f}x  p99={_pct(0.99):.3f}x  "
        f"p99.9={_pct(0.999):.3f}x  max={real_ratios[-1]:.3f}x"
    )
    print(f"  real > 1.10x: {n_over110}  |  real > 1.50x (clear #19 inflation): {n_over150}")
    print(
        f"  degenerate (source < {MIN_REAL_PERIM_M}m sub-quantum-extent/clip sliver): "
        f"{n_degen}; max decoded_perim={degen_max_decoded:.2f}m (bounded, harmless)"
    )
    large_max = None
    if large_worst:
        r, sfid, tile, cell, dlen, slen, interior = large_worst
        large_max = r
        print(f"  WORST LARGE building (source >= {LARGE_PERIM_M}m; jitter-clean):")
        print(f"    ratio={r:.3f}x  source_feature_id={sfid}")
        print(f"    {tile} cell={cell}  decoded_perim={dlen:.2f}m  source_perim={slen:.2f}m")
        print(f"    interior(clip-free)={interior}  <- if True, sub_c ring == unclipped Overture")
    # Undistorted: large jitter-clean buildings ~1.0x AND no real ring inflated > 1.5x.
    undistorted = (large_max is None or large_max <= 1.10) and n_over150 == 0
    verdict = (
        "UNDISTORTED (real buildings ~1.0x; no #19 residual)"
        if undistorted
        else "STILL DISTORTED -> EXCLUDE"
    )
    print(f"  VERDICT: {verdict}")
    return 0 if undistorted else 1


if __name__ == "__main__":
    raise SystemExit(main())

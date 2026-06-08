#!/usr/bin/env python3
"""Byte-compare reframe (condition #4): prove the #19 fix changed ONLY over-densified
geometry, on the least-changed city.

Since over-densification is pervasive (eindhoven changed 603/611 tiles), a fully
0-change city likely does not exist — so instead of "some city is untouched," we prove
the stronger, correct claim: every feature whose tokens CHANGED (1.1 -> 1.2) had a
sub-quantum segment in its sub_c source, and NO feature without one changed. A change to
a feature with no sub-quantum segment would be the inverse failure (the fix touching
representable geometry); this counts them and asserts ZERO.

Requires the pre-fix sub_f preserved at data/processed/sub_f/<R>/.preserve11_<city>
(1.1) alongside the re-derived live (1.2). Pairs old/new feature blocks (same count, same
order — de-densify changes vertices within a feature, never the feature count) and maps
each block to its sub_c source via the parts-walk to test the sub-quantum signature.
"""

from __future__ import annotations

import glob
import math
import sys
from pathlib import Path

from shapely.wkb import loads as wkb_loads

from cfm.data.sub_f.encoder import DEFAULT_MAGNITUDE_QUANTUM_M, canonicalize_geometry
from cfm.data.sub_g.readers import read_sub_c_features_by_cell, read_sub_f_cells
from cfm.data.sub_g.seam_decodability import _canon_parts, _part_coords, split_cell_into_features

RELEASE = "2026-04-15.0"
PROC = Path("data/processed")
Q = DEFAULT_MAGNITUDE_QUANTUM_M


def _has_subquantum(coords: list[tuple[float, float]]) -> bool:
    return any(
        math.hypot(coords[i][0] - coords[i - 1][0], coords[i][1] - coords[i - 1][1]) < Q
        for i in range(1, len(coords))
    )


def main() -> int:
    city = sys.argv[1]
    old_base = PROC / "sub_f" / RELEASE / f".preserve11_{city}"
    new_base = PROC / "sub_f" / RELEASE / city
    if not old_base.exists():
        print(f"no preserved 1.1 sub_f at {old_base}")
        return 1

    n_feat = changed = changed_subq = changed_no_subq = 0
    violations: list[str] = []
    for old_tile in sorted(glob.glob(str(old_base / "tile=*"))):
        tile = Path(old_tile).name
        old_cells = read_sub_f_cells(Path(old_tile) / "cells.parquet")
        new_cells = read_sub_f_cells(new_base / tile / "cells.parquet")
        feats = read_sub_c_features_by_cell(
            PROC / "sub_c" / RELEASE / city / tile / "features.parquet"
        )
        for cell, flist in feats.items():
            old_seq, new_seq = old_cells.get(cell), new_cells.get(cell)
            if not old_seq or not new_seq:
                continue
            old_blocks = split_cell_into_features(old_seq)
            new_blocks = split_cell_into_features(new_seq)
            if len(old_blocks) != len(new_blocks):
                violations.append(f"{tile} {cell}: block-count changed (unexpected)")
                continue
            bi = 0
            for f in flist:
                canon = canonicalize_geometry(wkb_loads(bytes(f["geometry"])))
                parts = _canon_parts(canon)
                for part in parts:
                    if bi >= len(old_blocks):
                        break
                    subq = _has_subquantum(_part_coords(part))
                    diff = old_blocks[bi] != new_blocks[bi]
                    n_feat += 1
                    if diff:
                        changed += 1
                        if subq:
                            changed_subq += 1
                        else:
                            changed_no_subq += 1
                            if len(violations) < 8:
                                violations.append(
                                    f"{tile} {cell} block{bi} sfid={f['source_feature_id']}: "
                                    f"CHANGED but NO sub-quantum segment"
                                )
                    bi += 1

    print(f"=== {city} byte-compare: changes confined to over-densified features? ===")
    print(f"  features paired: {n_feat}")
    print(f"  changed (1.1 -> 1.2): {changed}")
    print(f"  changed WITH sub-quantum segment (expected): {changed_subq}")
    print(f"  changed WITHOUT sub-quantum segment (inverse-failure; MUST be 0): {changed_no_subq}")
    if violations:
        print("  VIOLATIONS:")
        for v in violations:
            print(f"    {v}")
    ok = changed_no_subq == 0
    verdict = (
        "CONFINED — only over-densified geometry changed"
        if ok
        else "NOT CONFINED — fix touched representable geometry"
    )
    print(f"  VERDICT: {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

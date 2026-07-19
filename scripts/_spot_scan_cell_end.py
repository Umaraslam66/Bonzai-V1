"""On-Leonardo spot-scan of the REBUILT v2 shard cache (cell-EOS Tooth-2, live half).

Reads every cell of the sealed cache via the canonical read_city_cache and verifies
the 260 contract on real rebuilt data: every non-empty cell ends (...,510,260),
exactly one 260, never interior; empty cells stay (). Reports counts + any violation.
"""

from __future__ import annotations

import sys
from pathlib import Path

from cfm.data.training.shard_cache import read_city_cache

CELL_END = 260
FEATURE_END = 510
RELEASE_DIR = Path("data/processed/training_cache/2026-04-15.0")


def main() -> int:
    cities = sorted(p.name for p in RELEASE_DIR.iterdir() if p.is_dir())
    tot_cells = tot_nonempty = tot_empty = 0
    ends_260 = ends_510_260 = exactly_one_260 = 0
    v_not_end_260: list = []
    v_count_not_1: list = []
    v_interior_260: list = []
    v_not_510_260: list = []
    v_empty_with_260: list = []
    v_260_only: list = []  # the empty-guard-failure signature: a cell == (260,)

    for city in cities:
        shards = read_city_cache(RELEASE_DIR / city)
        for sh in shards:
            for c in sh.cells:
                t = c.tokens
                tot_cells += 1
                loc = (city, sh.tile_i, sh.tile_j, c.cell_i, c.cell_j)
                if len(t) == 0:
                    tot_empty += 1
                    continue
                tot_nonempty += 1
                if t == (CELL_END,):
                    v_260_only.append(loc)
                cnt = t.count(CELL_END)
                if t[-1] == CELL_END:
                    ends_260 += 1
                else:
                    v_not_end_260.append((*loc, t[-3:]))
                if cnt == 1:
                    exactly_one_260 += 1
                else:
                    v_count_not_1.append((*loc, cnt))
                if CELL_END in t[:-1]:
                    v_interior_260.append(loc)
                if len(t) >= 2 and t[-2] == FEATURE_END and t[-1] == CELL_END:
                    ends_510_260 += 1
                else:
                    v_not_510_260.append((*loc, t[-3:]))
        del shards

    def pct(n: int) -> str:
        return f"{100.0 * n / tot_nonempty:.6f}%" if tot_nonempty else "n/a"

    print(f"cities scanned          : {len(cities)}")
    print(f"total cells             : {tot_cells}")
    print(f"  non-empty             : {tot_nonempty}")
    print(f"  empty (())            : {tot_empty}")
    print(f"non-empty ending 260    : {ends_260}/{tot_nonempty} ({pct(ends_260)})")
    print(f"non-empty ending(510,260): {ends_510_260}/{tot_nonempty} ({pct(ends_510_260)})")
    print(f"non-empty w/ exactly 1 260: {exactly_one_260}/{tot_nonempty}")
    print("=== VIOLATIONS (expect all 0) ===")
    print(f"non-empty NOT ending 260   : {len(v_not_end_260)}")
    print(f"260 count != 1             : {len(v_count_not_1)}")
    print(f"interior 260               : {len(v_interior_260)}")
    print(f"non-empty NOT ending(510,260): {len(v_not_510_260)}")
    print(f"empty cell carrying 260    : {len(v_empty_with_260)}")
    print(f"cell == (260,) [empty-guard fail]: {len(v_260_only)}")
    for name, lst in [
        ("not_end_260", v_not_end_260),
        ("count_not_1", v_count_not_1),
        ("interior_260", v_interior_260),
        ("not_510_260", v_not_510_260),
        ("260_only", v_260_only),
    ]:
        if lst:
            print(f"  sample {name}: {lst[:5]}")
    clean = not (
        v_not_end_260 or v_count_not_1 or v_interior_260 or v_not_510_260 or v_empty_with_260
        or v_260_only
    )
    print("RESULT:", "ALL CLEAN" if clean else "VIOLATIONS FOUND")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())

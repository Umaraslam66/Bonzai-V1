#!/usr/bin/env python3
"""Gate helper: measured post-fix token count for one city (mirrors
build_g4_rollup._city_token_stats). Reports delta vs the ~26.9M inflated baseline
so a material shrink (checkpoint #3 risk) surfaces at the gate, not at final G4."""

from __future__ import annotations

import glob
import sys

import pyarrow.parquet as pq

RELEASE = "2026-04-15.0"
# Inflated (pre-fix) baselines from the failed add-cities G4 (2026-06-06).
_BASELINE = {"eindhoven": 26_900_000, "tilburg": 25_100_000, "szczecin": 7_200_000}


def main() -> int:
    city = sys.argv[1]
    files = sorted(glob.glob(f"data/processed/sub_f/{RELEASE}/{city}/tile=*/cells.parquet"))
    n = 0
    for f in files:
        t = pq.ParquetFile(f).read(columns=["token_sequence"])
        n += len(t.column("token_sequence").combine_chunks().flatten())
    print(f"  {city}: tiles={len(files)} tokens={n:,}")
    base = _BASELINE.get(city)
    if base:
        print(f"  inflated baseline ~{base:,} ; post-fix delta = {(n - base) / base * 100:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

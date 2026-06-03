"""Diagnose Berlin sub_c manifest-vs-disk tile mismatch (pilot finding #2)."""

from __future__ import annotations

import glob
import os

import yaml

B = "data/processed/sub_c/2026-04-15.0/berlin"


def parse(name: str) -> tuple[int, int]:
    n = os.path.basename(name).replace("tile=EPSG25833_", "")
    parts = n.split("_")
    return int(parts[0].lstrip("i")), int(parts[1].lstrip("j"))


m = yaml.safe_load(open(f"{B}/manifest.yaml"))
mt = {(t["tile_i"], t["tile_j"]) for t in m["tiles"]}
dt = {parse(d) for d in glob.glob(f"{B}/tile=EPSG25833_*")}

orphans = dt - mt
missing = mt - dt
print(f"manifest tiles: {len(mt)} | disk dirs: {len(dt)}")
print(f"orphans (disk not in manifest): {len(orphans)}")
print(f"missing (manifest not on disk): {len(missing)}")

mi = sorted(i for i, j in mt)
mj = sorted(j for i, j in mt)
print(f"manifest i: [{min(mi)}, {max(mi)}]  j: [{min(mj)}, {max(mj)}]")
if orphans:
    oi = sorted(i for i, j in orphans)
    oj = sorted(j for i, j in orphans)
    print(f"orphan   i: [{min(oi)}, {max(oi)}]  j: [{min(oj)}, {max(oj)}]")
    print(f"ALL orphans at i > manifest_max_i ({max(mi)})? {min(oi) > max(mi)}")
    print(f"ALL orphans at j > manifest_max_j ({max(mj)})? {min(oj) > max(mj)}")
    print(f"sample orphans: {sorted(orphans)[:8]}")
# Is the orphan set exactly one extra column (i = max_i+1) or row (j = max_j+1)?
extra_col = {(i, j) for (i, j) in orphans if i == max(mi) + 1}
extra_row = {(i, j) for (i, j) in orphans if j == max(mj) + 1}
print(f"orphans in extra column i={max(mi) + 1}: {len(extra_col)}")
print(f"orphans in extra row j={max(mj) + 1}: {len(extra_row)}")

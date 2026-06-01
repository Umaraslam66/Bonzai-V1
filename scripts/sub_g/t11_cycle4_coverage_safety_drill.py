"""sub-G T11 cycle-4 drill: coverage-safety of gating bref emission to roads only.

Cycle-4: the sub-F encoder emits a bref for ANY LineString clipped to endpoint on
an active road edge, including NON-road LineStrings (e.g. natural=coastline) ->
_check_non_road_non_emission halts (first at tile i10_j14 cell (4,6), key 'natural').

The proposed fix gates emission on the feature's semantic key == 'highway'. THE
load-bearing question (read-only): for every (cell, direction) that currently
emits a bref, does a ROAD (highway) feature ALSO emit there? If yes everywhere,
gating out non-road emissions never removes the last bref on an active road edge
-> coverage stays satisfied. If some (cell, direction) emits ONLY from non-road
features, gating would zero its brefs and coverage would fire there -> the gate
site / fix shape needs rethinking BEFORE scoping.

Reads sub-F cells.parquet directly (the encoder wrote all 494 tiles before the
validator halted). Read-only.

Run: uv run python scripts/sub_g/t11_cycle4_coverage_safety_drill.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from cfm.data.sub_f.validator_cross_tile import (
    _bref_id_to_dir_class,
    _emitted_brefs_by_cell,
    _semantic_id_to_tag,
)
from cfm.data.sub_f.vocab import unknown_family_tag_to_key

REPO_ROOT = Path(__file__).resolve().parents[2]
SUB_F = REPO_ROOT / "data" / "processed" / "sub_f" / "2026-04-15.0" / "singapore"
_ROAD_KEY = "highway"


def main() -> int:
    tile_dirs = sorted(SUB_F.glob("tile=EPSG3414_*"))
    if not tile_dirs:
        print(f"NO_TILES at {SUB_F}")
        return 2

    bref_decode = _bref_id_to_dir_class()
    sem_id_to_tag = _semantic_id_to_tag()
    unknown_tag_to_key = unknown_family_tag_to_key()

    total_emissions = 0
    nonroad_emissions = 0
    nonroad_keys: dict[str, int] = defaultdict(int)
    tiles_with_nonroad: set[str] = set()
    total_groups = 0
    coverage_risk: list[tuple[str, tuple[int, int], str, list[str]]] = []
    halt_cell_report: list[str] = []

    for td in tile_dirs:
        cpath = td / "cells.parquet"
        if not cpath.exists():
            continue
        cells_rows = pq.ParquetFile(cpath).read().to_pylist()
        emitted = _emitted_brefs_by_cell(cells_rows, bref_decode, sem_id_to_tag, unknown_tag_to_key)
        by_cell_dir: dict[tuple[tuple[int, int], str], set[str]] = defaultdict(set)
        for cell, items in emitted.items():
            for direction, _cls, fkey in items:
                total_emissions += 1
                by_cell_dir[(cell, direction)].add(fkey)
                if fkey != _ROAD_KEY:
                    nonroad_emissions += 1
                    nonroad_keys[fkey] += 1
                    tiles_with_nonroad.add(td.name)
                # Capture the exact halt cell for confirmation.
                if td.name == "tile=EPSG3414_i10_j14" and cell == (4, 6):
                    halt_cell_report.append(f"{td.name} cell (4,6) dir {direction}: key={fkey!r}")
        for (cell, direction), keys in by_cell_dir.items():
            total_groups += 1
            if _ROAD_KEY not in keys:
                coverage_risk.append((td.name, cell, direction, sorted(keys)))

    print(f"TILES={len(tile_dirs)}")
    print(f"TOTAL_BREF_EMISSIONS={total_emissions}")
    print(f"NONROAD_EMISSIONS={nonroad_emissions}")
    print(f"TILES_WITH_NONROAD_EMISSION={len(tiles_with_nonroad)}")
    print(f"NONROAD_KEY_BREAKDOWN={dict(sorted(nonroad_keys.items(), key=lambda kv: -kv[1]))}")
    print(f"TOTAL_CELL_DIR_BREF_GROUPS={total_groups}")
    print(f"COVERAGE_RISK_GROUPS_NO_ROAD_EMITTER={len(coverage_risk)}")
    for tile, cell, direction, keys in coverage_risk[:25]:
        print(f"  RISK: {tile} cell {cell} dir {direction}: keys={keys}")
    print("--- halt-cell confirmation (i10_j14 cell (4,6)) ---")
    for line in halt_cell_report:
        print(f"  {line}")
    print("COVERAGE_SAFE=" + ("YES" if not coverage_risk else "NO_SEE_RISK_GROUPS"))
    print("END_DRILL")
    return 0 if not coverage_risk else 1


if __name__ == "__main__":
    raise SystemExit(main())

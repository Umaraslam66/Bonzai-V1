"""sub-G T11: coverage-leg non-vacuity confirmation + motorway/multi-part presence.

The sub-F cross-tile validator PASSED on all 494 Singapore tiles, but the
coverage leg had NEVER executed before (the non-road leg always fired first in
earlier runs). "Did not raise" != "ran non-vacuously". This drill mirrors
_check_coverage's EXACT loop (reusing the validator's own helpers + DIRECTION_ORDER
+ _EMITTING_CLASSES + _neighbour_cell, NOT a re-implementation) and counts, across
494 tiles:
  - active road edges (MAJOR/MINOR per sub-E contract): the universe;
  - evaluated edges (active AND a road feature in this cell or the neighbour):
    the edges the coverage leg actually asserts on;
  - covered (>=1 bref emitted on that edge) vs uncovered (would have raised).

A non-zero evaluated count with zero uncovered => the coverage pass is REAL.

Also reports whether the motorway and multi-part (MultiLineString road) regimes
actually appeared in the 494 tiles (vs being simply absent), from sub-C.

Read-only. Run: uv run python scripts/sub_g/t11_coverage_nonvacuity_drill.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from shapely import wkb

from cfm.data.sub_c.enums import FEATURE_CLASS, encode_enum
from cfm.data.sub_f.boundary_contract import load_boundary_contract
from cfm.data.sub_f.rotation import DIRECTION_ORDER
from cfm.data.sub_f.validator_cross_tile import (
    _EMITTING_CLASSES,
    _bref_id_to_dir_class,
    _emitted_brefs_by_cell,
    _neighbour_cell,
    _road_cells,
    _semantic_id_to_tag,
)

REPO = Path(__file__).resolve().parents[2]
SUB_F = REPO / "data" / "processed" / "sub_f" / "2026-04-15.0" / "singapore"
SUB_E = REPO / "data" / "processed" / "sub_e" / "2026-04-15.0" / "singapore"
SUB_C = REPO / "data" / "processed" / "sub_c" / "2026-04-15.0" / "singapore"


def _coverage_nonvacuity() -> tuple[int, int, int, list, int]:
    tiles = sorted(SUB_F.glob("tile=EPSG3414_*"))
    bref_decode = _bref_id_to_dir_class()
    sem_id_to_tag = _semantic_id_to_tag()
    active_road_edges = 0
    evaluated = 0
    covered = 0
    uncovered: list[tuple] = []
    tiles_with_eval = 0
    for td in tiles:
        cells_rows = pq.ParquetFile(td / "cells.parquet").read().to_pylist()
        emitted_by_cell = _emitted_brefs_by_cell(cells_rows, bref_decode, sem_id_to_tag)
        road_cells = _road_cells(cells_rows, sem_id_to_tag)
        contract = load_boundary_contract(SUB_E / td.name / "boundary_contract.parquet")
        tile_eval = 0
        for cell, cell_edges in contract.items():
            emitted_dirs = {d for (d, _c, _fk) in emitted_by_cell.get(cell, [])}
            for direction in DIRECTION_ORDER:
                edge_class = cell_edges.get(direction, "NONE")
                if edge_class not in _EMITTING_CLASSES:
                    continue
                active_road_edges += 1
                neighbour = _neighbour_cell(cell, direction)
                road_here = cell in road_cells
                road_neighbour = neighbour is not None and neighbour in road_cells
                if not (road_here or road_neighbour):
                    continue
                evaluated += 1
                tile_eval += 1
                if direction in emitted_dirs:
                    covered += 1
                else:
                    uncovered.append((td.name, cell, direction, edge_class))
        if tile_eval:
            tiles_with_eval += 1
    return active_road_edges, evaluated, covered, uncovered, tiles_with_eval


def _regime_presence() -> tuple[int, int, Counter]:
    """Count sub-C road features that are motorway (class_raw) or multi-part."""
    road_code = encode_enum(FEATURE_CLASS, "road")
    motorway = 0
    multipart_roads = 0
    geom_types: Counter = Counter()
    for td in sorted(SUB_C.glob("tile=EPSG3414_*")):
        fp = td / "features.parquet"
        if not fp.exists():
            continue
        tbl = pq.ParquetFile(fp).read()
        cols = {n: tbl.column(n).to_pylist() for n in tbl.column_names}
        for i in range(tbl.num_rows):
            if cols["feature_class"][i] != road_code:
                continue
            if cols.get("class_raw", [None] * tbl.num_rows)[i] == "motorway":
                motorway += 1
            geom = wkb.loads(cols["geometry"][i])
            geom_types[geom.geom_type] += 1
            if geom.geom_type == "MultiLineString":
                multipart_roads += 1
    return motorway, multipart_roads, geom_types


def main() -> int:
    active, evaluated, covered, uncovered, tiles_eval = _coverage_nonvacuity()
    print("=== COVERAGE LEG NON-VACUITY (mirrors _check_coverage exactly) ===")
    print(f"ACTIVE_ROAD_EDGES={active}")
    print(f"COVERAGE_EVALUATED_EDGES={evaluated}  (active AND road in cell/neighbour)")
    print(f"COVERED={covered}")
    print(f"UNCOVERED_WOULD_HAVE_RAISED={len(uncovered)}")
    print(f"TILES_WITH_EVALUATED_EDGES={tiles_eval}/494")
    for u in uncovered[:10]:
        print(f"  UNCOVERED: {u}")
    nonvacuous = evaluated > 0 and not uncovered
    print("COVERAGE_NONVACUOUS=" + ("YES" if nonvacuous else "NO"))

    motorway, multipart, geom_types = _regime_presence()
    print("=== REGIME PRESENCE (sub-C road features) ===")
    print(f"MOTORWAY_ROAD_FEATURES={motorway}")
    print(f"MULTIPART_ROAD_FEATURES={multipart}")
    print(f"ROAD_GEOM_TYPES={dict(geom_types)}")
    print("END_DRILL")
    return 0 if nonvacuous else 1


if __name__ == "__main__":
    raise SystemExit(main())

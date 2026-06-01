#!/usr/bin/env python3
"""T11 BP7 symmetry-halt root-cause diagnosis (read-only, reproducible).

CONCLUSION (2026-05-31): the BP7 cross-tile symmetry halt is caused by the sub-F
ENCODER (`encoder._direction_of_endpoint`) — and latently sub-G seam-2
(`seam_contract_tokens._endpoint_edge`) — labelling cell faces by GEOGRAPHIC
compass (cell-local y=250 -> "N"). The BP7 direction AUTHORITY is the LOCKED
`configs/sub_f/boundary_reference_vocab.yaml`, whose `source_references` defer
N/S/E/W to `sub_e.rotation.cell_to_edge_ids`, which defines a cell's NORTH edge
as the one shared with `(i, j-1)` = the LOW-y edge. So the encoder looks up the
boundary contract on the OPPOSITE j-edge, dropping/mislabelling N/S brefs.
`cell_to_edge_ids` and `_neighbour_cell` are CONSISTENT with each other (56/56);
the encoder + seam-2 are the outliers. 100% of the 2728 symmetry failures are on
the N/S axis; swapping the ENCODER's N/S drives them to 0 (see `rederive_*`).
Fix lives in sub-F (encoder) + sub-G (seam-2), NOT sub-E.

This script reproduces three facts, all to stdout (run from repo root):
  (1) axis split of the symmetry failures in the SHIPPED cells.parquet,
  (2) re-derivation under the current vs N/S-reconciled convention (0 after),
  (3) the canonical worked example (road e7be7863, tile i10_j10).

It supersedes the earlier t11_f2_drill.py, whose nearest-vertex selection was
unreliable (removed 2026-05-31, see feedback_tool_output_trustworthiness_layer).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from shapely.wkb import loads as wkb_loads

sys.path.insert(0, "src")

from cfm.data.sub_f.boundary_contract import load_boundary_contract, resolve_bref_tag
from cfm.data.sub_f.rotation import cell_edge_directions
from cfm.data.sub_f.validator_cross_tile import (
    _OPPOSITE_DIRECTION,
    _bref_id_to_dir_class,
    _emitted_brefs_by_cell,
    _neighbour_cell,
    _semantic_id_to_tag,
)

R, G = "2026-04-15.0", "singapore"
SUBF = Path(f"data/processed/sub_f/{R}/{G}")
SUBC = Path(f"data/processed/sub_c/{R}/{G}")
SUBE = Path(f"data/processed/sub_e/{R}/{G}")
EPS, EXT = 1e-6, 250.0


def axis_split_shipped() -> dict[str, int]:
    """Fact (1): axis split of symmetry failures in the SHIPPED cells.parquet."""
    bd, sd = _bref_id_to_dir_class(), _semantic_id_to_tag()
    axis: dict[str, int] = defaultdict(int)
    for cells_path in sorted(SUBF.glob("tile=*/cells.parquet")):
        cells = pq.ParquetFile(cells_path).read().to_pylist()
        by_cd: dict[tuple, set[str]] = defaultdict(set)
        for cell, items in _emitted_brefs_by_cell(cells, bd, sd).items():
            for d, cls, _fk in items:
                by_cd[(cell, d)].add(cls)
        for (cell, d), classes in by_cd.items():
            nb = _neighbour_cell(cell, d)
            if nb is None:
                continue
            if classes != by_cd.get((nb, _OPPOSITE_DIRECTION[d]), set()):
                axis["NS" if d in ("N", "S") else "EW"] += 1
    return dict(axis)


def _dir_geographic(x: float, y: float) -> str | None:
    if abs(x) <= EPS:
        return "W"
    if abs(x - EXT) <= EPS:
        return "E"
    if abs(y) <= EPS:
        return "S"
    if abs(y - EXT) <= EPS:
        return "N"
    return None


def _dir_fixed(x: float, y: float) -> str | None:
    """The PROPOSED encoder fix: N/S swapped so cell-local y=0 -> "N", y=250 -> "S",

    matching `cell_to_edge_ids` (north = the (i, j-1) / low-y edge). Re-deriving
    with this drives symmetry failures to 0 — i.e. the fix belongs in the encoder.
    """
    if abs(x) <= EPS:
        return "W"
    if abs(x - EXT) <= EPS:
        return "E"
    if abs(y) <= EPS:
        return "N"
    if abs(y - EXT) <= EPS:
        return "S"
    return None


def _line_parts(geom):
    if geom.geom_type == "LineString":
        return [list(geom.coords)]
    if geom.geom_type == "MultiLineString":
        return [list(p.coords) for p in geom.geoms]
    return []


def _emit(dirfn, feats, contract) -> dict[tuple, set[str]]:
    by: dict[tuple, set[str]] = defaultdict(set)
    for r in feats:
        if int(r["feature_class"]) != 0:
            continue
        cell = (int(r["cell_i"]), int(r["cell_j"]))
        ce = contract.get(cell, {})
        for coords in _line_parts(wkb_loads(bytes(r["geometry"]))):
            if len(coords) < 2:
                continue
            for x, y in (coords[0], coords[-1]):
                d = dirfn(x, y)
                if d and ce.get(d) and resolve_bref_tag(d, ce[d]):
                    by[(cell, d)].add(ce[d])
    return by


def _fails(by) -> dict[str, int]:
    axis: dict[str, int] = defaultdict(int)
    for (cell, d), classes in by.items():
        nb = _neighbour_cell(cell, d)
        if nb is None:
            continue
        if classes != by.get((nb, _OPPOSITE_DIRECTION[d]), set()):
            axis["NS" if d in ("N", "S") else "EW"] += 1
    return dict(axis)


def rederive_current_vs_reconciled() -> tuple[dict, dict]:
    """Fact (2): re-derive from sub-C geometry, current vs N/S-reconciled."""
    cur: dict[str, int] = defaultdict(int)
    rec: dict[str, int] = defaultdict(int)
    for cells_path in sorted(SUBF.glob("tile=*/cells.parquet")):
        tile = cells_path.parent.name
        feats = pq.ParquetFile(SUBC / tile / "features.parquet").read().to_pylist()
        contract = load_boundary_contract(SUBE / tile / "boundary_contract.parquet")
        for k, v in _fails(_emit(_dir_geographic, feats, contract)).items():
            cur[k] += v
        for k, v in _fails(_emit(_dir_fixed, feats, contract)).items():
            rec[k] += v
    return dict(cur), dict(rec)


def worked_example() -> None:
    """Fact (3): road e7be7863 on edge (4,2,1), tile i10_j10 — geometry symmetric."""
    t = "tile=EPSG3414_i10_j10"
    bd = _bref_id_to_dir_class()
    cells = {
        (int(r["cell_i"]), int(r["cell_j"])): list(r["token_sequence"])
        for r in pq.ParquetFile(SUBF / t / "cells.parquet").read().to_pylist()
    }
    ct = load_boundary_contract(SUBE / t / "boundary_contract.parquet")
    for c in [(4, 2), (4, 3)]:
        brefs = [bd[tok] for tok in cells[c] if tok in bd]
        ed = {d: cell_edge_directions(*c)[d][:3] for d in "NESW"}
        print(f"  cell {c}: emitted={brefs}")
        print(f"           contract={ct[c]}  cell_to_edge_ids={ed}")
    print("  road e7be7863 reaches the shared edge (4,2,1) at cell-local y=250 in")
    print("  (4,2) and y=0 in (4,3): geometry is SYMMETRIC. (4,2) dropped the bref")
    print("  because cell_to_edge_ids names its y=250 edge 'S' but the encoder calls")
    print("  a y=250 endpoint 'N' and looks up contract['N']=(4,1,1)=NONE.")


def main() -> int:
    print("=== (1) axis split of symmetry failures in SHIPPED cells.parquet ===")
    print("   ", axis_split_shipped())
    print("=== (2) re-derive: current convention vs N/S-reconciled ===")
    cur, rec = rederive_current_vs_reconciled()
    print("    current      :", cur)
    print("    encoder-fixed:", rec, "  (0 => swapping the encoder N/S resolves it)")
    print("=== (3) worked example road e7be7863 @ tile i10_j10 ===")
    worked_example()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

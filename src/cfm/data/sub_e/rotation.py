"""Per-cell to per-edge rotation for sub-E.

For each cell `(cell_i, cell_j) ∈ [0, 8) x [0, 8)` the four boundary slots
N/E/S/W map to canonical `edge_id = (lower_cell_i, lower_cell_j, axis)` tuples
following sub-C's AXIS enum (0=x, 1=y). External slots (at tile boundary) are
tagged with EdgeKind.EXTERNAL; sub-E writes one row per external slot with
scope_marker driven by sub-D's macro_core, not a derivation here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

GRID_SIZE: Final[int] = 8

# Axis encoding matches cfm.data.sub_c.enums.AXIS = {0: "x", 1: "y"}.
AXIS_X: Final[int] = 0
AXIS_Y: Final[int] = 1


class EdgeKind(StrEnum):
    INTERNAL = "internal_edge"
    EXTERNAL = "external_edge"


EdgeIdTuple = tuple[int, int, int, EdgeKind]
"""(lower_cell_i, lower_cell_j, axis, kind)."""


@dataclass(frozen=True)
class CellEdgeIds:
    """Four edge_ids for one cell, ordered N/S/W/E for stable iteration."""

    north: EdgeIdTuple
    south: EdgeIdTuple
    west: EdgeIdTuple
    east: EdgeIdTuple


def cell_to_edge_ids(cell_i: int, cell_j: int) -> CellEdgeIds:
    """Map a cell to its four boundary slot edge_ids.

    Raises ValueError if `cell_i` or `cell_j` is outside [0, 8).
    """
    if not (0 <= cell_i < GRID_SIZE) or not (0 <= cell_j < GRID_SIZE):
        raise ValueError(f"cell ({cell_i}, {cell_j}) outside [0, {GRID_SIZE})^2")

    north_kind = EdgeKind.EXTERNAL if cell_j == 0 else EdgeKind.INTERNAL
    north_lower_j = cell_j if cell_j == 0 else cell_j - 1
    north = (cell_i, north_lower_j, AXIS_X, north_kind)

    south_kind = EdgeKind.EXTERNAL if cell_j == GRID_SIZE - 1 else EdgeKind.INTERNAL
    south = (cell_i, cell_j, AXIS_X, south_kind)

    west_kind = EdgeKind.EXTERNAL if cell_i == 0 else EdgeKind.INTERNAL
    west_lower_i = cell_i if cell_i == 0 else cell_i - 1
    west = (west_lower_i, cell_j, AXIS_Y, west_kind)

    east_kind = EdgeKind.EXTERNAL if cell_i == GRID_SIZE - 1 else EdgeKind.INTERNAL
    east = (cell_i, cell_j, AXIS_Y, east_kind)

    return CellEdgeIds(north=north, south=south, west=west, east=east)

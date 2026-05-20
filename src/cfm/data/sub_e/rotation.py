"""Per-cell to per-edge rotation for sub-E.

For each cell ``(cell_i, cell_j) ∈ [0, 8) × [0, 8)`` the four cell faces
N/E/S/W map to canonical ``edge_id = (lower_cell_i, lower_cell_j, axis)``
tuples following sub-C's AXIS enum (0=x, 1=y). External slots (at tile
boundary) are tagged with ``EdgeKind.EXTERNAL``; sub-E writes one row per
external slot with scope_marker driven by sub-D's macro_core, not a
derivation here.

**Convention (authoritative source: sub-D ``lattice.py:11-14`` docstring +
sub-C ``enums.py:23`` AXIS enum):**

- ``axis=0`` (x): face between i-neighbor cells ``(lower_i, lower_j)`` and
  ``(lower_i + 1, lower_j)``. This is the **west/east** face of a cell.
- ``axis=1`` (y): face between j-neighbor cells ``(lower_i, lower_j)`` and
  ``(lower_i, lower_j + 1)``. This is the **north/south** face of a cell.

External edge addressing uses sub-D's **off-grid neighbour** convention:
``lower_cell_i = -1`` denotes the off-grid neighbour above row 0 (west
boundary); ``lower_cell_i = 7`` denotes a real cell whose lower_i + 1 = 8
is off-grid (east boundary). Same for ``lower_cell_j`` on the y axis.

**Why this matters (lesson from Task 14 real-data integration, 2026-05-20):**
an earlier draft of this function had the axes swapped (north/south using
AXIS_X=0 instead of AXIS_Y=1, west/east using AXIS_Y=1 instead of AXIS_X=0)
AND used an in-grid pinning convention (lower_i = 0 for west, lower_i = 7
for east). Every synthetic sub-E fixture from Tasks 6–13 consulted this
function to build expected values, so the validator and the fixture agreed
self-consistently and 18+ verify-before-asserting catches missed the
defect. First contact with real cached sub-D Singapore output (Task 14)
surfaced it because sub-D's macro_core uses the correct convention.
``tests/data/sub_e/test_rotation.py::test_external_set_matches_sub_d_hand_enumeration``
is the sixth-gate guard: it cross-references this function's output
against sub-D's docstring as ground truth, without using this function
in the assertion logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

GRID_SIZE: Final[int] = 8

# Axis encoding matches cfm.data.sub_c.enums.AXIS = {0: "x", 1: "y"}.
# axis=0 (x) = i-neighbor face (west/east); axis=1 (y) = j-neighbor face (north/south).
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
    """Map a cell to its four cell-face edge_ids.

    Raises ValueError if ``cell_i`` or ``cell_j`` is outside ``[0, 8)``.

    Per sub-D's convention (lattice.py:11-14):
    - North face (j-neighbor): shared with cell ``(cell_i, cell_j - 1)``;
      axis=1 (y). External when ``cell_j == 0``; off-grid neighbour at
      ``lower_j = -1``.
    - South face (j-neighbor): shared with cell ``(cell_i, cell_j + 1)``;
      axis=1 (y). This cell is the LOWER index of the pair, so
      ``lower_j = cell_j``. External when ``cell_j == 7``; off-grid
      neighbour at ``j = 8``.
    - West face (i-neighbor): shared with cell ``(cell_i - 1, cell_j)``;
      axis=0 (x). External when ``cell_i == 0``; off-grid neighbour at
      ``lower_i = -1``.
    - East face (i-neighbor): shared with cell ``(cell_i + 1, cell_j)``;
      axis=0 (x). This cell is the LOWER index of the pair, so
      ``lower_i = cell_i``. External when ``cell_i == 7``; off-grid
      neighbour at ``i = 8``.
    """
    if not (0 <= cell_i < GRID_SIZE) or not (0 <= cell_j < GRID_SIZE):
        raise ValueError(f"cell ({cell_i}, {cell_j}) outside [0, {GRID_SIZE})^2")

    # North face: j-neighbor → axis=1 (y). Off-grid lower_j = -1 when external.
    north_kind = EdgeKind.EXTERNAL if cell_j == 0 else EdgeKind.INTERNAL
    north_lower_j = -1 if cell_j == 0 else cell_j - 1
    north = (cell_i, north_lower_j, AXIS_Y, north_kind)

    # South face: j-neighbor → axis=1. cell_j IS the lower index in the pair
    # (cell_j+1 is the other cell, off-grid if cell_j == 7).
    south_kind = EdgeKind.EXTERNAL if cell_j == GRID_SIZE - 1 else EdgeKind.INTERNAL
    south = (cell_i, cell_j, AXIS_Y, south_kind)

    # West face: i-neighbor → axis=0 (x). Off-grid lower_i = -1 when external.
    west_kind = EdgeKind.EXTERNAL if cell_i == 0 else EdgeKind.INTERNAL
    west_lower_i = -1 if cell_i == 0 else cell_i - 1
    west = (west_lower_i, cell_j, AXIS_X, west_kind)

    # East face: i-neighbor → axis=0. cell_i IS the lower index in the pair
    # (cell_i+1 is the other cell, off-grid if cell_i == 7).
    east_kind = EdgeKind.EXTERNAL if cell_i == GRID_SIZE - 1 else EdgeKind.INTERNAL
    east = (cell_i, cell_j, AXIS_X, east_kind)

    return CellEdgeIds(north=north, south=south, west=west, east=east)

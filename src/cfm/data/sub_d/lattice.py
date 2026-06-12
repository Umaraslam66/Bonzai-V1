"""Fixed 8x8 macro lattice utilities (sub-D spec section 5).

Three slot families share the same 8x8 cell grid:

- 64 cell slots, row-major: ``slot_index = cell_i * 8 + cell_j``.
- 112 internal edge slots between adjacent in-grid cells. Sub-C axis
  convention: axis=0 (x) edge sits between ``(lower_i, lower_j)`` and
  ``(lower_i + 1, lower_j)``; axis=1 (y) sits between ``(lower_i, lower_j)``
  and ``(lower_i, lower_j + 1)``. 56 axis-0 edges are emitted first
  (slots 0-55, row-major over ``lower_i in [0, 6], lower_j in [0, 7]``) then
  56 axis-1 edges (slots 56-111, row-major over
  ``lower_i in [0, 7], lower_j in [0, 6]``).
- 32 external edge slots on the perimeter. ``lower_cell_i = -1`` denotes the
  off-grid neighbour above row 0; ``lower_cell_i = 7`` denotes a real cell
  whose ``lower_i + 1 = 8`` is off-grid. Same convention for ``lower_cell_j``
  on the y axis. Order: axis=0 i=-1 (8), axis=0 i=7 (8), axis=1 j=-1 (8),
  axis=1 j=7 (8).
"""

from __future__ import annotations

from dataclasses import dataclass

from cfm.data.sub_d.enums import Scope, Side

CELL_GRID_SIZE: int = 8
CELL_SLOT_COUNT: int = 64
INTERNAL_EDGE_SLOT_COUNT: int = 112
EXTERNAL_EDGE_SLOT_COUNT: int = 32


@dataclass(frozen=True)
class CellSlot:
    """One slot in the 64-cell macro lattice."""

    slot_index: int
    cell_i: int
    cell_j: int


@dataclass(frozen=True)
class EdgeSlot:
    """One slot in the internal or external edge lattice.

    ``axis`` uses sub-C convention: ``0=x``, ``1=y``. For internal edges,
    both endpoint cells are in-grid. For external edges, exactly one of the
    endpoint cells is off-grid; the in-grid (interior) cell is the one with
    coordinates inside ``[0, 7]``.
    """

    slot_index: int
    lower_cell_i: int
    lower_cell_j: int
    axis: int

    @property
    def side(self) -> Side:
        """Perimeter side for an external edge slot.

        Derived from the canonical ``(lower_cell_i, lower_cell_j, axis)``
        fields so the external-edge convention has a single source of truth.
        Raises ``ValueError`` when called on an internal edge.
        """
        if self.axis == 0:
            if self.lower_cell_i == -1:
                return Side.TOP
            if self.lower_cell_i == 7:
                return Side.BOTTOM
        elif self.axis == 1:
            if self.lower_cell_j == -1:
                return Side.LEFT
            if self.lower_cell_j == 7:
                return Side.RIGHT
        raise ValueError(
            "EdgeSlot.side is only defined for external edges; "
            f"got (lower_cell_i={self.lower_cell_i}, "
            f"lower_cell_j={self.lower_cell_j}, axis={self.axis})"
        )


def iter_cell_slots() -> list[CellSlot]:
    """Return all 64 cell slots in row-major order."""
    return [
        CellSlot(slot_index=cell_i * CELL_GRID_SIZE + cell_j, cell_i=cell_i, cell_j=cell_j)
        for cell_i in range(CELL_GRID_SIZE)
        for cell_j in range(CELL_GRID_SIZE)
    ]


def iter_internal_edge_slots() -> list[EdgeSlot]:
    """Return all 112 internal edge slots.

    Axis 0 first (56 slots, indices 0-55), axis 1 second (56 slots, indices
    56-111). Within each axis, lower endpoints iterate row-major.
    """
    slots: list[EdgeSlot] = []
    # axis=0: edge between (lower_i, lower_j) and (lower_i+1, lower_j).
    for lower_i in range(CELL_GRID_SIZE - 1):
        for lower_j in range(CELL_GRID_SIZE):
            slots.append(
                EdgeSlot(
                    slot_index=len(slots),
                    lower_cell_i=lower_i,
                    lower_cell_j=lower_j,
                    axis=0,
                )
            )
    # axis=1: edge between (lower_i, lower_j) and (lower_i, lower_j+1).
    for lower_i in range(CELL_GRID_SIZE):
        for lower_j in range(CELL_GRID_SIZE - 1):
            slots.append(
                EdgeSlot(
                    slot_index=len(slots),
                    lower_cell_i=lower_i,
                    lower_cell_j=lower_j,
                    axis=1,
                )
            )
    return slots


def iter_external_edge_slots() -> list[EdgeSlot]:
    """Return all 32 external (perimeter) edge slots.

    Order: axis=0 with lower_i=-1 (8 slots, interior at row 0), axis=0 with
    lower_i=7 (8 slots, interior at row 7), axis=1 with lower_j=-1 (8 slots,
    interior at col 0), axis=1 with lower_j=7 (8 slots, interior at col 7).
    """
    slots: list[EdgeSlot] = []
    # axis=0, lower_i=-1: interior at (0, lower_j).
    for lower_j in range(CELL_GRID_SIZE):
        slots.append(EdgeSlot(slot_index=len(slots), lower_cell_i=-1, lower_cell_j=lower_j, axis=0))
    # axis=0, lower_i=7: interior at (7, lower_j); off-grid neighbour at i=8.
    for lower_j in range(CELL_GRID_SIZE):
        slots.append(EdgeSlot(slot_index=len(slots), lower_cell_i=7, lower_cell_j=lower_j, axis=0))
    # axis=1, lower_j=-1: interior at (lower_i, 0).
    for lower_i in range(CELL_GRID_SIZE):
        slots.append(EdgeSlot(slot_index=len(slots), lower_cell_i=lower_i, lower_cell_j=-1, axis=1))
    # axis=1, lower_j=7: interior at (lower_i, 7); off-grid neighbour at j=8.
    for lower_i in range(CELL_GRID_SIZE):
        slots.append(EdgeSlot(slot_index=len(slots), lower_cell_i=lower_i, lower_cell_j=7, axis=1))
    return slots


def derive_internal_edge_scope(lower_active: bool, upper_active: bool) -> Scope:
    """Scope marker for an internal edge given the two endpoint cell scopes."""
    if lower_active and upper_active:
        return Scope.ACTIVE
    if not lower_active and not upper_active:
        return Scope.FULLY_MASKED
    return Scope.SCOPE_BOUNDARY


def derive_external_edge_scope(interior_active: bool) -> Scope:
    """Scope marker for an external edge given the interior cell's scope.

    ``EXTERNAL_DEFERRED`` is reserved for external edges adjacent to an
    active interior cell — sub-E will own the real tile-to-tile generation
    semantics. If the interior cell is not in scope, the external edge is
    fully masked.
    """
    return Scope.EXTERNAL_DEFERRED if interior_active else Scope.FULLY_MASKED

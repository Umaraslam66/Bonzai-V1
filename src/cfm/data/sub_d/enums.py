"""Sub-D integer enums (spec section 5, section 11).

``SlotKind`` distinguishes which of the three macro lattices a slot belongs to
(cell, internal edge, external edge). ``Scope`` is the lattice-slot
availability marker: cells use ``ACTIVE``/``FULLY_MASKED``; internal edges
use those plus ``SCOPE_BOUNDARY``; external edges use ``EXTERNAL_DEFERRED``
or ``FULLY_MASKED``. Token integer codes are locked here so consumers can
treat them as schema-stable.
"""

from __future__ import annotations

from enum import IntEnum


class SlotKind(IntEnum):
    CELL = 0
    INTERNAL_EDGE = 1
    EXTERNAL_EDGE = 2


class Scope(IntEnum):
    ACTIVE = 0
    FULLY_MASKED = 1
    SCOPE_BOUNDARY = 2
    EXTERNAL_DEFERRED = 3


class Side(IntEnum):
    """Which perimeter of the 8x8 cell grid an external edge sits on.

    Derived from ``EdgeSlot.(lower_cell_i, lower_cell_j, axis)`` per the
    documented external-edge convention:

    - ``axis=0, lower_i=-1`` -> ``TOP``
    - ``axis=0, lower_i=7``  -> ``BOTTOM``
    - ``axis=1, lower_j=-1`` -> ``LEFT``
    - ``axis=1, lower_j=7``  -> ``RIGHT``
    """

    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3

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
    """Macro-lattice slot category.

    ``derivation_evidence.parquet`` (spec section 11.3) uses ``TILE`` for
    tile-level metrics (e.g. tile_population_density). ``macro_core.parquet``
    only emits rows with ``CELL``, ``INTERNAL_EDGE``, or ``EXTERNAL_EDGE``.
    """

    CELL = 0
    INTERNAL_EDGE = 1
    EXTERNAL_EDGE = 2
    TILE = 3


class MetricNamespace(IntEnum):
    """Metric namespace for derivation_evidence rows (spec section 11.3)."""

    ZONING = 0
    CELL_DENSITY = 1
    TILE_POPULATION_DENSITY = 2
    ROAD_SKELETON = 3


class FeatureClass(IntEnum):
    """Sub-C ``feature_class`` encoding.

    Mirrored in sub-D so this package never has to import from
    ``cfm.data.sub_c.*``. The integer codes are part of sub-C's on-disk
    contract; if sub-C ever changes them, both modules bump together.
    """

    ROAD = 0
    BUILDING = 1
    POI = 2
    BASE = 3


class Axis(IntEnum):
    """Sub-C ``axis`` encoding for crossings/edge slots."""

    X = 0
    Y = 1


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

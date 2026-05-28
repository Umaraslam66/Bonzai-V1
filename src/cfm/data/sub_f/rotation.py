"""Sub-F boundary-reference direction wrapper over sub-E cell edge ids."""

from __future__ import annotations

from typing import Final

from cfm.data.sub_e.derivation import BoundaryClass, load_class_grouping_map
from cfm.data.sub_e.rotation import CellEdgeIds, EdgeIdTuple, cell_to_edge_ids

DIRECTION_ORDER: Final[tuple[str, ...]] = ("N", "E", "S", "W")

SUB_F_BP7_HIGHWAY_OVERRIDES: Final[dict[str, BoundaryClass]] = {
    # Cascade #9: sub-F BP7 models drivable-network continuity for AV routing.
    # Values absent from sub-E grouping must be explicit, including deliberate
    # non-emitting NONE, so no locked BP1 highway value falls to NONE by omission.
    "*": BoundaryClass.NONE,
    "bridleway": BoundaryClass.NONE,
    "busway": BoundaryClass.NONE,
    "living_street": BoundaryClass.MINOR_ROAD,
    "motorway": BoundaryClass.MAJOR_ROAD,
    "motorway_link": BoundaryClass.NONE,
    "path": BoundaryClass.NONE,
    "pedestrian": BoundaryClass.NONE,
    "primary_link": BoundaryClass.NONE,
    "road": BoundaryClass.NONE,
    "secondary_link": BoundaryClass.NONE,
    "subway": BoundaryClass.NONE,
    "tertiary_link": BoundaryClass.NONE,
    "track": BoundaryClass.NONE,
    "trunk_link": BoundaryClass.NONE,
}


def cell_edge_directions(cell_i: int, cell_j: int) -> dict[str, EdgeIdTuple]:
    """Return sub-E edge ids keyed in BP7 boundary-reference vocab order."""
    edge_ids: CellEdgeIds = cell_to_edge_ids(cell_i, cell_j)
    return {
        "N": edge_ids.north,
        "E": edge_ids.east,
        "S": edge_ids.south,
        "W": edge_ids.west,
    }


def resolve_highway_boundary_class(highway_value: str) -> BoundaryClass:
    """Resolve a locked BP1 highway value to sub-F BP7's drivable class.

    sub-E's grouping is consumed first for values it already classifies.
    sub-F-local overrides cover locked BP1 values that sub-E leaves absent,
    including deliberate non-emitting NONE decisions for non-drivable classes.
    """
    grouping = load_class_grouping_map()
    if highway_value in grouping:
        return grouping[highway_value]
    if highway_value in SUB_F_BP7_HIGHWAY_OVERRIDES:
        return SUB_F_BP7_HIGHWAY_OVERRIDES[highway_value]
    raise KeyError(f"no BP7 boundary class mapping for highway={highway_value!r}")

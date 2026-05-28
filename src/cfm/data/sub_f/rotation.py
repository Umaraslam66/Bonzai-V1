"""Sub-F boundary-reference direction wrapper over sub-E cell edge ids."""

from __future__ import annotations

from typing import Final

from cfm.data.sub_e.rotation import CellEdgeIds, EdgeIdTuple, cell_to_edge_ids

DIRECTION_ORDER: Final[tuple[str, ...]] = ("N", "E", "S", "W")


def cell_edge_directions(cell_i: int, cell_j: int) -> dict[str, EdgeIdTuple]:
    """Return sub-E edge ids keyed in BP7 boundary-reference vocab order."""
    edge_ids: CellEdgeIds = cell_to_edge_ids(cell_i, cell_j)
    return {
        "N": edge_ids.north,
        "E": edge_ids.east,
        "S": edge_ids.south,
        "W": edge_ids.west,
    }

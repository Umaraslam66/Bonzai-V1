"""External-source-of-truth gate: the sub-F encoder's endpoint→direction
classification MUST agree with ``cell_to_edge_ids`` — the BP7 direction
authority that the LOCKED ``configs/sub_f/boundary_reference_vocab.yaml``
defers to (`source_references`) — NOT with the encoder's own past output.

The absence of exactly this test is why the geographic-vs-lattice N/S
convention bug (sub-G T11, 2026-05-31) lived undetected: the encoder, the
decoder, and the fixtures all shared the same wrong "+y is north" geographic
assumption, so every internal cross-check passed vacuously. This test traces
direction to the authority via lattice geometry, so no future encoder change
can silently re-introduce the geographic convention.

See `reports/2026-05-31-sub-G-T11-symmetry-root-cause.md` and
`feedback_independence_misses_shared_assumptions`.
"""

from __future__ import annotations

import pytest
from shapely.geometry import LineString

from cfm.data.sub_e.rotation import AXIS_X, AXIS_Y, cell_to_edge_ids
from cfm.data.sub_f.encoder import _classify_feature_for_bref

_EXT = 250.0  # cell extent (m)
_MID = 125.0  # an interior coordinate, on no edge


def _authority_direction(cell_i: int, cell_j: int, physical_edge_id: tuple[int, int, int]) -> str:
    """The N/S/E/W name `cell_to_edge_ids` assigns to a physical edge_id.

    `cell_to_edge_ids` is the authority (the vocab defers to it). We map each of
    its four edge_ids back to its field name; the caller supplies the physical
    edge_id derived purely from lattice geometry (below), never from the encoder.
    """
    e = cell_to_edge_ids(cell_i, cell_j)
    name_by_eid = {
        e.north[:3]: "N",
        e.south[:3]: "S",
        e.west[:3]: "W",
        e.east[:3]: "E",
    }
    return name_by_eid[physical_edge_id]


def _edge_cases(cell_i: int, cell_j: int) -> list[tuple[tuple[float, float], tuple[int, int, int]]]:
    """(cell-local endpoint, physical edge_id) for the 4 faces.

    Physical placement is lattice geometry, independent of any N/S naming: the
    edge at cell-local y=0 lies between (i, j-1) and (i, j) → edge_id
    (i, j-1, AXIS_Y); cell-local y=extent lies between (i, j) and (i, j+1) →
    (i, j, AXIS_Y); cell-local x=0 → (i-1, j, AXIS_X); x=extent → (i, j, AXIS_X).
    """
    return [
        ((_MID, 0.0), (cell_i, cell_j - 1, AXIS_Y)),  # bottom (low-y)
        ((_MID, _EXT), (cell_i, cell_j, AXIS_Y)),  # top (high-y)
        ((0.0, _MID), (cell_i - 1, cell_j, AXIS_X)),  # left (low-x)
        ((_EXT, _MID), (cell_i, cell_j, AXIS_X)),  # right (high-x)
    ]


@pytest.mark.parametrize("cell", [(4, 3), (2, 5), (6, 1)])
def test_encoder_endpoint_direction_matches_cell_to_edge_ids(cell: tuple[int, int]) -> None:
    """A road endpoint on a face emits the bref whose direction `cell_to_edge_ids`
    assigns to that physical face. Interior cells only (all four faces INTERNAL)."""
    cell_i, cell_j = cell
    for (x, y), physical_edge_id in _edge_cases(cell_i, cell_j):
        authority_dir = _authority_direction(cell_i, cell_j, physical_edge_id)
        cell_edges = {
            d: ("MAJOR_ROAD" if d == authority_dir else "NONE") for d in ("N", "E", "S", "W")
        }
        # endpoint under test is coords[0]; coords[-1] is interior (on no edge).
        geom = LineString([(x, y), (_MID, _MID)])
        inbound, outbound = _classify_feature_for_bref(geom, cell_edges)
        emitted = inbound or outbound
        assert emitted == f"<bref_{authority_dir}_MAJOR>", (
            f"cell {cell}: endpoint {(x, y)} lies on physical edge {physical_edge_id}, "
            f"which cell_to_edge_ids names {authority_dir!r}; encoder emitted {emitted!r}. "
            f"The encoder's endpoint→direction classification disagrees with the BP7 "
            f"direction authority (cell_to_edge_ids)."
        )

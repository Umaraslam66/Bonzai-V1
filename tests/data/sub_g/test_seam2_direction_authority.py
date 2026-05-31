"""External-source-of-truth gate for sub-G seam-2 `_endpoint_edge` — same lesson
as the sub-F encoder gate (`tests/data/sub_f/test_direction_authority.py`).

seam-2 independently reimplemented the SAME geographic "+y is north" assumption
as the sub-F encoder, so the transcription bijection could not catch the
convention bug (both sides agreed while both were wrong). This test pins
`_endpoint_edge`'s endpoint→direction mapping to the BP7 authority
(`cell_to_edge_ids`, which the locked vocab defers to), the third independent
source. See `reports/2026-05-31-sub-G-T11-symmetry-root-cause.md` and
`feedback_independence_misses_shared_assumptions`.
"""

from __future__ import annotations

import pytest

from cfm.data.sub_e.rotation import AXIS_X, AXIS_Y, cell_to_edge_ids
from cfm.data.sub_g.seam_contract_tokens import _endpoint_edge

_EXT = 250.0
_MID = 125.0


def _name_by_eid(cell_i: int, cell_j: int) -> dict[tuple[int, int, int], str]:
    e = cell_to_edge_ids(cell_i, cell_j)
    return {e.north[:3]: "N", e.south[:3]: "S", e.west[:3]: "W", e.east[:3]: "E"}


@pytest.mark.parametrize("cell", [(4, 3), (2, 5), (6, 1)])
def test_seam2_endpoint_edge_matches_cell_to_edge_ids(cell: tuple[int, int]) -> None:
    cell_i, cell_j = cell
    name_by_eid = _name_by_eid(cell_i, cell_j)
    cases = [
        ((_MID, 0.0), (cell_i, cell_j - 1, AXIS_Y)),  # bottom (low-y)
        ((_MID, _EXT), (cell_i, cell_j, AXIS_Y)),  # top (high-y)
        ((0.0, _MID), (cell_i - 1, cell_j, AXIS_X)),  # left (low-x)
        ((_EXT, _MID), (cell_i, cell_j, AXIS_X)),  # right (high-x)
    ]
    for (x, y), physical_edge_id in cases:
        authority_dir = name_by_eid[physical_edge_id]
        assert _endpoint_edge(x, y) == authority_dir, (
            f"cell {cell}: point {(x, y)} on physical edge {physical_edge_id} "
            f"(authority {authority_dir!r}) but _endpoint_edge returned "
            f"{_endpoint_edge(x, y)!r} — disagrees with cell_to_edge_ids."
        )

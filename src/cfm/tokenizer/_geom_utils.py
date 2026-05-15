from __future__ import annotations

BOUNDARY_TOL_M: float = 1e-3
"""Tolerance in metres for treating a coordinate as on a cell boundary.

Phase 0's fixture uses exact integer coords (0, 250); real Overture data
reprojected to a local metric frame will drift by sub-mm amounts. 1e-3 m
(1 mm) is far below the 1 m anchor grid so it cannot widen the contract,
yet absorbs all realistic reprojection noise.
"""


def on_cell_boundary(x: float, y: float, cell_size_m: float) -> bool:
    """True when (x, y) lies on any edge of the [0, cell_size_m]² square.

    Uses BOUNDARY_TOL_M tolerance so reprojection drift does not cause
    silent loss of EXIT markers.
    """
    return (
        abs(x) < BOUNDARY_TOL_M
        or abs(x - cell_size_m) < BOUNDARY_TOL_M
        or abs(y) < BOUNDARY_TOL_M
        or abs(y - cell_size_m) < BOUNDARY_TOL_M
    )

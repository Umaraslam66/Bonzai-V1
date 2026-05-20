from __future__ import annotations

from cfm.data.sub_e.rotation import (
    AXIS_X,
    AXIS_Y,
    GRID_SIZE,
    EdgeKind,
    cell_to_edge_ids,
)


def test_interior_cell_3_3_has_four_internal_edges() -> None:
    """Cell (3, 3): north shared with (3, 2), south with (3, 4), west with
    (2, 3), east with (4, 3). All four faces internal. axis=1 for N/S
    (j-neighbor), axis=0 for W/E (i-neighbor) per sub-D convention.
    """
    result = cell_to_edge_ids(cell_i=3, cell_j=3)
    assert result.north == (3, 2, AXIS_Y, EdgeKind.INTERNAL)
    assert result.south == (3, 3, AXIS_Y, EdgeKind.INTERNAL)
    assert result.west == (2, 3, AXIS_X, EdgeKind.INTERNAL)
    assert result.east == (3, 3, AXIS_X, EdgeKind.INTERNAL)


def test_edge_cell_0_3_has_west_external_three_internal() -> None:
    """Cell (0, 3): west face has off-grid neighbour at i=-1 (external)."""
    result = cell_to_edge_ids(cell_i=0, cell_j=3)
    assert result.north == (0, 2, AXIS_Y, EdgeKind.INTERNAL)
    assert result.south == (0, 3, AXIS_Y, EdgeKind.INTERNAL)
    assert result.west == (-1, 3, AXIS_X, EdgeKind.EXTERNAL)  # off-grid lower_i = -1
    assert result.east == (0, 3, AXIS_X, EdgeKind.INTERNAL)


def test_edge_cell_3_0_has_north_external_three_internal() -> None:
    """Cell (3, 0): north face has off-grid neighbour at j=-1 (external)."""
    result = cell_to_edge_ids(cell_i=3, cell_j=0)
    assert result.north == (3, -1, AXIS_Y, EdgeKind.EXTERNAL)  # off-grid lower_j = -1
    assert result.south == (3, 0, AXIS_Y, EdgeKind.INTERNAL)
    assert result.west == (2, 0, AXIS_X, EdgeKind.INTERNAL)
    assert result.east == (3, 0, AXIS_X, EdgeKind.INTERNAL)


def test_corner_cell_0_0_has_two_external_two_internal() -> None:
    """Cell (0, 0): NW corner. North and west faces external."""
    result = cell_to_edge_ids(cell_i=0, cell_j=0)
    assert result.north == (0, -1, AXIS_Y, EdgeKind.EXTERNAL)
    assert result.south == (0, 0, AXIS_Y, EdgeKind.INTERNAL)
    assert result.west == (-1, 0, AXIS_X, EdgeKind.EXTERNAL)
    assert result.east == (0, 0, AXIS_X, EdgeKind.INTERNAL)


def test_corner_cell_7_7_has_two_external_two_internal() -> None:
    """Cell (7, 7): SE corner. South and east faces external.
    Sub-D off-grid convention: lower_i = 7 means upper neighbour at i = 8
    is off-grid; same for lower_j = 7.
    """
    result = cell_to_edge_ids(cell_i=7, cell_j=7)
    assert result.north == (7, 6, AXIS_Y, EdgeKind.INTERNAL)
    assert result.south == (7, 7, AXIS_Y, EdgeKind.EXTERNAL)  # j=7 → off-grid at j=8
    assert result.west == (6, 7, AXIS_X, EdgeKind.INTERNAL)
    assert result.east == (7, 7, AXIS_X, EdgeKind.EXTERNAL)  # i=7 → off-grid at i=8


def test_full_lattice_counts_match_specification() -> None:
    """Aggregated check: 112 internal + 32 external across 64 cells x 4 slots."""
    internal: set[tuple[int, int, int]] = set()
    external_count = 0
    for cell_i in range(GRID_SIZE):
        for cell_j in range(GRID_SIZE):
            result = cell_to_edge_ids(cell_i, cell_j)
            for slot in (result.north, result.south, result.west, result.east):
                i, j, axis, kind = slot
                if kind is EdgeKind.INTERNAL:
                    internal.add((i, j, axis))
                else:
                    external_count += 1
    assert len(internal) == 112, "expected 112 unique internal edges"
    assert external_count == 32, "expected 32 external slot occurrences"


def test_external_set_matches_sub_d_hand_enumeration() -> None:
    """Sixth-gate cross-reference: rotation's external set matches sub-D's
    docstring as ground truth, without using rotation in the assertion logic.

    Expected values hand-enumerated from
    ``src/cfm/data/sub_d/lattice.py:130-149`` docstring:

    > axis=0 with lower_i=-1 (8 slots, interior at row 0), axis=0 with
    > lower_i=7 (8 slots, interior at row 7), axis=1 with lower_j=-1
    > (8 slots, interior at col 0), axis=1 with lower_j=7 (8 slots,
    > interior at col 7).

    This test exists because rotation.py was shipped with swapped axes
    and in-grid pinning that disagreed with sub-D's convention (Task 14
    real-data integration surfaced it 2026-05-20). Memory entry
    ``feedback_external_source_of_truth_gate.md`` documents the pattern:
    when introducing a new abstraction over an existing module, write
    a cross-reference test that hand-enumerates expected values from
    the upstream module's documentation.
    """
    # Hand-enumerated per sub-D lattice.py:130-149. Do NOT call
    # cell_to_edge_ids in the expected-value construction.
    expected_external: set[tuple[int, int, int]] = set()
    for lower_j in range(GRID_SIZE):
        expected_external.add((-1, lower_j, 0))  # west boundary, axis=0
    for lower_j in range(GRID_SIZE):
        expected_external.add((7, lower_j, 0))  # east boundary, axis=0 (i=8 off-grid)
    for lower_i in range(GRID_SIZE):
        expected_external.add((lower_i, -1, 1))  # north boundary, axis=1
    for lower_i in range(GRID_SIZE):
        expected_external.add((lower_i, 7, 1))  # south boundary, axis=1 (j=8 off-grid)
    assert len(expected_external) == 32, "hand-enumeration produces 32 unique tuples"

    # Now collect rotation's external set (the artifact under test).
    rotation_external: set[tuple[int, int, int]] = set()
    for cell_i in range(GRID_SIZE):
        for cell_j in range(GRID_SIZE):
            result = cell_to_edge_ids(cell_i, cell_j)
            for slot in (result.north, result.south, result.west, result.east):
                i, j, axis, kind = slot
                if kind is EdgeKind.EXTERNAL:
                    rotation_external.add((i, j, axis))

    assert rotation_external == expected_external, (
        f"rotation's external set disagrees with sub-D's docstring. "
        f"Only-in-rotation: {sorted(rotation_external - expected_external)}. "
        f"Only-in-sub-D: {sorted(expected_external - rotation_external)}."
    )


def test_internal_set_matches_sub_d_hand_enumeration() -> None:
    """Sixth-gate cross-reference for internal edges. Hand-enumerated from
    ``src/cfm/data/sub_d/lattice.py:6-10`` docstring:

    > axis=0 (x) edge sits between (lower_i, lower_j) and (lower_i + 1,
    > lower_j); axis=1 (y) sits between (lower_i, lower_j) and (lower_i,
    > lower_j + 1). 56 axis-0 edges are emitted first (slots 0-55, row-major
    > over lower_i in [0, 6], lower_j in [0, 7]) then 56 axis-1 edges
    > (slots 56-111, row-major over lower_i in [0, 7], lower_j in [0, 6]).

    Same discipline: hand-enumerate from upstream docs, do not derive
    from the new abstraction.
    """
    expected_internal: set[tuple[int, int, int]] = set()
    for lower_i in range(GRID_SIZE - 1):
        for lower_j in range(GRID_SIZE):
            expected_internal.add((lower_i, lower_j, 0))  # axis=0 internal
    for lower_i in range(GRID_SIZE):
        for lower_j in range(GRID_SIZE - 1):
            expected_internal.add((lower_i, lower_j, 1))  # axis=1 internal
    assert len(expected_internal) == 112, "hand-enumeration produces 112 unique tuples"

    rotation_internal: set[tuple[int, int, int]] = set()
    for cell_i in range(GRID_SIZE):
        for cell_j in range(GRID_SIZE):
            result = cell_to_edge_ids(cell_i, cell_j)
            for slot in (result.north, result.south, result.west, result.east):
                i, j, axis, kind = slot
                if kind is EdgeKind.INTERNAL:
                    rotation_internal.add((i, j, axis))

    assert rotation_internal == expected_internal, (
        f"rotation's internal set disagrees with sub-D's docstring. "
        f"Only-in-rotation: {sorted(rotation_internal - expected_internal)}. "
        f"Only-in-sub-D: {sorted(expected_internal - rotation_internal)}."
    )

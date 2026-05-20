from __future__ import annotations

from cfm.data.sub_e.rotation import (
    EdgeKind,
    cell_to_edge_ids,
)


def test_interior_cell_3_3_has_four_internal_edges() -> None:
    result = cell_to_edge_ids(cell_i=3, cell_j=3)
    assert result.north == (3, 2, 0, EdgeKind.INTERNAL)
    assert result.south == (3, 3, 0, EdgeKind.INTERNAL)
    assert result.west == (2, 3, 1, EdgeKind.INTERNAL)
    assert result.east == (3, 3, 1, EdgeKind.INTERNAL)


def test_edge_cell_0_3_has_west_external_three_internal() -> None:
    result = cell_to_edge_ids(cell_i=0, cell_j=3)
    assert result.north == (0, 2, 0, EdgeKind.INTERNAL)
    assert result.south == (0, 3, 0, EdgeKind.INTERNAL)
    assert result.west == (0, 3, 1, EdgeKind.EXTERNAL)  # i=0 → external
    assert result.east == (0, 3, 1, EdgeKind.INTERNAL)


def test_edge_cell_3_0_has_north_external_three_internal() -> None:
    result = cell_to_edge_ids(cell_i=3, cell_j=0)
    assert result.north == (3, 0, 0, EdgeKind.EXTERNAL)  # j=0 → external
    assert result.south == (3, 0, 0, EdgeKind.INTERNAL)
    assert result.west == (2, 0, 1, EdgeKind.INTERNAL)
    assert result.east == (3, 0, 1, EdgeKind.INTERNAL)


def test_corner_cell_0_0_has_two_external_two_internal() -> None:
    result = cell_to_edge_ids(cell_i=0, cell_j=0)
    assert result.north == (0, 0, 0, EdgeKind.EXTERNAL)
    assert result.south == (0, 0, 0, EdgeKind.INTERNAL)
    assert result.west == (0, 0, 1, EdgeKind.EXTERNAL)
    assert result.east == (0, 0, 1, EdgeKind.INTERNAL)


def test_corner_cell_7_7_has_two_external_two_internal() -> None:
    result = cell_to_edge_ids(cell_i=7, cell_j=7)
    assert result.north == (7, 6, 0, EdgeKind.INTERNAL)
    assert result.south == (7, 7, 0, EdgeKind.EXTERNAL)  # j=7 → external
    assert result.west == (6, 7, 1, EdgeKind.INTERNAL)
    assert result.east == (7, 7, 1, EdgeKind.EXTERNAL)  # i=7 → external


def test_full_lattice_counts_match_specification() -> None:
    """Aggregated check: 112 internal + 32 external across 64 cells x 4 slots."""
    internal: set[tuple[int, int, int]] = set()
    external_count = 0
    for cell_i in range(8):
        for cell_j in range(8):
            result = cell_to_edge_ids(cell_i, cell_j)
            for slot in (result.north, result.south, result.west, result.east):
                i, j, axis, kind = slot
                if kind is EdgeKind.INTERNAL:
                    internal.add((i, j, axis))
                else:
                    external_count += 1
    assert len(internal) == 112, "expected 112 unique internal edges"
    assert external_count == 32, "expected 32 external slot occurrences"

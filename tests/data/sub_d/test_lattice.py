"""Tests for the sub-D fixed macro lattice (spec section 5).

The 8x8 cell grid produces 64 cell slots, 112 internal-edge slots, and 32
external-edge slots. Sub-C axis convention is preserved: axis=0 (x) edge sits
between cells (lower_i, lower_j) and (lower_i+1, lower_j); axis=1 (y) sits
between (lower_i, lower_j) and (lower_i, lower_j+1).
"""

from __future__ import annotations

import pytest

from cfm.data.sub_d.enums import Scope, Side
from cfm.data.sub_d.lattice import (
    CELL_SLOT_COUNT,
    EXTERNAL_EDGE_SLOT_COUNT,
    INTERNAL_EDGE_SLOT_COUNT,
    CellSlot,
    derive_external_edge_scope,
    derive_internal_edge_scope,
    iter_cell_slots,
    iter_external_edge_slots,
    iter_internal_edge_slots,
)


def test_cell_lattice_has_64_slots_in_row_major_order():
    slots = iter_cell_slots()
    assert len(slots) == CELL_SLOT_COUNT == 64
    # Indices are contiguous 0..63.
    assert [s.slot_index for s in slots] == list(range(64))
    # Row-major: slot N has cell_i = N // 8, cell_j = N % 8.
    assert slots[0] == CellSlot(slot_index=0, cell_i=0, cell_j=0)
    assert slots[1] == CellSlot(slot_index=1, cell_i=0, cell_j=1)
    assert slots[8] == CellSlot(slot_index=8, cell_i=1, cell_j=0)
    assert slots[63] == CellSlot(slot_index=63, cell_i=7, cell_j=7)
    # No duplicates.
    coords = {(s.cell_i, s.cell_j) for s in slots}
    assert len(coords) == 64
    # All coords in [0, 7] x [0, 7].
    assert all(0 <= s.cell_i <= 7 and 0 <= s.cell_j <= 7 for s in slots)


def test_internal_edge_lattice_has_112_slots():
    slots = iter_internal_edge_slots()
    assert len(slots) == INTERNAL_EDGE_SLOT_COUNT == 112
    # Indices are contiguous 0..111.
    assert [s.slot_index for s in slots] == list(range(112))
    # Axis splits cleanly: 56 axis=0 edges + 56 axis=1 edges.
    axis0 = [s for s in slots if s.axis == 0]
    axis1 = [s for s in slots if s.axis == 1]
    assert len(axis0) == 56
    assert len(axis1) == 56
    # axis=0 edges sit between (lower_i, lower_j) and (lower_i+1, lower_j);
    # lower_i must be in [0, 6] and lower_j in [0, 7] (both endpoints in-grid).
    for s in axis0:
        assert 0 <= s.lower_cell_i <= 6
        assert 0 <= s.lower_cell_j <= 7
    # axis=1 edges sit between (lower_i, lower_j) and (lower_i, lower_j+1);
    # lower_i in [0, 7], lower_j in [0, 6].
    for s in axis1:
        assert 0 <= s.lower_cell_i <= 7
        assert 0 <= s.lower_cell_j <= 6
    # All (lower_i, lower_j, axis) tuples unique.
    keys = {(s.lower_cell_i, s.lower_cell_j, s.axis) for s in slots}
    assert len(keys) == 112


def test_external_edge_lattice_has_32_slots():
    slots = iter_external_edge_slots()
    assert len(slots) == EXTERNAL_EDGE_SLOT_COUNT == 32
    assert [s.slot_index for s in slots] == list(range(32))
    # Each external edge is on the perimeter: either axis=0 with lower_i in
    # {-1, 7} (so the off-grid neighbour is at lower_i or lower_i+1), or
    # axis=1 with lower_j in {-1, 7}.
    for s in slots:
        if s.axis == 0:
            assert s.lower_cell_i in (-1, 7)
            assert 0 <= s.lower_cell_j <= 7
        else:
            assert s.axis == 1
            assert s.lower_cell_j in (-1, 7)
            assert 0 <= s.lower_cell_i <= 7
    # 8 edges on each of the 4 sides.
    assert sum(1 for s in slots if s.axis == 0 and s.lower_cell_i == -1) == 8
    assert sum(1 for s in slots if s.axis == 0 and s.lower_cell_i == 7) == 8
    assert sum(1 for s in slots if s.axis == 1 and s.lower_cell_j == -1) == 8
    assert sum(1 for s in slots if s.axis == 1 and s.lower_cell_j == 7) == 8
    # All slot keys unique.
    keys = {(s.lower_cell_i, s.lower_cell_j, s.axis) for s in slots}
    assert len(keys) == 32


def test_internal_edge_scope_distinguishes_active_masked_and_boundary():
    # Both endpoints active: ordinary active edge.
    assert derive_internal_edge_scope(lower_active=True, upper_active=True) == Scope.ACTIVE
    # Both endpoints masked: fully masked.
    assert derive_internal_edge_scope(lower_active=False, upper_active=False) == Scope.FULLY_MASKED
    # Either side mixed: scope-boundary (kept-to-not-in-scope edge).
    assert derive_internal_edge_scope(lower_active=True, upper_active=False) == Scope.SCOPE_BOUNDARY
    assert derive_internal_edge_scope(lower_active=False, upper_active=True) == Scope.SCOPE_BOUNDARY
    # SCOPE_BOUNDARY is distinct from FULLY_MASKED and ACTIVE.
    assert Scope.SCOPE_BOUNDARY != Scope.FULLY_MASKED
    assert Scope.SCOPE_BOUNDARY != Scope.ACTIVE


def test_external_edge_side_distribution_is_exactly_eight_per_side():
    slots = iter_external_edge_slots()
    sides = [s.side for s in slots]
    counts = {side: sides.count(side) for side in Side}
    assert counts[Side.TOP] == 8
    assert counts[Side.BOTTOM] == 8
    assert counts[Side.LEFT] == 8
    assert counts[Side.RIGHT] == 8
    assert sum(counts.values()) == 32
    # Side derivation matches the documented (lower_i, lower_j, axis) convention.
    for s in slots:
        if s.axis == 0 and s.lower_cell_i == -1:
            assert s.side == Side.TOP
        elif s.axis == 0 and s.lower_cell_i == 7:
            assert s.side == Side.BOTTOM
        elif s.axis == 1 and s.lower_cell_j == -1:
            assert s.side == Side.LEFT
        elif s.axis == 1 and s.lower_cell_j == 7:
            assert s.side == Side.RIGHT
    # Internal edges have no meaningful side; accessing the property must fail.
    internal = iter_internal_edge_slots()[0]
    with pytest.raises(ValueError):
        _ = internal.side


def test_external_edge_scope_uses_deferred_only_when_interior_cell_active():
    # Interior in-scope: deferred placeholder (sub-E owns real semantics).
    assert derive_external_edge_scope(interior_active=True) == Scope.EXTERNAL_DEFERRED
    # Interior not-in-scope: fully masked external edge.
    assert derive_external_edge_scope(interior_active=False) == Scope.FULLY_MASKED
    # external_deferred must not collide with internal-edge scopes.
    assert Scope.EXTERNAL_DEFERRED != Scope.ACTIVE
    assert Scope.EXTERNAL_DEFERRED != Scope.SCOPE_BOUNDARY

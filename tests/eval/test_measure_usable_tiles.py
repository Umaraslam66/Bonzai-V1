"""Tests for the shared interior-road-graph builder + usable-tile predicate.

Synthetic fixtures only — NO real corpus is read here. The point is to pin the
ONE definition of interior / road / endpoints / road-graph so the T5 'usable'
predicate and the T9 coherence metric (which both import
``interior_road_graph``) cannot drift apart.
"""

from __future__ import annotations

import pytest

from cfm.data.sub_d.enums import Scope, SlotKind
from cfm.data.sub_d.io import MacroCoreRow
from cfm.eval.holdout.macro_graph import interior_road_graph
from cfm.eval.usable_tiles import tile_is_usable


def _internal_edge_row(
    slot_index: int,
    lower_cell_i: int,
    lower_cell_j: int,
    axis: int,
    road_skeleton_class: int | None,
) -> MacroCoreRow:
    """Build one INTERNAL_EDGE row. Edge slots use lower_cell_*/axis; cell_*
    are None per the macro_core schema (spec §11.2)."""
    return MacroCoreRow(
        slot_kind=SlotKind.INTERNAL_EDGE,
        slot_index=slot_index,
        cell_i=None,
        cell_j=None,
        lower_cell_i=lower_cell_i,
        lower_cell_j=lower_cell_j,
        axis=axis,
        scope=Scope.ACTIVE,
        zoning_class=None,
        cell_density_bucket=None,
        road_skeleton_class=road_skeleton_class,
    )


@pytest.fixture
def make_macro_core():
    """Factory for a synthetic ``list[MacroCoreRow]``.

    - ``interior_road_edges=N``: emit N road-carrying INTERNAL_EDGE rows whose
      endpoints are both interior (axis=0 chain rooted at (1,1) -> edges
      (1,1+k)-(2,1+k) for k in range(N), interior for N<=5).
    - ``active_cells=0``: water / inactive tile -> empty row list -> 0 edges.
    """

    def _make(*, interior_road_edges: int = 0, active_cells: int = 1) -> list[MacroCoreRow]:
        if active_cells == 0:
            # Water / inactive tile: no active rows at all.
            return []
        rows: list[MacroCoreRow] = []
        for k in range(interior_road_edges):
            rows.append(
                _internal_edge_row(
                    slot_index=k,
                    lower_cell_i=1,
                    lower_cell_j=1 + k,
                    axis=0,  # (1,1+k) <-> (2,1+k), both interior for k <= 5
                    road_skeleton_class=1,  # in ROAD = {1,2,3}
                )
            )
        return rows

    return _make


def test_usable_requires_3_interior_road_edges(make_macro_core):
    assert tile_is_usable(make_macro_core(interior_road_edges=3)) is True
    assert tile_is_usable(make_macro_core(interior_road_edges=2)) is False
    assert tile_is_usable(make_macro_core(active_cells=0)) is False  # water


def test_interior_road_graph_returns_endpoints(make_macro_core):
    graph = interior_road_graph(make_macro_core(interior_road_edges=3))
    assert graph == [((1, 1), (2, 1)), ((1, 2), (2, 2)), ((1, 3), (2, 3))]


def test_interior_road_graph_excludes_boundary_edge():
    """An edge with a non-interior endpoint is excluded (interior filter).

    lower_cell_i=0, lower_cell_j=1, axis=0 -> endpoints (0,1)-(1,1); (0,1) is a
    boundary cell (i=0), so the edge is NOT in the interior road graph.
    """
    rows = [
        _internal_edge_row(
            slot_index=0,
            lower_cell_i=0,
            lower_cell_j=1,
            axis=0,
            road_skeleton_class=2,  # road, but boundary -> excluded
        )
    ]
    assert interior_road_graph(rows) == []


def test_interior_road_graph_excludes_non_road_edge():
    """A road_skeleton_class=0 edge ([0,1) = no crossing) is excluded
    (ROAD filter), even with fully interior endpoints."""
    rows = [
        _internal_edge_row(
            slot_index=0,
            lower_cell_i=2,
            lower_cell_j=2,
            axis=1,  # (2,2)-(2,3), both interior
            road_skeleton_class=0,  # bucket 0 = no crossing -> not ROAD
        )
    ]
    assert interior_road_graph(rows) == []


def test_interior_road_graph_ignores_non_edge_and_none_road():
    """Non-INTERNAL_EDGE rows and edges with road_skeleton_class=None are
    skipped (only active road-carrying internal edges count)."""
    rows = [
        # A CELL row (not an internal edge) — ignored.
        MacroCoreRow(
            slot_kind=SlotKind.CELL,
            slot_index=0,
            cell_i=2,
            cell_j=2,
            lower_cell_i=None,
            lower_cell_j=None,
            axis=None,
            scope=Scope.ACTIVE,
            zoning_class=1,
            cell_density_bucket=1,
            road_skeleton_class=None,
        ),
        # An internal edge with no road target (inactive/masked) — ignored.
        _internal_edge_row(
            slot_index=1,
            lower_cell_i=3,
            lower_cell_j=3,
            axis=0,
            road_skeleton_class=None,
        ),
    ]
    assert interior_road_graph(rows) == []

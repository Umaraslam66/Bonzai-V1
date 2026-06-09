"""Coherence 3-way GATING teeth (Task 10, spec §3.3) — SYNTHETIC half.

These are a HALT-gate, not characterization tests. Each case is red-before /
green-after by construction, and the **disconnected-loops tooth is the
load-bearing non-redundancy proof** for the fragmentation term: it exhibits a
plan that continuity ALONE would pass (high ``continuity_real``) but the
fragmentation term catches (low ``giant_real``). The two terms DIVERGE there,
which is exactly the AR failure mode (locally plausible, globally disconnected)
continuity is blind to.

The real-vs-permuted-on-held-out-tiles tooth runs on Leonardo with the T11 tile
loader and is NOT in this file (the controller owns it).

The metric (``coherence_gap``, T9) is GATED here, never modified.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfm.data.sub_d.enums import Scope, SlotKind
from cfm.data.sub_d.io import MacroCoreRow
from cfm.eval.holdout.coherence import coherence_gap

Cell = tuple[int, int]


# --------------------------------------------------------------------------- #
# Row builders (mirror tests/eval/test_measure_usable_tiles.py construction).
# --------------------------------------------------------------------------- #
def _edge_row(
    slot_index: int,
    lower_cell_i: int,
    lower_cell_j: int,
    axis: int,
    road_skeleton_class: int,
) -> MacroCoreRow:
    """One INTERNAL_EDGE row. Edge slots carry lower_cell_*/axis; cell_* are
    None per the macro_core schema (spec §11.2)."""
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


def _square_loop_edges(start_index: int, anchor: Cell) -> list[MacroCoreRow]:
    """4 road edges forming a 2x2 cycle over cells (r,c),(r+1,c),(r+1,c+1),(r,c+1).

    Every cell in the square ends with road-degree 2 (a closed loop). The four
    edges are:
      (r,c)   <-> (r+1,c)    axis=0, lower=(r,c)
      (r+1,c) <-> (r+1,c+1)  axis=1, lower=(r+1,c)
      (r,c+1) <-> (r+1,c+1)  axis=0, lower=(r,c+1)
      (r,c)   <-> (r,c+1)    axis=1, lower=(r,c)
    """
    r, c = anchor
    return [
        _edge_row(start_index + 0, r, c, axis=0, road_skeleton_class=1),
        _edge_row(start_index + 1, r + 1, c, axis=1, road_skeleton_class=1),
        _edge_row(start_index + 2, r, c + 1, axis=0, road_skeleton_class=1),
        _edge_row(start_index + 3, r, c, axis=1, road_skeleton_class=1),
    ]


@pytest.fixture
def make_rows():
    """Factory for a synthetic ``list[MacroCoreRow]`` in one of four modes.

    All interior internal edges (both endpoints in the 6x6 interior). Road class
    1 is in ROAD={1,2,3}; class 0 is no-road.

    - ``all_edges_road=True`` (UNIFORM): EVERY interior internal edge carries a
      road. Permuting an all-road assignment yields an all-road assignment, so
      the road graph is unchanged -> gap ~ 0 by construction (anti-uniform
      property falling out of the permutation null).
    - ``random_edges=True`` (NOISE): each interior internal edge gets a RANDOM
      class scattered over {0 (no-road), 1, 2, 3} -> low continuity (dead-ends)
      and low gap.
    - ``disconnected_loops=True`` (NON-REDUNDANCY): K>=4 disjoint 2x2 square
      loops, no shared/adjacent cells -> continuity ~ 1.0 (every cell degree 2)
      but giant fraction = 4/(4K) <= 0.25 (multiple disconnected components).
    - ``connected_path=True`` (POSITIVE CONTROL): one connected path through
      interior cells -> giant fraction = 1.0 (a single component), to contrast
      with the disconnected-loops case (proves the metric DISTINGUISHES
      connected from fragmented, not just "low giant always").
    """

    def _make(
        *,
        all_edges_road: bool = False,
        random_edges: bool = False,
        disconnected_loops: bool = False,
        connected_path: bool = False,
    ) -> list[MacroCoreRow]:
        modes = [all_edges_road, random_edges, disconnected_loops, connected_path]
        if sum(bool(m) for m in modes) != 1:
            raise ValueError("make_rows: pass exactly one mode flag")

        rows: list[MacroCoreRow] = []

        if all_edges_road:
            # Every interior internal edge (both endpoints in 1..6) carries a road.
            idx = 0
            # axis=0 edges: (i,j)<->(i+1,j); interior both requires 1<=i, i+1<=6.
            for i in range(1, 6):  # i in 1..5 -> i+1 in 2..6
                for j in range(1, 7):  # j in 1..6
                    rows.append(_edge_row(idx, i, j, axis=0, road_skeleton_class=1))
                    idx += 1
            # axis=1 edges: (i,j)<->(i,j+1); interior both requires 1<=j, j+1<=6.
            for i in range(1, 7):  # i in 1..6
                for j in range(1, 6):  # j in 1..5 -> j+1 in 2..6
                    rows.append(_edge_row(idx, i, j, axis=1, road_skeleton_class=1))
                    idx += 1
            return rows

        if random_edges:
            # Random class scatter over the same interior internal-edge set.
            # Deterministic local RNG so the fixture is reproducible.
            r = np.random.default_rng(1234)
            idx = 0
            for i in range(1, 6):
                for j in range(1, 7):
                    cls = int(r.integers(0, 4))  # 0..3
                    rows.append(_edge_row(idx, i, j, axis=0, road_skeleton_class=cls))
                    idx += 1
            for i in range(1, 7):
                for j in range(1, 6):
                    cls = int(r.integers(0, 4))
                    rows.append(_edge_row(idx, i, j, axis=1, road_skeleton_class=cls))
                    idx += 1
            return rows

        if disconnected_loops:
            # K=4 disjoint 2x2 loops. Anchors chosen so no cell is shared and no
            # two squares are road-adjacent: each square occupies a 2x2 block, and
            # consecutive anchors are >=3 apart on each axis (one empty lane between
            # blocks). Anchor (4,4) -> cells up to (5,5), all interior (<=6).
            anchors: list[Cell] = [(1, 1), (1, 4), (4, 1), (4, 4)]
            idx = 0
            for a in anchors:
                rows.extend(_square_loop_edges(idx, a))
                idx += 4
            return rows

        # connected_path: one connected path through interior cells.
        # A horizontal chain along row 3: (3,1)-(3,2)-...-(3,6), all axis=1 edges
        # sharing cells -> a single connected component (giant fraction 1.0).
        idx = 0
        for j in range(1, 6):  # edges (3,j)<->(3,j+1) for j in 1..5
            rows.append(_edge_row(idx, 3, j, axis=1, road_skeleton_class=1))
            idx += 1
        return rows

    return _make


# --------------------------------------------------------------------------- #
# The gating teeth.
# --------------------------------------------------------------------------- #
def test_uniform_plan_fails(make_rows):
    # every edge a road -> shuffles to itself -> gap ~ 0 (FAILS the bar by construction)
    g = coherence_gap(make_rows(all_edges_road=True), rng=np.random.default_rng(0), n_shuffle=15)
    assert abs(g["continuity_gap"]) < 0.05 and abs(g["fragmentation_gap"]) < 0.05


def test_noise_plan_fails(make_rows):
    # random scatter -> low continuity, low gap
    g = coherence_gap(make_rows(random_edges=True), rng=np.random.default_rng(0), n_shuffle=15)
    assert g["continuity_gap"] < 0.1


def test_disconnected_loops_fragmentation_nonredundant(make_rows):
    # THE load-bearing tooth: locally-plausible (high continuity) but globally-broken (low giant
    # fraction). Continuity ALONE would PASS this plan (>0.8); only the fragmentation term catches
    # it (giant_real < 0.5). The two terms DIVERGE here -> fragmentation is NON-REDUNDANT, carrying
    # the exact AR failure mode (locally plausible, globally disconnected) continuity is blind to.
    g = coherence_gap(
        make_rows(disconnected_loops=True), rng=np.random.default_rng(0), n_shuffle=15
    )
    print(
        f"\n[disconnected-loops divergence] continuity_real={g['continuity_real']!r} "
        f"giant_real={g['giant_real']!r}"
    )
    assert g["continuity_real"] > 0.8  # continuity says "fine"
    assert g["giant_real"] < 0.5  # fragmentation says "broken" -> the divergence IS the proof


def test_connected_network_positive_control(make_rows):
    # POSITIVE CONTROL: one connected road network -> a single component (giant_real == 1.0) with
    # reasonable continuity. Contrasts with disconnected-loops (giant_real < 0.5): proves the
    # fragmentation term DISTINGUISHES connected from fragmented, not "low giant always".
    g = coherence_gap(make_rows(connected_path=True), rng=np.random.default_rng(0), n_shuffle=15)
    print(
        f"\n[connected-control] continuity_real={g['continuity_real']!r} "
        f"giant_real={g['giant_real']!r}"
    )
    assert g["giant_real"] == 1.0  # one connected component
    assert (
        g["continuity_real"] >= 0.5
    )  # interior cells are through-cells; only the 2 ends are stubs

"""Tests for the S1 macro-plan-coherence metric (spec §3.1).

Synthetic fixtures only — NO real corpus is read here. These pin the locked
design: skeleton = continuity + giant-component fraction over the 6x6 interior
road graph; zoning = ACTIVE-ACTIVE same-class agreement over interior-interior
internal edges; gap = score(real) - mean(score(interior-permuted)).

The term functions are imported from ``coherence``; the interior / road /
endpoints / road-graph definitions live in the SHARED ``macro_graph`` builder
(T5), so the "usable" power unit and the "scored" metric cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfm.data.sub_d.enums import Scope, SlotKind
from cfm.data.sub_d.io import MacroCoreRow
from cfm.eval.holdout.coherence import (
    coherence_gap,
    continuity,
    giant_component_fraction,
    zoning_agreement,
)


# --------------------------------------------------------------------------- #
# Pure term functions (edges / zoning in, score out).
# --------------------------------------------------------------------------- #
def test_continuity_counts_through_cells():
    edges = [((1, 1), (2, 1)), ((2, 1), (3, 1)), ((3, 1), (4, 1))]  # path: ends deg1, middles deg2
    assert continuity(edges) == pytest.approx(2 / 4)  # 4 touched cells, 2 with deg>=2


def test_continuity_none_on_empty():
    assert continuity([]) is None


def test_giant_component_fraction_drops_on_fragmentation():
    one_net = [((1, 1), (2, 1)), ((2, 1), (3, 1)), ((3, 1), (3, 2))]
    assert giant_component_fraction(one_net) == 1.0
    two_islands = [((1, 1), (2, 1)), ((5, 5), (6, 5)), ((6, 5), (6, 6))]
    assert giant_component_fraction(two_islands) < 1.0
    assert giant_component_fraction(two_islands) == pytest.approx(2 / 3)  # giant=2 of 3 edges


def test_giant_component_fraction_none_on_empty():
    assert giant_component_fraction([]) is None


def test_zoning_agreement_excludes_inactive_edges():
    # built(class 0) <-> empty(None) edge is EXCLUDED, not disagreement (active-active, spec §3.1)
    cells = {(1, 1): 0, (2, 1): None, (1, 2): 0}
    assert zoning_agreement(edges=[((1, 1), (2, 1)), ((1, 1), (1, 2))], zoning=cells) == 1.0


def test_zoning_agreement_counts_disagreement():
    cells = {(1, 1): 0, (2, 1): 1, (1, 2): 0}
    # (1,1)-(2,1) active-active class 0 vs 1 -> disagree; (1,1)-(1,2) 0 vs 0 -> agree => 1/2
    assert zoning_agreement(edges=[((1, 1), (2, 1)), ((1, 1), (1, 2))], zoning=cells) == 0.5


def test_zoning_agreement_none_when_no_active_pair():
    cells = {(1, 1): 0, (2, 1): None}
    assert zoning_agreement(edges=[((1, 1), (2, 1))], zoning=cells) is None


# --------------------------------------------------------------------------- #
# Row fixtures (CELL + INTERNAL_EDGE), adapted from test_measure_usable_tiles.
# --------------------------------------------------------------------------- #
def _edge_row(
    slot_index: int,
    lower_cell_i: int,
    lower_cell_j: int,
    axis: int,
    road_skeleton_class: int | None,
) -> MacroCoreRow:
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


def _cell_row(
    slot_index: int,
    cell_i: int,
    cell_j: int,
    zoning_class: int | None,
) -> MacroCoreRow:
    return MacroCoreRow(
        slot_kind=SlotKind.CELL,
        slot_index=slot_index,
        cell_i=cell_i,
        cell_j=cell_j,
        lower_cell_i=None,
        lower_cell_j=None,
        axis=None,
        scope=Scope.ACTIVE,
        zoning_class=zoning_class,
        cell_density_bucket=None,
        road_skeleton_class=None,
    )


def _hand_built_tile() -> list[MacroCoreRow]:
    """A small interior tile: a road path + a zoning patch, all interior (1..6).

    Road path along axis 0 at column j=1: edges (1,1)-(2,1), (2,1)-(3,1),
    (3,1)-(4,1) all road. Plus two non-road interior edges so the skeleton null
    has a real road/non-road marginal to permute.
    Zoning cells form interior-interior adjacencies with mixed classes.
    """
    rows: list[MacroCoreRow] = []
    si = 0
    # Road edges (class 1 = in ROAD).
    for k in range(3):
        rows.append(_edge_row(si, 1 + k, 1, axis=0, road_skeleton_class=1))
        si += 1
    # Non-road interior edges (class 0 = not ROAD) — part of the permutation marginal.
    rows.append(_edge_row(si, 3, 2, axis=0, road_skeleton_class=0))
    si += 1
    rows.append(_edge_row(si, 4, 2, axis=0, road_skeleton_class=0))
    si += 1
    # Zoning cells: interior, mixed classes giving real internal-edge adjacencies.
    ci = 0
    cell_zoning = {
        (1, 1): 0,
        (2, 1): 0,
        (3, 1): 1,
        (4, 1): 1,
        (3, 2): 0,
        (4, 2): 1,
    }
    for (ii, jj), z in cell_zoning.items():
        rows.append(_cell_row(ci, ii, jj, zoning_class=z))
        ci += 1
    # One inactive (None-zoning) interior cell to exercise active-active exclusion.
    rows.append(_cell_row(ci, 2, 2, zoning_class=None))
    return rows


def test_coherence_gap_returns_required_keys():
    rng = np.random.default_rng(0)
    out = coherence_gap(_hand_built_tile(), rng=rng, n_shuffle=50)
    for key in (
        "continuity_real",
        "continuity_gap",
        "giant_real",
        "fragmentation_gap",
        "zoning_real",
        "zoning_gap",
    ):
        assert key in out, f"missing key {key!r}"
    assert out["continuity_real"] is not None
    assert out["giant_real"] is not None
    assert out["zoning_real"] is not None


def test_coherence_gap_is_deterministic_given_rng():
    a = coherence_gap(_hand_built_tile(), rng=np.random.default_rng(7), n_shuffle=40)
    b = coherence_gap(_hand_built_tile(), rng=np.random.default_rng(7), n_shuffle=40)
    assert a == b


def test_coherence_gap_near_zero_when_arrangement_is_uninformative():
    """A tile whose attribute is the SAME everywhere shuffles to itself: any
    permutation reproduces the real arrangement, so every gap is ~0."""
    rows: list[MacroCoreRow] = []
    si = 0
    # All interior edges road (class 1): every permutation is identical -> gap 0.
    edge_cells = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
    for k in range(4):
        rows.append(_edge_row(si, 1 + k, 1, axis=0, road_skeleton_class=1))
        si += 1
    # All interior cells same zoning class -> zoning agreement permutes to itself.
    ci = 0
    for ii, jj in edge_cells:
        rows.append(_cell_row(ci, ii, jj, zoning_class=2))
        ci += 1
    out = coherence_gap(rows, rng=np.random.default_rng(1), n_shuffle=64)
    assert out["continuity_gap"] == pytest.approx(0.0, abs=1e-9)
    assert out["fragmentation_gap"] == pytest.approx(0.0, abs=1e-9)
    assert out["zoning_gap"] == pytest.approx(0.0, abs=1e-9)


def test_coherence_gap_none_real_yields_none_gap():
    """A tile with no active road edges / no active-active zoning pair returns
    None real scores and None gaps, without crashing."""
    rows = [
        # Only non-road interior edges -> empty road graph.
        _edge_row(0, 1, 1, axis=0, road_skeleton_class=0),
        # A single active zoning cell -> no active-active internal edge.
        _cell_row(0, 1, 1, zoning_class=0),
    ]
    out = coherence_gap(rows, rng=np.random.default_rng(0), n_shuffle=10)
    assert out["continuity_real"] is None
    assert out["continuity_gap"] is None
    assert out["giant_real"] is None
    assert out["fragmentation_gap"] is None
    assert out["zoning_real"] is None
    assert out["zoning_gap"] is None

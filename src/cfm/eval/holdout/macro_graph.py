"""The single shared definition of the interior road graph (spec §2.1/§3.5b).

This module is the ONE place that defines interior / road / endpoints /
road-graph for the eval set. Both the T5 'usable-tile' predicate
(``cfm.eval.usable_tiles``) and the T9 S1 coherence metric import
``interior_road_graph`` from here, so 'usable' and 'scored' are derived from a
single source and cannot drift apart.
"""

from __future__ import annotations

from cfm.data.sub_d.enums import SlotKind
from cfm.data.sub_d.io import MacroCoreRow

# road_skeleton bucket 0 = [0,1) = no crossing; {1,2,3} = road. Verified against
# configs/macro_plan/v1/macro_plan_vocab.yaml road_skeleton buckets (token_id 0..3):
# 0=[0,1), 1=[1,4), 2=[4,9), 3=[9,inf).
ROAD: frozenset[int] = frozenset({1, 2, 3})


def interior(i: int, j: int) -> bool:
    """6x6 interior of the 0-indexed 8x8 lattice (boundary cells excluded; their
    roads exit to neighbours = the cross-tile seam, deferred per spec §4.2)."""
    return 1 <= i <= 6 and 1 <= j <= 6


def endpoints(lower_i: int, lower_j: int, axis: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """The two cells an internal edge connects (sub_d/lattice axis convention:
    axis 0 -> (i,j)<->(i+1,j); axis 1 -> (i,j)<->(i,j+1))."""
    return (
        ((lower_i, lower_j), (lower_i + 1, lower_j))
        if axis == 0
        else ((lower_i, lower_j), (lower_i, lower_j + 1))
    )


def interior_road_graph(
    rows: list[MacroCoreRow],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Road-carrying interior-interior internal edges as ``[(cellA, cellB), ...]``.

    Single source for BOTH the 'usable' power unit (T5) and the S1 coherence
    metric (T9), so they cannot diverge.
    """
    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in rows:
        if r.slot_kind != SlotKind.INTERNAL_EDGE or r.road_skeleton_class is None:
            continue
        if int(r.road_skeleton_class) not in ROAD:
            continue
        a, b = endpoints(r.lower_cell_i, r.lower_cell_j, r.axis)
        if interior(*a) and interior(*b):
            # Bare append (no dedup) is correct: upstream macro_core guarantees
            # one row per (slot_kind, slot_index) pair, so duplicate edges cannot
            # arise from the source data.
            out.append((a, b))
    return out

"""S1 macro-plan-coherence metric — the net-new bake-off bar (spec §3.1).

Per held-out tile, per attribute, this scores the *arrangement* of the
generated macro plan over the 6x6 INTERIOR and compares it to an
interior-permutation null::

    gap = score(real arrangement) - mean(score(interior-permuted))

The null fixes the interior count/marginal and destroys arrangement, so a
positive gap means the real plan is *more coherent than chance given the same
amount of stuff* — it isolates arrangement, not density.

Two attributes are scored (density is DROPPED per spec §3.4 — its conditioning
vector is per-cell, so scoring it against the handed conditioning would be
circular; there is no density term here):

- **Skeleton (EDGE-keyed), two terms over the interior road graph:**
  - ``continuity`` = fraction of road-touched interior cells with road-degree
    ≥ 2 (dead-end avoidance);
  - ``giant_component_fraction`` = edges in the largest connected component /
    total interior road-edges (catches disconnected islands continuity alone
    cannot). ``fragmentation_gap`` is the gap on this term.
- **Zoning (CELL-keyed, categorical), ACTIVE-ACTIVE only:** per interior
  internal edge where BOTH incident cells are active (non-null zoning),
  agreement = same class. Active-inactive and inactive-inactive edges are
  EXCLUDED, not scored as disagreement (penalizing geography, not incoherence —
  the parallel of skeleton's road-touched guard). v1 = same-class only.

The interior / road / endpoints / road-graph definitions are imported from the
SHARED ``macro_graph`` builder, never re-defined here, so the T5 "usable" power
unit and this T9 "scored" metric are derived from one source and cannot drift.
"""

from __future__ import annotations

import numpy as np

from cfm.data.sub_d.enums import SlotKind
from cfm.data.sub_d.io import MacroCoreRow
from cfm.eval.holdout.macro_graph import ROAD, endpoints, interior, interior_road_graph

Cell = tuple[int, int]
Edge = tuple[Cell, Cell]


# --------------------------------------------------------------------------- #
# Pure term functions (edges / zoning in, score out). These score the GENERATED
# macro plan only — never the handed conditioning — per the §3.4 firewall.
# --------------------------------------------------------------------------- #
def continuity(edges: list[Edge]) -> float | None:
    """Fraction of road-touched interior cells with road-degree >= 2.

    Dead-end avoidance: a through-cell (degree >= 2) continues the road; a
    degree-1 cell is a stub. ``None`` if there are no road edges.
    """
    if not edges:
        return None
    deg: dict[Cell, int] = {}
    for a, b in edges:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    return sum(1 for c in deg if deg[c] >= 2) / len(deg)


def giant_component_fraction(edges: list[Edge]) -> float | None:
    """Edges in the largest connected component / total edges (union-find).

    1.0 = one connected network; a value < 1 means disconnected islands.
    ``None`` if there are no road edges.
    """
    if not edges:
        return None
    parent: dict[Cell, Cell] = {}

    def find(x: Cell) -> Cell:
        parent.setdefault(x, x)
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comp: dict[Cell, int] = {}
    for a, _b in edges:
        comp[find(a)] = comp.get(find(a), 0) + 1
    return max(comp.values()) / len(edges)


def zoning_agreement(edges: list[Edge], zoning: dict[Cell, int | None]) -> float | None:
    """Active-active same-class fraction (spec §3.1).

    Per edge where BOTH incident cells have non-null zoning, agreement = same
    class. Edges with either endpoint inactive (None) are EXCLUDED, not scored
    as disagreement. ``None`` if there is no active-active edge.
    """
    active = [(a, b) for (a, b) in edges if zoning.get(a) is not None and zoning.get(b) is not None]
    if not active:
        return None
    return sum(1 for a, b in active if zoning[a] == zoning[b]) / len(active)


# --------------------------------------------------------------------------- #
# Row -> graph/zoning adapters (built from the SAME interior/endpoints helpers).
# --------------------------------------------------------------------------- #
def interior_internal_edges(rows: list[MacroCoreRow]) -> list[Edge]:
    """Every INTERNAL_EDGE whose BOTH endpoints are interior (ROAD-AGNOSTIC).

    This is the zoning-adjacency set: the lattice adjacency over which
    active-active zoning agreement is measured. Road-agnostic by design (zoning
    lives on cells, not on whether the shared edge carries a road). Built from
    the SAME ``interior``/``endpoints`` helpers as the road graph so the two
    cannot drift. Deterministically ordered by endpoint pair.
    """
    out: list[Edge] = []
    for r in rows:
        if r.slot_kind != SlotKind.INTERNAL_EDGE:
            continue
        a, b = endpoints(r.lower_cell_i, r.lower_cell_j, r.axis)
        if interior(*a) and interior(*b):
            out.append((a, b))
    return sorted(out)


def interior_zoning(rows: list[MacroCoreRow]) -> dict[Cell, int]:
    """``{(cell_i, cell_j): zoning_class}`` for active interior CELL rows.

    Active = interior cell with non-None ``zoning_class``. Inactive interior
    cells (None zoning) are simply absent from the dict, so the active-active
    discipline in ``zoning_agreement`` excludes their edges.
    """
    out: dict[Cell, int] = {}
    for r in rows:
        if r.slot_kind != SlotKind.CELL or r.zoning_class is None:
            continue
        cell = (r.cell_i, r.cell_j)
        if interior(*cell):
            out[cell] = int(r.zoning_class)
    return out


def _active_interior_edge_classes(rows: list[MacroCoreRow]) -> list[tuple[Edge, int]]:
    """Ordered ``[(endpoint_pair, road_skeleton_class), ...]`` for the skeleton
    permutation: INTERNAL_EDGE rows with both endpoints interior AND non-None
    ``road_skeleton_class`` (the active edges eligible to permute).

    Deterministically ordered by endpoint pair so a given ``rng`` + rows is
    reproducible.
    """
    out: list[tuple[Edge, int]] = []
    for r in rows:
        if r.slot_kind != SlotKind.INTERNAL_EDGE or r.road_skeleton_class is None:
            continue
        a, b = endpoints(r.lower_cell_i, r.lower_cell_j, r.axis)
        if interior(*a) and interior(*b):
            out.append(((a, b), int(r.road_skeleton_class)))
    return sorted(out, key=lambda ec: ec[0])


def _road_graph_from_edge_classes(edge_classes: list[tuple[Edge, int]]) -> list[Edge]:
    """Road graph (edges whose class is in ROAD) from an edge->class list."""
    return [edge for edge, cls in edge_classes if cls in ROAD]


# --------------------------------------------------------------------------- #
# The shuffle-gap (the metric).
# --------------------------------------------------------------------------- #
def coherence_gap(
    rows: list[MacroCoreRow],
    *,
    rng: np.random.Generator,
    n_shuffle: int,
) -> dict[str, float | None]:
    """Per-tile coherence gaps = score(real) - mean(score(interior-permuted)).

    The null permutes each attribute among its ACTIVE interior slots only,
    fixing the interior count/marginal and destroying arrangement:

    - SKELETON: permute ``road_skeleton_class`` among the active interior edges,
      rebuild the road graph (edges whose permuted class ∈ ROAD), recompute
      continuity and giant-component fraction.
    - ZONING: permute ``zoning_class`` among the active interior cells, recompute
      active-active agreement over the SAME interior adjacency edges.

    All randomness flows through ``rng`` (no global RNG), so a given ``rng`` +
    ``rows`` is deterministic. Returns at least the keys ``continuity_real``,
    ``continuity_gap``, ``giant_real``, ``fragmentation_gap``, ``zoning_real``,
    ``zoning_gap``. If a real score is None (no active edges/cells), its gap is
    None. None shuffled scores are skipped when averaging.
    """
    # --- Real scores -------------------------------------------------------- #
    road_graph = interior_road_graph(rows)
    continuity_real = continuity(road_graph)
    giant_real = giant_component_fraction(road_graph)

    adjacency = interior_internal_edges(rows)
    zoning_map = interior_zoning(rows)
    zoning_real = zoning_agreement(adjacency, zoning_map)

    # --- Skeleton null: permute road class among active interior edges ------ #
    edge_classes = _active_interior_edge_classes(rows)
    edges_only = [edge for edge, _ in edge_classes]
    classes = np.array([cls for _, cls in edge_classes], dtype=int)

    shuffled_continuity: list[float] = []
    shuffled_giant: list[float] = []
    if len(classes) > 0:
        for _ in range(n_shuffle):
            permuted = rng.permutation(classes)
            perm_edge_classes = list(zip(edges_only, permuted.tolist(), strict=True))
            perm_road_graph = _road_graph_from_edge_classes(perm_edge_classes)
            sc = continuity(perm_road_graph)
            sg = giant_component_fraction(perm_road_graph)
            if sc is not None:
                shuffled_continuity.append(sc)
            if sg is not None:
                shuffled_giant.append(sg)

    # --- Zoning null: permute zoning class among active interior cells ------ #
    zoning_cells = sorted(zoning_map)  # deterministic cell order
    zoning_values = np.array([zoning_map[c] for c in zoning_cells], dtype=int)

    shuffled_zoning: list[float] = []
    if len(zoning_values) > 0:
        for _ in range(n_shuffle):
            permuted = rng.permutation(zoning_values)
            perm_map: dict[Cell, int | None] = {
                cell: int(val) for cell, val in zip(zoning_cells, permuted.tolist(), strict=True)
            }
            sz = zoning_agreement(adjacency, perm_map)
            if sz is not None:
                shuffled_zoning.append(sz)

    return {
        "continuity_real": continuity_real,
        "continuity_gap": _gap(continuity_real, shuffled_continuity),
        "giant_real": giant_real,
        "fragmentation_gap": _gap(giant_real, shuffled_giant),
        "zoning_real": zoning_real,
        "zoning_gap": _gap(zoning_real, shuffled_zoning),
    }


def _gap(real: float | None, shuffled: list[float]) -> float | None:
    """``real - mean(shuffled)``; None if the real score is None or no shuffled
    score was computable (None real scores propagate to a None gap)."""
    if real is None or not shuffled:
        return None
    return real - float(np.mean(shuffled))

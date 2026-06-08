"""Usable-tile predicate for coherence power (spec §2.1/§3.5b).

Consumes the shared ``interior_road_graph`` builder so 'usable' and 'scored'
are one definition — this module does NOT reimplement the graph.
"""

from __future__ import annotations

from cfm.data.sub_d.io import MacroCoreRow
from cfm.eval.holdout.macro_graph import interior_road_graph

MIN_ROAD_EDGES = 3


def tile_is_usable(rows: list[MacroCoreRow]) -> bool:
    """A tile is usable for coherence power iff it has >= 3 road-carrying
    interior edges in the SAME graph the coherence metric scores (so 'usable'
    and 'scored' are one definition). After-water-filter is implicit: a water /
    inactive tile has no active road edges -> 0 -> not usable."""
    return len(interior_road_graph(rows)) >= MIN_ROAD_EDGES

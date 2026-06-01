"""Locked macro-plan bucket cut-points + a bucketing function.

Cut-points are read from the LOCKED ``configs/macro_plan/v1/macro_plan_vocab.yaml``
(a sub-D contract *input*, not sub-D's runtime output). ``bucket_of`` reproduces
the vocab's lower-inclusive / upper-exclusive semantics so seam 1 can
independently bucket a recomputed metric and compare to sub-D's stored class.

Structure (verified 2026-05-31, Step-0): the bucket lists live under a top-level
``locked_buckets:`` key, e.g. ``locked_buckets.cell_density`` (lines 3472-3488)
and ``locked_buckets.road_skeleton`` (lines 3489-3505); each entry has
``lower_inclusive`` / ``token_id`` / ``token_name`` / ``upper_exclusive``.

DECISION: edges are the ``lower_inclusive`` values ordered by ``token_id``;
bucket k = the highest edge index whose lower_inclusive <= value. Boundary FP
sensitivity (a recomputed ratio landing exactly on a cut-point) is handled by
seam 1 (structural-boundary EPSILON), not here.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_VOCAB = (
    Path(__file__).resolve().parents[4] / "configs" / "macro_plan" / "v1" / "macro_plan_vocab.yaml"
)


def _load_edges(section: str) -> list[float]:
    data = yaml.safe_load(_VOCAB.read_text(encoding="utf-8"))
    rows = data["locked_buckets"][section]
    rows_sorted = sorted(rows, key=lambda r: r["token_id"])
    return [r["lower_inclusive"] for r in rows_sorted]


def load_density_edges() -> list[float]:
    return _load_edges("cell_density")


def load_road_skeleton_edges() -> list[int]:
    return [int(e) for e in _load_edges("road_skeleton")]


def bucket_of(value: float, edges: list[float]) -> int:
    """Return token_id for ``value`` given lower-inclusive ``edges`` (ascending).

    edges=[0.0, 0.05, 0.15, 0.35]: value 0.05 -> 1 (lower-inclusive), 0.049 -> 0.
    """
    idx = 0
    for k, lo in enumerate(edges):
        if value >= lo:
            idx = k
        else:
            break
    return idx

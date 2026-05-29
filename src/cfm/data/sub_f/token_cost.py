"""Case-A per-feature token-cost accounting for BP3 budget tooling.

Single source of truth for the chunked Case-A per-feature token count, shared
by the BP3 budget tools (``scripts/sub_f/compute_alpha_drop_report.py`` and the
chunked-budget audit). The runtime *emitter* is ``encoder.encode_feature`` /
``encoder.encode_cell``; this module is its budget-accounting *twin* — it counts
tokens without building them, so the budget tooling can run fast over hundreds
of thousands of features.

The two MUST agree. They are kept from drifting by the pin-tests in
``tests/data/sub_f/test_token_cost.py``, which grade this module's output
against the encoder's actual emission on a fixture battery. This is the guard
for the failure mode that shipped the T8.4 chunking miss and surfaced at T8.7
dispatch: a budget formula and an emitter that silently disagree
(``feedback_test_spec_not_just_plan``, ``feedback_external_source_of_truth_gate``).

Cost basis is Case A (no boundary refs): boundary-ref cases (B/C/D) shift the
count by a few tokens per through-road feature, but the BP3 budget surface and
drop report are computed on the Case-A basis (see
``configs/sub_f/sequence_length_analysis.yaml``). Keep this module Case-A only;
do not add bref accounting here without re-deriving the budget surface.
"""

from __future__ import annotations

import math

from shapely.geometry.base import BaseGeometry

from cfm.data.sub_f.encoder import (
    DEFAULT_CHUNK_THRESHOLD_M,
    DEFAULT_MAGNITUDE_QUANTUM_M,
    DEFAULT_N_ANCHOR_TOKENS,
    quantize_coord_m,
)

# Max magnitude quanta per (dir, mag) pair = chunk threshold / quantum.
# DERIVED from the BP2 locked constants (NOT a separate literal), so this
# module's chunk size cannot drift from the encoder's _MAX_MAGNITUDE_Q. The
# pin-test test_max_magnitude_quantum_does_not_drift_from_encoder asserts the
# two are equal.
_MAX_MAGNITUDE_Q: int = round(DEFAULT_CHUNK_THRESHOLD_M / DEFAULT_MAGNITUDE_QUANTUM_M)  # 64

# Per-feature structural overhead: <feature> + <semantic_tag> + <feature_end>.
_STRUCTURAL_TOKENS: int = 3


def chunked_segment_pairs(distance_m: float) -> int:
    """Number of (dir, mag) pairs the encoder emits for ONE segment, per §3.5.

    Mirrors ``encoder._direction_magnitude_pair``: quantize the distance with
    the same banker's-rounding ``quantize_coord_m``, floor to one quantum (a
    zero-length segment still emits one pair to preserve the vertex), then chunk
    into groups of ``_MAX_MAGNITUDE_Q`` quanta:

        total_q = max(1, quantize_coord_m(distance_m))
        pairs   = ceil(total_q / 64)
    """
    total_q = max(1, quantize_coord_m(distance_m))
    return math.ceil(total_q / _MAX_MAGNITUDE_Q)


def chunked_per_feature_tokens(
    coords: list[tuple[float, float]], n_anchor: int = DEFAULT_N_ANCHOR_TOKENS
) -> int:
    """Case-A token count for ONE feature given its coordinate list.

    Matches ``encoder.encode_feature(geom, semantic_tag=...)`` with no boundary
    refs (Case A)::

        tokens = 3 (structural) + n_anchor + 2 * sum_segments(chunked_segment_pairs)

    For ``len(coords) < 2`` (Point / degenerate): ``3 + n_anchor`` (no segments).
    """
    n = len(coords)
    if n < 2:
        return _STRUCTURAL_TOKENS + n_anchor
    pairs = 0
    for i in range(1, n):
        x1, y1 = coords[i - 1]
        x2, y2 = coords[i]
        pairs += chunked_segment_pairs(math.hypot(x2 - x1, y2 - y1))
    return _STRUCTURAL_TOKENS + n_anchor + 2 * pairs


def _part_coords(geom: BaseGeometry) -> list[list[tuple[float, float]]]:
    """Per-emitted-feature coordinate lists, matching ``encode_cell``'s split.

    ``encode_cell`` emits one ``EncodedFeature`` per part for MultiLineString /
    MultiPolygon / MultiPoint; Polygons encode their exterior ring; Points the
    single coordinate. Returns one coord list per feature the encoder emits, so
    summing ``chunked_per_feature_tokens`` over the result equals the encoder's
    total for ``geom``.
    """
    gt = geom.geom_type
    if gt == "LineString":
        return [list(geom.coords)]
    if gt == "Polygon":
        return [list(geom.exterior.coords)]
    if gt == "Point":
        return [[(geom.x, geom.y)]]
    if gt == "MultiLineString":
        return [list(part.coords) for part in geom.geoms]
    if gt == "MultiPolygon":
        return [list(part.exterior.coords) for part in geom.geoms]
    if gt == "MultiPoint":
        return [[(part.x, part.y)] for part in geom.geoms]
    raise ValueError(f"_part_coords: unsupported geom_type {gt!r}")


def feature_token_cost(geom: BaseGeometry, n_anchor: int = DEFAULT_N_ANCHOR_TOKENS) -> int:
    """Total Case-A tokens ``encode_cell`` emits for one sub-C feature geometry.

    Splits Multi* per-part exactly as ``encode_cell`` does (one ``<feature>``..
    ``<feature_end>`` per part) and sums the per-part chunked cost. For
    single-part geometries this equals ``chunked_per_feature_tokens`` on the
    feature's coord list.
    """
    return sum(chunked_per_feature_tokens(coords, n_anchor) for coords in _part_coords(geom))

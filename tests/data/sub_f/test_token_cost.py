"""Pin-tests for the BP3 budget-accounting token-cost module.

`src/cfm/data/sub_f/token_cost.py` is the budget-accounting twin of the
runtime encoder. The BP3 budget tooling (alpha drop report, budget audit)
must count per-feature tokens the SAME way the encoder emits them, or the
budget and the emitter drift silently — exactly the failure mode that shipped
the T8.4 chunking miss and surfaced at T8.7 dispatch (L_infinity = 183m). See
``feedback_test_spec_not_just_plan`` and ``feedback_external_source_of_truth_gate``.

These tests pin ``token_cost`` output against the ENCODER's actual token count
(``encode_cell`` / ``encode_feature``), never against an independently-derived
formula. The encoder is the source of truth; ``token_cost`` is graded against
it. ``cell_edges={}`` forces Case A (no boundary refs) on every feature, which
is exactly the cost basis the budget surface is computed on.
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)

# Battery of geometries paired with a resolvable semantic tag. The per-feature
# token COUNT is independent of which semantic tag is used (always exactly one
# semantic token in Case A), so a road tag on a Point/Polygon is fine for a
# count pin — we are pinning length, not semantics.
_BATTERY = [
    # label, geom, tag
    (
        "linestring_short_no_chunk",
        LineString([(10.0, 20.0), (15.0, 25.0), (20.0, 30.0)]),
        "highway=residential",
    ),
    ("linestring_50m_two_chunks", LineString([(0.0, 0.0), (50.0, 0.0)]), "highway=residential"),
    ("linestring_100m_four_chunks", LineString([(0.0, 0.0), (100.0, 0.0)]), "highway=primary"),
    (
        "linestring_zero_length_segment",
        LineString([(0.0, 0.0), (0.0, 0.0), (5.0, 0.0)]),
        "highway=service",
    ),
    (
        "polygon_square_10m",
        Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]),
        "building=residential",
    ),
    ("point_poi", Point(12.0, 34.0), "highway=residential"),
    (
        "multilinestring_two_short_parts",
        MultiLineString(
            [
                [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
                [(100.0, 100.0), (105.0, 100.0)],
            ]
        ),
        "highway=residential",
    ),
    (
        "multilinestring_long_part_chunks",
        MultiLineString(
            [
                [(0.0, 0.0), (60.0, 0.0)],  # 60m -> 2 chunks
                [(0.0, 50.0), (10.0, 50.0)],  # 10m -> 1 chunk
            ]
        ),
        "highway=primary",
    ),
    (
        "multipolygon_two_squares",
        MultiPolygon(
            [
                Polygon([(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)]),
                Polygon([(20.0, 20.0), (25.0, 20.0), (25.0, 25.0), (20.0, 25.0), (20.0, 20.0)]),
            ]
        ),
        "building=residential",
    ),
]


@pytest.mark.parametrize("label,geom,tag", _BATTERY, ids=[b[0] for b in _BATTERY])
def test_feature_token_cost_matches_encoder_emission(label, geom, tag):
    """feature_token_cost(geom) equals the encoder's actual Case-A token count.

    This is the external-source-of-truth pin: the budget twin is graded against
    encode_cell's real output, including the per-part split of Multi* geometries
    (each part is its own <feature>..<feature_end> in encode_cell).
    """
    from cfm.data.sub_f.encoder import encode_cell
    from cfm.data.sub_f.token_cost import feature_token_cost

    emitted = encode_cell(features=[(geom, tag)], cell_edges={})
    assert feature_token_cost(geom) == len(emitted.tokens), (
        f"{label}: budget twin {feature_token_cost(geom)} != encoder emission {len(emitted.tokens)}"
    )


def test_chunked_per_feature_tokens_matches_encode_feature_case_a():
    """The per-coord-list function pins directly against encode_feature (Case A).

    Pins the lower-level function (operating on one coord list, the unit the
    encoder emits one feature for), not only the geom-level wrapper.
    """
    from cfm.data.sub_f.encoder import encode_feature
    from cfm.data.sub_f.token_cost import chunked_per_feature_tokens

    for geom, tag in [
        (LineString([(10.0, 20.0), (15.0, 25.0), (20.0, 30.0)]), "highway=residential"),
        (LineString([(0.0, 0.0), (50.0, 0.0)]), "highway=residential"),
        (
            Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)]),
            "building=residential",
        ),
    ]:
        emitted = encode_feature(geom, semantic_tag=tag)  # Case A: no brefs
        coords = list(geom.coords) if geom.geom_type == "LineString" else list(geom.exterior.coords)
        assert chunked_per_feature_tokens(coords) == len(emitted.tokens)


def test_feature_token_cost_equals_per_feature_for_single_part_geoms():
    """For single-part geoms, feature_token_cost == chunked_per_feature_tokens.

    The two public functions must agree where there is no multi-part split, so
    a caller can use either interchangeably for LineString/Polygon/Point.
    """
    from cfm.data.sub_f.token_cost import chunked_per_feature_tokens, feature_token_cost

    line = LineString([(0.0, 0.0), (50.0, 0.0), (60.0, 10.0)])
    assert feature_token_cost(line) == chunked_per_feature_tokens(list(line.coords))

    poly = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)])
    assert feature_token_cost(poly) == chunked_per_feature_tokens(list(poly.exterior.coords))


@pytest.mark.parametrize(
    "distance_m,expected_pairs",
    [
        (0.0, 1),  # zero-length floor: still one pair (vertex preserved)
        (5.0, 1),  # short
        (16.0, 1),  # 32 quanta -> 1 pair
        (32.0, 1),  # 64 quanta = chunk threshold -> exactly 1 pair
        (32.5, 2),  # 65 quanta -> just over threshold -> 2 pairs
        (64.0, 2),  # 128 quanta -> 2 pairs
        (96.0, 3),  # 192 quanta -> 3 pairs
        (100.0, 4),  # 200 quanta -> ceil(200/64) = 4 pairs
    ],
)
def test_chunked_segment_pairs_matches_encoder_pair_count(distance_m, expected_pairs):
    """chunked_segment_pairs equals encoder._direction_magnitude_pair pair count.

    Pinned against the encoder's actual emitted pair count (len // 2), not only
    the analytic ceil(L/32). The 32m row is the boundary: exactly one pair.
    """
    from cfm.data.sub_f.encoder import _direction_magnitude_pair
    from cfm.data.sub_f.token_cost import chunked_segment_pairs

    emitted_pairs = len(_direction_magnitude_pair(distance_m, 0.0)) // 2
    assert chunked_segment_pairs(distance_m) == emitted_pairs == expected_pairs


def test_max_magnitude_quantum_does_not_drift_from_encoder():
    """token_cost's chunk size is derived from the same locked constants the
    encoder uses, so the two cannot drift apart silently.
    """
    from cfm.data.sub_f import encoder, token_cost

    assert token_cost._MAX_MAGNITUDE_Q == encoder._MAX_MAGNITUDE_Q == 64


def test_point_and_degenerate_cost_is_structural_plus_anchor():
    """A Point (or <2-vertex coord list) costs exactly 3 structural + n_anchor."""
    from cfm.data.sub_f.encoder import DEFAULT_N_ANCHOR_TOKENS
    from cfm.data.sub_f.token_cost import chunked_per_feature_tokens, feature_token_cost

    assert chunked_per_feature_tokens([(5.0, 5.0)]) == 3 + DEFAULT_N_ANCHOR_TOKENS
    assert feature_token_cost(Point(5.0, 5.0)) == 3 + DEFAULT_N_ANCHOR_TOKENS


def test_long_segment_cost_uses_quantized_distance_like_encoder():
    """sanity: 100m single segment -> 4 chunk pairs -> 3 + n_anchor + 8 tokens."""
    from cfm.data.sub_f.encoder import DEFAULT_N_ANCHOR_TOKENS
    from cfm.data.sub_f.token_cost import chunked_per_feature_tokens

    coords = [(0.0, 0.0), (100.0, 0.0)]
    expected = 3 + DEFAULT_N_ANCHOR_TOKENS + 2 * math.ceil(round(100.0 / 0.5) / 64)
    assert chunked_per_feature_tokens(coords) == expected

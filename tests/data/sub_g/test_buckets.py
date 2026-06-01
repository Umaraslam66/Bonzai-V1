from __future__ import annotations

import pytest

from cfm.data.sub_g.buckets import bucket_of, load_density_edges, load_road_skeleton_edges


def test_density_edges_match_locked_vocab():
    # macro_plan_vocab.yaml locked_buckets.cell_density (lines 3472-3488) — verbatim cut-points.
    assert load_density_edges() == [0.0, 0.05, 0.15, 0.35]


def test_road_skeleton_edges_match_locked_vocab():
    # macro_plan_vocab.yaml locked_buckets.road_skeleton (lines 3489-3505)
    assert load_road_skeleton_edges() == [0, 1, 4, 9]


@pytest.mark.parametrize(
    "value,expected",
    [(0.0, 0), (0.049, 0), (0.05, 1), (0.15, 2), (0.349, 2), (0.35, 3), (10.0, 3)],
)
def test_bucket_of_lower_inclusive_upper_exclusive(value, expected):
    assert bucket_of(value, [0.0, 0.05, 0.15, 0.35]) == expected

"""Per-feature geometry-realism KS distance (Phase-2 bake-off Task 2) +
building-ring promotion inside ``feature_samples`` (readiness-closure Task 26
step-1 (d)): the sealed decoder emits building closed rings as LineString BY
CONTRACT, so a non-promoting ``feature_samples`` silently reads 0 building
areas on real decoded data. The helper now promotes INTERNALLY via the ONE
construction-identity authority (``promote_building_rings``) when ``blocks``
are given, and is LOUD — never a silent skip — when a closed-ring LineString
appears with no blocks to disambiguate it (building ring vs road roundabout)."""

from __future__ import annotations

import pytest

from cfm.eval.emergence import building_token_ids
from cfm.eval.realism import FeatureMetric, feature_samples, ks_distance

_FEATURE = 509  # <feature> marker
_BUILDING = min(building_token_ids())  # a real building-class token id
_ROAD = 511  # direction-range id; NOT in building_token_ids (asserted below)
_CLOSED_RING = {"type": "LineString", "coordinates": [[0, 0], [0, 2], [2, 2], [2, 0], [0, 0]]}


def test_building_area_samples_extracted_from_polygon_geoms() -> None:
    geoms = [
        {"type": "Polygon", "coordinates": [[[0, 0], [0, 2], [2, 2], [2, 0], [0, 0]]]},  # area 4
        {"type": "LineString", "coordinates": [[0, 0], [3, 0]]},  # not a building
    ]
    assert feature_samples(geoms, metric=FeatureMetric.BUILDING_AREA) == [4.0]


def test_road_length_samples_extracted_from_linestring_geoms() -> None:
    geoms = [{"type": "LineString", "coordinates": [[0, 0], [3, 0], [3, 4]]}]  # 3 + 4 = 7
    assert feature_samples(geoms, metric=FeatureMetric.ROAD_LENGTH) == [7.0]


# --- Task 26 (d): promotion is INTERNAL; the building-ring regime is never silent ---


def test_building_closed_ring_with_blocks_yields_a_polygon_sample() -> None:
    """THE (d) TOOTH (red on the non-promoting helper): a building-class block
    whose decoded ring closes is PROMOTED internally and sampled as area 4.0 —
    not skipped as a LineString."""
    blocks = [[_FEATURE, _BUILDING, 1, 2]]
    out = feature_samples([dict(_CLOSED_RING)], metric=FeatureMetric.BUILDING_AREA, blocks=blocks)
    assert out == [4.0]


def test_closed_road_ring_with_blocks_stays_a_road_sample() -> None:
    """Regime-distinguishing twin: the identical closed ring on a ROAD block (a
    roundabout) is NOT promoted — it samples as road length 8.0 and contributes
    NO building area (construction identity, never bare ring-closure)."""
    assert _ROAD not in building_token_ids()
    blocks = [[_FEATURE, _ROAD, 1, 2]]
    assert (
        feature_samples([dict(_CLOSED_RING)], metric=FeatureMetric.BUILDING_AREA, blocks=blocks)
        == []
    )
    assert feature_samples(
        [dict(_CLOSED_RING)], metric=FeatureMetric.ROAD_LENGTH, blocks=blocks
    ) == [8.0]


@pytest.mark.parametrize("metric", [FeatureMetric.BUILDING_AREA, FeatureMetric.ROAD_LENGTH])
def test_closed_ring_without_blocks_is_loud_never_a_silent_skip(metric: FeatureMetric) -> None:
    """Without blocks a closed-ring LineString is AMBIGUOUS (unpromoted building
    ring vs road roundabout): under BUILDING_AREA the old helper silently
    dropped it; under ROAD_LENGTH it would silently COUNT a building as a road.
    Both regimes refuse, telling the caller to pass blocks."""
    with pytest.raises(ValueError, match="blocks"):
        feature_samples([dict(_CLOSED_RING)], metric=metric)


def test_open_lines_and_polygons_need_no_blocks() -> None:
    """No-ambiguity regimes are untouched: open LineStrings and Polygons sample
    exactly as before without blocks (the legacy callers' shapes)."""
    open_line = {"type": "LineString", "coordinates": [[0, 0], [3, 0], [3, 4]]}
    poly = {"type": "Polygon", "coordinates": [[[0, 0], [0, 2], [2, 2], [2, 0], [0, 0]]]}
    assert feature_samples([open_line, poly], metric=FeatureMetric.ROAD_LENGTH) == [7.0]
    assert feature_samples([open_line, poly], metric=FeatureMetric.BUILDING_AREA) == [4.0]


def test_promotion_uses_the_one_authority_by_identity() -> None:
    """One source: realism promotes via cfm.eval.geometry.promote_building_rings
    (no second promotion implementation to drift)."""
    import cfm.eval.geometry as g
    import cfm.eval.realism as r

    assert r.promote_building_rings is g.promote_building_rings


def test_ks_distance_is_zero_for_identical_distributions_and_grows_with_divergence() -> None:
    a = [1.0, 2.0, 3.0, 4.0]
    assert ks_distance(a, a) == 0.0
    assert ks_distance(a, [10.0, 20.0, 30.0, 40.0]) > ks_distance(a, [1.1, 2.1, 3.1, 4.1])


def test_ks_distance_of_disjoint_distributions_is_one() -> None:
    assert ks_distance([1.0, 2.0], [10.0, 20.0]) == 1.0


def test_ks_distance_empty_sample_is_maximally_far() -> None:
    assert ks_distance([], [1.0, 2.0]) == 1.0
    assert ks_distance([1.0], []) == 1.0

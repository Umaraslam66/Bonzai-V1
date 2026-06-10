"""Per-feature geometry-realism KS distance (Phase-2 bake-off Task 2)."""

from __future__ import annotations

from cfm.eval.realism import FeatureMetric, feature_samples, ks_distance


def test_building_area_samples_extracted_from_polygon_geoms() -> None:
    geoms = [
        {"type": "Polygon", "coordinates": [[[0, 0], [0, 2], [2, 2], [2, 0], [0, 0]]]},  # area 4
        {"type": "LineString", "coordinates": [[0, 0], [3, 0]]},  # not a building
    ]
    assert feature_samples(geoms, metric=FeatureMetric.BUILDING_AREA) == [4.0]


def test_road_length_samples_extracted_from_linestring_geoms() -> None:
    geoms = [{"type": "LineString", "coordinates": [[0, 0], [3, 0], [3, 4]]}]  # 3 + 4 = 7
    assert feature_samples(geoms, metric=FeatureMetric.ROAD_LENGTH) == [7.0]


def test_ks_distance_is_zero_for_identical_distributions_and_grows_with_divergence() -> None:
    a = [1.0, 2.0, 3.0, 4.0]
    assert ks_distance(a, a) == 0.0
    assert ks_distance(a, [10.0, 20.0, 30.0, 40.0]) > ks_distance(a, [1.1, 2.1, 3.1, 4.1])


def test_ks_distance_of_disjoint_distributions_is_one() -> None:
    assert ks_distance([1.0, 2.0], [10.0, 20.0]) == 1.0


def test_ks_distance_empty_sample_is_maximally_far() -> None:
    assert ks_distance([], [1.0, 2.0]) == 1.0
    assert ks_distance([1.0], []) == 1.0

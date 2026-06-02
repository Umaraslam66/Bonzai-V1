"""Tests for the building-emergence instrumentation (Phase-2 bake-off Task 1)."""

from __future__ import annotations

import pytest

from cfm.data.sub_f.vocab import semantic_tag_to_l1_key, vocab_tag_to_id
from cfm.eval.emergence import (
    BUILDING_L1_KEY,
    building_token_ids,
    buildings_emerged,
    emergence_floor_polygons_per_cell,
    sequence_has_building_tokens,
)


def test_building_token_ids_are_the_building_l1_key_tags() -> None:
    vocab = vocab_tag_to_id()
    expected = {i for tag, i in vocab.items() if semantic_tag_to_l1_key(tag) == BUILDING_L1_KEY}
    assert building_token_ids() == expected
    assert len(expected) == 77  # verified count against the sealed sub-F vocab


def test_building_token_ids_do_not_overlap_road_tokens() -> None:
    road_ids = {i for t, i in vocab_tag_to_id().items() if semantic_tag_to_l1_key(t) == "highway"}
    assert building_token_ids().isdisjoint(road_ids)


def test_sequence_with_a_building_token_is_detected() -> None:
    a_building_id = min(building_token_ids())
    assert sequence_has_building_tokens([1, 2, a_building_id, 3]) is True


def test_sequence_without_building_tokens_is_not_detected() -> None:
    non_building = sorted(set(range(686)) - building_token_ids())[:5]
    assert sequence_has_building_tokens(non_building) is False


def test_floor_is_a_fraction_of_holdout_density_not_an_absolute() -> None:
    assert emergence_floor_polygons_per_cell(holdout_polys_per_cell=4.0, frac=0.25) == 1.0


def test_buildings_emerged_requires_meeting_the_density_floor_not_one_stray() -> None:
    # 1 polygon across 100 cells = 0.01/cell -> below a 1.0 floor -> NOT emerged
    assert buildings_emerged(n_polygons=1, n_cells=100, floor_per_cell=1.0) is False
    # 120 polygons across 100 cells = 1.2/cell -> at/above floor -> emerged
    assert buildings_emerged(n_polygons=120, n_cells=100, floor_per_cell=1.0) is True


def test_buildings_emerged_handles_zero_cells() -> None:
    assert buildings_emerged(n_polygons=10, n_cells=0, floor_per_cell=1.0) is False


@pytest.mark.slow
def test_holdout_density_is_measured_from_real_roundtripped_geoms() -> None:
    # Real frozen holdout (Leonardo $WORK): round-trip real cells -> polygons / active cells.
    from cfm.eval.emergence import holdout_polygons_per_active_cell

    density = holdout_polygons_per_active_cell(release="2026-04-15.0", region="singapore")
    assert density > 0.0  # dense urban Singapore has buildings

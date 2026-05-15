from __future__ import annotations

import json
from pathlib import Path

import pytest

from cfm.tokenizer import (
    FeatureOutOfBounds,
    UnsupportedFeatureClass,
    UnsupportedGeometry,
    Vocabulary,
    encode_cell,
)


@pytest.fixture(scope="module")
def vocab(vocab_yaml_path: Path) -> Vocabulary:
    return Vocabulary.load(vocab_yaml_path)


def _load(fixtures_dir: Path, name: str) -> dict:
    with (fixtures_dir / "degenerate" / name).open() as f:
        return json.load(f)


@pytest.mark.parametrize(
    ("fixture_name", "expected_error"),
    [
        ("non_rectangular_building.geojson", UnsupportedGeometry),
        ("unknown_class.geojson", UnsupportedFeatureClass),
        ("out_of_bounds.geojson", FeatureOutOfBounds),
    ],
)
def test_degenerate_fixtures_raise_specific_error(
    fixture_name: str,
    expected_error: type[Exception],
    vocab: Vocabulary,
    fixtures_dir: Path,
) -> None:
    geo = _load(fixtures_dir, fixture_name)
    with pytest.raises(expected_error):
        encode_cell(geo, cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)

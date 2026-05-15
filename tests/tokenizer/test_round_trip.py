from __future__ import annotations

import json
from pathlib import Path

import pytest

from cfm.tokenizer import (
    Vocabulary,
    decode_cell,
    encode_cell,
    geometric_equal,
)


@pytest.fixture(scope="module")
def vocab(vocab_yaml_path: Path) -> Vocabulary:
    return Vocabulary.load(vocab_yaml_path)


def _load_fixture(fixtures_dir: Path) -> dict:
    with (fixtures_dir / "single_cell" / "input.geojson").open() as f:
        return json.load(f)


def test_single_cell_fixture_round_trips(vocab: Vocabulary, fixtures_dir: Path) -> None:
    original = _load_fixture(fixtures_dir)
    encoded = encode_cell(original, cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    decoded = decode_cell(encoded, vocab=vocab)
    assert geometric_equal(original, decoded, tol_m=0.5)


def test_round_trip_preserves_feature_count(vocab: Vocabulary, fixtures_dir: Path) -> None:
    original = _load_fixture(fixtures_dir)
    encoded = encode_cell(original, cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    decoded = decode_cell(encoded, vocab=vocab)
    assert len(decoded["features"]) == len(original["features"])


def test_round_trip_preserves_classes(vocab: Vocabulary, fixtures_dir: Path) -> None:
    original = _load_fixture(fixtures_dir)
    encoded = encode_cell(original, cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    decoded = decode_cell(encoded, vocab=vocab)
    orig_classes = sorted(f["properties"]["class"] for f in original["features"])
    decoded_classes = sorted(f["properties"]["class"] for f in decoded["features"])
    assert orig_classes == decoded_classes

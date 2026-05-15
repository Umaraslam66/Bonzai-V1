from __future__ import annotations

from pathlib import Path

import pytest

from cfm.tokenizer import (
    CellTokens,
    Vocabulary,
    VocabularyMismatch,
)
from cfm.tokenizer.decode import decode_cell
from cfm.tokenizer.encode import encode_cell


@pytest.fixture(scope="module")
def vocab(vocab_yaml_path: Path) -> Vocabulary:
    return Vocabulary.load(vocab_yaml_path)


def _fc(*features: dict) -> dict:
    return {"type": "FeatureCollection", "features": list(features)}


def test_decode_empty_returns_empty_collection(vocab: Vocabulary) -> None:
    encoded = encode_cell(_fc(), cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    decoded = decode_cell(encoded, vocab=vocab)
    assert decoded == {"type": "FeatureCollection", "features": []}


def test_decode_point(vocab: Vocabulary) -> None:
    feat = {
        "type": "Feature",
        "properties": {"class": "POI_restaurant"},
        "geometry": {"type": "Point", "coordinates": [50, 80]},
    }
    encoded = encode_cell(_fc(feat), cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    decoded = decode_cell(encoded, vocab=vocab)
    assert decoded["features"][0]["properties"]["class"] == "POI_restaurant"
    assert decoded["features"][0]["geometry"]["type"] == "Point"
    assert decoded["features"][0]["geometry"]["coordinates"] == [50.0, 80.0]


def test_unknown_token_id_raises(vocab: Vocabulary) -> None:
    bad = CellTokens(tokens=(999999,), cell_origin=(0.0, 0.0), cell_size_m=250.0)
    with pytest.raises(VocabularyMismatch):
        decode_cell(bad, vocab=vocab)

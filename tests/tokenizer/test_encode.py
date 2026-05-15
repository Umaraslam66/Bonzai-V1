from __future__ import annotations

from pathlib import Path

import pytest

from cfm.tokenizer import (
    FeatureOutOfBounds,
    UnsupportedFeatureClass,
    Vocabulary,
)
from cfm.tokenizer.encode import CellTokens, encode_cell


@pytest.fixture(scope="module")
def vocab(vocab_yaml_path: Path) -> Vocabulary:
    return Vocabulary.load(vocab_yaml_path)


def _fc(*features: dict) -> dict:
    return {"type": "FeatureCollection", "features": list(features)}


def _poi(x: float, y: float, cls: str = "POI_restaurant") -> dict:
    return {
        "type": "Feature",
        "properties": {"class": cls},
        "geometry": {"type": "Point", "coordinates": [x, y]},
    }


def test_empty_collection_produces_bos_cell_endcell_eos(vocab: Vocabulary) -> None:
    out = encode_cell(_fc(), cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    assert isinstance(out, CellTokens)
    expected = (
        vocab.token_to_id["BOS"],
        vocab.token_to_id["CELL"],
        vocab.token_to_id["END_CELL"],
        vocab.token_to_id["EOS"],
    )
    assert out.tokens == expected
    assert out.cell_origin == (0.0, 0.0)
    assert out.cell_size_m == 250.0


def test_single_poi_encodes_as_anchor_pair(vocab: Vocabulary) -> None:
    out = encode_cell(_fc(_poi(50, 80)), cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    expected = (
        vocab.token_to_id["BOS"],
        vocab.token_to_id["CELL"],
        vocab.token_to_id["FEATURE_START"],
        vocab.token_to_id["POI_restaurant"],
        vocab.token_to_id["ANCHOR_X_50"],
        vocab.token_to_id["ANCHOR_Y_80"],
        vocab.token_to_id["FEATURE_END"],
        vocab.token_to_id["END_CELL"],
        vocab.token_to_id["EOS"],
    )
    assert out.tokens == expected


def test_poi_outside_cell_raises_out_of_bounds(vocab: Vocabulary) -> None:
    with pytest.raises(FeatureOutOfBounds):
        encode_cell(_fc(_poi(300, 300)), cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)


def test_unknown_class_raises(vocab: Vocabulary) -> None:
    with pytest.raises(UnsupportedFeatureClass):
        encode_cell(
            _fc(_poi(50, 80, cls="POI_castle")),
            cell_origin=(0.0, 0.0),
            cell_size_m=250.0,
            vocab=vocab,
        )

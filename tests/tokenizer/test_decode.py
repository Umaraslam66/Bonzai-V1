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


def test_decode_rectangle_building(vocab: Vocabulary) -> None:
    feat = {
        "type": "Feature",
        "properties": {"class": "B_residential"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[40, 40], [60, 40], [60, 60], [40, 60], [40, 40]]],
        },
    }
    encoded = encode_cell(_fc(feat), cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    decoded = decode_cell(encoded, vocab=vocab)
    geom = decoded["features"][0]["geometry"]
    assert geom["type"] == "Polygon"
    ring = geom["coordinates"][0]
    assert ring[0] == [40.0, 40.0]
    # GeoJSON-valid: ring is closed
    assert ring[0] == ring[-1]


def test_decode_road_with_exit(vocab: Vocabulary) -> None:
    feat = {
        "type": "Feature",
        "properties": {"class": "R_residential"},
        "geometry": {"type": "LineString", "coordinates": [[0, 125], [250, 125]]},
    }
    encoded = encode_cell(_fc(feat), cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    decoded = decode_cell(encoded, vocab=vocab)
    coords = decoded["features"][0]["geometry"]["coordinates"]
    assert coords[0] == [0.0, 125.0]
    assert coords[-1] == [250.0, 125.0]


def test_decode_unclosed_polygon_raises(vocab: Vocabulary) -> None:
    # Hand-craft a token sequence whose polygon moves don't close.
    t = vocab.token_to_id
    tokens = (
        t["BOS"],
        t["CELL"],
        t["FEATURE_START"],
        t["B_residential"],
        t["ANCHOR_X_40"],
        t["ANCHOR_Y_40"],
        t["MOVE_E_16"],  # cursor at (56, 40) — no closure
        t["FEATURE_END"],
        t["END_CELL"],
        t["EOS"],
    )
    bad = CellTokens(tokens=tokens, cell_origin=(0.0, 0.0), cell_size_m=250.0)
    with pytest.raises(Exception) as excinfo:
        decode_cell(bad, vocab=vocab)
    # Expect UnsupportedGeometry (not VocabularyMismatch) — the IDs are valid.
    from cfm.tokenizer.errors import UnsupportedGeometry

    assert isinstance(excinfo.value, UnsupportedGeometry)

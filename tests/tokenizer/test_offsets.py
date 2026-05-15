from __future__ import annotations

from pathlib import Path

import pytest

from cfm.tokenizer import (
    FeatureOutOfBounds,
    Vocabulary,
    VocabularyMismatch,
    decode_cell,
    encode_cell,
    geometric_equal,
)


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


def _rect(x0: float, y0: float, x1: float, y1: float, cls: str = "B_residential") -> dict:
    return {
        "type": "Feature",
        "properties": {"class": cls},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
        },
    }


def _line(coords: list[list[float]], cls: str = "R_residential") -> dict:
    return {
        "type": "Feature",
        "properties": {"class": cls},
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def test_poi_at_non_zero_origin_uses_local_anchor_tokens(vocab: Vocabulary) -> None:
    # World (1050, 2080) with origin (1000, 2000) -> local (50, 80).
    out = encode_cell(
        _fc(_poi(1050, 2080)),
        cell_origin=(1000.0, 2000.0),
        cell_size_m=250.0,
        vocab=vocab,
    )
    t = vocab.token_to_id
    assert t["ANCHOR_X_50"] in out.tokens
    assert t["ANCHOR_Y_80"] in out.tokens


def test_poi_round_trip_at_non_zero_origin(vocab: Vocabulary) -> None:
    original = _fc(_poi(1050, 2080))
    encoded = encode_cell(
        original,
        cell_origin=(1000.0, 2000.0),
        cell_size_m=250.0,
        vocab=vocab,
    )
    decoded = decode_cell(encoded, vocab=vocab)
    assert geometric_equal(original, decoded, tol_m=0.5)
    # And explicit world-coord assertion on the restored point.
    coords = decoded["features"][0]["geometry"]["coordinates"]
    assert coords == [1050.0, 2080.0]


def test_poi_outside_offset_cell_raises_out_of_bounds(vocab: Vocabulary) -> None:
    # World (900, 2080) with origin (1000, 2000) -> local x = -100, out of cell.
    with pytest.raises(FeatureOutOfBounds):
        encode_cell(
            _fc(_poi(900, 2080)),
            cell_origin=(1000.0, 2000.0),
            cell_size_m=250.0,
            vocab=vocab,
        )


def test_building_round_trip_at_non_zero_origin(vocab: Vocabulary) -> None:
    # 20m x 20m building at world (1040,2040)-(1060,2060).
    original = _fc(_rect(1040, 2040, 1060, 2060))
    encoded = encode_cell(
        original,
        cell_origin=(1000.0, 2000.0),
        cell_size_m=250.0,
        vocab=vocab,
    )
    decoded = decode_cell(encoded, vocab=vocab)
    assert geometric_equal(original, decoded, tol_m=0.5)


def test_road_exiting_east_edge_at_non_zero_origin_emits_exit(vocab: Vocabulary) -> None:
    # Road from world (1000, 2125) to (1250, 2125): local (0, 125) -> (250, 125), east exit.
    out = encode_cell(
        _fc(_line([[1000, 2125], [1250, 2125]])),
        cell_origin=(1000.0, 2000.0),
        cell_size_m=250.0,
        vocab=vocab,
    )
    assert vocab.token_to_id["EXIT"] in out.tokens


def test_oversized_cell_size_raises_vocabulary_mismatch(vocab: Vocabulary) -> None:
    # cell_size_m=500 exceeds vocab.anchor_axis_count=250; must raise a clean
    # VocabularyMismatch rather than a bare KeyError on a missing ANCHOR token.
    with pytest.raises(VocabularyMismatch):
        encode_cell(
            _fc(_poi(400, 400)),
            cell_origin=(0.0, 0.0),
            cell_size_m=500.0,
            vocab=vocab,
        )


def test_smaller_cell_size_bounds_check_scales(vocab: Vocabulary) -> None:
    # POI at (150, 50) is OOB for a 100m cell; verifies the bounds check scales
    # with cell_size_m (downward, since upward exceeds vocab anchor range).
    with pytest.raises(FeatureOutOfBounds):
        encode_cell(
            _fc(_poi(150, 50)),
            cell_origin=(0.0, 0.0),
            cell_size_m=100.0,
            vocab=vocab,
        )

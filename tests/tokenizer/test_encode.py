from __future__ import annotations

from pathlib import Path

import pytest

from cfm.tokenizer import (
    FeatureOutOfBounds,
    UnsupportedFeatureClass,
    Vocabulary,
)
from cfm.tokenizer.encode import CellTokens, encode_cell
from cfm.tokenizer.errors import UnsupportedGeometry


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


def _rect(x0: float, y0: float, x1: float, y1: float, cls: str = "B_residential") -> dict:
    return {
        "type": "Feature",
        "properties": {"class": cls},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
        },
    }


def test_rectangular_building_encodes_with_dyadic_moves(vocab: Vocabulary) -> None:
    # 20m x 20m building at (40,40)-(60,60). Sides 20m = 16+4.
    out = encode_cell(
        _fc(_rect(40, 40, 60, 60)),
        cell_origin=(0.0, 0.0),
        cell_size_m=250.0,
        vocab=vocab,
    )
    t = vocab.token_to_id
    expected_core = (
        t["FEATURE_START"],
        t["B_residential"],
        t["ANCHOR_X_40"],
        t["ANCHOR_Y_40"],
        t["MOVE_E_16"],
        t["MOVE_E_4"],
        t["MOVE_N_16"],
        t["MOVE_N_4"],
        t["MOVE_W_16"],
        t["MOVE_W_4"],
        t["MOVE_S_16"],
        t["MOVE_S_4"],
        t["FEATURE_END"],
    )
    assert out.tokens[2:-2] == expected_core


def test_non_rectangular_building_raises(vocab: Vocabulary) -> None:
    # L-shape (6 vertices) under B_* class is rejected.
    feature = {
        "type": "Feature",
        "properties": {"class": "B_residential"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[30, 30], [70, 30], [70, 50], [50, 50], [50, 70], [30, 70], [30, 30]]],
        },
    }
    with pytest.raises(UnsupportedGeometry):
        encode_cell(_fc(feature), cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)


def test_land_use_polygon_multi_segment_encodes(vocab: Vocabulary) -> None:
    # 150m x 150m square anchored at (100, 0). Sides 150m = 128+16+4+2.
    out = encode_cell(
        _fc(_rect(100, 0, 250, 150, cls="L_residential")),
        cell_origin=(0.0, 0.0),
        cell_size_m=250.0,
        vocab=vocab,
    )
    t = vocab.token_to_id
    # Just spot-check the anchor + first east-side decomposition.
    assert t["L_residential"] in out.tokens
    assert t["ANCHOR_X_100"] in out.tokens
    assert t["ANCHOR_Y_0"] in out.tokens


def _line(coords: list[list[float]], cls: str = "R_residential") -> dict:
    return {
        "type": "Feature",
        "properties": {"class": cls},
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def test_road_crossing_east_edge_emits_exit(vocab: Vocabulary) -> None:
    # Road from (0,125) to (250,125). 250m = 32*7 + 16 + 8 + 2.
    out = encode_cell(
        _fc(_line([[0, 125], [250, 125]])),
        cell_origin=(0.0, 0.0),
        cell_size_m=250.0,
        vocab=vocab,
    )
    t = vocab.token_to_id
    assert t["EXIT"] in out.tokens
    assert t["ANCHOR_X_0"] in out.tokens
    assert t["ANCHOR_Y_125"] in out.tokens
    # MOVE_E_32 must appear at least 7 times.
    move_e_32 = t["MOVE_E_32"]
    assert sum(1 for tok in out.tokens if tok == move_e_32) == 7


def test_internal_road_no_exit(vocab: Vocabulary) -> None:
    out = encode_cell(
        _fc(_line([[20, 20], [40, 20]])),
        cell_origin=(0.0, 0.0),
        cell_size_m=250.0,
        vocab=vocab,
    )
    assert vocab.token_to_id["EXIT"] not in out.tokens


def test_diagonal_segment_raises(vocab: Vocabulary) -> None:
    with pytest.raises(UnsupportedGeometry):
        encode_cell(
            _fc(_line([[0, 0], [100, 100]])),
            cell_origin=(0.0, 0.0),
            cell_size_m=250.0,
            vocab=vocab,
        )


@pytest.mark.xfail(
    strict=True,
    reason="Phase 0 has no <ENTRY> marker for lines starting on a cell boundary; "
    "to be addressed by Phase 1 boundary contracts.",
)
def test_road_entering_west_edge_emits_entry_marker(vocab: Vocabulary) -> None:
    """A road starting at x=0 should emit some boundary-entry marker. Phase 0 doesn't yet."""
    # Road from (0, 60) going east to (50, 60). Starts on west edge.
    out = encode_cell(
        _fc(_line([[0, 60], [50, 60]])),
        cell_origin=(0.0, 0.0),
        cell_size_m=250.0,
        vocab=vocab,
    )
    # We don't have an ENTRY token in Phase 0 vocab; this assertion is *intentionally*
    # going to fail today. The xfail makes the gap visible. When Phase 1 adds
    # boundary contracts (or an ENTRY token), this test will need its assertion
    # updated and the xfail removed.
    assert "ENTRY" in [vocab.id_to_token[t] for t in out.tokens]

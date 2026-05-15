from __future__ import annotations

import json
from pathlib import Path

import pytest

from cfm.tokenizer import Vocabulary, encode_cell


@pytest.fixture(scope="module")
def vocab(vocab_yaml_path: Path) -> Vocabulary:
    return Vocabulary.load(vocab_yaml_path)


def test_encoding_fixture_is_deterministic(vocab: Vocabulary, fixtures_dir: Path) -> None:
    with (fixtures_dir / "single_cell" / "input.geojson").open() as f:
        original = json.load(f)
    a = encode_cell(original, cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    b = encode_cell(original, cell_origin=(0.0, 0.0), cell_size_m=250.0, vocab=vocab)
    assert a.tokens == b.tokens
    assert a.cell_origin == b.cell_origin
    assert a.cell_size_m == b.cell_size_m

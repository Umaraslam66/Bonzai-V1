from __future__ import annotations

from pathlib import Path

import pytest

from cfm.tokenizer.vocabulary import Vocabulary


def test_load_phase0_total_count(vocab_yaml_path: Path) -> None:
    vocab = Vocabulary.load(vocab_yaml_path)
    assert len(vocab) == 582


def test_shape_tokens_present(vocab_yaml_path: Path) -> None:
    vocab = Vocabulary.load(vocab_yaml_path)
    assert "POINT" in vocab.token_to_id
    assert "LINE" in vocab.token_to_id
    assert "POLYGON" in vocab.token_to_id


def test_first_eight_ids_are_control_tokens(vocab_yaml_path: Path) -> None:
    vocab = Vocabulary.load(vocab_yaml_path)
    expected = ("PAD", "BOS", "EOS", "CELL", "END_CELL", "FEATURE_START", "FEATURE_END", "EXIT")
    assert vocab.id_to_token[:8] == expected


def test_anchor_tokens_present(vocab_yaml_path: Path) -> None:
    vocab = Vocabulary.load(vocab_yaml_path)
    assert "ANCHOR_X_0" in vocab.token_to_id
    assert "ANCHOR_X_249" in vocab.token_to_id
    assert "ANCHOR_Y_0" in vocab.token_to_id
    assert "ANCHOR_Y_249" in vocab.token_to_id
    assert "ANCHOR_X_250" not in vocab.token_to_id


def test_move_tokens_present(vocab_yaml_path: Path) -> None:
    vocab = Vocabulary.load(vocab_yaml_path)
    assert "MOVE_E_1" in vocab.token_to_id
    assert "MOVE_NW_32" in vocab.token_to_id
    assert "MOVE_E_3" not in vocab.token_to_id


def test_feature_class_tokens_present(vocab_yaml_path: Path) -> None:
    vocab = Vocabulary.load(vocab_yaml_path)
    for name in ("R_residential", "B_residential", "POI_restaurant", "L_park"):
        assert name in vocab.token_to_id


def test_load_is_deterministic(vocab_yaml_path: Path) -> None:
    a = Vocabulary.load(vocab_yaml_path)
    b = Vocabulary.load(vocab_yaml_path)
    assert a.id_to_token == b.id_to_token
    assert a.token_to_id == b.token_to_id


def test_token_to_id_is_inverse_of_id_to_token(vocab_yaml_path: Path) -> None:
    vocab = Vocabulary.load(vocab_yaml_path)
    for i, name in enumerate(vocab.id_to_token):
        assert vocab.token_to_id[name] == i


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Vocabulary.load(tmp_path / "does_not_exist.yaml")


def test_vocabulary_exposes_anchor_axis_count(vocab_yaml_path: Path) -> None:
    assert Vocabulary.load(vocab_yaml_path).anchor_axis_count == 250

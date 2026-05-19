from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cfm.tokenizer import LoaderError, Vocabulary


def _phase1_yaml_minimal() -> dict:
    """Minimal valid Phase 1 YAML for loader exercising."""
    return {
        "schema_version": "1.0",
        "phase": 1,
        "vocab_version": "1.0",
        "control": [
            "PAD",
            "BOS",
            "EOS",
            "CELL",
            "END_CELL",
            "FEATURE_START",
            "FEATURE_END",
            "EXIT",
            "POINT",
            "LINE",
            "POLYGON",
        ],
        "hierarchy": ["MACRO", "END_MACRO", "MICRO", "END_MICRO"],
        "feature_class": {
            "road": {
                "tokens": ["R_motorway", "R_primary"],
            },
            "building": {
                "tokens": ["B__UNK__", "B_residential", "B_commercial"],
            },
            "poi": {
                "tokens": ["POI__UNK__", "POI_restaurant"],
            },
            "base": {
                "tokens": ["BASE_water", "BASE_park"],
            },
        },
        "anchor": {"axis_count": 250},
        "move": {
            "directions": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
            "steps_m": [1, 2, 4, 8, 16, 32],
        },
    }


def _write_yaml(tmp_path: Path, data: dict, name: str = "vocab.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def test_phase1_yaml_loads_with_vocabulary_loader(tmp_path):
    p = _write_yaml(tmp_path, _phase1_yaml_minimal())
    vocab = Vocabulary.load(p)
    # Should include the new section tokens.
    assert "R_motorway" in vocab.token_to_id
    assert "B__UNK__" in vocab.token_to_id
    assert "POI_restaurant" in vocab.token_to_id
    assert "BASE_water" in vocab.token_to_id


def test_phase1_loader_skips_metadata_keys_in_feature_class(tmp_path):
    data = _phase1_yaml_minimal()
    data["feature_class"]["building"]["floor_strategy"] = "Moderate"
    data["feature_class"]["building"]["notes"] = "some notes"
    data["feature_class"]["building"]["is_provisional"] = True
    p = _write_yaml(tmp_path, data)
    vocab = Vocabulary.load(p)
    # Metadata keys are ignored; tokens flatten as before.
    assert "B_residential" in vocab.token_to_id
    assert "floor_strategy" not in vocab.token_to_id
    assert "notes" not in vocab.token_to_id


def test_phase1_loader_preserves_token_order_per_section(tmp_path):
    p = _write_yaml(tmp_path, _phase1_yaml_minimal())
    vocab = Vocabulary.load(p)
    # In _flatten order: control(11) + hierarchy(4) = ids 0..14, then road tokens.
    assert vocab.id_to_token[15] == "R_motorway"
    assert vocab.id_to_token[16] == "R_primary"
    assert vocab.id_to_token[17] == "B__UNK__"


def test_phase0_yaml_still_loads_unchanged():
    """Phase 0 uses flat lists under feature_class; loader must still work."""
    phase0 = Path("configs/tokenizer/vocab_phase0.yaml")
    vocab = Vocabulary.load(phase0)
    # Phase 0 fixed-shape sanity checks.
    assert "B_residential" in vocab.token_to_id
    assert "R_motorway" in vocab.token_to_id
    assert vocab.anchor_axis_count == 250


def test_phase1_token_to_id_unique(tmp_path):
    p = _write_yaml(tmp_path, _phase1_yaml_minimal())
    vocab = Vocabulary.load(p)
    # Length of token_to_id mapping equals length of id_to_token (no collisions).
    assert len(vocab.token_to_id) == len(vocab.id_to_token)


def test_loader_rejects_duplicate_token_names_across_sections(tmp_path):
    data = _phase1_yaml_minimal()
    data["feature_class"]["road"]["tokens"].append("POI_restaurant")  # collide with poi.
    p = _write_yaml(tmp_path, data)
    with pytest.raises(LoaderError, match="duplicate token name"):
        Vocabulary.load(p)


def test_loader_rejects_unknown_token_not_at_section_position_zero_when_section_has_one(tmp_path):
    data = _phase1_yaml_minimal()
    # Put B__UNK__ at position 1 (not 0) in the building section.
    data["feature_class"]["building"]["tokens"] = ["B_residential", "B__UNK__", "B_commercial"]
    p = _write_yaml(tmp_path, data)
    with pytest.raises(LoaderError, match="__UNK__.*position 0"):
        Vocabulary.load(p)


def test_loader_accepts_data_derived_unknown_token_anywhere(tmp_path):
    """Tokens with the bare suffix `_unknown` (data category values like Overture's
    transportation.class 'unknown') are not placeholders; they can appear at any
    position within their section. Only the reserved __UNK__ marker is constrained.
    """
    data = _phase1_yaml_minimal()
    # Mimic the real-world case: R_unknown is a data-derived category token,
    # not a placeholder, and sits somewhere in the middle of the road section.
    data["feature_class"]["road"]["tokens"] = [
        "R_motorway",
        "R_primary",
        "R_unknown",
        "R_secondary",
    ]
    p = _write_yaml(tmp_path, data)
    vocab = Vocabulary.load(p)
    assert "R_unknown" in vocab.token_to_id

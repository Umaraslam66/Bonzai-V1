from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
VOCAB_PATH = REPO_ROOT / "configs" / "macro_plan" / "v1" / "boundary_vocab.yaml"

LOCKED_SUB_D_VOCAB_SHA256 = (
    "0b2d9eb4c1253b12f2fe50b32ec459e8fc25cdeafa05f4dda0a47240c0c9a1fd"
)


def test_boundary_vocab_loads_with_expected_structure() -> None:
    data = yaml.safe_load(VOCAB_PATH.read_text())
    assert data["boundary_vocab_schema_version"] == "1.0"
    assert data["boundary_vocab_version"] == "1.0"
    assert data["boundary_derivation_version"] == "1.0"
    assert data["phase"] == 1
    assert data["append_only_within_phase"] is True


def test_boundary_vocab_has_exactly_four_tokens_in_canonical_order() -> None:
    data = yaml.safe_load(VOCAB_PATH.read_text())
    tokens = data["tokens"]
    assert len(tokens) == 4
    assert tokens[0] == {"id": 0, "name": "BOUNDARY_NOT_APPLICABLE"}
    assert tokens[1] == {"id": 1, "name": "NONE"}
    assert tokens[2] == {"id": 2, "name": "MAJOR_ROAD"}
    assert tokens[3] == {"id": 3, "name": "MINOR_ROAD"}


def test_class_grouping_map_covers_all_named_class_raw_values() -> None:
    data = yaml.safe_load(VOCAB_PATH.read_text())
    cgm = data["class_grouping_map"]
    assert set(cgm["MAJOR_ROAD"]) == {"primary", "trunk", "secondary"}
    assert set(cgm["MINOR_ROAD"]) == {
        "tertiary",
        "residential",
        "service",
        "unclassified",
        "footway",
        "steps",
        "cycleway",
    }


def test_boundary_vocab_inherits_scope_from_locked_sub_d_artifact() -> None:
    data = yaml.safe_load(VOCAB_PATH.read_text())
    inheritance = data["scope_vocab_inherited_from"]
    assert inheritance["artifact"] == "configs/macro_plan/v1/macro_plan_vocab.yaml"
    assert inheritance["artifact_sha256"] == LOCKED_SUB_D_VOCAB_SHA256
    assert inheritance["block"] == "scope.tokens"

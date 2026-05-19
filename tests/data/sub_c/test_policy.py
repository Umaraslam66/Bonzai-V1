"""Tests for apply_missing_value_policy (spec §10.1 + §10.2 four-case schema).

9 named tests, all using synthetic pyarrow tables (no sub-A cache).
Tests 5-8 use the real configs YAML files.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest
import yaml

from cfm.data.sub_c.errors import PolicyError
from cfm.data.sub_c.policy import apply_missing_value_policy

# ---------------------------------------------------------------------------
# Paths to real config files (used for tests 5-8)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parents[3]
_POLICY_YAML = _REPO_ROOT / "configs" / "data" / "missing_value_policy.yaml"
_VOCAB_YAML = _REPO_ROOT / "configs" / "tokenizer" / "vocab_phase1.yaml"


# ---------------------------------------------------------------------------
# Helper builders for synthetic tables
# ---------------------------------------------------------------------------


def _make_buildings_table(classes: list[str | None]) -> pa.Table:
    """Minimal buildings table with just the 'class' column."""
    return pa.table({"class": pa.array(classes, type=pa.string())})


def _make_transportation_table(classes: list[str | None]) -> pa.Table:
    return pa.table({"class": pa.array(classes, type=pa.string())})


def _make_base_table(classes: list[str | None]) -> pa.Table:
    return pa.table({"class": pa.array(classes, type=pa.string())})


def _make_places_table(primaries: list[str | None]) -> pa.Table:
    """Minimal places table with categories struct{primary, alternate}."""
    n = len(primaries)
    primary_arr = pa.array(primaries, type=pa.string())
    alternate_arr = pa.array([[] for _ in range(n)], type=pa.list_(pa.string()))
    categories_arr = pa.StructArray.from_arrays(
        [primary_arr, alternate_arr],
        names=["primary", "alternate"],
    )
    return pa.table({"categories": categories_arr})


# ---------------------------------------------------------------------------
# Test 1: signature enforces non-mutation (returns NEW dict)
# ---------------------------------------------------------------------------


def test_apply_missing_value_policy_returns_new_themes_dict_signature_enforced_non_mutation(
    tmp_path: Path,
) -> None:
    """The function returns a NEW dict; original themes dict is unchanged."""
    # Minimal policy YAML with n_a for everything (no mutations)
    policy = {
        "fields": {
            "buildings.class": {
                "policies": {
                    "missing_value": {"type": "n_a"},
                    "not_in_vocab": {"type": "n_a"},
                }
            }
        }
    }
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.dump(policy))

    original_table = _make_buildings_table(["residential"])
    themes_in = {"buildings": original_table}

    result = apply_missing_value_policy(themes_in, policy_path)

    # Must return a NEW dict object
    assert result is not themes_in
    # The original dict must be unmodified
    assert themes_in["buildings"] is original_table
    # The result contains an entry for buildings
    assert "buildings" in result


# ---------------------------------------------------------------------------
# Test 2: drop_row for transportation missing class
# ---------------------------------------------------------------------------


def test_apply_missing_value_policy_drops_transportation_null_class_rows(
    tmp_path: Path,
) -> None:
    """transportation.class missing_value=drop_row → NULL rows are removed."""
    policy = {
        "fields": {
            "transportation.class": {
                "policies": {
                    "missing_value": {"type": "drop_row"},
                    "not_in_vocab": {"type": "n_a"},
                }
            }
        }
    }
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.dump(policy))

    themes = {"transportation": _make_transportation_table(["residential", None, "primary"])}
    result = apply_missing_value_policy(themes, policy_path)

    out = result["transportation"]
    # NULL row dropped; 2 rows remain
    assert out.num_rows == 2
    classes = out.column("class").to_pylist()
    assert None not in classes
    assert "residential" in classes
    assert "primary" in classes


# ---------------------------------------------------------------------------
# Test 3: emit_unknown_token for buildings.class NULL
# ---------------------------------------------------------------------------


def test_apply_missing_value_policy_assigns_b_unk_to_null_buildings_class(
    tmp_path: Path,
) -> None:
    """buildings.class missing_value=emit_unknown_token → NULL → B__UNK__."""
    policy = {
        "fields": {
            "buildings.class": {
                "policies": {
                    "missing_value": {"type": "emit_unknown_token"},
                    "not_in_vocab": {"type": "emit_unknown_token"},
                }
            }
        }
    }
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.dump(policy))

    themes = {"buildings": _make_buildings_table([None, "residential", None])}
    result = apply_missing_value_policy(themes, policy_path)

    classes = result["buildings"].column("class").to_pylist()
    assert classes == ["B__UNK__", "residential", "B__UNK__"]


# ---------------------------------------------------------------------------
# Test 4: emit_unknown_token for places.categories.primary NULL
# ---------------------------------------------------------------------------


def test_apply_missing_value_policy_assigns_poi_unk_to_null_places_primary(
    tmp_path: Path,
) -> None:
    """places.categories.primary missing_value=emit_unknown_token → NULL → POI__UNK__."""
    policy = {
        "fields": {
            "places.categories.primary": {
                "policies": {
                    "missing_value": {"type": "emit_unknown_token"},
                    "not_in_vocab": {"type": "emit_unknown_token"},
                }
            },
            "places.categories.alternate": {
                "policies": {
                    "missing_value": {"type": "n_a"},
                    "not_in_vocab": {"type": "drop_element"},
                }
            },
        }
    }
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.dump(policy))

    themes = {"places": _make_places_table([None, "restaurant", None])}
    result = apply_missing_value_policy(themes, policy_path)

    cats = result["places"].column("categories")
    primaries = [cats[i].as_py()["primary"] for i in range(cats.__len__())]
    assert primaries == ["POI__UNK__", "restaurant", "POI__UNK__"]


# ---------------------------------------------------------------------------
# Test 5: sea-defining base rows dropped from features (real YAML files)
# ---------------------------------------------------------------------------


def test_apply_missing_value_policy_drops_sea_defining_base_rows_from_features() -> None:
    """base.class not_in_vocab=drop_row → ocean row dropped; river row kept.

    Uses real configs YAML files. This verifies §9.1 + §10.2: sea-defining rows
    (ocean, etc.) land below the Strict-300 floor and are correctly dropped from
    feature emission. The sea-mask view separately sources them pre-policy.
    """
    themes = {
        "buildings": _make_buildings_table([]),
        "transportation": _make_transportation_table([]),
        "places": _make_places_table([]),
        "base": _make_base_table(["ocean", "river"]),
    }
    result = apply_missing_value_policy(themes, _POLICY_YAML, vocab_yaml_path=_VOCAB_YAML)

    # "river" is in vocab (BASE_river); "ocean" is not → dropped
    out_classes = result["base"].column("class").to_pylist()
    assert "ocean" not in out_classes
    assert "river" in out_classes
    assert result["base"].num_rows == 1


# ---------------------------------------------------------------------------
# Test 6: not_in_vocab buildings.class → row KEPT, class column UNCHANGED
# ---------------------------------------------------------------------------


def test_not_in_vocab_buildings_class_stored_as_class_raw() -> None:
    """buildings.class not_in_vocab=emit_unknown_token → rare class stored raw.

    Sub-C does NOT drop or remap not-in-vocab buildings classes. The raw value
    survives into features.parquet; the tokenizer maps it to B__UNK__ at encode.
    """
    rare = "some_rare_class_definitely_not_in_phase_1_vocab"
    themes = {
        "buildings": _make_buildings_table([rare, "residential"]),
        "transportation": _make_transportation_table([]),
        "places": _make_places_table([]),
        "base": _make_base_table([]),
    }
    result = apply_missing_value_policy(themes, _POLICY_YAML, vocab_yaml_path=_VOCAB_YAML)

    out = result["buildings"]
    # Row is KEPT (not dropped)
    assert out.num_rows == 2
    classes = out.column("class").to_pylist()
    # Class is UNCHANGED (raw value preserved)
    assert rare in classes
    assert "residential" in classes


# ---------------------------------------------------------------------------
# Test 7: not_in_vocab transportation.class → row DROPPED (symmetric extension)
# ---------------------------------------------------------------------------


def test_not_in_vocab_transportation_class_dropped_symmetric_extension() -> None:
    """transportation.class not_in_vocab=drop_row → rare class row dropped.

    Symmetric extension of the NULL drop policy to not-in-vocab (spec §10.2).
    """
    rare = "some_rare_class_definitely_not_in_phase_1_vocab"
    themes = {
        "buildings": _make_buildings_table([]),
        "transportation": _make_transportation_table([rare, "residential"]),
        "places": _make_places_table([]),
        "base": _make_base_table([]),
    }
    result = apply_missing_value_policy(themes, _POLICY_YAML, vocab_yaml_path=_VOCAB_YAML)

    out = result["transportation"]
    classes = out.column("class").to_pylist()
    # Rare class row is DROPPED
    assert rare not in classes
    # In-vocab class is kept
    assert "residential" in classes
    assert out.num_rows == 1


# ---------------------------------------------------------------------------
# Test 8: not_in_vocab base.class → row DROPPED (Strict-300 floor decision)
# ---------------------------------------------------------------------------


def test_not_in_vocab_base_class_dropped_per_strict_decision() -> None:
    """base.class not_in_vocab=drop_row → out-of-vocab class row dropped.

    Strict-300 floor explicit decision (spec §10.2); ~4.69% of SG base rows.
    """
    rare = "some_class_not_in_phase_1_vocab"
    themes = {
        "buildings": _make_buildings_table([]),
        "transportation": _make_transportation_table([]),
        "places": _make_places_table([]),
        "base": _make_base_table([rare, "river"]),
    }
    result = apply_missing_value_policy(themes, _POLICY_YAML, vocab_yaml_path=_VOCAB_YAML)

    out = result["base"]
    classes = out.column("class").to_pylist()
    assert rare not in classes
    assert "river" in classes
    assert out.num_rows == 1


# ---------------------------------------------------------------------------
# Test 9: unknown policy type → PolicyError
# ---------------------------------------------------------------------------


def test_apply_missing_value_policy_raises_policy_error_on_unknown_policy_type(
    tmp_path: Path,
) -> None:
    """Unknown missing_value type → PolicyError with closed handler-map."""
    policy = {
        "fields": {
            "buildings.class": {
                "policies": {
                    "missing_value": {"type": "bogus"},
                    "not_in_vocab": {"type": "n_a"},
                }
            }
        }
    }
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.dump(policy))

    themes = {"buildings": _make_buildings_table(["residential"])}

    with pytest.raises(PolicyError):
        apply_missing_value_policy(themes, policy_path)

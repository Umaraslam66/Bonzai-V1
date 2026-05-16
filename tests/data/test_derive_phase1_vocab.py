from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "derive_phase1_vocab.py"


def _run_script(output_dir: Path, *, rerun_reason: str = "initial") -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir", str(output_dir),
            "--rerun-reason", rerun_reason,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def _load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text())


def test_script_runs_against_cached_singapore_and_produces_well_formed_artifacts(tmp_path):
    _run_script(tmp_path)
    vocab_path = tmp_path / "configs" / "tokenizer" / "vocab_phase1.yaml"
    policy_path = tmp_path / "configs" / "data" / "missing_value_policy.yaml"
    assert vocab_path.exists()
    assert policy_path.exists()

    vocab = _load_yaml(vocab_path)

    # Top-level version fields.
    assert vocab["schema_version"] == "1.0"
    assert vocab["phase"] == 1
    assert vocab["vocab_version"] == "1.0"
    assert len(vocab["vocab_sha256"]) == 64
    assert vocab["generated_from"]["overture_release"] == "2026-04-15.0"
    assert vocab["generated_from"]["regions"] == ["singapore"]

    # Feature class sections, in expected order.
    fc = vocab["feature_class"]
    assert set(fc.keys()) == {"road", "building", "poi", "base"}

    # Pinned counts against the locked B2 decisions (§7 of the spec).
    # Investigate-first if any of these change.
    assert len(fc["building"]["tokens"]) == 23   # 22 Moderate-kept + B_unknown
    assert len(fc["road"]["tokens"]) == 17       # 17 Moderate-kept (includes R_unknown as data value, not placeholder)
    assert len(fc["base"]["tokens"]) == 7        # 7 Strict-kept, no unknown
    poi_len = len(fc["poi"]["tokens"])
    assert 291 <= poi_len <= 341, (
        f"POI section size {poi_len} outside expected [291, 341]; "
        "investigate union-size shift before re-pinning."
    )

    # Building section: B_unknown at position 0 (placeholder for missing values).
    assert fc["building"]["tokens"][0] == "B_unknown"
    # POI section: POI_unknown at position 0 (placeholder for missing values).
    assert fc["poi"]["tokens"][0] == "POI_unknown"
    # Road: transportation.class has drop_row policy, so no placeholder unknown.
    # However, "unknown" is an actual data value in Overture (6,066 rows).
    assert "R_unknown" in fc["road"]["tokens"]
    # Base: base.class has n_a policy and is 100% present; no unknown.
    assert all(not t.endswith("_unknown") for t in fc["base"]["tokens"])


def test_policy_yaml_field_set_matches_expected(tmp_path):
    _run_script(tmp_path)
    policy = _load_yaml(tmp_path / "configs" / "data" / "missing_value_policy.yaml")
    expected = {
        "buildings.class",
        "transportation.class",
        "base.class",
        "places.categories.primary",
        "places.categories.alternate",
    }
    actual = set(policy["fields"].keys())
    assert actual == expected, f"unexpected diff: added={actual-expected}, removed={expected-actual}"


def test_cross_artifact_consistency_unknown_tokens(tmp_path):
    """For every emit_unknown_token field, the vocab section has *_unknown at position 0.
    For every drop_row / n_a field, the section should not have a *_unknown PLACEHOLDER
    at position 0 (though data values named 'unknown' may appear elsewhere)."""
    _run_script(tmp_path)
    vocab = _load_yaml(tmp_path / "configs" / "tokenizer" / "vocab_phase1.yaml")
    policy = _load_yaml(tmp_path / "configs" / "data" / "missing_value_policy.yaml")

    # Map: source field name → vocab section name.
    field_to_section = {
        "buildings.class": "building",
        "transportation.class": "road",
        "base.class": "base",
        "places.categories.primary": "poi",
        "places.categories.alternate": "poi",
    }

    for field, entry in policy["fields"].items():
        section_name = field_to_section[field]
        section = vocab["feature_class"][section_name]
        mv = entry["policies"]["missing_value"]
        if mv["type"] == "emit_unknown_token":
            # Placeholder token at position 0 for missing-value handling
            assert section["tokens"][0].endswith("_unknown"), (
                f"{field} policy=emit_unknown_token but {section_name} has no _unknown at index 0"
            )
        elif mv["type"] in ("drop_row", "n_a"):
            # No placeholder unknown at position 0. However, "unknown" may appear
            # as a data value elsewhere in the section, and alternate fields in
            # the POI section may have _unknown from primary's policy.
            # The check: position 0 should NOT be a placeholder unknown.
            if not section["tokens"][0].endswith("_unknown"):
                # Good, no placeholder
                pass
            else:
                # Position 0 is _unknown, which violates drop_row/n_a.
                # Exception: POI section where primary has emit_unknown_token,
                # so position 0 is a legitimate placeholder.
                if section_name == "poi" and field in ("places.categories.alternate",):
                    # Alternate's n_a policy is OK with POI_unknown from primary
                    pass
                else:
                    raise AssertionError(
                        f"{field} policy={mv['type']} but {section_name} has _unknown at index 0 "
                        f"(expected drop_row/n_a to have no placeholder)"
                    )


def test_script_byte_deterministic_modulo_generated_utc(tmp_path):
    """Two consecutive runs produce byte-identical artifacts after stripping
    generated_utc and the embedded sha256 lines."""
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    _run_script(out1)
    _run_script(out2, rerun_reason="rerun-test")

    def _stripped(path: Path) -> str:
        lines = path.read_text().splitlines()
        return "\n".join(
            line for line in lines
            if not line.lstrip().startswith(("generated_utc:", "vocab_sha256:", "policy_sha256:"))
        )

    vocab1 = _stripped(out1 / "configs" / "tokenizer" / "vocab_phase1.yaml")
    vocab2 = _stripped(out2 / "configs" / "tokenizer" / "vocab_phase1.yaml")
    assert vocab1 == vocab2

    policy1 = _stripped(out1 / "configs" / "data" / "missing_value_policy.yaml")
    policy2 = _stripped(out2 / "configs" / "data" / "missing_value_policy.yaml")
    assert policy1 == policy2

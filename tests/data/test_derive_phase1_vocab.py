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
    assert len(fc["building"]["tokens"]) == 23   # 22 Moderate-kept + B__UNK__
    assert len(fc["road"]["tokens"]) == 17       # 17 Moderate-kept (no __UNK__ placeholder; drop_row policy)
    assert len(fc["base"]["tokens"]) == 7        # 7 Strict-kept (no __UNK__ placeholder; n_a policy)
    poi_len = len(fc["poi"]["tokens"])
    assert 291 <= poi_len <= 341, (
        f"POI section size {poi_len} outside expected [291, 341]; "
        "investigate union-size shift before re-pinning."
    )

    # Building section: B__UNK__ placeholder at position 0 (emit_unknown_token policy).
    assert fc["building"]["tokens"][0] == "B__UNK__"
    # POI section: POI__UNK__ placeholder at position 0 (primary emit_unknown_token policy).
    assert fc["poi"]["tokens"][0] == "POI__UNK__"
    # Road and base sections must not contain a __UNK__ placeholder anywhere
    # (their fields all use drop_row or n_a policies). Note: data-derived tokens
    # like R_unknown (Overture's transportation.class literal "unknown" category)
    # are NOT placeholders — they end in `_unknown` but do not contain `__UNK__`.
    assert all("__UNK__" not in t for t in fc["road"]["tokens"])
    assert all("__UNK__" not in t for t in fc["base"]["tokens"])
    # Verify R_unknown is present as a data token (Overture data category, not placeholder).
    assert "R_unknown" in fc["road"]["tokens"]


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

    # Every field carries BOTH four-case axes per sub-C spec §10.2 (B2 follow-up).
    for field_name in expected:
        field_block = policy["fields"][field_name]
        assert "policies" in field_block, f"{field_name}: missing policies block"
        assert "missing_value" in field_block["policies"], (
            f"{field_name}: missing missing_value axis"
        )
        assert "not_in_vocab" in field_block["policies"], (
            f"{field_name}: missing not_in_vocab axis (B2 follow-up incomplete?)"
        )


def test_cross_artifact_consistency_unknown_tokens(tmp_path):
    """Cross-artifact consistency between policy and vocab:

    For each vocab section, look at the policies of all source fields that
    contributed to it. If ANY source field has emit_unknown_token policy, the
    section MUST have a __UNK__ placeholder at position 0. If ALL source fields
    have drop_row or n_a policies, the section MUST NOT contain any __UNK__
    token anywhere.

    Data-derived tokens like R_unknown (Overture's "unknown" category in
    transportation.class) are NOT placeholders — they contain the substring
    `_unknown` but not `__UNK__`, and are allowed regardless of policy.
    """
    _run_script(tmp_path)
    vocab = _load_yaml(tmp_path / "configs" / "tokenizer" / "vocab_phase1.yaml")
    policy = _load_yaml(tmp_path / "configs" / "data" / "missing_value_policy.yaml")

    # Map: vocab section name → list of source fields contributing to it.
    section_to_fields = {
        "building": ["buildings.class"],
        "road": ["transportation.class"],
        "base": ["base.class"],
        "poi": ["places.categories.primary", "places.categories.alternate"],
    }

    for section_name, source_fields in section_to_fields.items():
        section = vocab["feature_class"][section_name]
        policies = [
            policy["fields"][f]["policies"]["missing_value"]["type"] for f in source_fields
        ]
        has_emit = any(p == "emit_unknown_token" for p in policies)
        all_drop_or_na = all(p in ("drop_row", "n_a") for p in policies)

        unk_tokens = [t for t in section["tokens"] if "__UNK__" in t]
        if has_emit:
            assert len(unk_tokens) == 1, (
                f"section {section_name!r} has source fields {source_fields} with "
                f"policies {policies}; expected exactly 1 __UNK__ placeholder, got {unk_tokens}"
            )
            assert section["tokens"][0] == unk_tokens[0], (
                f"section {section_name!r}: __UNK__ placeholder must be at index 0; "
                f"got tokens[0]={section['tokens'][0]!r}"
            )
        elif all_drop_or_na:
            assert unk_tokens == [], (
                f"section {section_name!r} has policies {policies} (all drop_row/n_a) "
                f"but contains __UNK__ tokens: {unk_tokens}"
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

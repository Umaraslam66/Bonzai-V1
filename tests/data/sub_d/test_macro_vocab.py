"""Tests for the locked macro vocab loader + promote-script contract (Task 8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cfm.data.io import canonicalize_yaml
from cfm.data.sub_d.errors import SubDValidationError
from cfm.data.sub_d.macro_vocab import (
    LOCKED_VOCAB_NAMESPACES,
    load_macro_vocab,
    token_id_to_name,
    token_name_to_id,
)

# Promote-script entry points (imported at function call time to avoid
# requiring the script to be on PYTHONPATH at collection).


def _build_minimal_proposal(status: str = "proposal") -> dict:
    """A minimal-but-validatable proposal index dict for tests.

    Carries the bare set of fields the validator and promote-script touch.
    Real proposals (from build_frequency_analysis + write_proposal_artifacts)
    carry many more fields, but those don't change validator behavior.
    """
    return {
        "status": status,
        "analysis_version": "1.0",
        "derivation_versions": {
            "zoning": "1.0",
            "cell_density": "1.0",
            "tile_population_density": "1.0",
            "road_skeleton": "1.0",
        },
        "tile_count": 2,
        "input_digests": [],
        "per_tile_evidence": [],
        "zoning_orthogonality": {},
        "namespace_files": [],
        "selected_layer3_tiles": [],
        "locked_buckets": {
            "zoning": [
                {"token_id": 0, "token_name": "road", "count": 100},
                {"token_id": 1, "token_name": "building", "count": 50},
            ],
            "cell_density": [
                {
                    "token_id": 0,
                    "token_name": "bucket_0",
                    "lower_inclusive": 0.0,
                    "upper_exclusive": 0.1,
                },
                {
                    "token_id": 1,
                    "token_name": "bucket_1",
                    "lower_inclusive": 0.1,
                    "upper_exclusive": None,
                },
            ],
            "tile_population_density": [
                {
                    "token_id": 0,
                    "token_name": "bucket_0",
                    "lower_inclusive": 0.0,
                    "upper_exclusive": 0.05,
                },
                {
                    "token_id": 1,
                    "token_name": "bucket_1",
                    "lower_inclusive": 0.05,
                    "upper_exclusive": None,
                },
            ],
            "road_skeleton": [
                {
                    "token_id": 0,
                    "token_name": "bucket_0",
                    "lower_inclusive": 0,
                    "upper_exclusive": 1,
                },
                {
                    "token_id": 1,
                    "token_name": "bucket_1",
                    "lower_inclusive": 1,
                    "upper_exclusive": None,
                },
            ],
        },
        "locked_proxy": {
            "tile_population_density": "p75_building_footprint_ratio",
        },
        "append_only_within_phase": {
            "cell_density": True,
            "road_skeleton": True,
            "tile_population_density": True,
            "zoning": True,
        },
    }


def _write_proposal(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "proposal.yaml"
    path.write_text(canonicalize_yaml(data), encoding="utf-8")
    return path


def test_macro_vocab_loads_locked_artifact(tmp_path: Path):
    locked = _build_minimal_proposal(status="locked")
    path = tmp_path / "macro_plan_vocab.yaml"
    path.write_text(canonicalize_yaml(locked), encoding="utf-8")

    data = load_macro_vocab(path)
    assert data["status"] == "locked"
    for ns in LOCKED_VOCAB_NAMESPACES:
        assert ns in data["locked_buckets"]
        assert len(data["locked_buckets"][ns]) >= 1
    assert data["locked_proxy"]["tile_population_density"]


def test_macro_vocab_rejects_duplicate_token_ids(tmp_path: Path):
    bad = _build_minimal_proposal(status="locked")
    # Force duplicate token_id within zoning.
    bad["locked_buckets"]["zoning"] = [
        {"token_id": 0, "token_name": "road", "count": 100},
        {"token_id": 0, "token_name": "building", "count": 50},
    ]
    path = tmp_path / "bad.yaml"
    path.write_text(canonicalize_yaml(bad), encoding="utf-8")

    with pytest.raises(SubDValidationError, match="duplicate token_id"):
        load_macro_vocab(path)


def test_macro_vocab_rejects_duplicate_token_names(tmp_path: Path):
    bad = _build_minimal_proposal(status="locked")
    # Force duplicate token_name within cell_density.
    bad["locked_buckets"]["cell_density"] = [
        {
            "token_id": 0,
            "token_name": "bucket_0",
            "lower_inclusive": 0.0,
            "upper_exclusive": 0.1,
        },
        {
            "token_id": 1,
            "token_name": "bucket_0",
            "lower_inclusive": 0.1,
            "upper_exclusive": None,
        },
    ]
    path = tmp_path / "bad.yaml"
    path.write_text(canonicalize_yaml(bad), encoding="utf-8")

    with pytest.raises(SubDValidationError, match="duplicate token_name"):
        load_macro_vocab(path)


def test_macro_vocab_records_frequency_analysis_digests(tmp_path: Path):
    locked = _build_minimal_proposal(status="locked")
    # The locked artifact carries the four namespace_files records pinned by
    # sha256; reviewers and validators read these to cross-check provenance.
    locked["namespace_files"] = [
        {"filename": "zoning_analysis.yaml", "section_key": "zoning_proposal", "sha256": "z" * 64},
        {
            "filename": "cell_density_analysis.yaml",
            "section_key": "cell_density_proposal",
            "sha256": "c" * 64,
        },
        {
            "filename": "tile_population_density_analysis.yaml",
            "section_key": "tile_population_density_proposal",
            "sha256": "t" * 64,
        },
        {
            "filename": "road_skeleton_analysis.yaml",
            "section_key": "road_skeleton_proposal",
            "sha256": "r" * 64,
        },
    ]
    path = tmp_path / "macro_plan_vocab.yaml"
    path.write_text(canonicalize_yaml(locked), encoding="utf-8")

    data = load_macro_vocab(path)
    assert len(data["namespace_files"]) == 4
    filenames = {entry["filename"] for entry in data["namespace_files"]}
    assert filenames == {
        "zoning_analysis.yaml",
        "cell_density_analysis.yaml",
        "tile_population_density_analysis.yaml",
        "road_skeleton_analysis.yaml",
    }
    for entry in data["namespace_files"]:
        assert "sha256" in entry and len(entry["sha256"]) == 64


def test_macro_vocab_has_append_only_flags_for_every_enum(tmp_path: Path):
    locked = _build_minimal_proposal(status="locked")
    path = tmp_path / "macro_plan_vocab.yaml"
    path.write_text(canonicalize_yaml(locked), encoding="utf-8")
    data = load_macro_vocab(path)

    for ns in LOCKED_VOCAB_NAMESPACES:
        assert data["append_only_within_phase"][ns] is True, (
            f"namespace {ns!r} missing append_only_within_phase=true"
        )

    # Removing a flag must fail validation.
    bad = _build_minimal_proposal(status="locked")
    del bad["append_only_within_phase"]["zoning"]
    path_bad = tmp_path / "bad.yaml"
    path_bad.write_text(canonicalize_yaml(bad), encoding="utf-8")
    with pytest.raises(SubDValidationError, match="append_only_within_phase"):
        load_macro_vocab(path_bad)


def test_promote_macro_vocab_derives_locked_artifact_from_proposal(tmp_path: Path):
    from scripts.promote_macro_vocab import promote  # type: ignore[import-not-found]

    proposal = _build_minimal_proposal(status="proposal")
    proposal_path = _write_proposal(tmp_path, proposal)
    output_path = tmp_path / "macro_plan_vocab.yaml"

    promote(proposal_path, output_path)

    locked = load_macro_vocab(output_path)
    assert locked["status"] == "locked"
    # Token IDs/names round-trip via the public API.
    assert token_name_to_id("zoning", "road", locked) == 0
    assert token_id_to_name("zoning", 0, locked) == "road"
    assert token_name_to_id("cell_density", "bucket_0", locked) == 0
    assert token_id_to_name("cell_density", 1, locked) == "bucket_1"


def test_promote_macro_vocab_diff_is_status_marker_only(tmp_path: Path):
    """The byte-identity-modulo-status-marker contract.

    Reads both files as bytes, normalizes the locked file's status line
    back to ``status: proposal``, and asserts byte equality with the
    proposal file. Any hand-edit between proposal and locked beyond the
    status line fails this test — that is the entire point of routing
    reviewer edits through the proposal file rather than the locked one.
    """
    from scripts.promote_macro_vocab import (  # type: ignore[import-not-found]
        LOCKED_STATUS_LINE,
        PROPOSAL_STATUS_LINE,
        promote,
    )

    proposal = _build_minimal_proposal(status="proposal")
    proposal_path = _write_proposal(tmp_path, proposal)
    output_path = tmp_path / "macro_plan_vocab.yaml"

    promote(proposal_path, output_path)

    proposal_bytes = proposal_path.read_bytes()
    locked_bytes = output_path.read_bytes()
    normalized_locked = locked_bytes.replace(LOCKED_STATUS_LINE, PROPOSAL_STATUS_LINE, 1)
    assert proposal_bytes == normalized_locked, (
        "byte-identity-modulo-status-marker violated: locked artifact diverges "
        "from proposal beyond the single status line"
    )

    # And conversely, if the proposal is hand-tampered with after promote
    # (e.g. someone slips a bucket edit into the locked file), the
    # normalization must NOT make the bytes match.
    tampered = locked_bytes.replace(b"bucket_0", b"bucket_X", 1)
    output_path.write_bytes(tampered)
    locked_bytes_tampered = output_path.read_bytes()
    normalized_tampered = locked_bytes_tampered.replace(LOCKED_STATUS_LINE, PROPOSAL_STATUS_LINE, 1)
    assert proposal_bytes != normalized_tampered, (
        "byte-identity test must catch tampering beyond the status line"
    )

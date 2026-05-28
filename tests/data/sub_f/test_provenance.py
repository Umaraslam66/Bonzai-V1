"""Tests for sub-F provenance hashing exclusions."""

from __future__ import annotations

from copy import deepcopy

from cfm.data.sub_f.provenance import SUB_F_EXCLUDED_FROM_SHA, provenance_sha256


def _sample_provenance() -> dict:
    return {
        "provenance_schema_version": "1.0",
        "tile_i": 1,
        "tile_j": 2,
        "extraction": {
            "commit_sha": "12b1cdf8838d9f8b601ea4b2a859f905ee5ab368",
            "extracted_utc": "2026-05-23T00:00:00Z",
        },
        "inputs": {
            "sub_c": {
                "overture_release": "2026-04-15.0",
                "sub_c_schema_version": "1.1",
                "sub_c_commit_sha": "12b1cdf8838d9f8b601ea4b2a859f905ee5ab368",
            },
            "semantic_vocab_sha256": "a" * 64,
        },
        "outputs": {
            "token_stream_parquet_sha256": "b" * 64,
            "nested": {"metadata_sha256": "c" * 64},
        },
    }


def test_provenance_sha_excludes_live_extraction_timestamp():
    base = _sample_provenance()
    changed = deepcopy(base)
    changed["extraction"]["extracted_utc"] = "2099-12-31T23:59:59Z"

    assert provenance_sha256(base) == provenance_sha256(changed)


def test_provenance_sha_excludes_nested_final_segment_sha_fields():
    base = _sample_provenance()
    changed = deepcopy(base)
    changed["inputs"]["semantic_vocab_sha256"] = "d" * 64
    changed["outputs"]["token_stream_parquet_sha256"] = "e" * 64
    changed["outputs"]["nested"]["metadata_sha256"] = "f" * 64

    assert provenance_sha256(base) == provenance_sha256(changed)
    assert "*_sha256" in SUB_F_EXCLUDED_FROM_SHA["*"]


def test_provenance_sha_changes_on_semantic_content_change():
    base = _sample_provenance()
    changed = deepcopy(base)
    changed["tile_j"] = 3

    assert provenance_sha256(base) != provenance_sha256(changed)

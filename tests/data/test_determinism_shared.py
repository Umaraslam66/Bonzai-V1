"""Tests for the neutral shared determinism helpers in cfm.data.determinism.

These helpers take an explicit ``exclusions`` table so that sub-C, sub-D, and
any later sidecar layer can reuse the grammar without sharing a global
exclusion list.
"""

from __future__ import annotations

from cfm.data.determinism import (
    compute_sha256,
    compute_sha256_excluding,
    path_in_excluded,
)
from cfm.data.io import canonicalize_yaml

EXCLUSIONS = {
    "*": ["*_sha256"],
    "manifest.yaml": ["initial_extraction.started_utc"],
}


def test_path_in_excluded_uses_final_segment_sha_suffix():
    assert path_in_excluded("tiles[0].provenance_sha256", "*", EXCLUSIONS)
    assert not path_in_excluded("sha256_input", "*", EXCLUSIONS)


def test_compute_sha256_excluding_uses_supplied_exclusion_table():
    with_digest = {"a": 1, "nested": {"file_sha256": "abc"}}
    clean = {"a": 1, "nested": {}}
    assert compute_sha256_excluding(with_digest, "*", EXCLUSIONS) == compute_sha256(
        canonicalize_yaml(clean).encode("utf-8")
    )


def test_file_specific_timestamp_exclusion():
    first = {"initial_extraction": {"started_utc": "2026-01-01T00:00:00Z", "tile_count": 2}}
    second = {"initial_extraction": {"started_utc": "2026-01-02T00:00:00Z", "tile_count": 2}}
    assert compute_sha256_excluding(first, "manifest.yaml", EXCLUSIONS) == (
        compute_sha256_excluding(second, "manifest.yaml", EXCLUSIONS)
    )

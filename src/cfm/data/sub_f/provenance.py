"""Sub-F provenance hashing helpers."""

from __future__ import annotations

from cfm.data.determinism import (
    compute_sha256_excluding as _compute_sha256_excluding,
)

SUB_F_EXCLUDED_FROM_SHA: dict[str, list[str]] = {
    "*": ["*_sha256"],
    "provenance.yaml": [
        "extraction.extracted_utc",
    ],
    "manifest.yaml": [
        "initial_extraction.started_utc",
        "initial_extraction.completed_utc",
    ],
}


def provenance_sha256(data: dict) -> str:
    """Compute a timestamp-stable self-integrity hash for provenance.yaml."""

    return _compute_sha256_excluding(data, "provenance.yaml", SUB_F_EXCLUDED_FROM_SHA)

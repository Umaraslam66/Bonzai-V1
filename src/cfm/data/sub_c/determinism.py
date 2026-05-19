"""Determinism contract for sub-C tile extraction.

EXCLUDED_FROM_SHA defines which fields are stripped before computing content
hashes. This is the single source of truth for both production sha computation
and test comparison helpers.

Wildcard semantics per spec §14.6:
- The "*" key applies to ALL files.
- Entries under "*" that start with "*" use final-segment suffix-match.
  e.g. "*_sha256" matches any dotted path whose last segment ends in "_sha256".
- All other entries (under "*" or under a specific file key) use exact
  dotted-path match.

The grammar and primitive helpers live in ``cfm.data.determinism``; this
module pins sub-C's exclusion table and wraps the neutral helpers so the
public API stays single-argument (``file_key``).
"""

from __future__ import annotations

from cfm.data.determinism import (
    compute_sha256,
    compute_sha256_excluding as _compute_sha256_excluding,
    path_in_excluded as _path_in_excluded,
)

# ---------------------------------------------------------------------------
# Exclusion table (spec §14.6)
# ---------------------------------------------------------------------------

EXCLUDED_FROM_SHA: dict[str, list[str]] = {
    "*": ["*_sha256"],  # final-segment suffix-match across all files
    "manifest.yaml": [
        "initial_extraction.started_utc",
        "initial_extraction.completed_utc",
    ],
    "provenance.yaml": [
        "extraction.extracted_utc",
    ],
}

# One source of truth: test helpers reference this same dict.
EXCLUDED_FROM_TEST_COMPARE = EXCLUDED_FROM_SHA


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def path_in_excluded(dotted_path: str, file_key: str) -> bool:
    """Return True if *dotted_path* should be stripped before hashing.

    Thin wrapper that pins sub-C's EXCLUDED_FROM_SHA so callers keep the
    single-argument file_key API.
    """
    return _path_in_excluded(dotted_path, file_key, EXCLUDED_FROM_SHA)


def compute_sha256_excluding(data: dict, file_key: str) -> str:
    """Strip EXCLUDED_FROM_SHA paths, canonicalize to YAML, hash the bytes."""
    return _compute_sha256_excluding(data, file_key, EXCLUDED_FROM_SHA)


__all__ = [
    "EXCLUDED_FROM_SHA",
    "EXCLUDED_FROM_TEST_COMPARE",
    "compute_sha256",
    "compute_sha256_excluding",
    "path_in_excluded",
]

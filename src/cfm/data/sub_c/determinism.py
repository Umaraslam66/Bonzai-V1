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
"""

from __future__ import annotations

import hashlib
import re

from cfm.data.sub_c.io import canonicalize_yaml

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


def compute_sha256(content_bytes: bytes) -> str:
    """Return the hex SHA-256 digest of *content_bytes*."""
    return hashlib.sha256(content_bytes).hexdigest()


def path_in_excluded(dotted_path: str, file_key: str) -> bool:
    """Return True if *dotted_path* should be stripped before hashing.

    Checks in two passes:
    1. Universal entries under EXCLUDED_FROM_SHA["*"].
    2. File-specific entries under EXCLUDED_FROM_SHA[file_key].

    Wildcard semantics (spec §14.6):
    - Pattern "*_suffix": matches when the final segment of dotted_path ends
      with "_suffix" (after stripping any trailing list index like "[N]").
    - All other patterns: exact dotted-path equality.

    Examples:
        path_in_excluded("vocab_sha256", "*")          -> True   (suffix match)
        path_in_excluded("tiles[3].provenance_sha256", "*") -> True
        path_in_excluded("sha256_input", "*")          -> False  (suffix only)
        path_in_excluded("initial_extraction.started_utc", "manifest.yaml") -> True
        path_in_excluded("extraction.rerun_reason", "provenance.yaml")      -> False
    """
    final_seg = _final_segment(dotted_path)

    for pattern in EXCLUDED_FROM_SHA.get("*", []):
        if _matches(pattern, dotted_path, final_seg):
            return True

    for pattern in EXCLUDED_FROM_SHA.get(file_key, []):
        if _matches(pattern, dotted_path, final_seg):
            return True

    return False


def compute_sha256_excluding(data: dict, file_key: str) -> str:
    """Strip EXCLUDED_FROM_SHA paths, canonicalize to YAML, hash the bytes.

    This is the standard hash used for content-addressable tile artefacts.
    """
    stripped = _strip_excluded(data, file_key)
    canonical = canonicalize_yaml(stripped)
    return compute_sha256(canonical.encode("utf-8"))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _final_segment(dotted_path: str) -> str:
    """Return the last segment of *dotted_path*, stripping any trailing [N] index."""
    last = dotted_path.rsplit(".", 1)[-1]
    return re.sub(r"\[\d+\]$", "", last)


def _matches(pattern: str, dotted_path: str, final_seg: str) -> bool:
    """Return True if *pattern* matches *dotted_path* / *final_seg*."""
    if pattern.startswith("*"):
        # Suffix-match on final segment: "*_sha256" → ends with "_sha256"
        suffix = pattern[1:]
        return final_seg.endswith(suffix)
    # Exact dotted-path match
    return pattern == dotted_path


def _strip_excluded(data: object, file_key: str, prefix: str = "") -> object:
    """Recursively strip excluded paths from *data* (returns a new structure)."""
    if isinstance(data, dict):
        result: dict = {}
        for k, v in data.items():
            child_path = f"{prefix}.{k}" if prefix else k
            if path_in_excluded(child_path, file_key):
                continue
            result[k] = _strip_excluded(v, file_key, child_path)
        return result

    if isinstance(data, list):
        return [_strip_excluded(item, file_key, f"{prefix}[{i}]") for i, item in enumerate(data)]

    return data

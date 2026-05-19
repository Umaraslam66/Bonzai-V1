"""Neutral determinism helpers shared by sub-C and later sidecar layers.

The grammar matches sub-C spec §14.6:
- ``"*"`` key in the exclusion table applies to all files.
- Patterns starting with ``"*"`` use final-segment suffix-match
  (e.g. ``"*_sha256"`` matches any path whose last segment ends in
  ``_sha256``). List indices ``[N]`` on the final segment are stripped.
- All other patterns use exact dotted-path equality.

The exclusion table is passed in explicitly so each sidecar layer can carry
its own table without sharing global state.
"""

from __future__ import annotations

import hashlib
import re

from cfm.data.io import canonicalize_yaml

ExclusionTable = dict[str, list[str]]


def compute_sha256(content_bytes: bytes) -> str:
    """Return the hex SHA-256 digest of *content_bytes*."""
    return hashlib.sha256(content_bytes).hexdigest()


def path_in_excluded(dotted_path: str, file_key: str, exclusions: ExclusionTable) -> bool:
    """Return True if *dotted_path* should be stripped before hashing.

    Checks two passes:
    1. Universal entries under ``exclusions["*"]``.
    2. File-specific entries under ``exclusions[file_key]``.
    """
    final_seg = _final_segment(dotted_path)
    for pattern in exclusions.get("*", []):
        if _matches(pattern, dotted_path, final_seg):
            return True
    for pattern in exclusions.get(file_key, []):
        if _matches(pattern, dotted_path, final_seg):
            return True
    return False


def compute_sha256_excluding(data: dict, file_key: str, exclusions: ExclusionTable) -> str:
    """Strip excluded paths from *data*, canonicalise to YAML, hash the bytes."""
    stripped = _strip_excluded(data, file_key, exclusions)
    return compute_sha256(canonicalize_yaml(stripped).encode("utf-8"))


def _final_segment(dotted_path: str) -> str:
    last = dotted_path.rsplit(".", 1)[-1]
    return re.sub(r"\[\d+\]$", "", last)


def _matches(pattern: str, dotted_path: str, final_seg: str) -> bool:
    if pattern.startswith("*"):
        return final_seg.endswith(pattern[1:])
    return pattern == dotted_path


def _strip_excluded(
    data: object,
    file_key: str,
    exclusions: ExclusionTable,
    prefix: str = "",
) -> object:
    if isinstance(data, dict):
        result: dict = {}
        for key, value in data.items():
            child_path = f"{prefix}.{key}" if prefix else str(key)
            if path_in_excluded(child_path, file_key, exclusions):
                continue
            result[key] = _strip_excluded(value, file_key, exclusions, child_path)
        return result
    if isinstance(data, list):
        return [
            _strip_excluded(item, file_key, exclusions, f"{prefix}[{index}]")
            for index, item in enumerate(data)
        ]
    return data

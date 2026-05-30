"""Sub-F provisional region manifest helpers for Halt 6.

Task 6 locks the six-axis version surface but leaves vocab source completeness
partial until BP7 locks. Tile entries with ``tile_i`` and ``tile_j`` are sorted
by that coordinate pair. If tile coordinates are absent, caller order is
preserved because there is no stable sortable key.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from cfm.data.determinism import (
    compute_sha256,
)
from cfm.data.determinism import (
    compute_sha256_excluding as _compute_sha256_excluding,
)
from cfm.data.sub_f.provenance import SUB_F_EXCLUDED_FROM_SHA
from cfm.data.sub_f.versions import (
    SUB_F_ARTIFACT_FORMAT_VERSION,
    SUB_F_DERIVATION_VERSION,
    SUB_F_SCHEMA_VERSION,
    SUB_F_VALIDATOR_VERSION,
    SUB_F_VOCAB_VERSION,
    load_sub_f_source_version,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]

TASK6_VOCAB_SOURCE_PATHS: tuple[str, ...] = (
    "configs/sub_f/semantic_vocab.yaml",
    "configs/sub_f/unknown_family.yaml",
    "configs/sub_f/encoding_primitives.yaml",
    "configs/sub_f/boundary_reference_vocab.yaml",
)

_TASK6_VOCAB_SOURCE_KEYS: tuple[str, ...] = (
    "bp1_semantic_vocab",
    "bp4_unknown_family",
    "bp2_encoding_primitives",
    "bp7_boundary_reference_vocab",
)


def task6_vocab_sources() -> dict[str, dict[str, str]]:
    """Return the COMPLETE BP1/BP2/BP4/BP7 vocab-source set.

    BP7 (`boundary_reference_vocab.yaml`) was provisional-pending at Halt 6 and
    added here once Task 7 locked the boundary-reference vocab (close-checklist
    line 8, discharged at T15). The four locked vocab blueprints are now covered.
    """

    sources: dict[str, dict[str, str]] = {}
    for key, rel_path in zip(_TASK6_VOCAB_SOURCE_KEYS, TASK6_VOCAB_SOURCE_PATHS, strict=False):
        path = _REPO_ROOT / rel_path
        sources[key] = {
            "path": rel_path,
            "sha256": compute_sha256(path.read_bytes()),
        }
    return sources


def build_region_manifest(
    region: str,
    release: str,
    tile_entries: list[dict],
    vocab_sources: dict[str, Any],
) -> dict:
    """Build a sub-F region manifest.

    ``vocab_sources`` is region-scope metadata covering the four locked vocab
    blueprints (BP1/BP2/BP4/BP7). It was partial at Halt 6 (BP7 pending); BP7's
    source was added once Task 7 locked it (close-checklist line 8, T15).
    """

    manifest = {
        "region": region,
        "release": release,
        "sub_f_artifact_format_version": SUB_F_ARTIFACT_FORMAT_VERSION,
        "sub_f_schema_version": SUB_F_SCHEMA_VERSION,
        "sub_f_vocab_version": SUB_F_VOCAB_VERSION,
        "sub_f_derivation_version": SUB_F_DERIVATION_VERSION,
        "sub_f_validator_version": SUB_F_VALIDATOR_VERSION,
        "sub_f_source_version": load_sub_f_source_version(),
        "vocab_sources_status": "complete",
        "vocab_sources": deepcopy(vocab_sources),
        "initial_extraction": {
            "started_utc": None,
            "completed_utc": None,
        },
        "tiles": _sort_tile_entries(tile_entries),
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return manifest


def manifest_sha256(data: dict) -> str:
    """Compute a timestamp-stable self-integrity hash for manifest.yaml."""

    return _compute_sha256_excluding(data, "manifest.yaml", SUB_F_EXCLUDED_FROM_SHA)


def _sort_tile_entries(tile_entries: list[dict]) -> list[dict]:
    copied = [deepcopy(entry) for entry in tile_entries]
    if all("tile_i" in entry and "tile_j" in entry for entry in copied):
        return sorted(copied, key=lambda entry: (entry["tile_i"], entry["tile_j"]))
    return copied

"""Per-region manifest writer.

Tiles sorted by ``(tile_i, tile_j)`` at write-time (not assumed from input).
Self-integrity sha (``manifest_sha256``) available for cross-environment
determinism checks; uses ``SUB_E_EXCLUDED_FROM_SHA`` from provenance.py
to strip ``initial_extraction.started_utc/completed_utc`` and final-segment
``*_sha256`` before hashing (spec §9.2 mandate).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from cfm.data.determinism import (
    compute_sha256_excluding as _compute_sha256_excluding,
)
from cfm.data.io import canonicalize_yaml
from cfm.data.sub_e.provenance import SUB_E_EXCLUDED_FROM_SHA


@dataclass(frozen=True)
class SubEManifestInputs:
    sub_c_manifest_sha256: str
    sub_c_region_dir: str
    sub_d_manifest_sha256: str
    sub_d_region_dir: str
    boundary_vocab_sha256: str


@dataclass(frozen=True)
class SubEManifestVersions:
    boundary_vocab_version: str
    boundary_derivation_version: str


@dataclass(frozen=True)
class SubEManifestConfig:
    cell_grid: tuple[int, int]
    internal_edge_count: int
    external_edge_count: int


@dataclass(frozen=True)
class SubEManifestExtraction:
    commit_sha: str
    started_utc: str
    completed_utc: str
    tile_count: int


@dataclass(frozen=True)
class SubEManifestTile:
    tile_i: int
    tile_j: int
    provenance_sha256: str


@dataclass(frozen=True)
class SubEManifest:
    manifest_schema_version: str
    sub_e_schema_version: str
    release: str
    region: str
    region_crs: str
    inputs: SubEManifestInputs
    versions: SubEManifestVersions
    config_source: str
    config: SubEManifestConfig
    initial_extraction: SubEManifestExtraction
    tiles: list[SubEManifestTile] = field(default_factory=list)


def manifest_to_dict(manifest: SubEManifest) -> dict:
    """Serialise SubEManifest to its on-disk YAML dict shape.

    Enforces canonical tile sort by ``(tile_i, tile_j)`` at write-time —
    does not trust input ordering. Same discipline as Task 6's per-edge
    canonical sort key.
    """
    sorted_tiles = sorted(manifest.tiles, key=lambda t: (t.tile_i, t.tile_j))
    return {
        "manifest_schema_version": manifest.manifest_schema_version,
        "sub_e_schema_version": manifest.sub_e_schema_version,
        "release": manifest.release,
        "region": manifest.region,
        "region_crs": manifest.region_crs,
        "inputs": asdict(manifest.inputs),
        "versions": asdict(manifest.versions),
        "config_source": manifest.config_source,
        "config": {
            "cell_grid": list(manifest.config.cell_grid),
            "internal_edge_count": manifest.config.internal_edge_count,
            "external_edge_count": manifest.config.external_edge_count,
        },
        "initial_extraction": asdict(manifest.initial_extraction),
        "tiles": [
            {
                "tile_i": t.tile_i,
                "tile_j": t.tile_j,
                "provenance_sha256": t.provenance_sha256,
            }
            for t in sorted_tiles
        ],
    }


def write_manifest(path: Path, manifest: SubEManifest) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonicalize_yaml(manifest_to_dict(manifest)), encoding="utf-8")
    return path


def manifest_sha256(data: dict) -> str:
    """Compute the self-integrity sha for a manifest.yaml dict.

    Strips ``initial_extraction.started_utc/completed_utc`` and
    final-segment ``*_sha256`` per ``SUB_E_EXCLUDED_FROM_SHA``,
    canonicalises the remainder, hashes the bytes. Spec §9.2 mandate.
    Not in Phase-1's digest chain — exposed so cross-environment
    determinism checks (spec §14) can compare manifests timestamp-free.
    """
    return _compute_sha256_excluding(data, "manifest.yaml", SUB_E_EXCLUDED_FROM_SHA)

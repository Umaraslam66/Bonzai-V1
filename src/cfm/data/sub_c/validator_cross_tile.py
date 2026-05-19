"""Cross-tile validator for sub-C tile extraction.

Implements the four mandatory invariants from spec §12.2 that gate _SUCCESS
write. This module contains only library logic; the CLI wrapper that calls it
lives in scripts/validate_extraction.py.

Public API:
    validate_extraction_cross_tile(region_dir) -> None

Invariants (spec §12.2):
    1. sub_c_schema_version_consistency
       manifest.sub_c_schema_version must equal every tile's meta.yaml
       schema_version AND every tile's provenance.yaml schema_version.
       (Rationale: manifest's sub_c_schema_version is the authoritative
       version covering all sub-C output shapes, per §11.7. meta/provenance
       each carry schema_version that records the YAML-format version at
       extraction time. These must agree for the entire extraction to be
       consistently versioned.)

    2. manifest_tiles_match_filesystem
       manifest.tiles[] inventory must match the tile=* dirs on disk exactly:
       no orphan dirs not listed in the manifest; no manifest entries missing
       from disk.

    3. manifest_provenance_sha_matches_disk
       For each tile in the manifest, manifest.tiles[i].provenance_sha256 must
       equal compute_sha256_excluding(yaml.safe_load(tile_dir/provenance.yaml),
       "provenance.yaml"). This re-derives the sha the same way
       aggregate_tile_inventory did — stripping EXCLUDED_FROM_SHA paths,
       canonicalizing to YAML, hashing the UTF-8 bytes.

    4. provenance_outputs_sha_match_files
       For each tile, provenance.yaml's outputs.*_sha256 fields must equal the
       SHA-256 of the corresponding file's raw bytes on disk. This checks the
       full integrity chain: _SUCCESS → manifest.tiles[*].provenance_sha256 →
       provenance.outputs.*_sha256 → file bytes (spec §11.7).
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from cfm.data.sub_c.determinism import compute_sha256, compute_sha256_excluding
from cfm.data.sub_c.errors import TileValidationError
from cfm.data.sub_c.manifest import RegionManifest, read_manifest

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_OUTPUTS_FILENAMES: dict[str, str] = {
    "cells_parquet_sha256": "cells.parquet",
    "features_parquet_sha256": "features.parquet",
    "crossings_parquet_sha256": "crossings.parquet",
    "meta_yaml_sha256": "meta.yaml",
}


def validate_extraction_cross_tile(region_dir: Path) -> None:
    """Run all 4 cross-tile invariants. Raises TileValidationError on failure.

    Per spec §12.2 — mandatory _SUCCESS gate. Runs AFTER manifest is written,
    BEFORE _SUCCESS is written. _SUCCESS write is the responsibility of
    extract_tiles.py CLI; this function only raises on failure.

    Invariant IDs used in TileValidationError.invariant:
        "sub_c_schema_version_consistency"
        "manifest_tiles_match_filesystem"
        "manifest_provenance_sha_matches_disk"
        "provenance_outputs_sha_match_files"
    """
    manifest_path = region_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise TileValidationError(
            tile="region",
            invariant="manifest_tiles_match_filesystem",
            failed_row={},
            detail={"message": f"manifest.yaml missing at {manifest_path}"},
        )

    manifest = read_manifest(manifest_path)
    log.debug(
        "cross-tile validator: region=%s tiles=%d",
        manifest.region,
        len(manifest.tiles),
    )

    # Run filesystem-completeness check FIRST so that subsequent invariants
    # (which read tile files by following the manifest) don't encounter
    # FileNotFoundError for manifest entries whose dirs were removed or are
    # missing. The spec numbers the invariants 1-4 for logical grouping, not
    # for mandatory execution order; checking #2 before #1 is safe because #2
    # only reads directory names, not file contents.
    _check_manifest_tiles_match_filesystem(manifest, region_dir)
    _check_schema_version_consistency(manifest, region_dir)
    _check_manifest_provenance_sha_matches_disk(manifest, region_dir)
    _check_provenance_outputs_sha_match_files(manifest, region_dir)


# ---------------------------------------------------------------------------
# Invariant checkers
# ---------------------------------------------------------------------------


def _check_schema_version_consistency(manifest: RegionManifest, region_dir: Path) -> None:
    """Invariant 1: sub_c_schema_version consistency.

    manifest.sub_c_schema_version must equal every tile's meta.yaml
    schema_version AND every tile's provenance.yaml schema_version.
    """
    expected = manifest.sub_c_schema_version

    for tile_entry in manifest.tiles:
        tile_i = tile_entry["tile_i"]
        tile_j = tile_entry["tile_j"]
        tile_name = f"tile=EPSG3414_i{tile_i}_j{tile_j}"
        tile_dir = region_dir / tile_name

        # Check meta.yaml schema_version
        meta_path = tile_dir / "meta.yaml"
        meta_data = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        meta_schema_version = str(meta_data.get("schema_version", ""))
        if meta_schema_version != expected:
            raise TileValidationError(
                tile=tile_name,
                invariant="sub_c_schema_version_consistency",
                failed_row={"tile_i": tile_i, "tile_j": tile_j, "file": "meta.yaml"},
                detail={
                    "expected": expected,
                    "actual": meta_schema_version,
                    "source": "manifest.sub_c_schema_version vs meta.yaml.schema_version",
                },
            )

        # Check provenance.yaml schema_version
        prov_path = tile_dir / "provenance.yaml"
        prov_data = yaml.safe_load(prov_path.read_text(encoding="utf-8"))
        prov_schema_version = str(prov_data.get("schema_version", ""))
        if prov_schema_version != expected:
            raise TileValidationError(
                tile=tile_name,
                invariant="sub_c_schema_version_consistency",
                failed_row={"tile_i": tile_i, "tile_j": tile_j, "file": "provenance.yaml"},
                detail={
                    "expected": expected,
                    "actual": prov_schema_version,
                    "source": "manifest.sub_c_schema_version vs provenance.yaml.schema_version",
                },
            )


def _check_manifest_tiles_match_filesystem(manifest: RegionManifest, region_dir: Path) -> None:
    """Invariant 2: manifest.tiles[] inventory matches filesystem tree.

    No orphan tile=* dirs on disk not in the manifest.
    No missing tile dirs that ARE in the manifest.
    """
    # Build expected set from manifest
    manifest_tile_names: set[str] = set()
    for tile_entry in manifest.tiles:
        tile_i = tile_entry["tile_i"]
        tile_j = tile_entry["tile_j"]
        manifest_tile_names.add(f"tile=EPSG3414_i{tile_i}_j{tile_j}")

    # Build actual set from filesystem
    disk_tile_names: set[str] = {
        d.name for d in region_dir.iterdir() if d.is_dir() and d.name.startswith("tile=")
    }

    # Orphan dirs: on disk but not in manifest
    orphans = disk_tile_names - manifest_tile_names
    if orphans:
        orphan = sorted(orphans)[0]  # deterministic: report first alphabetically
        raise TileValidationError(
            tile=orphan,
            invariant="manifest_tiles_match_filesystem",
            failed_row={"tile_dir": orphan},
            detail={
                "reason": "tile dir on disk not listed in manifest.tiles[]",
                "orphan_dirs": sorted(orphans),
            },
        )

    # Missing dirs: in manifest but not on disk
    missing = manifest_tile_names - disk_tile_names
    if missing:
        missing_tile = sorted(missing)[0]  # deterministic: report first alphabetically
        raise TileValidationError(
            tile=missing_tile,
            invariant="manifest_tiles_match_filesystem",
            failed_row={"tile_dir": missing_tile},
            detail={
                "reason": "manifest.tiles[] entry has no corresponding dir on disk",
                "missing_dirs": sorted(missing),
            },
        )


def _check_manifest_provenance_sha_matches_disk(manifest: RegionManifest, region_dir: Path) -> None:
    """Invariant 3: manifest.tiles[i].provenance_sha256 matches disk.

    For each tile, re-derive the provenance sha the same way aggregate_tile_inventory
    did: yaml.safe_load(provenance.yaml) → compute_sha256_excluding(..., "provenance.yaml").
    This strips EXCLUDED_FROM_SHA paths (extracted_utc, any *_sha256 fields) before
    hashing, canonicalizes to YAML, and hashes the UTF-8 bytes.
    """
    for tile_entry in manifest.tiles:
        tile_i = tile_entry["tile_i"]
        tile_j = tile_entry["tile_j"]
        stored_sha = tile_entry["provenance_sha256"]
        tile_name = f"tile=EPSG3414_i{tile_i}_j{tile_j}"
        tile_dir = region_dir / tile_name

        prov_path = tile_dir / "provenance.yaml"
        prov_data = yaml.safe_load(prov_path.read_text(encoding="utf-8"))
        computed_sha = compute_sha256_excluding(prov_data, "provenance.yaml")

        if computed_sha != stored_sha:
            raise TileValidationError(
                tile=tile_name,
                invariant="manifest_provenance_sha_matches_disk",
                failed_row={"tile_i": tile_i, "tile_j": tile_j},
                detail={
                    "stored_sha": stored_sha,
                    "computed_sha": computed_sha,
                },
            )


def _check_provenance_outputs_sha_match_files(manifest: RegionManifest, region_dir: Path) -> None:
    """Invariant 4: provenance.outputs.*_sha256 match actual file bytes on disk.

    For each tile, for each key in {cells_parquet_sha256, features_parquet_sha256,
    crossings_parquet_sha256, meta_yaml_sha256}: read the stored sha from
    provenance.yaml and compare it to compute_sha256(path.read_bytes()).

    NOTE: outputs digests hash the RAW FILE BYTES, not canonicalized content.
    The orchestrator writes the file then reads path.read_bytes() and hashes;
    the cross-tile validator does the same comparison.
    """
    for tile_entry in manifest.tiles:
        tile_i = tile_entry["tile_i"]
        tile_j = tile_entry["tile_j"]
        tile_name = f"tile=EPSG3414_i{tile_i}_j{tile_j}"
        tile_dir = region_dir / tile_name

        prov_path = tile_dir / "provenance.yaml"
        prov_data = yaml.safe_load(prov_path.read_text(encoding="utf-8"))
        outputs = prov_data.get("outputs", {})

        for sha_key, filename in _OUTPUTS_FILENAMES.items():
            stored_sha = outputs.get(sha_key, "")
            file_path = tile_dir / filename
            actual_sha = compute_sha256(file_path.read_bytes())
            if actual_sha != stored_sha:
                raise TileValidationError(
                    tile=tile_name,
                    invariant="provenance_outputs_sha_match_files",
                    failed_row={"tile_i": tile_i, "tile_j": tile_j, "file": filename},
                    detail={
                        "sha_key": sha_key,
                        "stored_sha": stored_sha,
                        "actual_sha": actual_sha,
                    },
                )

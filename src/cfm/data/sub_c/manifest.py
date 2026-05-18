"""RegionManifest dataclass + write/read helpers for sub-C tile extraction.

Implements spec §11.7 (manifest.yaml schema), §11.8 (_SUCCESS semantics + write
order), and §14.6 (digest chain — EXCLUDED_FROM_SHA already defined in Task 8).

Write order (locked, from spec §11.8):
  1. Per-tile writers (cells, features, crossings, meta, provenance) — Task 9.
  2. write_manifest — called in main process after all tiles are done.
  3. Cross-tile validator (§12.2) — external gate (Task 12/14).
  4. write_success_marker — called IFF cross-tile validator passes.

IMPORTANT: initial_extraction.commit_sha is frozen after the first full
extraction.  write_manifest does NOT enforce this; the orchestrator (Task 12/14)
is responsible.  started_utc and completed_utc are excluded from sha computation
(see EXCLUDED_FROM_SHA in determinism.py) but are always written to the file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from cfm.data.sub_c.determinism import compute_sha256_excluding
from cfm.data.sub_c.io import TileProvenance, canonicalize_yaml, write_provenance_yaml  # noqa: F401

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionManifest:
    """Structured representation of manifest.yaml (spec §11.7).

    Fields map 1:1 to the YAML top-level keys.  Nested structures (config,
    conditioning_defaults, initial_extraction, tiles) are plain dicts so that
    callers can evolve their contents without a schema_version bump on the
    dataclass itself.

    tiles is a list of {tile_i, tile_j, provenance_sha256} dicts.  write_manifest
    sorts them by (tile_i, tile_j) before serialisation — callers need not
    pre-sort.
    """

    schema_version: str
    sub_c_schema_version: str
    release: str
    region: str
    region_crs: str
    admin_polygon_source: str
    admin_polygon_sha256: str
    densified_admin_polygon_sha256: str
    sea_polygons_sha256: str
    policy_yaml_sha256: str
    vocab_yaml_sha256: str
    config: dict
    conditioning_defaults: dict
    initial_extraction: dict
    tiles: list[dict]  # sorted by (tile_i, tile_j) for byte-determinism


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_manifest(manifest: RegionManifest, path: Path) -> None:
    """Write the manifest to *path* (typically <region_dir>/manifest.yaml).

    Tiles are sorted by (tile_i, tile_j) before serialisation regardless of the
    order they appear in the dataclass.  canonicalize_yaml from io.py is used for
    byte-determinism (sorted keys, block style).

    NOTE: initial_extraction.commit_sha must be set by the orchestrator before
    calling this function.  It is frozen after the first full extraction; this
    function does not enforce that invariant — it just writes whatever is passed.
    """
    sorted_tiles = sorted(manifest.tiles, key=lambda t: (t["tile_i"], t["tile_j"]))

    data: dict = {
        "schema_version": manifest.schema_version,
        "sub_c_schema_version": manifest.sub_c_schema_version,
        "release": manifest.release,
        "region": manifest.region,
        "region_crs": manifest.region_crs,
        "admin_polygon_source": manifest.admin_polygon_source,
        "admin_polygon_sha256": manifest.admin_polygon_sha256,
        "densified_admin_polygon_sha256": manifest.densified_admin_polygon_sha256,
        "sea_polygons_sha256": manifest.sea_polygons_sha256,
        "policy_yaml_sha256": manifest.policy_yaml_sha256,
        "vocab_yaml_sha256": manifest.vocab_yaml_sha256,
        "config": manifest.config,
        "conditioning_defaults": manifest.conditioning_defaults,
        "initial_extraction": manifest.initial_extraction,
        "tiles": sorted_tiles,
    }

    path.write_text(canonicalize_yaml(data), encoding="utf-8")


def write_success_marker(region_dir: Path) -> None:
    """Touch <region_dir>/_SUCCESS as a zero-byte sentinel file.

    Per spec §11.8: this MUST be called LAST — after write_manifest AND after
    the cross-tile validator passes.  Callers (Task 14 orchestrator) are
    responsible for enforcing that ordering.

    Idempotent: if _SUCCESS already exists it is overwritten with zero bytes.
    This supports the full-re-extraction protocol (delete _SUCCESS first is
    recommended but not required here).
    """
    success_path = region_dir / "_SUCCESS"
    success_path.write_bytes(b"")


def read_manifest(path: Path) -> RegionManifest:
    """Load *path* (manifest.yaml) and return a RegionManifest.

    Tiles list is returned as-is from the YAML (already sorted by the writer).
    No validation beyond YAML parsing is performed here; the cross-tile
    validator (Task 12) handles semantic checks.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    return RegionManifest(
        schema_version=str(data["schema_version"]),
        sub_c_schema_version=str(data["sub_c_schema_version"]),
        release=str(data["release"]),
        region=str(data["region"]),
        region_crs=str(data["region_crs"]),
        admin_polygon_source=str(data["admin_polygon_source"]),
        admin_polygon_sha256=str(data["admin_polygon_sha256"]),
        densified_admin_polygon_sha256=str(data["densified_admin_polygon_sha256"]),
        sea_polygons_sha256=str(data["sea_polygons_sha256"]),
        policy_yaml_sha256=str(data["policy_yaml_sha256"]),
        vocab_yaml_sha256=str(data["vocab_yaml_sha256"]),
        config=data["config"],
        conditioning_defaults=data["conditioning_defaults"],
        initial_extraction=data["initial_extraction"],
        tiles=data["tiles"],
    )


def aggregate_tile_inventory(tile_provenances: list[TileProvenance]) -> list[dict]:
    """Build the sorted tiles[] list from per-tile TileProvenance objects.

    For each TileProvenance:
    - Reconstruct the provenance dict in the same shape as write_provenance_yaml.
    - Hash it via compute_sha256_excluding(data, "provenance.yaml") — this strips
      EXCLUDED_FROM_SHA paths (extracted_utc, any *_sha256 final-segment fields)
      before hashing, matching the integrity chain from spec §14.6.
    - Record {tile_i, tile_j, provenance_sha256} in the result.

    Result is sorted by (tile_i, tile_j) for byte-determinism; callers can rely
    on this ordering without a second sort.
    """
    entries: list[dict] = []
    for prov in tile_provenances:
        prov_data: dict = {
            "schema_version": prov.schema_version,
            "tile_i": prov.tile_i,
            "tile_j": prov.tile_j,
            "crs": prov.crs,
            "extraction": prov.extraction,
            "inputs": prov.inputs,
            "outputs": prov.outputs,
        }
        sha = compute_sha256_excluding(prov_data, "provenance.yaml")
        entries.append(
            {
                "tile_i": prov.tile_i,
                "tile_j": prov.tile_j,
                "provenance_sha256": sha,
            }
        )

    return sorted(entries, key=lambda e: (e["tile_i"], e["tile_j"]))

"""Tests for cfm.data.sub_c.manifest — RegionManifest + write/read protocol.

Named tests per plan Task 10:
1. test_write_manifest_yaml_keys_match_spec_11_7
2. test_manifest_tiles_sorted_by_ij
3. test_write_success_marker_creates_zero_byte_file
4. test_manifest_byte_deterministic_across_writes
5. test_manifest_excludes_started_utc_completed_utc_from_sha
6. test_aggregate_tile_inventory_computes_provenance_sha256
7. test_read_manifest_round_trip
8. test_manifest_config_section_pins_epsilon_constants
"""

from __future__ import annotations

from pathlib import Path

import yaml

from cfm.data.sub_c.determinism import compute_sha256_excluding
from cfm.data.sub_c.io import TileProvenance, write_provenance_yaml
from cfm.data.sub_c.manifest import (
    RegionManifest,
    aggregate_tile_inventory,
    read_manifest,
    write_manifest,
    write_success_marker,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_manifest(tiles: list[dict] | None = None) -> RegionManifest:
    """Build a minimal but spec-complete RegionManifest for tests."""
    if tiles is None:
        tiles = [
            {"tile_i": 12, "tile_j": 18, "provenance_sha256": "aaaa" * 16},
            {"tile_i": 12, "tile_j": 17, "provenance_sha256": "bbbb" * 16},
        ]
    return RegionManifest(
        schema_version="1.0",
        sub_c_schema_version="1.0",
        release="2026-04-15.0",
        region="singapore",
        region_crs="EPSG:3414",
        admin_polygon_source="overture://divisions:country:SG",
        admin_polygon_sha256="a" * 64,
        densified_admin_polygon_sha256="b" * 64,
        sea_polygons_sha256="c" * 64,
        policy_yaml_sha256="d" * 64,
        vocab_yaml_sha256="e" * 64,
        config={
            "tile_size_m": 2000,
            "cell_size_m": 250,
            "cell_grid": [8, 8],
            "epsilon_ratio": 1.0e-9,
            "epsilon_coord_m": 1.0e-6,
            "epsilon_area_m2": 1.0e-6,
            "epsilon_length_m": 1.0e-6,
            "sea_definition": "base.class IN {ocean, strait, bay} OR base.subtype = ocean",
            "sea_water_fraction_threshold": 1.0,
            "coastal_inland_river_min_river_length_m": 500.0,
            "pipeline_order": ["clip", "reproject", "partition", "sliver_drop", "sea_mask"],
        },
        conditioning_defaults={
            "country": "SG",
            "climate_zone": "tropical_rainforest",
        },
        initial_extraction={
            "commit_sha": "f" * 40,
            "started_utc": "2026-05-17T08:00:00Z",
            "completed_utc": "2026-05-17T08:42:31Z",
            "tile_count": 2,
        },
        tiles=tiles,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_write_manifest_yaml_keys_match_spec_11_7(tmp_path: Path) -> None:
    """All top-level keys required by spec §11.7 are present in the written YAML."""
    manifest = _make_manifest()
    manifest_path = tmp_path / "manifest.yaml"
    write_manifest(manifest, manifest_path)

    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    expected_top_level_keys = {
        "schema_version",
        "sub_c_schema_version",
        "release",
        "region",
        "region_crs",
        "admin_polygon_source",
        "admin_polygon_sha256",
        "densified_admin_polygon_sha256",
        "sea_polygons_sha256",
        "policy_yaml_sha256",
        "vocab_yaml_sha256",
        "config",
        "conditioning_defaults",
        "initial_extraction",
        "tiles",
    }
    assert expected_top_level_keys == set(data.keys())

    # tiles[] entries must carry all three required keys
    assert all("tile_i" in t and "tile_j" in t and "provenance_sha256" in t for t in data["tiles"])


def test_manifest_tiles_sorted_by_ij(tmp_path: Path) -> None:
    """tiles[] are sorted by (tile_i, tile_j) even when input is in scrambled order."""
    scrambled_tiles = [
        {"tile_i": 5, "tile_j": 3, "provenance_sha256": "cc" * 32},
        {"tile_i": 1, "tile_j": 9, "provenance_sha256": "dd" * 32},
        {"tile_i": 5, "tile_j": 1, "provenance_sha256": "ee" * 32},
        {"tile_i": 1, "tile_j": 2, "provenance_sha256": "ff" * 32},
    ]
    manifest = _make_manifest(tiles=scrambled_tiles)
    manifest_path = tmp_path / "manifest.yaml"
    write_manifest(manifest, manifest_path)

    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    tile_tuples = [(t["tile_i"], t["tile_j"]) for t in data["tiles"]]
    assert tile_tuples == sorted(tile_tuples), "tiles[] must be sorted by (tile_i, tile_j)"


def test_write_success_marker_creates_zero_byte_file(tmp_path: Path) -> None:
    """_SUCCESS is a zero-byte file; calling twice is idempotent."""
    write_success_marker(tmp_path)
    success_path = tmp_path / "_SUCCESS"
    assert success_path.exists(), "_SUCCESS must exist after write_success_marker"
    assert success_path.stat().st_size == 0, "_SUCCESS must be zero bytes"

    # Calling again is idempotent (no exception, still zero bytes)
    write_success_marker(tmp_path)
    assert success_path.stat().st_size == 0


def test_manifest_byte_deterministic_across_writes(tmp_path: Path) -> None:
    """Same RegionManifest written twice produces identical bytes."""
    manifest = _make_manifest()

    path_a = tmp_path / "manifest_a.yaml"
    path_b = tmp_path / "manifest_b.yaml"
    write_manifest(manifest, path_a)
    write_manifest(manifest, path_b)

    assert path_a.read_bytes() == path_b.read_bytes(), (
        "write_manifest must be byte-deterministic across calls with the same input"
    )


def test_manifest_excludes_started_utc_completed_utc_from_sha(tmp_path: Path) -> None:
    """initial_extraction.started_utc and .completed_utc are excluded from sha.

    Two manifests identical except for started_utc / completed_utc must produce
    the same sha256 when hashed via compute_sha256_excluding.
    """
    manifest_path_a = tmp_path / "manifest_a.yaml"
    manifest_path_b = tmp_path / "manifest_b.yaml"

    m_a = _make_manifest()
    write_manifest(m_a, manifest_path_a)

    # Build a second manifest with different timestamps but otherwise identical
    import dataclasses

    m_b = dataclasses.replace(
        m_a,
        initial_extraction={
            **m_a.initial_extraction,
            "started_utc": "2099-01-01T00:00:00Z",
            "completed_utc": "2099-01-01T01:00:00Z",
        },
    )
    write_manifest(m_b, manifest_path_b)

    data_a = yaml.safe_load(manifest_path_a.read_text(encoding="utf-8"))
    data_b = yaml.safe_load(manifest_path_b.read_text(encoding="utf-8"))

    sha_a = compute_sha256_excluding(data_a, "manifest.yaml")
    sha_b = compute_sha256_excluding(data_b, "manifest.yaml")

    assert sha_a == sha_b, (
        "started_utc and completed_utc must not affect the manifest sha256 "
        "(both are in EXCLUDED_FROM_SHA for 'manifest.yaml')"
    )


def test_aggregate_tile_inventory_computes_provenance_sha256(tmp_path: Path) -> None:
    """aggregate_tile_inventory: each tile's provenance_sha256 matches
    compute_sha256_excluding on the provenance.yaml content dict."""
    prov = TileProvenance(
        schema_version="1.0",
        tile_i=3,
        tile_j=7,
        crs="EPSG:3414",
        extraction={
            "commit_sha": "a" * 40,
            "extracted_utc": "2026-05-17T09:00:00Z",
            "rerun_count": 0,
            "rerun_reason": None,
        },
        inputs={
            "release": "2026-04-15.0",
            "admin_polygon_sha256": "b" * 64,
            "policy_yaml_sha256": "c" * 64,
            "vocab_yaml_sha256": "d" * 64,
        },
        outputs={
            "cells_parquet_sha256": "e" * 64,
            "features_parquet_sha256": "f" * 64,
            "crossings_parquet_sha256": "g" * 64,
            "meta_yaml_sha256": "h" * 64,
        },
    )

    # Write the provenance to a real file so we can load its dict
    prov_path = tmp_path / "provenance.yaml"
    write_provenance_yaml(prov, prov_path)
    prov_data = yaml.safe_load(prov_path.read_text(encoding="utf-8"))

    expected_sha = compute_sha256_excluding(prov_data, "provenance.yaml")

    tile_list = aggregate_tile_inventory([prov])
    assert len(tile_list) == 1
    entry = tile_list[0]

    assert entry["tile_i"] == 3
    assert entry["tile_j"] == 7
    assert entry["provenance_sha256"] == expected_sha


def test_read_manifest_round_trip(tmp_path: Path) -> None:
    """write_manifest then read_manifest returns a RegionManifest equal to input.

    Tiles are sorted during write so input tiles are pre-sorted for this test;
    comparison is on the parsed dataclass.
    """
    sorted_tiles = [
        {"tile_i": 1, "tile_j": 2, "provenance_sha256": "aa" * 32},
        {"tile_i": 1, "tile_j": 5, "provenance_sha256": "bb" * 32},
    ]
    manifest = _make_manifest(tiles=sorted_tiles)
    manifest_path = tmp_path / "manifest.yaml"
    write_manifest(manifest, manifest_path)

    loaded = read_manifest(manifest_path)

    assert loaded.schema_version == manifest.schema_version
    assert loaded.sub_c_schema_version == manifest.sub_c_schema_version
    assert loaded.release == manifest.release
    assert loaded.region == manifest.region
    assert loaded.region_crs == manifest.region_crs
    assert loaded.admin_polygon_source == manifest.admin_polygon_source
    assert loaded.admin_polygon_sha256 == manifest.admin_polygon_sha256
    assert loaded.densified_admin_polygon_sha256 == manifest.densified_admin_polygon_sha256
    assert loaded.sea_polygons_sha256 == manifest.sea_polygons_sha256
    assert loaded.policy_yaml_sha256 == manifest.policy_yaml_sha256
    assert loaded.vocab_yaml_sha256 == manifest.vocab_yaml_sha256
    assert loaded.config == manifest.config
    assert loaded.conditioning_defaults == manifest.conditioning_defaults
    assert loaded.initial_extraction == manifest.initial_extraction
    assert loaded.tiles == manifest.tiles


def test_manifest_config_section_pins_epsilon_constants(tmp_path: Path) -> None:
    """manifest.config contains the exact epsilon constants from spec §11.7."""
    manifest = _make_manifest()
    manifest_path = tmp_path / "manifest.yaml"
    write_manifest(manifest, manifest_path)

    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    cfg = data["config"]

    assert cfg["epsilon_ratio"] == 1.0e-9
    assert cfg["epsilon_coord_m"] == 1.0e-6
    assert cfg["epsilon_area_m2"] == 1.0e-6
    assert cfg["epsilon_length_m"] == 1.0e-6
    assert cfg["tile_size_m"] == 2000
    assert cfg["cell_size_m"] == 250
    assert cfg["cell_grid"] == [8, 8]
    assert cfg["sea_water_fraction_threshold"] == 1.0
    assert cfg["coastal_inland_river_min_river_length_m"] == 500.0
    assert cfg["pipeline_order"] == ["clip", "reproject", "partition", "sliver_drop", "sea_mask"]

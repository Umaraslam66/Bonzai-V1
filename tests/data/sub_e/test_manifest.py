from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

from cfm.data.sub_e.manifest import (
    SubEManifest,
    SubEManifestConfig,
    SubEManifestExtraction,
    SubEManifestInputs,
    SubEManifestTile,
    SubEManifestVersions,
    manifest_sha256,
    manifest_to_dict,
    write_manifest,
)


def _make_manifest(
    tile_count: int = 3,
    *,
    started_utc: str = "2026-05-21T12:00:00Z",
    completed_utc: str = "2026-05-21T12:05:00Z",
) -> SubEManifest:
    tiles = [
        SubEManifestTile(tile_i=i, tile_j=0, provenance_sha256="z" * 64) for i in range(tile_count)
    ]
    return SubEManifest(
        manifest_schema_version="1.0",
        sub_e_schema_version="1.0",
        release="2026-04-15.0",
        region="singapore",
        region_crs="EPSG:3414",
        inputs=SubEManifestInputs(
            sub_c_manifest_sha256="b" * 64,
            sub_c_region_dir="data/processed/sub_c/2026-04-15.0/singapore",
            sub_d_manifest_sha256="e" * 64,
            sub_d_region_dir="data/processed/sub_d/2026-04-15.0/singapore",
            boundary_vocab_sha256="0" * 64,
        ),
        versions=SubEManifestVersions(
            boundary_vocab_version="1.0",
            boundary_derivation_version="1.0",
        ),
        config_source="sub_d_manifest.config",
        config=SubEManifestConfig(
            cell_grid=(8, 8),
            internal_edge_count=112,
            external_edge_count=32,
        ),
        initial_extraction=SubEManifestExtraction(
            commit_sha="a" * 40,
            started_utc=started_utc,
            completed_utc=completed_utc,
            tile_count=tile_count,
        ),
        tiles=tiles,
    )


def test_manifest_writes_with_all_fields(tmp_path: Path) -> None:
    p = tmp_path / "manifest.yaml"
    write_manifest(p, _make_manifest())
    data = yaml.safe_load(p.read_text())
    assert data["region"] == "singapore"
    assert data["initial_extraction"]["tile_count"] == 3
    assert len(data["tiles"]) == 3
    assert data["config"]["cell_grid"] == [8, 8]


def test_manifest_tiles_sorted_by_tile_i_tile_j(tmp_path: Path) -> None:
    p = tmp_path / "manifest.yaml"
    manifest = _make_manifest(tile_count=3)
    # Shuffle tiles before write; manifest writer must sort.
    manifest = manifest.__class__(**{**manifest.__dict__, "tiles": list(reversed(manifest.tiles))})
    write_manifest(p, manifest)
    data = yaml.safe_load(p.read_text())
    ijs = [(t["tile_i"], t["tile_j"]) for t in data["tiles"]]
    assert ijs == sorted(ijs)


def test_manifest_is_byte_deterministic_on_rerun(tmp_path: Path) -> None:
    import hashlib

    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    m = _make_manifest()
    write_manifest(a, m)
    write_manifest(b, m)
    assert hashlib.sha256(a.read_bytes()).hexdigest() == hashlib.sha256(b.read_bytes()).hexdigest()


def test_manifest_sha256_excludes_started_and_completed_utc() -> None:
    """Spec §9.2: initial_extraction.started_utc and completed_utc are
    stripped before manifest_sha256. Two manifests differing ONLY in
    those timestamps must produce the same sha — otherwise cross-env
    determinism checks (spec §14) become noise.
    """
    a = _make_manifest()
    b = _make_manifest(started_utc="2026-12-01T09:30:42Z", completed_utc="2026-12-01T09:35:01Z")
    assert manifest_sha256(manifest_to_dict(a)) == manifest_sha256(manifest_to_dict(b))


def test_manifest_sha256_excludes_nested_sha_fields() -> None:
    """Spec §9.2: final-segment *_sha256 fields are stripped before manifest_sha256."""
    a = _make_manifest()
    a_inputs = replace(a.inputs, sub_c_manifest_sha256="9" * 64)
    b = replace(a, inputs=a_inputs)
    assert manifest_sha256(manifest_to_dict(a)) == manifest_sha256(manifest_to_dict(b))


def test_manifest_sha256_sensitive_to_semantic_changes() -> None:
    """Inverse guard: non-excluded field changes MUST shift the sha."""
    a = _make_manifest()
    b = replace(a, region="zurich")
    assert manifest_sha256(manifest_to_dict(a)) != manifest_sha256(manifest_to_dict(b))

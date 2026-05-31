"""Tests for sub-F Halt 6 version manifest and provisional region manifest."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from cfm.data.sub_d.versions import VersionNamespace, VersionRef, compare_version
from cfm.data.sub_f.manifest import (
    TASK6_VOCAB_SOURCE_PATHS,
    build_region_manifest,
    manifest_sha256,
    task6_vocab_sources,
)
from cfm.data.sub_f.versions import (
    SUB_F_ARTIFACT_FORMAT_VERSION,
    SUB_F_DERIVATION_VERSION,
    SUB_F_SCHEMA_VERSION,
    SUB_F_VALIDATOR_VERSION,
    SUB_F_VOCAB_VERSION,
    encode_sub_f_source_version,
    load_sub_f_source_version,
    sub_f_version_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OVERTURE_PIN_PATH = REPO_ROOT / "configs" / "data" / "overture_release.yaml"
SUB_C_SINGAPORE_MANIFEST_PATH = (
    REPO_ROOT / "data" / "processed" / "sub_c" / "2026-04-15.0" / "singapore" / "manifest.yaml"
)
SENTINEL_INVENTORY_PATH = REPO_ROOT / "configs" / "sub_f" / "sentinel_inventory.yaml"


def test_version_namespace_now_has_six_members_with_source():
    assert [m.name for m in VersionNamespace] == [
        "ARTIFACT_FORMAT",
        "DATA_SHAPE",
        "VOCAB",
        "DERIVATION",
        "VALIDATOR",
        "SOURCE",
    ]
    assert VersionNamespace.SOURCE.value == "source"


def test_compare_version_accepts_all_six_axes_including_source():
    for namespace in VersionNamespace:
        expected = VersionRef(namespace, "1.0")
        actual = VersionRef(namespace, "1.0")
        compare_version(namespace, expected, actual)

    source = load_sub_f_source_version()
    encoded = encode_sub_f_source_version(source)
    compare_version(
        VersionNamespace.SOURCE,
        VersionRef(VersionNamespace.SOURCE, encoded),
        VersionRef(VersionNamespace.SOURCE, encoded),
    )


def test_sub_f_version_manifest_returns_matching_namespaced_refs():
    refs = sub_f_version_manifest()

    assert set(refs) == set(VersionNamespace)
    for namespace, ref in refs.items():
        assert ref.namespace == namespace

    assert refs[VersionNamespace.SOURCE].value == encode_sub_f_source_version(
        load_sub_f_source_version()
    )


def test_source_version_is_composite_from_overture_pin_and_sub_c_manifest():
    source = load_sub_f_source_version()
    overture_pin = yaml.safe_load(OVERTURE_PIN_PATH.read_text(encoding="utf-8"))
    sub_c_manifest = yaml.safe_load(SUB_C_SINGAPORE_MANIFEST_PATH.read_text(encoding="utf-8"))

    assert source == {
        "overture_release": overture_pin["release"],
        "sub_c_schema_version": sub_c_manifest["sub_c_schema_version"],
        "sub_c_commit_sha": sub_c_manifest["initial_extraction"]["commit_sha"],
    }
    assert encode_sub_f_source_version(source) == (
        "overture=2026-04-15.0;subc_schema=1.1;subc_commit=12b1cdf8838d9f8b601ea4b2a859f905ee5ab368"
    )


def test_sub_f_non_source_version_constants():
    assert SUB_F_ARTIFACT_FORMAT_VERSION == "1.0"
    assert SUB_F_SCHEMA_VERSION == "1.0"
    assert SUB_F_VOCAB_VERSION == "1.0"
    # 1.1: cycle-1 N/S encoder fix (commit 98cdeb0) changed bref output for the
    # same input → derivation-axis bump distinguishes pre/post-cycle-1 artifacts.
    assert SUB_F_DERIVATION_VERSION == "1.1"
    # 1.1: cycle-3 validator fix (BP4 <unknown_*> key resolution) — verdict-only.
    assert SUB_F_VALIDATOR_VERSION == "1.1"


def test_region_manifest_has_six_version_fields_and_region_vocab_sources():
    manifest = build_region_manifest(
        region="singapore",
        release="2026-04-15.0",
        tile_entries=[
            {"tile_i": 2, "tile_j": 1, "provenance_sha256": "b" * 64},
            {"tile_i": 1, "tile_j": 2, "provenance_sha256": "a" * 64},
        ],
        vocab_sources=task6_vocab_sources(),
    )

    for field in [
        "sub_f_artifact_format_version",
        "sub_f_schema_version",
        "sub_f_vocab_version",
        "sub_f_derivation_version",
        "sub_f_validator_version",
        "sub_f_source_version",
    ]:
        assert field in manifest

    assert manifest["sub_f_source_version"] == load_sub_f_source_version()
    assert manifest["vocab_sources_status"] == "complete"
    assert "vocab_sources" in manifest
    assert all("vocab_sources" not in tile for tile in manifest["tiles"])
    assert [(t["tile_i"], t["tile_j"]) for t in manifest["tiles"]] == [(1, 2), (2, 1)]
    assert manifest["manifest_sha256"] == manifest_sha256(manifest)


def test_task6_vocab_sources_cover_all_locked_blueprints():
    # Renamed from ..._bp1_bp2_bp4_only: BP7 (boundary_reference_vocab.yaml) was
    # added once Task 7 locked it (close-checklist line 8, discharged at T15).
    sources = task6_vocab_sources()

    assert set(sources) == {
        "bp1_semantic_vocab",
        "bp4_unknown_family",
        "bp2_encoding_primitives",
        "bp7_boundary_reference_vocab",
    }
    assert [sources[key]["path"] for key in sources] == list(TASK6_VOCAB_SOURCE_PATHS)
    # BP7 boundary-ref source is now present (was explicitly absent pre-T15).
    assert sources["bp7_boundary_reference_vocab"]["path"] == (
        "configs/sub_f/boundary_reference_vocab.yaml"
    )
    assert all(Path(source["path"]).exists() for source in sources.values())


def test_manifest_sha_excludes_live_clock_and_sha_fields():
    manifest = build_region_manifest(
        region="singapore",
        release="2026-04-15.0",
        tile_entries=[{"tile_i": 0, "tile_j": 0, "provenance_sha256": "a" * 64}],
        vocab_sources=task6_vocab_sources(),
    )
    changed = deepcopy(manifest)
    changed["initial_extraction"]["started_utc"] = "2099-12-31T00:00:00Z"
    changed["initial_extraction"]["completed_utc"] = "2099-12-31T00:01:00Z"
    changed["tiles"][0]["provenance_sha256"] = "b" * 64
    changed["manifest_sha256"] = "c" * 64

    assert manifest_sha256(manifest) == manifest_sha256(changed)


def test_manifest_sha_changes_on_semantic_content_change():
    manifest = build_region_manifest(
        region="singapore",
        release="2026-04-15.0",
        tile_entries=[{"tile_i": 0, "tile_j": 0, "provenance_sha256": "a" * 64}],
        vocab_sources=task6_vocab_sources(),
    )
    changed = deepcopy(manifest)
    changed["region"] = "jakarta"

    assert manifest_sha256(manifest) != manifest_sha256(changed)


def test_source_and_derivation_axes_are_independent(tmp_path: Path):
    overture_pin = tmp_path / "overture_release.yaml"
    sub_c_manifest = tmp_path / "manifest.yaml"
    overture_pin.write_text("release: '2099-01-01.0'\n", encoding="utf-8")
    sub_c_manifest.write_text(
        yaml.safe_dump(
            {
                "sub_c_schema_version": "9.9",
                "initial_extraction": {"commit_sha": "f" * 40},
            }
        ),
        encoding="utf-8",
    )

    source = load_sub_f_source_version(
        overture_pin_path=overture_pin,
        sub_c_manifest_path=sub_c_manifest,
    )

    assert encode_sub_f_source_version(source) == (
        "overture=2099-01-01.0;subc_schema=9.9;subc_commit=" + "f" * 40
    )
    assert SUB_F_DERIVATION_VERSION == "1.1"

    manifest = build_region_manifest(
        region="singapore",
        release="2026-04-15.0",
        tile_entries=[],
        vocab_sources=task6_vocab_sources(),
    )
    manifest["sub_f_derivation_version"] = "2.0"
    assert manifest["sub_f_source_version"] == load_sub_f_source_version()


def test_sentinel_inventory_keeps_bp2_and_bp7_locked():
    data = yaml.safe_load(SENTINEL_INVENTORY_PATH.read_text(encoding="utf-8"))

    # Post commit 4c4f880 (2026-05-28 sentinel-inventory fix): BP2 status
    # carries the structural_sentinels-consumed note. Halt-2 revisit 2026-05-29
    # appended the direction 48->360 + relocation note.
    assert data["bp2_encoding_primitives"]["status"] == (
        "LOCKED at Halt 2 approval; "
        "structural_sentinels consumed at T8 plan-write 2026-05-28; "
        "direction widened 48->360 + relocated 396..443->511..870 at Halt-2 revisit 2026-05-29"
    )
    assert data["bp2_encoding_primitives"]["start_id"] == 300
    assert data["bp2_encoding_primitives"]["end_id"] == 1499
    assert "bp7_boundary_ref_placeholder" not in data
    assert data["bp7_boundary_ref"]["status"] == "LOCKED at Halt 7 approval"
    assert data["bp7_boundary_ref"]["start_id"] == 1500
    assert data["bp7_boundary_ref"]["end_id"] == 1599

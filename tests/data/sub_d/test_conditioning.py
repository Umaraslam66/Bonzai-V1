"""Tests for the sub-D effective_conditioning.yaml overlay (Task 10)."""

from __future__ import annotations

from pathlib import Path

import yaml

from cfm.data.sub_d.conditioning import (
    SUB_D_OWNED_FIELDS,
    build_effective_conditioning,
    write_effective_conditioning,
)


def _sub_c_meta_fixture(tile_i: int = 12, tile_j: int = 17) -> dict:
    """A realistic sub-C meta.yaml for one tile.

    Mirrors what sub-C wrote on the real Singapore extraction: schema_version
    bare (sub-C convention), conditioning_per_tile carrying admin_region,
    morphology_class, era_class, coastal_inland_river plus the sub-D-owned
    placeholder population_density_bucket with its _owner marker.
    """
    return {
        "schema_version": "1.1",
        "tile_i": tile_i,
        "tile_j": tile_j,
        "aggregates": {"kept_cell_count": 64},
        "config": {"sliver_drop_rule": "drop iff area < 0.01"},
        "conditioning_per_tile": {
            "admin_region": "Central Region",
            "morphology_class": "Asian-megacity",
            "era_class": "contemporary",
            "coastal_inland_river": 1,  # int8 enum: 0=inland, 1=coastal, etc.
            "population_density_bucket": None,  # sub-D fills this
            "population_density_bucket_owner": "sub-D",  # marker; not conditioning
        },
    }


def _sub_c_manifest_fixture() -> dict:
    return {
        "sub_c_schema_version": "1.1",
        "release": "2026-04-15.0",
        "region": "singapore",
        "region_crs": "EPSG:3414",
        "conditioning_defaults": {
            "country": "SG",
            "climate_zone": "tropical_rainforest",
        },
        "config": {
            "cell_grid": [8, 8],
            "cell_size_m": 250,
            "tile_size_m": 2000,
        },
    }


def _versions_fixture() -> dict:
    return {
        "sub_c_conditioning_schema_version": "1.1",
        "tile_population_density_vocab_version": "1.0",
        "tile_population_density_derivation_version": "1.0",
    }


def _digests_fixture() -> dict:
    return {
        "manifest_sha256": "a" * 64,
        "tile_meta_sha256": "b" * 64,
        "tile_provenance_sha256": "c" * 64,
    }


def test_effective_conditioning_copies_schema_driven_sub_c_owned_fields():
    """The copy rule is schema-driven, not a hand-picked allowlist.

    Every field in ``conditioning_per_tile`` and ``conditioning_defaults``
    that is not (a) suffixed ``_owner`` and (b) in ``SUB_D_OWNED_FIELDS``
    must appear verbatim in the output conditioning dict.
    """
    meta = _sub_c_meta_fixture()
    # Inject a synthetic future sub-C field that didn't exist when this test
    # was written; the schema-driven rule must copy it without any static
    # allowlist update.
    meta["conditioning_per_tile"]["future_sub_c_field"] = "some_value"

    data = build_effective_conditioning(
        meta=meta,
        manifest=_sub_c_manifest_fixture(),
        population_density_bucket=3,
        versions=_versions_fixture(),
        digests=_digests_fixture(),
    )

    cond = data["conditioning"]
    # All sub-C-owned per-tile fields copied.
    assert cond["admin_region"] == "Central Region"
    assert cond["morphology_class"] == "Asian-megacity"
    assert cond["era_class"] == "contemporary"
    assert cond["coastal_inland_river"] == 1
    # Manifest's conditioning_defaults copied too.
    assert cond["country"] == "SG"
    assert cond["climate_zone"] == "tropical_rainforest"
    # The injected future field is forwarded — proves the rule is schema-driven.
    assert cond["future_sub_c_field"] == "some_value"


def test_effective_conditioning_fills_population_density_bucket():
    """The sub-D-owned ``population_density_bucket`` field is filled from
    the caller's pre-computed value (Task 14's pipeline derives it from the
    locked p75 proxy + locked_buckets boundaries; Task 10 just accepts it).
    """
    data = build_effective_conditioning(
        meta=_sub_c_meta_fixture(),
        manifest=_sub_c_manifest_fixture(),
        population_density_bucket=2,
        versions=_versions_fixture(),
        digests=_digests_fixture(),
    )
    assert data["conditioning"]["population_density_bucket"] == 2
    # It must be one of the locked-vocab token_ids (small int, non-null).
    assert isinstance(data["conditioning"]["population_density_bucket"], int)
    # population_density_bucket is the only sub-D-owned conditioning field today.
    assert SUB_D_OWNED_FIELDS == {"population_density_bucket"}


def test_effective_conditioning_does_not_copy_owner_marker_as_conditioning():
    """Fields ending in ``_owner`` are markers about who fills a field, not
    conditioning values. They must not leak into the output conditioning
    dict.
    """
    meta = _sub_c_meta_fixture()
    # Add a second owner-suffix field to prove the rule is suffix-based,
    # not a hardcoded "population_density_bucket_owner" check.
    meta["conditioning_per_tile"]["climate_zone_owner"] = "sub-D-future"

    data = build_effective_conditioning(
        meta=meta,
        manifest=_sub_c_manifest_fixture(),
        population_density_bucket=3,
        versions=_versions_fixture(),
        digests=_digests_fixture(),
    )
    cond = data["conditioning"]
    assert "population_density_bucket_owner" not in cond
    assert "climate_zone_owner" not in cond
    # No field in cond ends with "_owner".
    assert not any(k.endswith("_owner") for k in cond.keys())


def test_effective_conditioning_records_composite_versions_and_digests():
    """The composite version surface (sub-C upstream + sub-D's own) and the
    sub-C input digests are recorded as separate blocks so validators can
    cross-check provenance without rummaging through conditioning.
    """
    data = build_effective_conditioning(
        meta=_sub_c_meta_fixture(),
        manifest=_sub_c_manifest_fixture(),
        population_density_bucket=3,
        versions=_versions_fixture(),
        digests=_digests_fixture(),
    )
    assert data["versions"]["sub_c_conditioning_schema_version"] == "1.1"
    assert data["versions"]["tile_population_density_vocab_version"] == "1.0"
    assert data["versions"]["tile_population_density_derivation_version"] == "1.0"

    assert data["sub_c_inputs"]["manifest_sha256"] == "a" * 64
    assert data["sub_c_inputs"]["tile_meta_sha256"] == "b" * 64
    assert data["sub_c_inputs"]["tile_provenance_sha256"] == "c" * 64


def test_effective_conditioning_schema_uses_effective_conditioning_schema_version():
    """Per spec §11.4, the artifact carries a namespaced
    ``effective_conditioning_schema_version`` field — not a bare
    ``schema_version`` like sub-C uses. This pins the per-artifact-format
    versioning discipline (one version per file format, not one global).
    """
    data = build_effective_conditioning(
        meta=_sub_c_meta_fixture(),
        manifest=_sub_c_manifest_fixture(),
        population_density_bucket=3,
        versions=_versions_fixture(),
        digests=_digests_fixture(),
    )
    assert "effective_conditioning_schema_version" in data
    assert data["effective_conditioning_schema_version"] == "1.0"
    # And conversely: no bare ``schema_version`` field at the top level.
    assert "schema_version" not in data


def test_effective_conditioning_yaml_is_canonical(tmp_path: Path):
    """write_effective_conditioning produces byte-deterministic YAML via
    the neutral canonicalize_yaml helper.
    """
    data = build_effective_conditioning(
        meta=_sub_c_meta_fixture(),
        manifest=_sub_c_manifest_fixture(),
        population_density_bucket=3,
        versions=_versions_fixture(),
        digests=_digests_fixture(),
    )
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    write_effective_conditioning(data, a)
    write_effective_conditioning(data, b)
    assert a.read_bytes() == b.read_bytes()

    # Round-trip parse.
    loaded = yaml.safe_load(a.read_text(encoding="utf-8"))
    assert loaded["tile_i"] == 12
    assert loaded["tile_j"] == 17
    assert loaded["conditioning"]["country"] == "SG"
    # YAML keys at every level sorted alphabetically (canonicalize_yaml
    # contract).
    top_keys = list(loaded.keys())
    assert top_keys == sorted(top_keys)

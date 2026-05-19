"""Tests for the sub-D region manifest.yaml (Task 12).

Sub-D's manifest is the region-level rollup written by Task 14's pipeline after
all per-tile artifacts are on disk. It carries:

- ``manifest_schema_version`` (namespaced per B7; no bare ``schema_version``).
- ``sub_d_schema_version`` at the top level (per spec §11.6).
- ``inputs``: sub-D's view of the consumed sub-C region (sha + path).
- ``versions``: locked vocab + per-namespace derivation versions (A1).
- ``config_source`` + ``config``: the entire sub-C ``manifest.config`` dict
  copied verbatim (B6, schema-driven; future sub-C config keys inherit with
  zero code change here). build_manifest enforces equality with
  ``sub_c_manifest["config"]`` at build time; the Task 13 validator
  cross-checks again at validation time.
- ``initial_extraction``: extraction provenance (commit_sha, started_utc,
  completed_utc, tile_count).
- ``tiles``: per-tile inventory sorted by (tile_i, tile_j), each entry
  carrying ``provenance_sha256`` from Task 11's
  ``provenance_sha256(tile_prov)`` — the chain-of-custody anchor the Task 13
  validator uses to detect drift between manifest and on-disk
  provenance.yaml.

``_SUCCESS`` is written by a separate explicit function — never by
``write_manifest`` — so the Task 14 pipeline can gate it on the cross-tile
validator's pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cfm.data.sub_d.errors import SubDValidationError
from cfm.data.sub_d.manifest import (
    MANIFEST_SCHEMA_VERSION,
    SUB_D_SCHEMA_VERSION,
    aggregate_tile_inventory,
    build_manifest,
    read_manifest,
    write_manifest,
    write_success_marker,
)
from cfm.data.sub_d.provenance import build_tile_provenance, provenance_sha256

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sub_c_config_fixture(**extra) -> dict:
    """A realistic sub-C ``manifest.config`` dict. Includes ``sliver_drop_rule``
    — a real sub-C key NOT shown in the spec §11.6 example. The "surprise key"
    is intentional: B6 says copy verbatim, so a future or undocumented sub-C
    config key must flow through with zero code change here.
    """
    base = {
        "cell_grid": [8, 8],
        "cell_size_m": 250,
        "tile_size_m": 2000,
        "internal_edge_count": 112,
        "external_edge_count": 32,
        "sliver_drop_rule": "drop iff area < 0.01",
    }
    base.update(extra)
    return base


def _sub_c_manifest_fixture(config: dict | None = None) -> dict:
    return {
        "schema_version": "1.1",
        "sub_c_schema_version": "1.1",
        "release": "2026-04-15.0",
        "region": "singapore",
        "region_crs": "EPSG:3414",
        "config": config if config is not None else _sub_c_config_fixture(),
        "conditioning_defaults": {
            "country": "SG",
            "climate_zone": "tropical_rainforest",
        },
    }


def _inputs_fixture() -> dict:
    return {
        "sub_c_manifest_sha256": "1" * 64,
        "sub_c_region_dir": "data/processed/sub_c/2026-04-15.0/singapore",
    }


def _versions_fixture() -> dict:
    return {
        "macro_plan_vocab_version": "1.0",
        "zoning_vocab_version": "1.0",
        "zoning_derivation_version": "1.0",
        "cell_density_vocab_version": "1.0",
        "cell_density_derivation_version": "1.0",
        "tile_population_density_vocab_version": "1.0",
        "tile_population_density_derivation_version": "1.0",
        "road_skeleton_vocab_version": "1.0",
        "road_skeleton_derivation_version": "1.0",
    }


def _initial_extraction_fixture(tile_count: int = 3) -> dict:
    return {
        "commit_sha": "abc" + "0" * 37,
        "started_utc": "2026-05-19T12:00:00Z",
        "completed_utc": "2026-05-19T12:10:00Z",
        "tile_count": tile_count,
    }


def _build_provenance(tile_i: int, tile_j: int) -> dict:
    """Build one tile-provenance dict via Task 11's builder, so the manifest
    test exercises the actual chain-of-custody hashing path."""
    return build_tile_provenance(
        tile_i=tile_i,
        tile_j=tile_j,
        extraction={
            "commit_sha": "abc" + "0" * 37,
            "extracted_utc": "2026-05-19T12:00:00Z",
            "rerun_count": 0,
            "rerun_reason": "initial",
        },
        inputs={
            "release": "2026-04-15.0",
            "sub_c_manifest_sha256": "1" * 64,
            "sub_c_tile_provenance_sha256": "2" * 64,
            "sub_c_cells_parquet_sha256": "3" * 64,
            "sub_c_features_parquet_sha256": "4" * 64,
            "sub_c_crossings_parquet_sha256": "5" * 64,
            "sub_c_meta_yaml_sha256": "6" * 64,
            "macro_vocab_sha256": "7" * 64,
            "derivation_config_sha256": "8" * 64,
        },
        versions={
            "sub_d_schema_version": SUB_D_SCHEMA_VERSION,
            "macro_plan_vocab_version": "1.0",
            "zoning_vocab_version": "1.0",
            "zoning_derivation_version": "1.0",
            "cell_density_vocab_version": "1.0",
            "cell_density_derivation_version": "1.0",
            "tile_population_density_vocab_version": "1.0",
            "tile_population_density_derivation_version": "1.0",
            "road_skeleton_vocab_version": "1.0",
            "road_skeleton_derivation_version": "1.0",
        },
        outputs={
            "macro_core_parquet_sha256": "a" * 64,
            "derivation_evidence_parquet_sha256": "b" * 64,
            "effective_conditioning_yaml_sha256": "c" * 64,
        },
    )


def _build_manifest_fixture(**overrides) -> dict:
    """Build a manifest dict using fixture defaults. Tests can override any
    named arg of ``build_manifest``."""
    sub_c_manifest = overrides.pop("sub_c_manifest", _sub_c_manifest_fixture())
    config = overrides.pop("config", sub_c_manifest["config"])
    tile_provenances = overrides.pop(
        "tile_provenances",
        [_build_provenance(1, 2), _build_provenance(1, 7), _build_provenance(5, 3)],
    )
    kwargs: dict = {
        "release": "2026-04-15.0",
        "region": "singapore",
        "region_crs": "EPSG:3414",
        "sub_c_manifest": sub_c_manifest,
        "inputs": _inputs_fixture(),
        "versions": _versions_fixture(),
        "config": config,
        "initial_extraction": _initial_extraction_fixture(tile_count=len(tile_provenances)),
        "tile_provenances": tile_provenances,
    }
    kwargs.update(overrides)
    return build_manifest(**kwargs)


# ---------------------------------------------------------------------------
# Plan-named tests
# ---------------------------------------------------------------------------


def test_manifest_schema_uses_manifest_schema_version():
    """Per spec §11.6 + tension flag B7, sub-D's manifest.yaml carries a
    namespaced ``manifest_schema_version`` field. The bare ``schema_version``
    convention from sub-C must NOT leak in (each per-artifact format has its
    own namespaced version per the sub-D discipline established at Task 10
    and Task 11).
    """
    data = _build_manifest_fixture()
    assert "manifest_schema_version" in data
    assert data["manifest_schema_version"] == "1.0"
    assert data["manifest_schema_version"] == MANIFEST_SCHEMA_VERSION
    # Conversely: no bare ``schema_version`` at the top level (same explicit-
    # negative-assertion pattern as Task 10's effective_conditioning and
    # Task 11's provenance).
    assert "schema_version" not in data
    # sub_d_schema_version IS at the top level per spec §11.6 — that field
    # name is namespaced, so it does not violate B7.
    assert data["sub_d_schema_version"] == SUB_D_SCHEMA_VERSION


def test_manifest_tiles_sorted_by_tile_i_tile_j():
    """``tiles`` is sorted by ``(tile_i, tile_j)`` for byte-determinism,
    regardless of the order ``tile_provenances`` are passed in. Two runs that
    pass the same provenances in different orders must produce the same
    manifest bytes.
    """
    p_5_3 = _build_provenance(5, 3)
    p_1_7 = _build_provenance(1, 7)
    p_1_2 = _build_provenance(1, 2)

    data = _build_manifest_fixture(tile_provenances=[p_5_3, p_1_7, p_1_2])

    sort_keys = [(t["tile_i"], t["tile_j"]) for t in data["tiles"]]
    assert sort_keys == [(1, 2), (1, 7), (5, 3)]

    # Cross-check: aggregate_tile_inventory by itself also sorts (so callers
    # outside build_manifest can rely on the same ordering — Task 13's
    # validator reads the tiles[] list back).
    inventory = aggregate_tile_inventory([p_5_3, p_1_7, p_1_2])
    inv_keys = [(t["tile_i"], t["tile_j"]) for t in inventory]
    assert inv_keys == [(1, 2), (1, 7), (5, 3)]


def test_manifest_config_copied_from_sub_c_and_validated():
    """B6: sub-D's manifest copies the entire ``sub_c_manifest["config"]``
    dict verbatim. NOT a hand-picked subset of keys (cell_grid, cell_size_m,
    etc.) from the spec §11.6 example. A future sub-C config key inherits
    into sub-D's manifest with zero code change here — the rule is
    schema-driven, not allowlist-driven.

    "validated": ``build_manifest`` enforces ``config ==
    sub_c_manifest["config"]`` at build time and raises
    ``SubDValidationError`` if the caller passes a drifted config. (Task 13's
    region validator will check again at validation time; defense in depth.)
    """
    # Happy path: config_extra simulates a hypothetical future sub-C key plus
    # the real ``sliver_drop_rule`` key that is NOT in the spec §11.6 example.
    sub_c_config = _sub_c_config_fixture(
        future_sub_c_config_key="some_future_value",
    )
    sub_c_manifest = _sub_c_manifest_fixture(config=sub_c_config)
    data = _build_manifest_fixture(sub_c_manifest=sub_c_manifest, config=sub_c_config)

    # Full dict copied verbatim — no hand-picked subset.
    assert data["config"] == sub_c_config
    # The validator-facing marker is recorded.
    assert data["config_source"] == "sub_c_manifest.config"
    # Specifically: the keys that the spec §11.6 example LISTS are present.
    for spec_key in (
        "cell_grid",
        "cell_size_m",
        "tile_size_m",
        "internal_edge_count",
        "external_edge_count",
    ):
        assert spec_key in data["config"], spec_key
    # AND the keys NOT shown in the spec example are also present — proving
    # the rule is schema-driven, not a hand-picked allowlist.
    assert data["config"]["sliver_drop_rule"] == "drop iff area < 0.01"
    assert data["config"]["future_sub_c_config_key"] == "some_future_value"

    # Sad path: drifted config (caller hand-edits one key) raises at build
    # time. This catches a caller bug before any artifact is written.
    drifted_config = dict(sub_c_config)
    drifted_config["tile_size_m"] = 9999  # drift
    with pytest.raises(SubDValidationError):
        _build_manifest_fixture(sub_c_manifest=sub_c_manifest, config=drifted_config)

    # Sad path #2: hand-picked subset of sub-C config (the exact failure B6
    # warns against) also raises. A subset that drops keys is itself drift.
    subset_config = {
        "cell_grid": sub_c_config["cell_grid"],
        "cell_size_m": sub_c_config["cell_size_m"],
        "tile_size_m": sub_c_config["tile_size_m"],
        "internal_edge_count": sub_c_config["internal_edge_count"],
        "external_edge_count": sub_c_config["external_edge_count"],
        # Deliberately drops sliver_drop_rule + future_sub_c_config_key.
    }
    with pytest.raises(SubDValidationError):
        _build_manifest_fixture(sub_c_manifest=sub_c_manifest, config=subset_config)


def test_manifest_provenance_sha_matches_tile_provenance_sha():
    """The inventory digest chain: each ``tiles[]`` entry records
    ``provenance_sha256`` derived from Task 11's ``provenance_sha256(prov)``.
    This is the chain-of-custody anchor that Task 13's validator will use to
    detect drift between the manifest and the on-disk ``provenance.yaml``.

    Tested by exercising the actual ``build_tile_provenance`` →
    ``provenance_sha256`` → ``aggregate_tile_inventory`` chain, not a
    pre-computed sha — so if any link's hash semantics change, this test
    fails.
    """
    prov_a = _build_provenance(3, 4)
    prov_b = _build_provenance(7, 1)
    expected_a = provenance_sha256(prov_a)
    expected_b = provenance_sha256(prov_b)

    data = _build_manifest_fixture(tile_provenances=[prov_a, prov_b])

    # Map tile to its recorded sha for assertion robustness against ordering.
    tile_shas = {(t["tile_i"], t["tile_j"]): t["provenance_sha256"] for t in data["tiles"]}
    assert tile_shas[(3, 4)] == expected_a
    assert tile_shas[(7, 1)] == expected_b

    # Different tile inputs ⇒ different shas (sanity: the test is comparing
    # real shas, not a hard-coded constant).
    assert expected_a != expected_b


def test_success_marker_written_only_by_explicit_function(tmp_path: Path):
    """Spec §11.8 (sub-C precedent): ``_SUCCESS`` is written ONLY by
    ``write_success_marker`` — never as a side effect of ``write_manifest``.
    The Task 14 pipeline gates the explicit ``write_success_marker`` call on
    the cross-tile validator passing. If ``write_manifest`` created
    ``_SUCCESS`` itself, that gate would be bypassed.
    """
    data = _build_manifest_fixture()
    manifest_path = tmp_path / "manifest.yaml"
    write_manifest(data, manifest_path)

    # write_manifest does NOT create _SUCCESS.
    assert manifest_path.exists()
    assert not (tmp_path / "_SUCCESS").exists()

    # Only an explicit write_success_marker call creates it.
    write_success_marker(tmp_path)
    assert (tmp_path / "_SUCCESS").exists()
    assert (tmp_path / "_SUCCESS").read_bytes() == b""


# ---------------------------------------------------------------------------
# Companion: byte-determinism + round-trip read. Not in the plan's 5 named
# tests, but small and pinned the same way Task 10 did for conditioning and
# Task 11 did for provenance.
# ---------------------------------------------------------------------------


def test_write_manifest_is_byte_deterministic_and_round_trips(tmp_path: Path):
    data = _build_manifest_fixture()
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    write_manifest(data, a)
    write_manifest(data, b)
    assert a.read_bytes() == b.read_bytes()

    loaded = read_manifest(a)
    assert loaded["manifest_schema_version"] == "1.0"
    assert loaded["region"] == "singapore"
    assert len(loaded["tiles"]) == 3
    # Tiles still sorted on read.
    assert [(t["tile_i"], t["tile_j"]) for t in loaded["tiles"]] == [
        (1, 2),
        (1, 7),
        (5, 3),
    ]

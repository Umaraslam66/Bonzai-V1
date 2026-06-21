from __future__ import annotations

import pytest

from cfm.eval import lane_s_sampler as ls


def _minimal_payload() -> dict:
    return {
        "sampler_schema_version": ls.SAMPLER_SCHEMA_VERSION,
        "release": "test.0",
        "floor_sha256": "deadbeef",
        "methodology": {
            "target_features": 50,
            "headroom": 2.0,
            "seed": 7,
            "selection": "blake2b_hash_rank",
        },
        "held_out_cities": ["glasgow"],
        "strata": [],
        "cells": [],
    }


def test_seal_then_verified_load_round_trips(tmp_path):
    path = tmp_path / "sampler-manifest.yaml"
    ls.seal_manifest(_minimal_payload(), path)
    assert (tmp_path / ls.SAMPLER_LOCK_NAME).exists()
    loaded = ls.load_verified_manifest(path)
    assert loaded["release"] == "test.0"
    assert loaded["methodology"]["target_features"] == 50


def test_verified_load_refuses_tampered_content(tmp_path):
    path = tmp_path / "sampler-manifest.yaml"
    ls.seal_manifest(_minimal_payload(), path)
    text = path.read_text().replace("release: test.0", "release: tampered.9")
    path.write_text(text)
    with pytest.raises(ls.SamplerArtifactError, match="sha mismatch"):
        ls.load_verified_manifest(path)


def test_seal_is_write_once(tmp_path):
    path = tmp_path / "sampler-manifest.yaml"
    ls.seal_manifest(_minimal_payload(), path)
    with pytest.raises(FileExistsError):
        ls.seal_manifest(_minimal_payload(), path)


def test_binding_metric_is_building_when_owed():
    assert ls.binding_metric(frozenset({ls.BUILDING_METRIC, ls.ROAD_METRIC})) == ls.BUILDING_METRIC
    assert ls.binding_metric(frozenset({ls.ROAD_METRIC})) == ls.ROAD_METRIC


def test_size_stratum_ceiling_bound_depends_only_on_floor_n():
    # floor_n=50, target=50, headroom=1.0 -> raw=ceil(50*1*A/50)=A -> NOT ceiling-bound
    r = ls.size_stratum(target_features=50, headroom=1.0, floor_n_binding=50, available_cells=200)
    assert not r.ceiling_bound and r.n_cells_selected == 200 and r.n_cells_target == 200
    # floor_n=59 (the real min), headroom=2.0 -> target*headroom=100 > 59 -> ceiling-bound
    r2 = ls.size_stratum(target_features=50, headroom=2.0, floor_n_binding=59, available_cells=40)
    assert r2.ceiling_bound and r2.n_cells_selected == 40  # take-all
    # plentiful stratum: floor_n=3000, headroom=2.0, available big -> small draw, not ceiling
    r3 = ls.size_stratum(
        target_features=50, headroom=2.0, floor_n_binding=3000, available_cells=2000
    )
    assert not r3.ceiling_bound and r3.n_cells_selected < 2000


def test_size_stratum_rejects_unfloored_n():
    with pytest.raises(ValueError, match="floor_n_binding"):
        ls.size_stratum(target_features=50, headroom=2.0, floor_n_binding=0, available_cells=10)

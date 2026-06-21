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

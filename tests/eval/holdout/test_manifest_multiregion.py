from __future__ import annotations

import pytest

from cfm.eval.holdout.manifest import (
    HoldoutDeclarationError,
    build_holdout_manifest_multiregion,
)

REG = {  # minimal per-region payload (real build sources these from G4 + sub-D provenance)
    "krakow": dict(
        morphology="medieval-organic",
        density="moderate",
        geography="PL",
        crs="EPSG:25834",
        n_tiles=2,
        tokens=100,
        tiles=[
            dict(tile_i=0, tile_j=0, provenance_sha256="a", macro_vocab_sha256="v"),
            dict(tile_i=0, tile_j=1, provenance_sha256="b", macro_vocab_sha256="v"),
        ],
    ),
}


def test_schema_and_whole_city_declaration():
    m = build_holdout_manifest_multiregion(
        REG,
        corpus_release="2026-04-15.0",
        derivation_version="1.2",
        train_cities={"hamburg"},
        corpus_tile_counts={"krakow": 2},
    )
    assert m["manifest_schema_version"] == "2.0"
    assert m["regions"]["krakow"]["holdout_kind"] == "whole_city"
    assert m["held_out_cities"] == ["krakow"]


def test_assertion_a_holdout_train_disjoint():
    with pytest.raises(HoldoutDeclarationError, match="on both sides"):
        build_holdout_manifest_multiregion(
            REG,
            corpus_release="2026-04-15.0",
            derivation_version="1.2",
            train_cities={"krakow"},
            corpus_tile_counts={"krakow": 2},
        )


def test_assertion_b_tiles_match_frozen_corpus_set():
    # enumerated 2 tiles but corpus has 3 -> NOT a match (must not drop tiles)
    with pytest.raises(HoldoutDeclarationError, match="matches frozen-corpus tile set"):
        build_holdout_manifest_multiregion(
            REG,
            corpus_release="2026-04-15.0",
            derivation_version="1.2",
            train_cities={"hamburg"},
            corpus_tile_counts={"krakow": 3},
        )


def test_sg_builder_still_schema_1_0():
    # SG schema is independent; never bumped by the 2.0 work (the frozen SG manifest's
    # manifest_sha256 is computed over manifest_schema_version: '1.0').
    from cfm.eval.holdout.manifest import build_holdout_manifest

    m = build_holdout_manifest(
        region="singapore",
        selected_tiles=[(1, 7)],
        per_tile_provenance={(1, 7): {"provenance_sha256": "x", "macro_vocab_sha256": "y"}},
    )
    assert m["manifest_schema_version"] == "1.0"

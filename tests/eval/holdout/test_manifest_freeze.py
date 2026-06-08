"""Write-once + byte-determinism freeze test on a multi-region manifest (plan T6).

The freeze is the point-of-no-return for the eval set, so two properties are
non-negotiable and tested here on a synthetic (no-corpus) multiregion manifest:

- **write-once:** ``freeze_holdout_manifest`` refuses to overwrite a locked path
  (raises ``FileExistsError``) — a contaminated/re-derived holdout invalidates every
  eval number, so the artifact never moves once frozen.
- **byte-determinism:** freezing the SAME manifest dict to two distinct paths
  produces byte-identical files. This holds WITHOUT touching ``canonicalize_yaml``
  because the multiregion builder already sorts tiles by (tile_i, tile_j) and
  ``held_out_cities``, and ``canonicalize_yaml`` writes with ``sort_keys=True``.
"""

from __future__ import annotations

import pytest

from cfm.eval.holdout.manifest import (
    build_holdout_manifest_multiregion,
    freeze_holdout_manifest,
)

REG = {
    "krakow": dict(
        morphology="medieval-organic",
        density="moderate",
        geography="PL",
        crs="EPSG:25834",
        tokens=100,
        tiles=[
            dict(tile_i=0, tile_j=0, provenance_sha256="a", macro_vocab_sha256="v"),
            dict(tile_i=0, tile_j=1, provenance_sha256="b", macro_vocab_sha256="v"),
        ],
    )
}


def test_freeze_is_write_once_and_byte_deterministic(tmp_path):
    m = build_holdout_manifest_multiregion(
        REG,
        corpus_release="2026-04-15.0",
        derivation_version="1.2",
        train_cities={"hamburg"},
        corpus_tile_counts={"krakow": 2},
    )
    p = tmp_path / "holdout_manifest.yaml"
    freeze_holdout_manifest(m, p)
    first = p.read_bytes()
    with pytest.raises(FileExistsError):  # write-once
        freeze_holdout_manifest(m, p)
    p2 = tmp_path / "again.yaml"
    freeze_holdout_manifest(m, p2)
    assert p2.read_bytes() == first  # byte-deterministic

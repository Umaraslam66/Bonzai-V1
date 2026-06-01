from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cfm.eval.holdout import manifest


def _provs() -> dict:
    return {
        (1, 7): {"provenance_sha256": "a" * 64, "macro_vocab_sha256": "b" * 64},
        (1, 8): {"provenance_sha256": "c" * 64, "macro_vocab_sha256": "b" * 64},
    }


def test_build_manifest_is_region_keyed_and_sorted():
    data = manifest.build_holdout_manifest(
        region="singapore", selected_tiles=[(1, 8), (1, 7)], per_tile_provenance=_provs()
    )
    assert data["regions"]["singapore"]["partition_path"] == "holdout/region=singapore"
    tiles = data["regions"]["singapore"]["tiles"]
    assert [(t["tile_i"], t["tile_j"]) for t in tiles] == [(1, 7), (1, 8)]  # sorted
    assert tiles[0]["provenance_sha256"] == "a" * 64


def test_freeze_computes_sha_excluding_the_sha_field_and_writes_once(tmp_path: Path):
    data = manifest.build_holdout_manifest(
        region="singapore", selected_tiles=[(1, 7)], per_tile_provenance=_provs()
    )
    path = tmp_path / "holdout_manifest.yaml"
    manifest.freeze_holdout_manifest(data, path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert len(loaded["manifest_sha256"]) == 64
    # recompute over the loaded dict minus its sha -> identical (sha excludes itself):
    assert manifest.manifest_sha256(loaded) == loaded["manifest_sha256"]
    # written once: a second freeze refuses to overwrite the locked artifact.
    with pytest.raises(FileExistsError):
        manifest.freeze_holdout_manifest(data, path)

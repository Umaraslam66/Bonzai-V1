"""Task 6 real-data integration (slow): the full CellDataModule.setup() path on a
small real built manifest — proves the audit + in-memory build + split wire up
against actual Singapore sub-D/sub-F tiles, and that the real holdout audit passes
(the built tiles are validated-minus-holdout, so no leak)."""

from __future__ import annotations

import pytest

from cfm.data.training.build_shards import (
    _validated_inventory,
    _write_training_manifest,
    build_shards_in_memory,
    compute_training_tile_ids,
)
from cfm.data.training.datamodule import CellDataModule
from cfm.data.training.paths import training_manifest_path
from cfm.eval.holdout.paths import holdout_manifest_path

_RELEASE, _REGION = "2026-04-15.0", "singapore"


@pytest.fixture
def small_real_manifest(tmp_path):
    """A real training manifest restricted to the first 40 training tiles (fast)."""
    ids = compute_training_tile_ids(_RELEASE, _REGION)[:40]
    shards = build_shards_in_memory(_RELEASE, _REGION, tile_ids=ids)
    prov = {
        (int(e["tile_i"]), int(e["tile_j"])): e["provenance_sha256"]
        for e in _validated_inventory(_RELEASE, _REGION)
    }
    _write_training_manifest(tmp_path, _RELEASE, _REGION, shards, prov)
    return tmp_path / training_manifest_path(_RELEASE, _REGION).name


@pytest.mark.slow
def test_real_setup_audits_and_builds_disjoint_split(small_real_manifest):
    import yaml

    dm = CellDataModule(
        training_manifest=small_real_manifest,
        holdout_manifest=holdout_manifest_path(_RELEASE),
        seed=7,
        val_fraction=0.2,
        # Legacy SG thin-slice: audits the FROZEN, IMMUTABLE Singapore holdout manifest (schema 1.0;
        # can never be re-stamped to 2.0). This "1.0" opt-down makes THIS site ACCEPT a 1.0
        # manifest — correct ONLY for the SG set. DANGER on EU/bake-off reuse: leaving "1.0" here
        # while the manifest is (or defaults to) the SG 1.0 set silently audits the EU corpus
        # against the WRONG holdout (the #16 failure, one layer over). EU reuse MUST set "2.0" AND
        # re-point to multiregion_holdout_manifest_path. (Re-pointing to the EU 2.0 manifest but
        # forgetting to flip "1.0" fails loud — 2.0≠1.0 — which is fine.) See handoff residual.
        expected_holdout_schema="1.0",
    )
    dm.setup("fit")  # real holdout audit must pass (built tiles exclude holdout)

    assert len(dm.train_cells) > 0 and len(dm.val_cells) > 0  # non-vacuous

    holdout = yaml.safe_load(holdout_manifest_path(_RELEASE).read_text())
    held = {(t["tile_i"], t["tile_j"]) for t in holdout["regions"][_REGION]["tiles"]}
    val_tiles = {(c.tile_i, c.tile_j) for c in dm.val_cells}
    train_tiles = {(c.tile_i, c.tile_j) for c in dm.train_cells}
    assert val_tiles.isdisjoint(held)  # selection-loop leak guard
    assert train_tiles.isdisjoint(held)
    assert val_tiles.isdisjoint(train_tiles)  # tile-level split

    # a real batch is yieldable, and every cell is non-empty (flatten dropped empties)
    batch = next(iter(dm.train_dataloader()))
    assert batch["ids"].shape[0] >= 1
    assert (batch["seq_len"] > batch["prefix_len"]).all()  # body has >=1 cell token

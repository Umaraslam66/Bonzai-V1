"""Task 6 discrimination tests for CellDataModule.

Covers, fast (synthetic shards / synthetic manifests):
  * flatten drops empty + over-length cells (logged, not silent);
  * tile-level train/val split is deterministic and val is disjoint from train
    (=> disjoint from holdout by construction, since holdout tiles are never in
    the shards);
  * seeded order is reproducible across processes (resume-safe);
  * the conditioning prefix is the field-slot id-block (value-agnostic, slice v1);
  * collate right-pads and reports per-example prefix_len + seq_len.

Plus audit-halt wiring (synthetic planted-leak manifest): setup() RAISES before any
batch (zero steps) and a clean twin yields >=1 batch. The real 4->4 DDP resume +
all-ranks-halt with world_size==4 is validated on Slurm (Task 12), not here.
"""

from __future__ import annotations

import pytest
import yaml

from cfm.data.training import datamodule as DM
from cfm.data.training.conditioning import (
    CONDITIONING_ID_BASE,
    build_value_bearing_prefix,
    conditioning_field_to_id,
)
from cfm.data.training.shard_schema import CellPayload, TrainingShard
from cfm.eval.holdout.lineage_audit import HoldoutLeakError

_REGION = "singapore"

#: The full 6-key tile conditioning the production writer always emits
#: (build_shards._tile_conditioning_dict); flatten indexes these keys STRICTLY,
#: so a partial fixture dict would KeyError — which is the wanted loud failure.
_TILE_CONDITIONING = {
    "population_density_bucket": 0,
    "dominant_zoning_class": 0,
    "modal_road_skeleton_class": 1,
    "admin_region": None,
    "coastal_inland_river": 0,
    "sub_c_morphology_class": "Asian-megacity",
}


def _shard(ti, tj, cells, tile_conditioning=None):
    return TrainingShard(
        region=_REGION,
        tile_i=ti,
        tile_j=tj,
        tile_conditioning=dict(
            _TILE_CONDITIONING if tile_conditioning is None else tile_conditioning
        ),
        macro_tokens=(),
        cells=tuple(cells),
        lineage=frozenset({(_REGION, ti, tj)}),
    )


def _cell(ci, cj, n_tokens, density=1):
    return CellPayload(
        cell_i=ci,
        cell_j=cj,
        cell_slot_index=ci * 8 + cj,
        tokens=tuple(range(n_tokens)),
        cell_density_bucket=density,
        boundary_contracts=(),
    )


def _synthetic_shards():
    # 4 tiles; tile (0,0) has an empty cell + an over-length cell + a good cell.
    return [
        _shard(0, 0, [_cell(0, 0, 0), _cell(0, 1, 10_000), _cell(0, 2, 50)]),
        _shard(0, 1, [_cell(1, 1, 30)]),
        _shard(0, 2, [_cell(2, 2, 40)]),
        _shard(0, 3, [_cell(3, 3, 20)]),
    ]


def test_legacy_slot_builder_is_the_field_slot_id_block():
    """Pure unit test of the LEGACY value-agnostic slot builder — NOT the live
    training path (flatten now builds the value-bearing prefix; F6 delivery).
    Deliberate re-aim of the former test_conditioning_prefix_is_the_field_slot_id_block."""
    prefix = DM.build_conditioning_prefix()
    n = len(conditioning_field_to_id())
    assert prefix == [CONDITIONING_ID_BASE + i for i in range(n)]


def test_model_input_prefix_differs_across_differing_tile_conditioning():
    """THE F6 tooth: two shards differing in tile_conditioning must produce different
    example prefixes. RED on the constant slot prefix, GREEN after the value wire-in."""
    a = _shard(0, 0, [_cell(0, 0, 10)])
    b = _shard(0, 0, [_cell(0, 0, 10)], {**_TILE_CONDITIONING, "dominant_zoning_class": 1})
    ca = DM.flatten_shards_to_cells([a])[0][0]
    cb = DM.flatten_shards_to_cells([b])[0][0]
    assert ca.prefix_ids != cb.prefix_ids


def test_prefix_is_exactly_the_value_bearing_layout():
    """Mutual-exclusivity: the live prefix must equal build_value_bearing_prefix(...) of
    the shard's conditioning — no slot ids, no mixing (slot block == field-0 value block,
    so equality against the value layout rules the slot layout out entirely)."""
    shard = _shard(0, 0, [_cell(0, 0, 10, density=2)])
    examples, _ = DM.flatten_shards_to_cells([shard])
    ex = examples[0]
    expected = build_value_bearing_prefix(
        population_density_bucket=0,
        zoning_class=0,
        road_skeleton_class=1,
        cell_density_bucket=2,  # per-CELL value, from the cell payload (not the tile dict)
        region=None,
        coastal_inland_river=0,
        sub_c_morphology_class="Asian-megacity",
        seed=0,  # flatten's default seed; inert — the seed slot is constant-bucketed
    )
    assert list(ex.prefix_ids) == expected


def test_flatten_drops_empty_and_overlength_cells():
    examples, dropped = DM.flatten_shards_to_cells(_synthetic_shards(), max_cell_tokens=5760)
    # 6 cells total; 1 empty + 1 over-length dropped => 4 examples
    assert len(examples) == 4
    assert dropped == {"empty": 1, "too_long": 1}
    # every example carries the conditioning prefix + its cell tokens
    ex = next(e for e in examples if (e.tile_i, e.tile_j, e.cell_i, e.cell_j) == (0, 0, 0, 2))
    assert ex.prefix_len == len(conditioning_field_to_id())
    assert ex.seq_len == ex.prefix_len + 50
    # value-bearing layout (F6 delivery): the fixture's conditioning + this cell's density
    assert list(ex.ids[: ex.prefix_len]) == build_value_bearing_prefix(
        population_density_bucket=0,
        zoning_class=0,
        road_skeleton_class=1,
        cell_density_bucket=1,  # _cell default density
        region=None,
        coastal_inland_river=0,
        sub_c_morphology_class="Asian-megacity",
        seed=0,  # inert (constant-bucketed)
    )


def test_split_is_tile_level_and_val_disjoint_from_train():
    examples, _ = DM.flatten_shards_to_cells(_synthetic_shards(), max_cell_tokens=5760)
    train, val = DM.split_train_val(examples, seed=7, val_fraction=0.5)
    train_tiles = {(e.tile_i, e.tile_j) for e in train}
    val_tiles = {(e.tile_i, e.tile_j) for e in val}
    assert train_tiles.isdisjoint(val_tiles)  # no tile spans train+val (=> val disjoint holdout)
    assert train and val  # non-vacuous


def test_split_is_deterministic_under_seed():
    examples, _ = DM.flatten_shards_to_cells(_synthetic_shards(), max_cell_tokens=5760)
    a = DM.split_train_val(examples, seed=7, val_fraction=0.5)
    b = DM.split_train_val(examples, seed=7, val_fraction=0.5)
    c = DM.split_train_val(examples, seed=8, val_fraction=0.5)

    def keys(exs):
        return [(e.tile_i, e.tile_j, e.cell_i, e.cell_j) for e in exs]

    assert keys(a[0]) == keys(b[0]) and keys(a[1]) == keys(b[1])  # same seed -> identical
    assert keys(a[1]) != keys(c[1]) or keys(a[0]) != keys(c[0])  # different seed -> different


def test_collate_right_pads_and_reports_lengths():
    import torch

    examples, _ = DM.flatten_shards_to_cells(_synthetic_shards(), max_cell_tokens=5760)
    short, long = (
        sorted(examples, key=lambda e: e.seq_len)[0],
        sorted(examples, key=lambda e: e.seq_len)[-1],
    )
    batch = DM.collate_cells([DM._as_item(short), DM._as_item(long)])
    assert batch["ids"].shape == (2, long.seq_len)  # padded to batch max
    assert batch["seq_len"].tolist() == [short.seq_len, long.seq_len]
    assert torch.all(batch["prefix_len"] == len(conditioning_field_to_id()))


def test_collated_batch_carries_value_prefixes_not_a_constant_block():
    """F6 collate-layer guard: batches must carry VALUE prefixes. Two shards with
    DIFFERENT tile_conditioning -> the two collated rows' first-8 ids must differ
    (a regression to the constant slot block would make them equal, and would only
    fail the flatten tests, not any batch-level check — this closes that gap)."""
    import torch

    a = _shard(0, 0, [_cell(0, 0, 10)])
    b = _shard(0, 1, [_cell(0, 0, 10)], {**_TILE_CONDITIONING, "dominant_zoning_class": 1})
    ea = DM.flatten_shards_to_cells([a])[0][0]
    eb = DM.flatten_shards_to_cells([b])[0][0]
    batch = DM.collate_cells([DM._as_item(ea), DM._as_item(eb)])
    assert not torch.equal(batch["ids"][0, :8], batch["ids"][1, :8])  # value-bearing, not constant
    assert batch["prefix_len"].tolist() == [8, 8]


# ----- audit-halt wiring (synthetic manifests; no real data) -----

_HOLDOUT = {
    "manifest_schema_version": "2.0",
    "regions": {_REGION: {"tiles": [{"tile_i": 1, "tile_j": 7}]}},
}


def _write_holdout(tmp):
    p = tmp / "holdout.yaml"
    p.write_text(yaml.safe_dump(_HOLDOUT))
    return p


def _write_training_manifest(tmp, *, lineage_for_2_2):
    """A 1-tile training manifest. lineage_for_2_2 is the stamped lineage of tile
    (2,2): a holdout ref => leak; its own ref => clean."""
    manifest = {
        "release": "2026-04-15.0",
        "region": _REGION,
        "tiles": [{"tile_i": 2, "tile_j": 2, "lineage": [list(r) for r in lineage_for_2_2]}],
    }
    p = tmp / "training_manifest.yaml"
    p.write_text(yaml.safe_dump(manifest))
    return p


def test_planted_leak_halts_setup_before_any_batch(tmp_path):
    leak = _write_training_manifest(tmp_path, lineage_for_2_2=[(_REGION, 1, 7)])  # holdout ref
    dm = DM.CellDataModule(
        training_manifest=leak, holdout_manifest=_write_holdout(tmp_path), seed=7
    )
    with pytest.raises(HoldoutLeakError):
        dm.setup("fit")
    assert dm._batches_yielded == 0  # zero training steps possible


def test_clean_manifest_audit_passes_setup(tmp_path, monkeypatch):
    """Twin: a clean manifest passes the audit. We stub the in-memory build so the
    test stays data-free and proves the audit is non-blocking on clean lineage."""
    clean = _write_training_manifest(tmp_path, lineage_for_2_2=[(_REGION, 2, 2)])
    monkeypatch.setattr(DM, "build_shards_in_memory", lambda *a, **k: _synthetic_shards())
    dm = DM.CellDataModule(
        training_manifest=clean, holdout_manifest=_write_holdout(tmp_path), seed=7
    )
    dm.setup("fit")  # must not raise
    assert len(dm.train_cells) + len(dm.val_cells) > 0  # built examples


def test_train_order_is_reproducible_across_instances(tmp_path, monkeypatch):
    clean = _write_training_manifest(tmp_path, lineage_for_2_2=[(_REGION, 2, 2)])
    monkeypatch.setattr(DM, "build_shards_in_memory", lambda *a, **k: _synthetic_shards())
    hp = _write_holdout(tmp_path)
    o1 = DM.CellDataModule(training_manifest=clean, holdout_manifest=hp, seed=7)
    o2 = DM.CellDataModule(training_manifest=clean, holdout_manifest=hp, seed=7)
    o1.setup("fit")
    o2.setup("fit")
    assert o1.train_order() == o2.train_order()  # same seed -> same order (resume-safe)

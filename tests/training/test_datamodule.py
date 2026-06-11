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

import logging

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


#: Task 24b: synthetic cells carry a healthy non-constant stats vector (flags
#: present; varies with the cell coords so single-region fixtures never look
#: constant-across-regions to the §4.5 guard).
def _stats(ci, cj):
    return (1.0 + ci, 0.5 + cj, 0.2, 1.1, 0.9, 1.0, 1.0)


def _cell(ci, cj, n_tokens, density=1, character_stats=None):
    return CellPayload(
        cell_i=ci,
        cell_j=cj,
        cell_slot_index=ci * 8 + cj,
        tokens=tuple(range(n_tokens)),
        cell_density_bucket=density,
        boundary_contracts=(),
        character_stats=_stats(ci, cj) if character_stats is None else character_stats,
    )


def _synthetic_shards():
    # 4 tiles; tile (0,0) has an empty cell + an over-length cell + a good cell.
    return [
        _shard(0, 0, [_cell(0, 0, 0), _cell(0, 1, 10_000), _cell(0, 2, 50)]),
        _shard(0, 1, [_cell(1, 1, 30)]),
        _shard(0, 2, [_cell(2, 2, 40)]),
        _shard(0, 3, [_cell(3, 3, 20)]),
    ]


def _clean_synthetic_shards():
    """_synthetic_shards MINUS the over-length cell — for setup()-path tests. The F13
    drop-rate contract (DropRateExceeded at >0.5% too_long over the union) makes the
    1-in-5 over-length fixture a legitimate halt in setup(); flatten-level tests keep
    using _synthetic_shards (flatten itself never raises, it only counts)."""
    return [
        _shard(0, 0, [_cell(0, 0, 0), _cell(0, 2, 50)]),
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
    so equality against the value layout rules the slot layout out entirely) — PLUS the
    Task-24b character placeholder at position 9 (total prefix: 10 positions)."""
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
        city_identity=_REGION,  # Task 24a: the shard's city name (TrainingShard.region)
    )
    assert list(ex.prefix_ids) == [*expected, DM.CHARACTER_PLACEHOLDER_ID]  # Task 24b


def test_flatten_drops_empty_and_overlength_cells():
    examples, dropped = DM.flatten_shards_to_cells(_synthetic_shards(), max_cell_tokens=5760)
    # 6 cells total; 1 empty + 1 over-length dropped => 4 examples
    assert len(examples) == 4
    assert dropped == {"empty": 1, "too_long": 1}
    # every example carries the conditioning prefix + its cell tokens (Task 24b:
    # the 9 id positions + the character placeholder => prefix_len 10)
    ex = next(e for e in examples if (e.tile_i, e.tile_j, e.cell_i, e.cell_j) == (0, 0, 0, 2))
    assert ex.prefix_len == len(conditioning_field_to_id()) + 1
    assert ex.seq_len == ex.prefix_len + 50
    # value-bearing layout (F6 delivery): the fixture's conditioning + this cell's density
    assert list(ex.ids[: ex.prefix_len - 1]) == build_value_bearing_prefix(
        population_density_bucket=0,
        zoning_class=0,
        road_skeleton_class=1,
        cell_density_bucket=1,  # _cell default density
        region=None,
        coastal_inland_river=0,
        sub_c_morphology_class="Asian-megacity",
        seed=0,  # inert (constant-bucketed)
        city_identity=_REGION,  # Task 24a: from TrainingShard.region
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
    # Task 24b: prefix = 9 id positions + the character placeholder position
    assert torch.all(batch["prefix_len"] == len(conditioning_field_to_id()) + 1)
    assert batch["char_stats"].shape == (2, 7)  # the continuous channel rides the batch


def test_collated_batch_carries_value_prefixes_not_a_constant_block():
    """F6 collate-layer guard: batches must carry VALUE prefixes. Two shards with
    DIFFERENT tile_conditioning -> the two collated rows' 9 conditioning-prefix ids must differ
    (a regression to the constant slot block would make them equal, and would only
    fail the flatten tests, not any batch-level check — this closes that gap)."""
    import torch

    a = _shard(0, 0, [_cell(0, 0, 10)])
    b = _shard(0, 1, [_cell(0, 0, 10)], {**_TILE_CONDITIONING, "dominant_zoning_class": 1})
    ea = DM.flatten_shards_to_cells([a])[0][0]
    eb = DM.flatten_shards_to_cells([b])[0][0]
    batch = DM.collate_cells([DM._as_item(ea), DM._as_item(eb)])
    n = len(conditioning_field_to_id())  # 9 id positions (Task-24a city_identity append)
    assert not torch.equal(batch["ids"][0, :n], batch["ids"][1, :n])  # value-bearing, not constant
    # Task 24b reverse-lock: TOTAL prefix positions = 9 ids + 1 character placeholder
    assert batch["prefix_len"].tolist() == [n + 1, n + 1] == [10, 10]


# ----- audit-halt wiring (synthetic manifests; no real data) -----

_HOLDOUT = {
    "manifest_schema_version": "2.0",
    "regions": {_REGION: {"tiles": [{"tile_i": 1, "tile_j": 7}]}},
}


def _write_holdout(tmp):
    """Strengthened to the F9 reader contract (Task 20): setup() now verifies the
    manifest's stamped manifest_sha256 + the _EVAL_SET_LOCKED marker beside it, so
    the synthetic holdout is stamped with the REAL freeze grammar and sealed."""
    from cfm.eval.holdout.manifest import manifest_sha256

    stamped = dict(_HOLDOUT)
    stamped["manifest_sha256"] = manifest_sha256(stamped)
    p = tmp / "holdout.yaml"
    p.write_text(yaml.safe_dump(stamped))
    (tmp / "_EVAL_SET_LOCKED").touch()
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
    monkeypatch.setattr(DM, "build_shards_in_memory", lambda *a, **k: _clean_synthetic_shards())
    dm = DM.CellDataModule(
        training_manifest=clean, holdout_manifest=_write_holdout(tmp_path), seed=7
    )
    dm.setup("fit")  # must not raise
    assert len(dm.train_cells) + len(dm.val_cells) > 0  # built examples


def test_train_order_is_reproducible_across_instances(tmp_path, monkeypatch):
    clean = _write_training_manifest(tmp_path, lineage_for_2_2=[(_REGION, 2, 2)])
    monkeypatch.setattr(DM, "build_shards_in_memory", lambda *a, **k: _clean_synthetic_shards())
    hp = _write_holdout(tmp_path)
    o1 = DM.CellDataModule(training_manifest=clean, holdout_manifest=hp, seed=7)
    o2 = DM.CellDataModule(training_manifest=clean, holdout_manifest=hp, seed=7)
    o1.setup("fit")
    o2.setup("fit")
    assert o1.train_order() == o2.train_order()  # same seed -> same order (resume-safe)


# ----- F13 drop-rate action contract (readiness-closure Task 15) -----
#
# setup() must accumulate flatten's `dropped` counts across the manifest UNION and
# raise DropRateExceeded when too_long / (kept + too_long) > 0.005 (strict >).
# Denominator = NON-EMPTY cells: empty cells are a different defect class and would
# dilute the length signal.
#
# NOTE on the union tests: the plan's literal construction ("each manifest
# individually under threshold, together over") is mathematically impossible — the
# union rate is a convex combination of per-manifest rates, so it can never exceed
# their max. The accumulate-then-check ordering is pinned by its two possible
# discriminating directions instead: (a) the raise's denominator includes the OTHER
# manifest's kept cells; (b) a manifest that is over threshold ALONE does not raise
# when the union dilutes it back under (the check runs on the union, not per city).


def _mixed_cells(kept: int, too_long: int = 0, empty: int = 0):
    """Unique-keyed synthetic cells: `kept` short cells, then `too_long` cells one
    token over DEFAULT_MAX_CELL_TOKENS, then `empty` cells."""
    cells = []
    for idx in range(kept + too_long + empty):
        if idx < kept:
            n = 10
        elif idx < kept + too_long:
            n = DM.DEFAULT_MAX_CELL_TOKENS + 1
        else:
            n = 0
        cells.append(_cell(idx // 8, idx % 8, n))
    return cells


def _write_manifest_at(tmp, name, ti, tj):
    manifest = {
        "release": "2026-04-15.0",
        "region": _REGION,
        "tiles": [{"tile_i": ti, "tile_j": tj, "lineage": [[_REGION, ti, tj]]}],
    }
    p = tmp / name
    p.write_text(yaml.safe_dump(manifest))
    return p


def test_setup_raises_drop_rate_exceeded_above_threshold(tmp_path, monkeypatch):
    """1 too_long / 100 non-empty = 0.01 > 0.005 -> raise. The 100 EMPTY cells prove
    the denominator excludes empties (an empty-diluted denominator would read
    1/200 == 0.005, NOT > threshold, and this test would fail to raise)."""
    clean = _write_training_manifest(tmp_path, lineage_for_2_2=[(_REGION, 2, 2)])
    monkeypatch.setattr(
        DM,
        "build_shards_in_memory",
        lambda *a, **k: [_shard(2, 2, _mixed_cells(kept=99, too_long=1, empty=100))],
    )
    dm = DM.CellDataModule(
        training_manifest=clean, holdout_manifest=_write_holdout(tmp_path), seed=7
    )
    with pytest.raises(DM.DropRateExceeded) as ei:
        dm.setup("fit")
    msg = str(ei.value)
    assert "1/100" in msg  # counts: too_long / non-empty
    assert str(DM.MAX_TOO_LONG_DROP_RATE) in msg  # the threshold
    assert "0.01" in msg  # the rate
    # the named escalation (F13 action contract)
    assert "raise DEFAULT_MAX_CELL_TOKENS via a recorded decision or re-chunk" in msg
    assert "see readiness F13" in msg


def test_setup_at_exact_threshold_does_not_raise(tmp_path, monkeypatch):
    """Exactly AT the threshold (1/200 == 0.005) must NOT raise — strict >."""
    clean = _write_training_manifest(tmp_path, lineage_for_2_2=[(_REGION, 2, 2)])
    monkeypatch.setattr(
        DM,
        "build_shards_in_memory",
        lambda *a, **k: [_shard(2, 2, _mixed_cells(kept=199, too_long=1))],
    )
    dm = DM.CellDataModule(
        training_manifest=clean, holdout_manifest=_write_holdout(tmp_path), seed=7
    )
    dm.setup("fit")  # must not raise
    assert len(dm.train_cells) + len(dm.val_cells) == 199


def test_sub_design_budget_opt_down_is_exempt_from_drop_rate(tmp_path, monkeypatch):
    """A caller that deliberately opts the budget DOWN below DEFAULT_MAX_CELL_TOKENS
    (run_smoke's tiny-budget max_len=256 loop drops ~64% of real SG cells BY DESIGN)
    accepts tail amputation by construction. The 0.005 threshold is calibrated to the
    design point — and its escalation ('raise DEFAULT_MAX_CELL_TOKENS') is not even
    the right action for an opt-down — so the contract must not fire there."""
    clean = _write_training_manifest(tmp_path, lineage_for_2_2=[(_REGION, 2, 2)])
    # 5 cells of 300 tokens + 5 of 50: at max_cell_tokens=256 the rate is 0.5 >> 0.005
    cells = [_cell(i // 8, i % 8, 300 if i < 5 else 50) for i in range(10)]
    monkeypatch.setattr(DM, "build_shards_in_memory", lambda *a, **k: [_shard(2, 2, cells)])
    dm = DM.CellDataModule(
        training_manifest=clean,
        holdout_manifest=_write_holdout(tmp_path),
        seed=7,
        max_cell_tokens=256,
    )
    dm.setup("fit")  # must not raise: sub-design budget regime
    assert len(dm.train_cells) + len(dm.val_cells) == 5


def test_exempt_regime_logs_union_drop_stats(tmp_path, monkeypatch, caplog):
    """The sub-design exemption must not be SILENT: setup() logs the union too_long
    drop stats at INFO unconditionally (before the enforcement branch), so exempt
    opt-down runs (the --max-len 2048 sbatch entry points; run_smoke's tiny-budget
    loop) show their accepted tail amputation in job logs. The dirty
    _synthetic_shards at max_cell_tokens=256 is exempt (256 < DEFAULT) with known
    counts: 1 too_long / 5 non-empty = rate 0.2, plus 1 empty."""
    clean = _write_training_manifest(tmp_path, lineage_for_2_2=[(_REGION, 2, 2)])
    monkeypatch.setattr(DM, "build_shards_in_memory", lambda *a, **k: _synthetic_shards())
    dm = DM.CellDataModule(
        training_manifest=clean,
        holdout_manifest=_write_holdout(tmp_path),
        seed=7,
        max_cell_tokens=256,
    )
    with caplog.at_level(logging.INFO, logger="cfm.data.training.datamodule"):
        dm.setup("fit")  # exempt regime: must not raise
    [rec] = [r for r in caplog.records if "union" in r.getMessage()]  # exactly one
    msg = rec.getMessage()
    assert rec.levelno == logging.INFO
    assert "1/5" in msg  # counts: too_long / non-empty union denominator
    assert "0.2" in msg  # the rate
    assert "256" in msg  # the budget in force


def test_union_drop_rate_denominator_accumulates_across_manifests(tmp_path, monkeypatch):
    """Manifest A: 8 kept + 2 too_long; manifest B: 92 kept. The raise must report
    2/102 — B's kept cells in the denominator prove the counts accumulate across
    the WHOLE union before the check runs (accumulate-then-check ordering)."""
    mf_a = _write_manifest_at(tmp_path, "a.yaml", 2, 2)
    mf_b = _write_manifest_at(tmp_path, "b.yaml", 3, 3)

    def fake_build(release, region, *, tile_ids=None):
        if tile_ids == [(2, 2)]:
            return [_shard(2, 2, _mixed_cells(kept=8, too_long=2))]
        return [_shard(3, 3, _mixed_cells(kept=92))]

    monkeypatch.setattr(DM, "build_shards_in_memory", fake_build)
    dm = DM.CellDataModule(
        training_manifests=[mf_a, mf_b], holdout_manifest=_write_holdout(tmp_path), seed=7
    )
    with pytest.raises(DM.DropRateExceeded) as ei:
        dm.setup("fit")
    assert "2/102" in str(ei.value)


def test_union_check_runs_on_the_union_not_per_manifest(tmp_path, monkeypatch):
    """Manifest A alone is way over (1/10 = 0.1); the union dilutes it to
    1/1000 = 0.001 <= 0.005 -> setup must NOT raise. A per-manifest (inside-the-loop)
    check would raise on A — this pins the check to the accumulated union."""
    mf_a = _write_manifest_at(tmp_path, "a.yaml", 2, 2)
    mf_b = _write_manifest_at(tmp_path, "b.yaml", 3, 3)

    def fake_build(release, region, *, tile_ids=None):
        if tile_ids == [(2, 2)]:
            return [_shard(2, 2, _mixed_cells(kept=9, too_long=1))]
        return [_shard(3, 3, _mixed_cells(kept=990))]

    monkeypatch.setattr(DM, "build_shards_in_memory", fake_build)
    dm = DM.CellDataModule(
        training_manifests=[mf_a, mf_b], holdout_manifest=_write_holdout(tmp_path), seed=7
    )
    dm.setup("fit")  # must not raise: 1/1000 is under the union threshold
    assert len(dm.train_cells) + len(dm.val_cells) == 999


# ----- Task 24a: conditioning_ablation threading (Lane S instrument) -----
#
# The generation-side conditioning path consumes dm.val_cells' prefix_ids verbatim
# (train_scaffold._generate_and_score, matched conditioning), so proving BOTH
# train_cells and val_cells carry the ablated prefix proves the ablation acts
# identically train-side and generation-side, by construction.


def _city_slot_ids(examples):
    from cfm.data.training.conditioning import (
        _CONDITIONING_FIELDS,
        _VALUE_STRIDE,
        CONDITIONING_VALUE_BASE,
    )

    city_slot = _CONDITIONING_FIELDS.index("city_identity")
    city_block_base = CONDITIONING_VALUE_BASE + city_slot * _VALUE_STRIDE
    return city_block_base, [e.prefix_ids[city_slot] for e in examples]


def test_flatten_no_city_ablation_zeroes_only_the_city_slot():
    shard = _shard(0, 0, [_cell(0, 0, 10)])
    full = DM.flatten_shards_to_cells([shard])[0][0]
    ablated = DM.flatten_shards_to_cells([shard], ablation="no_city")[0][0]
    assert ablated.prefix_ids[:8] == full.prefix_ids[:8]  # other 8 ids untouched
    base, [city_id] = _city_slot_ids([ablated])
    assert city_id == base  # bucket 0 (ablated)
    assert full.prefix_ids[8] != city_id  # full carries a real registry id


def test_setup_threads_conditioning_ablation_to_train_and_val_prefixes(tmp_path, monkeypatch):
    """End-to-end through CellDataModule(conditioning_ablation="no_city"): every
    train AND val example (the generation-side conditioning source) has the city
    slot at bucket 0; the full twin carries the real singapore registry id."""
    clean = _write_training_manifest(tmp_path, lineage_for_2_2=[(_REGION, 2, 2)])
    monkeypatch.setattr(DM, "build_shards_in_memory", lambda *a, **k: _clean_synthetic_shards())
    hp = _write_holdout(tmp_path)

    dm_full = DM.CellDataModule(training_manifest=clean, holdout_manifest=hp, seed=7)
    dm_ablated = DM.CellDataModule(
        training_manifest=clean, holdout_manifest=hp, seed=7, conditioning_ablation="no_city"
    )
    dm_full.setup("fit")
    dm_ablated.setup("fit")

    base, ablated_ids = _city_slot_ids(dm_ablated.train_cells + dm_ablated.val_cells)
    assert dm_ablated.val_cells, "generation-side source must be non-vacuous"
    assert all(i == base for i in ablated_ids)  # city ablated everywhere
    _, full_ids = _city_slot_ids(dm_full.train_cells + dm_full.val_cells)
    assert all(i != base for i in full_ids)  # full run carries the real city id
    # the ablation touches ONLY the city slot: everything else identical
    for fe, ae in zip(
        dm_full.train_cells + dm_full.val_cells,
        dm_ablated.train_cells + dm_ablated.val_cells,
        strict=True,
    ):
        assert fe.key == ae.key
        assert fe.prefix_ids[:8] == ae.prefix_ids[:8]
        assert fe.tokens == ae.tokens


def test_setup_runs_the_city_identity_guard_on_the_union(tmp_path, monkeypatch):
    """Constant-across-cities regime at the LIVE training seam: two manifests
    declaring distinct regions whose built shards all carry the same city ->
    setup halts loud (CityIdentityError), before any split/loader."""
    from cfm.data.training.build_shards import CityIdentityError

    mf_a = tmp_path / "a.yaml"
    mf_a.write_text(
        yaml.safe_dump(
            {
                "release": "2026-04-15.0",
                "region": "almere",
                "tiles": [{"tile_i": 2, "tile_j": 2, "lineage": [["almere", 2, 2]]}],
            }
        )
    )
    mf_b = tmp_path / "b.yaml"
    mf_b.write_text(
        yaml.safe_dump(
            {
                "release": "2026-04-15.0",
                "region": "welwyn",
                "tiles": [{"tile_i": 3, "tile_j": 3, "lineage": [["welwyn", 3, 3]]}],
            }
        )
    )
    # the wiring bug under test: the builder stamps "almere" regardless of region
    monkeypatch.setattr(
        DM,
        "build_shards_in_memory",
        lambda release, region, *, tile_ids=None: [
            TrainingShard(
                region="almere",
                tile_i=tile_ids[0][0],
                tile_j=tile_ids[0][1],
                tile_conditioning=dict(_TILE_CONDITIONING),
                macro_tokens=(),
                cells=(_cell(0, 0, 10),),
                lineage=frozenset({(region, tile_ids[0][0], tile_ids[0][1])}),
            )
        ],
    )
    dm = DM.CellDataModule(
        training_manifests=[mf_a, mf_b], holdout_manifest=_write_holdout(tmp_path), seed=7
    )
    with pytest.raises(CityIdentityError, match="almere"):
        dm.setup("fit")

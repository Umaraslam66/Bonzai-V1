"""Task 2 — driver core: ordered generation over ``sharded_eval`` (fast, no torch).

The driver only *delegates* sharding to ``cfm.eval.shard.sharded_eval`` (whose real
body lazy-imports ``torch.distributed``). These tests patch that symbol on the driver
module with an in-process single-rank stand-in — ``[score_one(i) for i in range(n)]``
— which is exactly ``sharded_eval``'s single-process (world_size=1) semantics: the
full, ordered, count-conserved list. So the suite exercises the driver's contracts
(global-index seed keying, manifest order, self-terminated derivation, write-once I/O)
with a pure deterministic fake ``gen_fn`` and never touches torch or a GPU.
"""

from __future__ import annotations

import sys

import pytest

from cfm.eval.realism_driver import driver
from cfm.eval.realism_driver.conditioning import ConditionedCell

_CELL_END = 260  # <cell_end> sentinel (mirror of cfm.data.sub_f.vocab.CELL_END_TOKEN_ID)


def _cell(i: int, *, density_bucket: int = 3) -> ConditionedCell:
    """A ConditionedCell whose cell_key encodes its global index ``i`` (cell_i slot)."""
    return ConditionedCell(
        cell_key=("glasgow", 0, 0, i, 0),
        density_bucket=density_bucket,
        prefix_ids=tuple(range(10)),
        char_stats=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
        real_body_tokens=(11, 12, 13),
    )


def _single_rank(n_items, score_one, *, rank=None, world_size=None):
    """In-process stand-in for ``sharded_eval`` single-process (world_size=1) path."""
    return [score_one(i) for i in range(n_items)]


@pytest.fixture(autouse=True)
def _patch_sharded_eval(monkeypatch):
    monkeypatch.setattr(driver, "sharded_eval", _single_rank)


def test_driver_module_is_torch_free():
    """Importing the driver must not pull torch (GPU-free core; A: no torch in driver)."""
    assert "cfm.eval.realism_driver.driver" in sys.modules
    # The module under test was imported at top; torch must not have come with it.
    # (A fresh import would be ideal, but conftest/other tests may load torch; assert the
    # module's own namespace holds no torch reference.)
    assert not hasattr(driver, "torch")


def test_seed_is_global_index_keyed():
    """gen_fn(cell, seed): seed == base_seed + i for every cell (rank-independence, A5)."""
    base_seed = 5000
    cells = [_cell(i) for i in range(6)]
    seen: list[tuple[int, int]] = []

    def gen_fn(cell: ConditionedCell, seed: int) -> dict:
        seen.append((cell.cell_key[3], seed))  # (global index, seed)
        return {"tokens": [_CELL_END], "blocks": [], "geoms": []}

    driver.run_generation(cells, gen_fn, base_seed=base_seed)

    assert sorted(seen) == [(i, base_seed + i) for i in range(6)]


def test_records_in_manifest_order_and_count_conserved():
    """N=7 (ragged vs a would-be 4-shard): full ordered, count-conserved list."""
    cells = [_cell(i) for i in range(7)]

    def gen_fn(cell: ConditionedCell, seed: int) -> dict:
        return {"tokens": [cell.cell_key[3], _CELL_END], "blocks": [], "geoms": []}

    out = driver.run_generation(cells, gen_fn, base_seed=0)

    assert len(out) == 7
    assert [r.cell_key for r in out] == [c.cell_key for c in cells]
    assert [r.tokens[0] for r in out] == list(range(7))


def test_self_terminated_flag():
    """tokens ending in 260 -> True; a cap-length list not ending in 260 -> False."""
    cells = [_cell(0), _cell(1)]

    def gen_fn(cell: ConditionedCell, seed: int) -> dict:
        if cell.cell_key[3] == 0:
            tokens = [11, 12, _CELL_END]  # self-terminated (ends in <cell_end>)
        else:
            tokens = [11, 12, 13]  # hit the cap: no <cell_end>
        return {"tokens": tokens, "blocks": [], "geoms": []}

    out = driver.run_generation(cells, gen_fn, base_seed=0)

    assert out[0].self_terminated is True
    assert out[1].self_terminated is False


def test_self_terminated_empty_tokens_is_false():
    """A degenerate empty generation is not a self-termination."""
    cells = [_cell(0)]

    def gen_fn(cell: ConditionedCell, seed: int) -> dict:
        return {"tokens": [], "blocks": [], "geoms": []}

    out = driver.run_generation(cells, gen_fn, base_seed=0)
    assert out[0].self_terminated is False


def _sample_records() -> list[driver.GenCellRecord]:
    return [
        driver.GenCellRecord(
            cell_key=("glasgow", 0, 0, 0, 0),
            density_bucket=3,
            tokens=[11, 12, _CELL_END],
            blocks=[[509, 1, 2], [510, 3, 4]],
            geoms=[{"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0]]}],
            self_terminated=True,
        ),
        driver.GenCellRecord(
            cell_key=("krakow", 2, 5, 7, 9),
            density_bucket=1,
            tokens=[11, 12, 13],
            blocks=[],
            geoms=[],
            self_terminated=False,
        ),
    ]


def test_write_artifact_is_write_once(tmp_path):
    """A second write to the same path raises FileExistsError (write-once discipline)."""
    path = tmp_path / "gen.json"
    records = _sample_records()
    driver.write_gen_artifact(records, path, meta={"release": "r1"})
    with pytest.raises(FileExistsError):
        driver.write_gen_artifact(records, path, meta={"release": "r1"})


def test_roundtrip_read_write(tmp_path):
    """read(write(records, meta)) reproduces both the records and the meta."""
    path = tmp_path / "gen.json"
    records = _sample_records()
    meta = {"release": "r1", "base_seed": 5000, "n_cells": len(records)}

    driver.write_gen_artifact(records, path, meta=meta)
    got_meta, got_records = driver.read_gen_artifact(path)

    assert got_meta == meta
    assert got_records == records
    # cell_key must survive as a tuple (JSON would otherwise hand back a list).
    assert all(isinstance(r.cell_key, tuple) for r in got_records)

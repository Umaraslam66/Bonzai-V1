"""F6 (generation side): matched-conditioning generation in _generate_and_score.

Each generated cell must be conditioned on a REAL val example's value-bearing
prefix (not the legacy constant slot prefix) and scored in that example's
stratum (CellExample.stratum: cell_density_bucket, -1 when unknown).

Stub boundary: monkeypatch ``generate_cell_tokens`` / ``split_cell_into_features``
/ ``try_decode_block`` / ``slice_eval`` in the ``scripts.train_scaffold``
namespace — recording the prefixes/seeds generation was called with and the
strata handed to slice_eval — so no real model or decoder runs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import scripts.train_scaffold as ts
from cfm.data.training.datamodule import CellExample, build_conditioning_prefix
from cfm.training.config import ScaffoldConfig

_PREFIX_A = (101, 102, 103)
_PREFIX_B = (201, 202, 203)
#: Task 24b: distinct per-example character stats — the generation side must
#: receive each sampled example's OWN stats verbatim (matched conditioning).
_STATS_A = (1.5, 0.4, 0.2, 1.1, 0.9, 1.0, 1.0)
_STATS_B = (0.0,) * 7  # e.g. an ablated/absent example


def _example(cell_i: int, prefix: tuple[int, ...], bucket: int | None) -> CellExample:
    return CellExample(
        region="singapore",
        tile_i=0,
        tile_j=0,
        cell_i=cell_i,
        cell_j=0,
        prefix_ids=prefix,
        tokens=(1, 2, 3),
        cell_density_bucket=bucket,
        character_stats=_STATS_A if prefix == _PREFIX_A else _STATS_B,
    )


class _FakeDM:
    """Minimal stand-in exposing the one attribute _generate_and_score reads."""

    def __init__(self, val: list[CellExample]) -> None:
        self._val = val

    @property
    def val_cells(self) -> list[CellExample]:
        return self._val


@pytest.fixture()
def capture(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    cap = SimpleNamespace(prefixes=[], seeds=[], char_stats=[], strata=None)

    def fake_generate(model, *, prefix, max_new, seed, char_stats=None):
        cap.prefixes.append(list(prefix))
        cap.seeds.append(seed)
        cap.char_stats.append(char_stats)
        return [7, 8, 9]  # fixed generated tail

    def fake_slice_eval(blocks, geoms, strata, **kwargs):
        cap.strata = list(strata)
        return {"n_decoded": len(blocks)}

    monkeypatch.setattr(ts, "generate_cell_tokens", fake_generate)
    monkeypatch.setattr(ts, "split_cell_into_features", lambda tokens: [list(tokens)])
    monkeypatch.setattr(ts, "try_decode_block", lambda block: {"type": "Polygon"})
    monkeypatch.setattr(ts, "slice_eval", fake_slice_eval)
    return cap


def _run(cap: SimpleNamespace, n_cells: int = 4) -> dict:
    cfg = ScaffoldConfig(region="singapore", devices=1, accelerator="cpu")
    dm = _FakeDM([_example(0, _PREFIX_A, 2), _example(1, _PREFIX_B, None)])
    model = SimpleNamespace(model=object())  # generation is stubbed; inner model unused
    return ts._generate_and_score(model, dm, cfg, n_cells=n_cells, max_new=16)


def test_generation_uses_real_val_prefixes_not_constant_slot_block(capture):
    _run(capture, n_cells=4)
    slot_block = build_conditioning_prefix()
    assert len(capture.prefixes) == 4
    allowed = {_PREFIX_A, _PREFIX_B}
    assert all(tuple(p) in allowed for p in capture.prefixes)
    # cycling over 2 val examples for 4 cells must use BOTH conditionings
    assert {tuple(p) for p in capture.prefixes} == allowed
    assert all(p != slot_block for p in capture.prefixes)


def test_generation_receives_each_examples_own_character_stats(capture):
    """Task 24b (train/gen identity by construction): generation gets the SAME
    character_stats the sampled val example carries — already ablation-applied by
    the datamodule, exactly like the prefix ids. Alignment is per-call: the stats
    captured at call i must be _example(prefix_i)'s stats, never a constant."""
    _run(capture, n_cells=4)
    assert len(capture.char_stats) == 4
    expected = [
        {_PREFIX_A: list(_STATS_A), _PREFIX_B: list(_STATS_B)}[tuple(p)] for p in capture.prefixes
    ]
    assert capture.char_stats == expected
    # both distinct stats vectors were exercised (cycling covers both examples)
    assert {tuple(c) for c in capture.char_stats} == {_STATS_A, _STATS_B}


def test_strata_match_sampled_examples_buckets(capture):
    _run(capture, n_cells=4)
    # one block per cell (stubbed split), so strata align 1:1 with generation calls
    expected = [{_PREFIX_A: 2, _PREFIX_B: -1}[tuple(p)] for p in capture.prefixes]
    assert capture.strata == expected
    assert set(capture.strata) == {2, -1}  # None bucket -> -1 (CellExample.stratum)


def test_per_cell_seeding_kept(capture):
    cfg_seed = ScaffoldConfig(region="singapore", devices=1, accelerator="cpu").seed
    _run(capture, n_cells=4)
    assert capture.seeds == [cfg_seed + i for i in range(4)]


def test_sampling_is_deterministic(capture):
    _run(capture, n_cells=4)
    first = [list(p) for p in capture.prefixes]
    capture.prefixes.clear()
    capture.seeds.clear()
    _run(capture, n_cells=4)
    assert [list(p) for p in capture.prefixes] == first


def test_empty_val_fails_loud(capture):
    cfg = ScaffoldConfig(region="singapore", devices=1, accelerator="cpu")
    model = SimpleNamespace(model=object())
    with pytest.raises(ValueError, match="val"):
        ts._generate_and_score(model, _FakeDM([]), cfg, n_cells=2, max_new=16)

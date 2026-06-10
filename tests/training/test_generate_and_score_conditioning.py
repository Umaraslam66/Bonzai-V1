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
    cap = SimpleNamespace(prefixes=[], seeds=[], strata=None)

    def fake_generate(model, *, prefix, max_new, seed):
        cap.prefixes.append(list(prefix))
        cap.seeds.append(seed)
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
    cfg = ScaffoldConfig(devices=1, accelerator="cpu")
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


def test_strata_match_sampled_examples_buckets(capture):
    _run(capture, n_cells=4)
    # one block per cell (stubbed split), so strata align 1:1 with generation calls
    expected = [{_PREFIX_A: 2, _PREFIX_B: -1}[tuple(p)] for p in capture.prefixes]
    assert capture.strata == expected
    assert set(capture.strata) == {2, -1}  # None bucket -> -1 (CellExample.stratum)


def test_per_cell_seeding_kept(capture):
    cfg_seed = ScaffoldConfig(devices=1, accelerator="cpu").seed
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
    cfg = ScaffoldConfig(devices=1, accelerator="cpu")
    model = SimpleNamespace(model=object())
    with pytest.raises(ValueError, match="val"):
        ts._generate_and_score(model, _FakeDM([]), cfg, n_cells=2, max_new=16)

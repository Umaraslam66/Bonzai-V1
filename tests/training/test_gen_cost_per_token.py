"""Task 8 (bake-off): the diagnostic must capture per-token GENERATION cost so the
§6 eval-budget can be projected to the locked 13,312-token scored-eval regime.

The diagnostic times the WHOLE 2048-token eval (eval_seconds), but the scored eval
generates 13,312 tokens/cell. AR generation is one sequential forward per token, so
cost scales ~linearly in tokens — the projection extrapolates from the cost PER
GENERATED TOKEN, not the 2048-total. This test pins:

  1. the pure per-token arithmetic helper (incl. the n_tokens==0 -> 0.0 guard),
  2. that _generate_and_score surfaces gen_seconds + n_tokens_generated, counting
     only GENERATED tokens (generate_cell_tokens returns the prefix-stripped tail),
  3. that the cost path references gen_seconds_per_token (wiring can't silently drop).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

import scripts.train_scaffold as ts
from cfm.data.training.datamodule import CellExample
from cfm.training.config import ScaffoldConfig

_PREFIX_A = (101, 102, 103)
_STATS_A = (1.5, 0.4, 0.2, 1.1, 0.9, 1.0, 1.0)
#: distinct count so n_tokens_generated == sum(len(tail)) is unambiguous: 5 tokens/cell
_GEN_TAIL = [7, 8, 9, 10, 11]


def _example(cell_i: int) -> CellExample:
    return CellExample(
        region="singapore",
        tile_i=0,
        tile_j=0,
        cell_i=cell_i,
        cell_j=0,
        prefix_ids=_PREFIX_A,
        tokens=(1, 2, 3),
        cell_density_bucket=2,
        character_stats=_STATS_A,
    )


class _FakeDM:
    def __init__(self, val: list[CellExample]) -> None:
        self._val = val

    @property
    def val_cells(self) -> list[CellExample]:
        return self._val


# --- 1. the pure helper -----------------------------------------------------


def test_gen_seconds_per_token_divides():
    assert ts._gen_seconds_per_token(10.0, 5) == pytest.approx(2.0)


def test_gen_seconds_per_token_zero_tokens_is_zero_not_error():
    # cfg.eval_cells == 0 generates nothing: must guard the divide-by-zero, not raise.
    assert ts._gen_seconds_per_token(0.0, 0) == 0.0
    assert ts._gen_seconds_per_token(5.0, 0) == 0.0


# --- 2. _generate_and_score surfaces gen timing + count ---------------------


@pytest.fixture()
def capture(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    cap = SimpleNamespace(n_generate_calls=0)

    def fake_generate(model, *, prefix, max_new, seed, char_stats=None):
        cap.n_generate_calls += 1
        return list(_GEN_TAIL)  # prefix-stripped tail (the real contract)

    monkeypatch.setattr(ts, "generate_cell_tokens", fake_generate)
    monkeypatch.setattr(ts, "split_cell_into_features", lambda tokens: [list(tokens)])
    monkeypatch.setattr(ts, "try_decode_block", lambda block: {"type": "Polygon"})
    monkeypatch.setattr(
        ts, "slice_eval", lambda blocks, geoms, strata, **kw: {"n_decoded": len(blocks)}
    )
    return cap


def test_metrics_carry_gen_seconds_and_token_count(capture):
    cfg = ScaffoldConfig(devices=1, accelerator="cpu")
    dm = _FakeDM([_example(0), _example(1)])
    model = SimpleNamespace(model=object())
    metrics = ts._generate_and_score(model, dm, cfg, n_cells=4, max_new=16)

    assert "gen_seconds" in metrics
    assert "n_tokens_generated" in metrics
    # counts only GENERATED tokens (tail), across all 4 sampled cells: 4 * len(_GEN_TAIL)
    assert metrics["n_tokens_generated"] == 4 * len(_GEN_TAIL)
    assert metrics["gen_seconds"] >= 0.0


# --- 3. the cost path references the projection field -----------------------


def test_cost_path_wires_gen_seconds_per_token():
    """Content guard: run_short must set cost["gen_seconds_per_token"]. A pure
    unit test of the helper can't catch the wiring being dropped, so assert the
    field name appears in the run_short source (the field that the §6 13,312
    projection extrapolates from)."""
    src = inspect.getsource(ts.run_short)
    assert 'cost["gen_seconds_per_token"]' in src
    assert "n_tokens_generated" in src

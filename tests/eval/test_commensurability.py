"""F15 length-commensurability guard on the §2 emergence verdict (readiness Task 14).

A 512-token-capped generation cannot be scored against a floor derived in the
full-cell regime (5760 tokens): the model never got the budget to emit what the
floor counts. When ``generated_length_cap < floor_regime_cell_length`` the §2
verdict is REFUSED — ``emergence_verdict == "INCOMMENSURATE"`` (loud, distinct
from the legacy verdict-None) and the building metrics are neither floored nor
blessed. Both lengths become config/report facts (F9 reproducibility), and the
metrics dict records them plus the floor's denominator convention.
"""

from __future__ import annotations

from types import SimpleNamespace

import cfm.eval.slice_metrics as S
import scripts.train_scaffold as ts
from cfm.data.training.datamodule import DEFAULT_MAX_CELL_TOKENS, CellExample
from cfm.eval.slice_metrics import EmergenceVerdict, slice_eval
from cfm.training.config import ScaffoldConfig
from cfm.training.lit_module import ScaffoldLit

# A valid unit square (1 polygon — clears any sub-1.0 floor at n_cells=1) and a
# roads-only LineString (same fixtures as tests/eval/test_slice_metrics.py).
_SQUARE_BLOCK = [509, 510]
_SQUARE = {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}
_LINE = {"type": "LineString", "coordinates": [[0, 0], [1, 0]]}

_PROV = {
    "region": "krakow",
    "floor": 2.0,
    "holdout_density": 8.0,
    "frac": 0.25,
    "derived_at": "abc123",
    "derivation_regime": {"cell_length": "full", "denominator": "all_nonempty_cells"},
}


# --- (a) eval generation lengths are config (and so hparams/report) facts -------------


def test_config_carries_eval_lengths_with_argparse_defaults() -> None:
    cfg = ScaffoldConfig()
    assert cfg.eval_cells == 64
    assert cfg.eval_max_new == 512


def test_eval_lengths_land_in_model_dump_hence_every_report() -> None:
    # scripts/train_scaffold.py::_write_report renders cfg.model_dump() wholesale, so
    # presence in the dump == presence in every reports/ summary (F9).
    dump = ScaffoldConfig().model_dump()
    assert dump["eval_cells"] == 64
    assert dump["eval_max_new"] == 512


def test_checkpoint_hparams_carry_eval_lengths() -> None:
    # save_hyperparameters(cfg.model_dump()) puts the lengths into every checkpoint's
    # hparams (same wire as the F16 conditioning_scheme tag).
    cfg = ScaffoldConfig(
        d_model=64, n_layers=2, n_heads=2, max_len=128, accelerator="cpu", devices=1
    )
    lit = ScaffoldLit(cfg)
    assert lit.hparams["eval_max_new"] == 512
    assert lit.hparams["eval_cells"] == 64


# --- (b) the refuse rule: L < R => INCOMMENSURATE, never a §2 verdict -----------------


def test_incommensurate_refuses_verdict_even_when_polygons_clear_the_floor(
    monkeypatch,
) -> None:
    # The distinguishing fixture: 1 polygon / 1 cell clears floor 0.5, so the PURE
    # verdict function would say SCOREABLE — but a 512-cap generation scored against
    # a 5760-regime floor is refused BEFORE that function is ever consulted.
    def _boom(**kwargs):
        raise AssertionError("emergence_verdict must NOT be consulted when lengths refuse")

    monkeypatch.setattr(S, "emergence_verdict", _boom)
    out = slice_eval(
        [_SQUARE_BLOCK],
        [_SQUARE],
        [1],
        n_cells=1,
        emergence_floor_per_cell=0.5,
        generated_length_cap=512,
        floor_regime_cell_length=5760,
    )
    assert out["emergence_verdict"] == "INCOMMENSURATE"
    # REFUSED, not floored: the metrics were never compared against the floor.
    assert out["building_metrics_floored"] is False


def test_incommensurate_is_loud_and_distinct_from_legacy_none() -> None:
    assert EmergenceVerdict.INCOMMENSURATE.value == "INCOMMENSURATE"
    out = slice_eval(
        [[1]],
        [_LINE],
        [0],
        n_cells=50,
        emergence_floor_per_cell=1.0,
        generated_length_cap=512,
        floor_regime_cell_length=5760,
    )
    assert out["emergence_verdict"] == "INCOMMENSURATE"
    assert out["emergence_verdict"] is not None  # loud, never the legacy None


def test_commensurate_lengths_run_the_normal_scoreable_path() -> None:
    out = slice_eval(
        [_SQUARE_BLOCK],
        [_SQUARE],
        [1],
        n_cells=1,
        emergence_floor_per_cell=0.5,
        generated_length_cap=5760,  # L >= R: commensurate
        floor_regime_cell_length=5760,
    )
    assert out["emergence_verdict"] == EmergenceVerdict.SCOREABLE.value


def test_commensurate_lengths_run_the_normal_roads_only_path() -> None:
    out = slice_eval(
        [[1]],
        [_LINE],
        [0],
        n_cells=50,
        emergence_floor_per_cell=1.0,
        generated_length_cap=5760,
        floor_regime_cell_length=5760,
    )
    assert out["emergence_verdict"] == EmergenceVerdict.ROADS_ONLY.value
    assert out["building_metrics_floored"] is True


def test_lengths_absent_legacy_behavior_unchanged() -> None:
    # Legacy callers (no lengths) keep the exact pre-F15 behavior: verdict runs when
    # the floor inputs are given, stays None when they are not.
    floorless = slice_eval([[1]], [_LINE], [0])
    assert floorless["emergence_verdict"] is None
    assert floorless["building_metrics_floored"] is False

    floored = slice_eval([[1]], [_LINE], [0], n_cells=50, emergence_floor_per_cell=1.0)
    assert floored["emergence_verdict"] == EmergenceVerdict.ROADS_ONLY.value
    assert floored["building_metrics_floored"] is True


# --- (c) both lengths + the floor's denominator convention land in the metrics -------


def test_metrics_record_lengths_and_denominator_convention_with_provenance() -> None:
    out = slice_eval(
        [[1]],
        [_LINE],
        [0],
        n_cells=50,
        emergence_floor_per_cell=2.0,
        emergence_floor_provenance=_PROV,
        generated_length_cap=512,
        floor_regime_cell_length=5760,
    )
    assert out["generated_length_cap"] == 512
    assert out["floor_regime_cell_length"] == 5760
    assert out["floor_denominator_convention"] == "all_nonempty_cells"


def test_metrics_length_keys_present_but_none_when_absent() -> None:
    # present-but-None (report-stable shape, matching emergence_floor_provenance).
    out = slice_eval([[1]], [_LINE], [0])
    assert "generated_length_cap" in out and out["generated_length_cap"] is None
    assert "floor_regime_cell_length" in out and out["floor_regime_cell_length"] is None
    # no provenance -> no denominator convention to quote
    assert "floor_denominator_convention" in out
    assert out["floor_denominator_convention"] is None


# --- scaffold plumbing: the lengths flow through _generate_and_score to slice_eval ----


def test_generate_and_score_passes_lengths_to_slice_eval(monkeypatch) -> None:
    captured: dict = {}

    def fake_slice_eval(blocks, geoms, strata, **kwargs):
        captured.update(kwargs)
        return {"n_decoded": len(blocks)}

    monkeypatch.setattr(ts, "slice_eval", fake_slice_eval)
    monkeypatch.setattr(
        ts,
        "generate_cell_tokens",
        lambda model, *, prefix, max_new, seed, char_stats=None: [7, 8, 9],
    )
    monkeypatch.setattr(ts, "split_cell_into_features", lambda tokens: [])
    example = CellExample(
        region="singapore",
        tile_i=0,
        tile_j=0,
        cell_i=0,
        cell_j=0,
        prefix_ids=(101, 102),
        tokens=(1, 2, 3),
        cell_density_bucket=2,
        character_stats=(0.0,) * 7,  # Task 24b required field
    )
    dm = SimpleNamespace(val_cells=[example])
    cfg = ScaffoldConfig(devices=1, accelerator="cpu")
    model = SimpleNamespace(model=object())
    ts._generate_and_score(
        model,
        dm,
        cfg,
        n_cells=1,
        max_new=4,
        generated_length_cap=4,
        floor_regime_cell_length=DEFAULT_MAX_CELL_TOKENS,
    )
    assert captured["generated_length_cap"] == 4
    assert captured["floor_regime_cell_length"] == DEFAULT_MAX_CELL_TOKENS == 5760


# --- the refuse rule NESTS inside "a verdict was requested" (Task 14 follow-up) -------


def test_incommensurate_lengths_without_floor_inputs_keep_verdict_none() -> None:
    # Load-bearing nesting: the F15 length check lives INSIDE the "floor inputs
    # given" branch. Incommensurate lengths with NO floor inputs must not promote
    # the legacy verdict-None to INCOMMENSURATE — verdict-never-requested wins
    # over the refuse rule.
    out = slice_eval(
        [[1]],
        [_LINE],
        [0],
        generated_length_cap=512,
        floor_regime_cell_length=5760,
    )
    assert out["emergence_verdict"] is None
    assert out["building_metrics_floored"] is False

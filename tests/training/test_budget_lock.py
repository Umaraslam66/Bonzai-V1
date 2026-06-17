"""W1 budget-lock guards (token-budget coupled decision, locked 2026-06-11).

The locked quadruple (reports/2026-06-11-token-budget-coupled-decision-memo.md,
flash probe PASS reports/2026-06-11-sdpa-window-probe.yaml):
DEFAULT_MAX_CELL_TOKENS = max_len = 13,312 (EU p99.9-cover; worst city valencia
p99.9 = 13,268); scored runs require eval_max_new >= the budget (else the F15
emergence verdict is INCOMMENSURATE by construction); MAX_TOO_LONG_DROP_RATE
stays 0.005 (the "5x the per-city p99.9 design point" calibration carries over).

Lock-and-guards travel together: these tests pin the constants to the recorded
decision, and pin the scored-entrypoint commensurability gate (memo section 4:
a scored run must not train short / evaluate long; 2048 opt-downs stay legal
for NON-scored jobs only).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cfm.data.training.datamodule import DEFAULT_MAX_CELL_TOKENS, MAX_TOO_LONG_DROP_RATE
from cfm.training.config import ScaffoldConfig

_REPO = Path(__file__).resolve().parents[2]


def _load_scaffold():
    spec = importlib.util.spec_from_file_location(
        "train_scaffold_budget_lock", _REPO / "scripts" / "train_scaffold.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- the locked constants ------------------------------------------------------------


def test_default_max_cell_tokens_is_the_locked_13312():
    assert DEFAULT_MAX_CELL_TOKENS == 13_312


def test_scaffold_max_len_default_equals_the_budget_lock():
    # ONE authoritative number (memo section 4): the data-side drop threshold and the
    # model-side window must never drift apart silently.
    assert ScaffoldConfig(region="singapore").max_len == DEFAULT_MAX_CELL_TOKENS


def test_drop_rate_contract_is_unchanged():
    # Deliberately NOT changed by the lock: 0.005 keeps its "5x the per-city p99.9
    # design point" calibration because 13,312 is again a p99.9-class budget.
    assert MAX_TOO_LONG_DROP_RATE == 0.005


# --- the scored-entrypoint commensurability gate (memo section 4 tooth) ---------------


def test_scored_gate_refuses_max_len_below_the_budget():
    mod = _load_scaffold()
    cfg = ScaffoldConfig(
        region="singapore", max_len=2048, eval_max_new=DEFAULT_MAX_CELL_TOKENS, accelerator="cpu"
    )
    with pytest.raises(ValueError, match=r"2048.*13312|13312.*2048"):
        mod.assert_scored_commensurate(cfg)


def test_scored_gate_refuses_eval_max_new_below_the_budget():
    mod = _load_scaffold()
    cfg = ScaffoldConfig(
        region="singapore", max_len=DEFAULT_MAX_CELL_TOKENS, eval_max_new=512, accelerator="cpu"
    )
    with pytest.raises(ValueError, match=r"eval_max_new"):
        mod.assert_scored_commensurate(cfg)


def test_scored_gate_passes_a_compliant_config():
    mod = _load_scaffold()
    cfg = ScaffoldConfig(
        region="singapore",
        max_len=DEFAULT_MAX_CELL_TOKENS,
        eval_max_new=DEFAULT_MAX_CELL_TOKENS,
        accelerator="cpu",
    )
    mod.assert_scored_commensurate(cfg)  # must not raise


def test_parser_scored_run_flag_default_off():
    mod = _load_scaffold()
    assert mod._build_parser().parse_args([]).scored_run is False
    assert mod._build_parser().parse_args(["--scored-run"]).scored_run is True


def test_main_refuses_a_scored_run_at_an_optdown_window():
    # The wiring tooth: --scored-run + --max-len 2048 must die at the gate, BEFORE
    # any training/smoke executes (the gate is the first thing after config build).
    mod = _load_scaffold()
    # --region supplied (now REQUIRED) so the run reaches the scored-commensurability
    # gate — the subject under test — rather than the earlier region-required guard.
    with pytest.raises(ValueError, match=r"scored"):
        mod.main(["--scored-run", "--region", "singapore", "--max-len", "2048", "--devices", "1"])

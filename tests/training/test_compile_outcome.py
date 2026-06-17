"""Compile OUTCOME recorded in the report, not intent (readiness A-3 / Task 26).

``maybe_compile`` swallows torch.compile failures by design (disable rather
than fight the compiler) — but the report previously recorded only
``cfg.compile`` (the INTENT). A run that silently fell back to eager would
publish cost/throughput numbers that look compiled. The OUTCOME of the wrap
step is now stamped on the lit module and rendered into the reports/ summary.
(Note: torch.compile backend failures can also surface lazily at first
forward; the outcome records the WRAP step — the failure mode this project
actually hit — and the training log carries the rest.)
"""

from __future__ import annotations

import torch

import scripts.train_scaffold as ts
from cfm.training.config import ScaffoldConfig
from cfm.training.lit_module import ScaffoldLit
from cfm.training.train import maybe_compile

_TINY = dict(
    region="singapore",  # region is REQUIRED (no default); SG keeps these tests region-agnostic
    devices=1,
    accelerator="cpu",
    d_model=64,
    n_layers=2,
    n_heads=2,
    max_len=256,
)


def _tiny_cfg(**overrides) -> ScaffoldConfig:
    return ScaffoldConfig(**{**_TINY, **overrides})


def test_outcome_off_when_config_disables() -> None:
    cfg = _tiny_cfg(compile=False)
    lit = maybe_compile(ScaffoldLit(cfg), cfg)
    assert lit.compile_outcome == "off (cfg.compile=False)"


def test_outcome_compiled_on_successful_wrap(monkeypatch) -> None:
    monkeypatch.setattr(torch, "compile", lambda m: m)
    cfg = _tiny_cfg(compile=True)
    lit = maybe_compile(ScaffoldLit(cfg), cfg)
    assert lit.compile_outcome == "compiled"


def test_outcome_names_the_failure_when_wrap_raises(monkeypatch) -> None:
    def _boom(_m):
        raise RuntimeError("no inductor here")

    monkeypatch.setattr(torch, "compile", _boom)
    cfg = _tiny_cfg(compile=True)
    lit = maybe_compile(ScaffoldLit(cfg), cfg)
    assert lit.compile_outcome == "disabled: RuntimeError: no inductor here"


def test_report_records_the_outcome_not_the_intent(tmp_path, monkeypatch) -> None:
    """cfg.compile=True (intent) + a failed wrap must read 'disabled: ...' in
    the report — the regime where intent and outcome DISAGREE."""
    monkeypatch.chdir(tmp_path)  # _write_report writes to a relative reports/ dir
    cfg = _tiny_cfg(compile=True)
    report = ts._write_report(
        cfg,
        {},
        trained_steps=1,
        cost={"n_params_M": 1.0},
        compile_outcome="disabled: RuntimeError: no inductor here",
    )
    text = report.read_text(encoding="utf-8")
    assert "compile outcome" in text
    assert "disabled: RuntimeError: no inductor here" in text
    assert '"compile": true' in text  # the intent still sits in the Config dump


def test_run_short_wires_the_outcome_into_the_report() -> None:
    """Wiring pin: run_short must hand the lit module's outcome to
    _write_report (dropping the kwarg would silently revert to intent-only)."""
    import inspect

    src = inspect.getsource(ts.run_short)
    assert "compile_outcome" in src

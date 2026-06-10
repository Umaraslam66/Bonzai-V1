"""Across-job $WORK resume wired into run_short (readiness-closure Task 18, F8).

The 1B bake-off runs span multiple Slurm jobs, so a relaunched job must continue
from the latest checkpoint on $WORK instead of silently restarting from step 0
(F8's failure mode). This pins the wiring:

- ``run_short`` hands ``trainer.fit`` ``ckpt_path=resume_ckpt_path(ckpt_dir)`` —
  the latest checkpoint when one exists under ``work_checkpoint_dir(...)``, and
  ``None`` on a fresh run (Lightning's fresh-start semantics);
- write/read coherence: the dir handed to ``build_trainer(ckpt_dirpath=...)`` is
  the SAME dir the resume decision reads — one variable, so the write side and
  the read side cannot diverge;
- the highest-step fallback survives Lightning's ``auto_insert_metric_name``
  naming quirk (real files are named like ``stepstep=N.ckpt``);
- the resume decision is LOGGED at INFO either way, so an operator reading job
  logs sees which branch fired — a silent restart-from-0 is exactly F8.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.train_scaffold as ts
from cfm.training.config import ScaffoldConfig
from cfm.training.resume import work_checkpoint_dir

_TINY = dict(
    devices=1,
    accelerator="cpu",
    d_model=64,
    n_layers=2,
    n_heads=2,
    max_len=256,
    compile=False,
)

_PARAMS = 89_700_000  # -> _scale_label(89.7) == "90M"


def _tiny_cfg(**overrides) -> ScaffoldConfig:
    return ScaffoldConfig(**{**_TINY, **overrides})


class _Boom(Exception):
    """Sentinel: stops run_short right after trainer.fit records its kwargs."""


def _run_recording(monkeypatch, tmp_path, cfg) -> dict:
    """Run run_short with recorder stubs; returns what build_trainer/fit were handed.

    Same stub idiom as test_report_naming's run_short tests, except the sentinel is
    raised from ``fit`` (not ``build_trainer``) so BOTH the write-side
    ``ckpt_dirpath`` and the read-side ``ckpt_path`` are captured.
    """
    recorded: dict = {}

    def fake_fit(*args, **kwargs):
        # "MISSING" (not None) distinguishes "kwarg absent" from "ckpt_path=None".
        recorded["fit_ckpt_path"] = kwargs.get("ckpt_path", "MISSING")
        raise _Boom

    fake_trainer = SimpleNamespace(fit=fake_fit)

    def fake_build_trainer(cfg, **kwargs):
        recorded["ckpt_dirpath"] = kwargs["ckpt_dirpath"]
        return fake_trainer

    monkeypatch.setenv("WORK", str(tmp_path))
    monkeypatch.setattr(ts, "build_trainer", fake_build_trainer)
    monkeypatch.setattr(ts, "_datamodule", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(ts, "ScaffoldLit", lambda cfg: SimpleNamespace())
    monkeypatch.setattr(ts, "maybe_compile", lambda lit, cfg: lit)
    monkeypatch.setattr(ts, "_param_count", lambda cfg: _PARAMS)

    with pytest.raises(_Boom):
        ts.run_short(cfg, build_shards=False)
    return recorded


def _expected_ckpt_dir(cfg: ScaffoldConfig) -> Path:
    # The label computed the SAME way run_short does (one-sourced, Task 17).
    label = ts._scale_label(_PARAMS / 1e6)
    return work_checkpoint_dir(cfg.backbone, label, region=cfg.region, seed=cfg.seed)


# --- (a) fresh run: no checkpoint on $WORK -> ckpt_path=None ---------------------------


def test_fresh_run_passes_ckpt_path_none(tmp_path, monkeypatch) -> None:
    cfg = _tiny_cfg(eval_cells=0)
    recorded = _run_recording(monkeypatch, tmp_path, cfg)
    assert recorded["fit_ckpt_path"] is None  # kwarg present AND None -> fresh start


# --- (b) resume: last.ckpt on $WORK -> ckpt_path points at it --------------------------


def test_resume_passes_last_ckpt_path(tmp_path, monkeypatch) -> None:
    cfg = _tiny_cfg(eval_cells=0)
    monkeypatch.setenv("WORK", str(tmp_path))  # set BEFORE computing the plant dir
    ckpt_dir = _expected_ckpt_dir(cfg)
    ckpt_dir.mkdir(parents=True)
    last = ckpt_dir / "last.ckpt"
    last.touch()  # resume_ckpt_path only checks existence

    recorded = _run_recording(monkeypatch, tmp_path, cfg)
    assert recorded["fit_ckpt_path"] == last
    assert str(tmp_path) in str(recorded["fit_ckpt_path"])  # really WORK-rooted


# --- (c) write/read coherence: build_trainer dir IS the resume-read dir ----------------


def test_write_and_read_sides_share_one_checkpoint_dir(tmp_path, monkeypatch) -> None:
    cfg = _tiny_cfg(eval_cells=0)
    monkeypatch.setenv("WORK", str(tmp_path))
    ckpt_dir = _expected_ckpt_dir(cfg)
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "last.ckpt").touch()

    recorded = _run_recording(monkeypatch, tmp_path, cfg)
    # ModelCheckpoint writes where resume reads: one variable, no divergence possible.
    assert recorded["ckpt_dirpath"] == ckpt_dir
    assert Path(recorded["fit_ckpt_path"]).parent == recorded["ckpt_dirpath"]


# --- (d) highest-step fallback under Lightning's auto_insert_metric_name naming --------


def test_highest_step_fallback_without_last_ckpt(tmp_path, monkeypatch) -> None:
    cfg = _tiny_cfg(eval_cells=0)
    monkeypatch.setenv("WORK", str(tmp_path))
    ckpt_dir = _expected_ckpt_dir(cfg)
    ckpt_dir.mkdir(parents=True)
    # Real Lightning files are named "stepstep=N.ckpt" (auto_insert_metric_name quirk);
    # resume.py's _STEP_RE matches the trailing "step=(\d+)" regardless.
    (ckpt_dir / "stepstep=500.ckpt").touch()
    (ckpt_dir / "stepstep=1500.ckpt").touch()

    recorded = _run_recording(monkeypatch, tmp_path, cfg)
    assert recorded["fit_ckpt_path"] == ckpt_dir / "stepstep=1500.ckpt"


# --- (e) the decision is logged: an operator can see which branch fired ----------------


def test_resume_branch_logs_checkpoint_path(tmp_path, monkeypatch, caplog) -> None:
    cfg = _tiny_cfg(eval_cells=0)
    monkeypatch.setenv("WORK", str(tmp_path))
    ckpt_dir = _expected_ckpt_dir(cfg)
    ckpt_dir.mkdir(parents=True)
    last = ckpt_dir / "last.ckpt"
    last.touch()

    with caplog.at_level(logging.INFO, logger=ts.logger.name):
        _run_recording(monkeypatch, tmp_path, cfg)
    assert any("resum" in r.getMessage() and str(last) in r.getMessage() for r in caplog.records)


def test_fresh_branch_logs_fresh_run(tmp_path, monkeypatch, caplog) -> None:
    cfg = _tiny_cfg(eval_cells=0)
    with caplog.at_level(logging.INFO, logger=ts.logger.name):
        _run_recording(monkeypatch, tmp_path, cfg)
    expected_dir = _expected_ckpt_dir(cfg)
    assert any(
        "fresh" in r.getMessage() and str(expected_dir) in r.getMessage() for r in caplog.records
    )

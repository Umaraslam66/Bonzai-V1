"""Run-keyed reports + per-run checkpoint dirs (readiness-closure Task 17, F17).

The bake-off runs 4 backbones x 3 scales (x seeds later) against the SAME
release+region, but the report filename was keyed only on release+region (+ an
optional params suffix) and the checkpoint dirs were Lightning defaults — so a
second run silently overwrote the first run's report and could collide on
checkpoints. This pins the fix:

- the report filename carries the full run key (backbone, integer params-M, seed);
- two configs differing ONLY in backbone (or ONLY in seed) produce different paths;
- ``build_trainer(cfg, ckpt_dirpath=...)`` routes BOTH ModelCheckpoint callbacks to
  the given per-run dir (and ``None`` keeps today's Lightning-default behavior);
- ``run_short`` wires ``ckpt_dirpath=work_checkpoint_dir(backbone, scale_label,
  region=..., seed=...)`` — the FULL run key, so two runs differing only in seed or
  region never resume each other's ``last.ckpt`` (Task 17 follow-up);
- ``run_short`` one-sources the scale label: the SAME ``_scale_label(params_m)`` keys
  the checkpoint dir and the report filename, so they can never diverge by rounding
  (``cost["n_params_M"]`` is ``round(params_m, 1)``; labelling from it double-rounds
  and flips in the x.45-x.50 band).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from lightning.pytorch.callbacks import ModelCheckpoint

import scripts.train_scaffold as ts
from cfm.training.config import ScaffoldConfig
from cfm.training.resume import work_checkpoint_dir
from cfm.training.train import build_trainer

_TINY = dict(
    devices=1,
    accelerator="cpu",
    d_model=64,
    n_layers=2,
    n_heads=2,
    max_len=256,
    compile=False,
)


def _tiny_cfg(**overrides) -> ScaffoldConfig:
    return ScaffoldConfig(**{**_TINY, **overrides})


# --- (a) report path carries the run key ---------------------------------------------


def test_report_path_contains_backbone_params_and_seed(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # _write_report writes to a relative reports/ dir
    cfg = _tiny_cfg(seed=11)
    cost = {"n_params_M": 89.7}
    report = ts._write_report(cfg, {"n_decoded": 0}, trained_steps=1, cost=cost)
    assert report.exists()
    name = report.name
    assert cfg.backbone in name
    assert "90M" in name  # integer params-M, rounded from cost["n_params_M"]
    assert "seed11" in name


def test_report_path_prefers_cost_params_over_rebuilding_the_model(tmp_path, monkeypatch) -> None:
    # When cost carries n_params_M, _write_report must NOT rebuild the model.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        ts, "_param_count", lambda cfg: pytest.fail("model rebuilt despite cost given")
    )
    cfg = _tiny_cfg()
    report = ts._write_report(cfg, {}, trained_steps=1, cost={"n_params_M": 311.4})
    assert "311M" in report.name


def test_report_path_without_cost_computes_params_from_cfg(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ts, "_param_count", lambda cfg: 12_300_000)
    cfg = _tiny_cfg()
    report = ts._write_report(cfg, {}, trained_steps=1)
    assert "12M" in report.name


# --- (b) run-key injectivity: backbone / seed flips change the path -------------------


def test_configs_differing_only_in_backbone_yield_different_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cost = {"n_params_M": 89.7}
    a = ts._write_report(_tiny_cfg(backbone="transformer-ar"), {}, trained_steps=1, cost=cost)
    b = ts._write_report(_tiny_cfg(backbone="mamba-hybrid"), {}, trained_steps=1, cost=cost)
    assert a != b


def test_configs_differing_only_in_seed_yield_different_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cost = {"n_params_M": 89.7}
    a = ts._write_report(_tiny_cfg(seed=7), {}, trained_steps=1, cost=cost)
    b = ts._write_report(_tiny_cfg(seed=8), {}, trained_steps=1, cost=cost)
    assert a != b


# --- _scale_label: deterministic params-M -> label ------------------------------------


def test_scale_label_rounds_to_integer_megaparams() -> None:
    assert ts._scale_label(89.7) == "90M"
    assert ts._scale_label(311.4) == "311M"
    assert ts._scale_label(12.3) == "12M"


# --- (c) build_trainer routes BOTH checkpoint callbacks to ckpt_dirpath ---------------


def _checkpoint_callbacks(trainer) -> list[ModelCheckpoint]:
    return [cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)]


def test_build_trainer_ckpt_dirpath_routes_both_checkpoint_callbacks(tmp_path) -> None:
    cfg = _tiny_cfg()
    trainer = build_trainer(cfg, ckpt_dirpath=tmp_path, ckpt_every_n_steps=5)
    ckpts = _checkpoint_callbacks(trainer)
    assert len(ckpts) == 2  # time-interval + step-interval
    for cb in ckpts:
        assert cb.dirpath is not None
        assert Path(cb.dirpath) == tmp_path


def test_build_trainer_default_keeps_lightning_default_dirpath() -> None:
    cfg = _tiny_cfg()
    trainer = build_trainer(cfg, ckpt_every_n_steps=5)
    ckpts = _checkpoint_callbacks(trainer)
    assert len(ckpts) == 2
    for cb in ckpts:
        assert cb.dirpath is None  # unset at construction -> Lightning default at fit


# --- (d) run_short wires ckpt_dirpath = work_checkpoint_dir(full run key) -------------


class _Boom(Exception):
    """Sentinel: stops run_short right after build_trainer records its kwargs."""


def test_run_short_passes_work_checkpoint_dir_to_build_trainer(tmp_path, monkeypatch) -> None:
    recorded: dict = {}

    def fake_build_trainer(cfg, **kwargs):
        recorded.update(kwargs)
        raise _Boom

    monkeypatch.setenv("WORK", str(tmp_path))
    monkeypatch.setattr(ts, "build_trainer", fake_build_trainer)
    monkeypatch.setattr(ts, "_datamodule", lambda *a, **k: SimpleNamespace())
    # the stub models ScaffoldLit's declared interface: compile_outcome is a real
    # attribute (Task-26 quality review #6), no longer a getattr-with-default read
    monkeypatch.setattr(
        ts, "ScaffoldLit", lambda cfg: SimpleNamespace(compile_outcome="unknown (stub)")
    )
    monkeypatch.setattr(ts, "maybe_compile", lambda lit, cfg: lit)
    monkeypatch.setattr(ts, "_param_count", lambda cfg: 89_700_000)

    cfg = _tiny_cfg(eval_cells=0)  # no generation -> no emergence floor needed
    with pytest.raises(_Boom):
        ts.run_short(cfg, build_shards=False)

    expected = work_checkpoint_dir(cfg.backbone, "90M", region=cfg.region, seed=cfg.seed)
    assert recorded["ckpt_dirpath"] == expected
    assert str(tmp_path) in str(expected)  # really on the WORK-rooted path


# --- (e) one-sourced scale label: report and ckpt dir can never diverge by rounding ---


def test_write_report_uses_passed_scale_label_over_cost_derived(tmp_path, monkeypatch) -> None:
    # cost carries round(params_m, 1) = 89.5 -> labelling from it double-rounds to
    # "90M"; the label one-sourced from raw params_m (89.45) is "89M". The passed
    # label must win, and the model must still never be rebuilt.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        ts, "_param_count", lambda cfg: pytest.fail("model rebuilt despite scale_label given")
    )
    cfg = _tiny_cfg()
    report = ts._write_report(
        cfg, {}, trained_steps=1, cost={"n_params_M": 89.5}, scale_label="89M"
    )
    assert "89M" in report.name
    assert "90M" not in report.name


def test_run_short_report_label_matches_ckpt_dir_label(tmp_path, monkeypatch) -> None:
    # End-to-end through run_short with raw params_m = 89.45, the x.45-x.50 band where
    # the old double-rounding path (round -> 89.5 -> "90M") diverges from the
    # one-sourced label (89.45 -> "89M"). Report filename and ckpt dir must agree.
    recorded: dict = {}
    fake_trainer = SimpleNamespace(fit=lambda *a, **k: None, is_global_zero=True, global_step=1)

    def fake_build_trainer(cfg, **kwargs):
        recorded["ckpt_dirpath"] = kwargs["ckpt_dirpath"]
        return fake_trainer

    monkeypatch.setenv("WORK", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ts, "build_trainer", fake_build_trainer)
    monkeypatch.setattr(ts, "_datamodule", lambda *a, **k: SimpleNamespace())
    # stub models ScaffoldLit's declared interface (compile_outcome, review #6)
    monkeypatch.setattr(
        ts, "ScaffoldLit", lambda cfg: SimpleNamespace(compile_outcome="unknown (stub)")
    )
    monkeypatch.setattr(ts, "maybe_compile", lambda lit, cfg: lit)
    monkeypatch.setattr(ts, "_param_count", lambda cfg: 89_450_000)
    monkeypatch.setattr(ts, "_generate_and_score", lambda *a, **k: {})

    out = ts.run_short(_tiny_cfg(eval_cells=0), build_shards=False)

    assert "89M" in str(recorded["ckpt_dirpath"])
    assert "89M" in Path(out["report"]).name
    assert "90M" not in Path(out["report"]).name  # the double-rounded label never ships

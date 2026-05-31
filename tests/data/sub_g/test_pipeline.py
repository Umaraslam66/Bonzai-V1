from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cfm.data.sub_g import pipeline
from cfm.data.sub_g.pipeline import ChainConfig, run_chain
from cfm.data.sub_g.validator import ValidationResult

_VOLATILE = {"run_timestamp": "T", "host": "h", "run_uuid": "u", "sub_g_commit_sha": "s"}


def _mk_success(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "_SUCCESS").touch()


def _cfg(tmp_path: Path, force: bool = False) -> ChainConfig:
    return ChainConfig(
        region="singapore",
        release="2026-04-15.0",
        sub_c_region_dir=tmp_path / "sub_c",
        sub_d_region_dir=tmp_path / "sub_d",
        sub_e_region_dir=tmp_path / "sub_e",
        sub_f_region_dir=tmp_path / "sub_f",
        output_dir=tmp_path / "out",
        volatile=_VOLATILE,
        force=force,
    )


def _stub_validate(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline,
        "validate_region",
        lambda *a, **k: ValidationResult(True, 0, False, 0, True),
    )


def test_run_chain_requires_sub_c_success(tmp_path):
    cfg = _cfg(tmp_path)
    _mk_success(cfg.sub_d_region_dir)  # sub-D present, sub-C absent
    with pytest.raises(FileNotFoundError, match="sub-C"):
        run_chain(cfg)


def test_run_chain_skips_stages_with_success_when_not_forced(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, force=False)
    for d in (
        cfg.sub_c_region_dir,
        cfg.sub_d_region_dir,
        cfg.sub_e_region_dir,
        cfg.sub_f_region_dir,
    ):
        _mk_success(d)
    calls: list = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    _stub_validate(monkeypatch)
    res = run_chain(cfg)
    assert calls == []  # both sub-E and sub-F skipped (already _SUCCESS)
    assert res.passed is True


def test_run_chain_forces_rerun(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, force=True)
    for d in (
        cfg.sub_c_region_dir,
        cfg.sub_d_region_dir,
        cfg.sub_e_region_dir,
        cfg.sub_f_region_dir,
    ):
        _mk_success(d)
    invoked: list[str] = []

    def fake_run(args, **kw):
        # record the script being invoked; (re)create the stage _SUCCESS.
        script = args[1]
        invoked.append(Path(script).name)
        return None

    monkeypatch.setattr(subprocess, "run", fake_run)
    _stub_validate(monkeypatch)
    run_chain(cfg)
    assert "derive_boundary_contracts.py" in invoked  # sub-E re-run
    assert "derive.py" in invoked  # sub-F re-run


def test_run_chain_halts_on_stage_failure(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    _mk_success(cfg.sub_c_region_dir)
    _mk_success(cfg.sub_d_region_dir)
    # sub-E absent -> will "run"; make the subprocess fail.
    validate_called = []
    monkeypatch.setattr(pipeline, "validate_region", lambda *a, **k: validate_called.append(1))

    def boom(args, **kw):
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(subprocess.CalledProcessError):
        run_chain(cfg)
    assert validate_called == []  # halted before validation

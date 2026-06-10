"""Across-job $WORK checkpoint resume logic (Phase-2 bake-off Task 10)."""

from __future__ import annotations

from pathlib import Path

from cfm.training.resume import latest_checkpoint, resume_ckpt_path, work_checkpoint_dir


def test_latest_checkpoint_prefers_last_ckpt(tmp_path: Path) -> None:
    (tmp_path / "epoch=0-step=100.ckpt").write_text("x")
    (tmp_path / "epoch=0-step=500.ckpt").write_text("x")
    (tmp_path / "last.ckpt").write_text("x")
    assert latest_checkpoint(tmp_path).name == "last.ckpt"  # Lightning's last.ckpt preferred


def test_latest_checkpoint_picks_highest_step_when_no_last(tmp_path: Path) -> None:
    (tmp_path / "epoch=0-step=100.ckpt").write_text("x")
    (tmp_path / "epoch=0-step=500.ckpt").write_text("x")
    assert latest_checkpoint(tmp_path).name == "epoch=0-step=500.ckpt"


def test_resume_returns_none_on_fresh_run(tmp_path: Path) -> None:
    assert resume_ckpt_path(tmp_path) is None  # empty dir -> fresh, no false resume


def test_resume_returns_none_when_dir_absent(tmp_path: Path) -> None:
    assert resume_ckpt_path(tmp_path / "does-not-exist") is None


def test_resume_returns_checkpoint_when_present(tmp_path: Path) -> None:
    (tmp_path / "last.ckpt").write_text("x")
    assert resume_ckpt_path(tmp_path) == tmp_path / "last.ckpt"


def test_work_checkpoint_dir_is_under_work_and_per_run(tmp_path: Path) -> None:
    d = work_checkpoint_dir("mamba-hybrid", "1B", work_root=tmp_path)
    assert d == tmp_path / "Bonzai-OSM" / "checkpoints" / "bakeoff" / "mamba-hybrid-1B"


def test_work_checkpoint_dir_reads_WORK_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WORK", str(tmp_path / "work"))
    d = work_checkpoint_dir("transformer-ar", "300M")
    assert str(d).startswith(str(tmp_path / "work"))

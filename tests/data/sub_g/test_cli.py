from __future__ import annotations

from cfm.data.sub_g import cli
from cfm.data.sub_g.validator import ValidationResult

_ARGS = [
    "--region",
    "singapore",
    "--release",
    "2026-04-15.0",
    "--sub-c-region-dir",
    "/tmp/c",
    "--sub-d-region-dir",
    "/tmp/d",
    "--sub-e-region-dir",
    "/tmp/e",
    "--sub-f-region-dir",
    "/tmp/f",
    "--output-dir",
    "/tmp/out",
]


def test_exit_code_for_clean_and_quarantine():
    assert cli.exit_code_for(ValidationResult(True, 0, False, 0, True)) == cli.EXIT_CLEAN
    assert cli.exit_code_for(ValidationResult(False, 3, False, 3, False)) == cli.EXIT_QUARANTINE


def test_build_volatile_has_required_keys():
    v = cli.build_volatile()
    assert set(v) == {"run_timestamp", "host", "run_uuid", "sub_g_commit_sha"}


def test_derive_main_returns_clean(monkeypatch):
    monkeypatch.setattr(cli, "run_chain", lambda cfg: ValidationResult(True, 0, False, 0, True))
    assert cli.derive_main(_ARGS) == cli.EXIT_CLEAN


def test_derive_main_returns_quarantine_nonempty(monkeypatch):
    monkeypatch.setattr(cli, "run_chain", lambda cfg: ValidationResult(False, 5, False, 5, False))
    assert cli.derive_main(_ARGS) == cli.EXIT_QUARANTINE


def test_derive_main_precondition_failure_exits_2(monkeypatch):
    def boom(cfg):
        raise FileNotFoundError("sub-C _SUCCESS missing")

    monkeypatch.setattr(cli, "run_chain", boom)
    assert cli.derive_main(_ARGS) == cli.EXIT_PRECONDITION


def test_validate_main_returns_clean(monkeypatch):
    monkeypatch.setattr(
        cli, "validate_region", lambda *a, **k: ValidationResult(True, 0, False, 0, True)
    )
    assert cli.validate_main(_ARGS) == cli.EXIT_CLEAN

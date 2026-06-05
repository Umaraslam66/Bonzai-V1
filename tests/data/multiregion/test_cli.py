"""extract_region_batch CLI: --dry-run prints the per-city plan and submits
nothing; a boost partition is rejected loud (exit nonzero), even in dry-run."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "extract_region_batch.py"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args], capture_output=True, text=True, cwd=_REPO
    )


def test_cli_dry_run_lists_planned_stages_per_city():
    out = _run(["--cities", "berlin", "--partition", "lrd_all_serial", "--dry-run"])
    assert out.returncode == 0, out.stderr
    assert "berlin" in out.stdout
    assert "fetch" in out.stdout  # the plan is printed (no city state → full chain)


def test_cli_rejects_boost_partition_even_in_dry_run():
    out = _run(["--cities", "berlin", "--partition", "boost_usr_prod", "--dry-run"])
    assert out.returncode != 0
    assert "boost_usr_prod" in (out.stdout + out.stderr)

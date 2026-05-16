from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _script_path() -> Path:
    return _REPO_ROOT / "scripts" / "cfm_data_invalidate.py"


def _write_release_pin(root: Path, release: str = "2026-04-15.0") -> None:
    cfg = root / "configs" / "data"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "overture_release.yaml").write_text(
        yaml.safe_dump(
            {
                "release": release,
                "overture_schema_version": "v1.16.0",
                "release_date": "2026-04-15",
                "release_subversion": 0,
            }
        )
    )


def _populate_cache(root: Path, release: str, region: str) -> Path:
    cache = root / "data" / "cache" / "overture" / release / region
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "manifest.yaml").write_text("dummy")
    (cache / "buildings.parquet").write_text("dummy")
    return cache


def test_invalidate_removes_region_dir(tmp_path: Path) -> None:
    _write_release_pin(tmp_path)
    cache = _populate_cache(tmp_path, "2026-04-15.0", "singapore")
    assert cache.exists()
    script = _script_path()
    result = subprocess.run(
        [sys.executable, str(script), "singapore", "--repo-root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not cache.exists()


def test_invalidate_refuses_path_traversal(tmp_path: Path) -> None:
    _write_release_pin(tmp_path)
    script = _script_path()
    result = subprocess.run(
        [sys.executable, str(script), "../etc", "--repo-root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "refus" in combined or "invalid" in combined


def test_invalidate_no_cache_is_a_no_op(tmp_path: Path) -> None:
    _write_release_pin(tmp_path)
    script = _script_path()
    result = subprocess.run(
        [sys.executable, str(script), "singapore", "--repo-root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "nothing to remove" in (result.stdout + result.stderr).lower()

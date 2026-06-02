from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cfm.data.overture.backend import LocalFixtureBackend
from cfm.data.overture.errors import (
    CacheCorrupt,
    OversizedFetch,
    RegionNotFound,
    ReleaseNotConfigured,
)
from cfm.data.overture.loader import THEMES_TO_LOAD, load_region


class _CountingBackend:
    """Wraps a backend and counts estimate_size / read_theme calls.

    Used to prove the confirm=True COUNT(*) optimization skips the pre-fetch
    size estimate while leaving the actual fetch (read_theme) untouched.
    """

    def __init__(self, inner: LocalFixtureBackend) -> None:
        self.inner = inner
        self.estimate_calls = 0
        self.read_calls = 0

    def read_theme(self, **kw):  # type: ignore[no-untyped-def]
        self.read_calls += 1
        return self.inner.read_theme(**kw)

    def estimate_size(self, **kw):  # type: ignore[no-untyped-def]
        self.estimate_calls += 1
        return self.inner.estimate_size(**kw)


def _write_release_pin(repo_root: Path, release: str = "2026-04-15.0") -> Path:
    cfg_dir = repo_root / "configs" / "data"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / "overture_release.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "release": release,
                "overture_schema_version": "v1.16.0",
                "release_date": "2026-04-15",
                "release_subversion": 0,
            }
        )
    )
    return path


def _write_singapore_region(repo_root: Path) -> Path:
    cfg_dir = repo_root / "configs" / "data" / "regions"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / "singapore.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "singapore",
                "admin": {
                    "source": "overture://divisions",
                    "country_code": "SG",
                    "level": "country",
                },
                "fallback_bbox": [103.6, 1.16, 104.05, 1.48],
                "crs": "EPSG:4326",
                "projected_crs": "EPSG:3414",
            }
        )
    )
    return path


@pytest.fixture()
def isolated_repo(tmp_path: Path, overture_mini_dir: Path) -> Path:
    """A throwaway repo root with configs + a fixtures dir pointed at the
    real overture_mini parquets. Cache lives under tmp_path/data/cache/."""
    _write_release_pin(tmp_path)
    _write_singapore_region(tmp_path)
    # Symlink overture_mini fixtures so LocalFixtureBackend can read them.
    fix = tmp_path / "tests" / "fixtures" / "overture_mini"
    fix.parent.mkdir(parents=True, exist_ok=True)
    fix.symlink_to(overture_mini_dir)
    return tmp_path


def test_load_region_happy_path_first_fetch(isolated_repo: Path, overture_mini_dir: Path) -> None:
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    region = load_region("singapore", backend=backend, repo_root=isolated_repo)
    assert region.name == "singapore"
    assert region.release == "2026-04-15.0"
    assert set(region.themes) == {"buildings", "places", "transportation", "base", "divisions"}
    assert region.manifest_path.exists()
    # Manifest contains the right release.
    data = yaml.safe_load(region.manifest_path.read_text())
    assert data["release"] == "2026-04-15.0"
    assert "themes" in data and "buildings" in data["themes"]


def test_load_region_second_call_uses_cache(isolated_repo: Path, overture_mini_dir: Path) -> None:
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    first = load_region("singapore", backend=backend, repo_root=isolated_repo)
    # Mutate the fixture's mtime so we'd notice a re-read; the loader should
    # NOT touch the source on a cache hit.
    second = load_region("singapore", backend=backend, repo_root=isolated_repo)
    assert first.manifest_path == second.manifest_path


# --- Multi-region (Q1): projected_crs is a required region-config field ------
# Singapore pins EPSG:3414; new European cities pin their conformal UTM zone.
# It rides on the Region object so sub-C reads region.projected_crs (one source).


def test_load_region_exposes_projected_crs(isolated_repo: Path, overture_mini_dir: Path) -> None:
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    region = load_region("singapore", backend=backend, repo_root=isolated_repo)
    assert region.projected_crs == "EPSG:3414"


def test_load_region_projected_crs_survives_cache_path(
    isolated_repo: Path, overture_mini_dir: Path
) -> None:
    # The cache-hit reconstruction must also carry projected_crs (read from the
    # region config, not the cache manifest), not silently drop it.
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    load_region("singapore", backend=backend, repo_root=isolated_repo)  # populate cache
    cached = load_region("singapore", backend=backend, repo_root=isolated_repo)
    assert cached.projected_crs == "EPSG:3414"


def test_load_region_missing_projected_crs_raises(
    isolated_repo: Path, overture_mini_dir: Path
) -> None:
    # projected_crs is REQUIRED — no silent default to Singapore's 3414, which
    # would be a footgun for a new European region.
    region_path = isolated_repo / "configs" / "data" / "regions" / "singapore.yaml"
    cfg = yaml.safe_load(region_path.read_text())
    del cfg["projected_crs"]
    region_path.write_text(yaml.safe_dump(cfg))
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    with pytest.raises(ValueError):
        load_region("singapore", backend=backend, repo_root=isolated_repo)


# --- COUNT(*) fetch optimization: confirm=True skips the pre-fetch estimate --
# The per-theme estimate_size COUNT(*) scans every Overture S3 partition and
# only feeds the OversizedFetch guard. When confirm=True the guard is overridden,
# so the COUNT is pure wasted work — skip it. read_theme is untouched, so the
# fetched data is byte-identical (the determinism requirement).


def test_load_region_confirm_true_skips_size_estimate(
    isolated_repo: Path, overture_mini_dir: Path
) -> None:
    backend = _CountingBackend(LocalFixtureBackend(fixtures_dir=overture_mini_dir))
    load_region("singapore", backend=backend, repo_root=isolated_repo, confirm=True)
    assert backend.estimate_calls == 0  # COUNT(*) pre-estimate skipped
    assert backend.read_calls == len(THEMES_TO_LOAD)  # actual fetch still ran


def test_load_region_confirm_false_runs_size_estimate(
    isolated_repo: Path, overture_mini_dir: Path
) -> None:
    backend = _CountingBackend(LocalFixtureBackend(fixtures_dir=overture_mini_dir))
    load_region("singapore", backend=backend, repo_root=isolated_repo, confirm=False)
    assert backend.estimate_calls == len(THEMES_TO_LOAD)  # guard active -> estimate ran


def test_confirm_true_fetches_byte_identical_themes_to_confirm_false(
    isolated_repo: Path, overture_mini_dir: Path
) -> None:
    # The COUNT(*) skip must NOT change WHAT is fetched.
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    r_false = load_region("singapore", backend=backend, repo_root=isolated_repo, confirm=False)
    r_true = load_region(
        "singapore", backend=backend, repo_root=isolated_repo, refresh=True, confirm=True
    )
    assert set(r_false.themes) == set(r_true.themes)
    for theme in r_false.themes:
        assert r_false.themes[theme].equals(r_true.themes[theme])


def test_load_region_unknown_region_raises(isolated_repo: Path, overture_mini_dir: Path) -> None:
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    with pytest.raises(RegionNotFound):
        load_region("atlantis", backend=backend, repo_root=isolated_repo)


def test_load_region_missing_release_pin_raises(tmp_path: Path, overture_mini_dir: Path) -> None:
    _write_singapore_region(tmp_path)
    # No release pin written.
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    with pytest.raises(ReleaseNotConfigured):
        load_region("singapore", backend=backend, repo_root=tmp_path)


def test_load_region_release_mismatch_silently_refetches(
    isolated_repo: Path, overture_mini_dir: Path
) -> None:
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    # First fetch at the pinned release.
    region = load_region("singapore", backend=backend, repo_root=isolated_repo)
    # Re-pin to a different release.
    _write_release_pin(isolated_repo, release="2099-01-01.0")
    region2 = load_region("singapore", backend=backend, repo_root=isolated_repo)
    assert region2.release == "2099-01-01.0"
    assert region2.manifest_path != region.manifest_path  # different release subdir


def test_load_region_sha_mismatch_raises_cache_corrupt(
    isolated_repo: Path, overture_mini_dir: Path
) -> None:
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    region = load_region("singapore", backend=backend, repo_root=isolated_repo)
    # Tamper with one cached parquet.
    buildings = region.manifest_path.parent / "buildings.parquet"
    buildings.write_bytes(buildings.read_bytes() + b"corruption")
    with pytest.raises(CacheCorrupt):
        load_region("singapore", backend=backend, repo_root=isolated_repo)


def test_load_region_refresh_true_ignores_cache(
    isolated_repo: Path, overture_mini_dir: Path
) -> None:
    backend = LocalFixtureBackend(fixtures_dir=overture_mini_dir)
    first = load_region("singapore", backend=backend, repo_root=isolated_repo)
    # Snapshot the first manifest before refresh overwrites it; both
    # load_region calls write to the same path under the same release.
    first_data = yaml.safe_load(first.manifest_path.read_text())
    second = load_region("singapore", backend=backend, repo_root=isolated_repo, refresh=True)
    assert first.manifest_path == second.manifest_path
    # Manifest fetched_at differs after a refresh.
    second_data = yaml.safe_load(second.manifest_path.read_text())
    assert first_data["fetched_at"] != second_data["fetched_at"]


def test_oversized_fetch_aborts_without_confirm(
    isolated_repo: Path, overture_mini_dir: Path
) -> None:
    # Use a backend that reports huge estimates.
    class FakeBackend(LocalFixtureBackend):
        def estimate_size(self, **kw):  # type: ignore[override]
            from cfm.data.overture.region import SizeEstimate

            return SizeEstimate(rows=10_000_000, bytes=3 * 1024 * 1024 * 1024)  # 3 GB

    backend = FakeBackend(fixtures_dir=overture_mini_dir)
    with pytest.raises(OversizedFetch):
        load_region("singapore", backend=backend, repo_root=isolated_repo)


def test_oversized_fetch_proceeds_with_confirm(
    isolated_repo: Path, overture_mini_dir: Path
) -> None:
    class FakeBackend(LocalFixtureBackend):
        def estimate_size(self, **kw):  # type: ignore[override]
            from cfm.data.overture.region import SizeEstimate

            return SizeEstimate(rows=10_000_000, bytes=3 * 1024 * 1024 * 1024)

    backend = FakeBackend(fixtures_dir=overture_mini_dir)
    # confirm=True should proceed.
    region = load_region("singapore", backend=backend, repo_root=isolated_repo, confirm=True)
    assert region.name == "singapore"

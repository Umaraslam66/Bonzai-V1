"""Regression: the sub_c cross-tile validator must derive the tile-dir EPSG
label from ``manifest.region_crs``, NOT a hardcoded ``EPSG3414``.

Caught by the Berlin (EPSG:25833) pilot on 2026-06-03: the all-Singapore test
suite was structurally blind to this because the ``EPSG3414`` hardcode trivially
matches Singapore's own label (sample-regime-blind). These tests exercise a
non-Singapore regime so the hardcode can never silently return.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from cfm.data.sub_c.validator_cross_tile import _check_manifest_tiles_match_filesystem


def _manifest(region_crs: str, tiles: list[dict]) -> SimpleNamespace:
    # Duck-types the RegionManifest fields the checker reads.
    return SimpleNamespace(region_crs=region_crs, tiles=tiles)


def test_cross_tile_match_uses_region_crs_label_for_utm_region(tmp_path: Path) -> None:
    # A non-Singapore region (EPSG:25833) whose tile dirs are tile=EPSG25833_*
    # must validate clean — the label comes from region_crs, not a constant.
    m = _manifest(
        "EPSG:25833",
        [{"tile_i": 5, "tile_j": 3}, {"tile_i": 184, "tile_j": 2900}],
    )
    (tmp_path / "tile=EPSG25833_i5_j3").mkdir()
    (tmp_path / "tile=EPSG25833_i184_j2900").mkdir()
    _check_manifest_tiles_match_filesystem(m, tmp_path)  # must NOT raise


def test_cross_tile_match_still_clean_for_singapore(tmp_path: Path) -> None:
    m = _manifest("EPSG:3414", [{"tile_i": 5, "tile_j": 15}])
    (tmp_path / "tile=EPSG3414_i5_j15").mkdir()
    _check_manifest_tiles_match_filesystem(m, tmp_path)  # must NOT raise

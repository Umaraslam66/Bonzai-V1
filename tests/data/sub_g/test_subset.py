from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from cfm.data.io import write_parquet
from cfm.data.sub_g.subset import (
    assert_gap_empty,
    density_distribution,
    select_densest,
)


def test_select_densest_deterministic_and_excludes_empty():
    counts = {"tile=a": 5, "tile=b": 10, "tile=c": 0, "tile=d": 10}
    # 10-tie broken by name (b<d), then 5; the 0-count tile is excluded.
    assert select_densest(counts, 3) == ["tile=b", "tile=d", "tile=a"]


def test_select_densest_cutoff_is_nth_densest():
    counts = {f"tile={i:02d}": 100 - i for i in range(10)}
    top5 = select_densest(counts, 5)
    assert len(top5) == 5
    assert counts[top5[-1]] == 96  # 5th densest = cutoff


def test_density_distribution():
    assert density_distribution([1, 2, 3, 4]) == {
        "n": 4,
        "min": 1,
        "median": 2.5,
        "p99": 4,
        "max": 4,
        "total": 10,
    }


def test_density_distribution_empty():
    assert density_distribution([])["n"] == 0


def test_assert_gap_empty_passes_when_gap_tiles_are_empty(tmp_path: Path):
    sub_c = tmp_path / "sub_c"
    sub_d = tmp_path / "sub_d"
    schema = pa.schema([pa.field("x", pa.int8())])
    # shared tile (in both) + a gap tile (sub-C only), both with 0 features.
    for tile in ("tile=EPSG3414_i0_j0",):
        (sub_d / tile).mkdir(parents=True)
    for tile in ("tile=EPSG3414_i0_j0", "tile=EPSG3414_i9_j9"):
        (sub_c / tile).mkdir(parents=True)
        write_parquet(pa.table({"x": []}, schema=schema), sub_c / tile / "features.parquet")
    res = assert_gap_empty(sub_c, sub_d)
    assert res.gap_tile_count == 1  # i9_j9 only
    assert res.passed is True
    assert res.max_feature_count == 0


def test_assert_gap_empty_fails_on_nonempty_gap_tile(tmp_path: Path):
    sub_c = tmp_path / "sub_c"
    sub_d = tmp_path / "sub_d"
    schema = pa.schema([pa.field("x", pa.int8())])
    (sub_d / "tile=EPSG3414_i0_j0").mkdir(parents=True)
    (sub_c / "tile=EPSG3414_i0_j0").mkdir(parents=True)
    write_parquet(
        pa.table({"x": []}, schema=schema), sub_c / "tile=EPSG3414_i0_j0" / "features.parquet"
    )
    # gap tile with 2 features -> sub-D skipped a non-empty tile (bug).
    (sub_c / "tile=EPSG3414_i9_j9").mkdir(parents=True)
    write_parquet(
        pa.table({"x": [1, 2]}, schema=schema), sub_c / "tile=EPSG3414_i9_j9" / "features.parquet"
    )
    res = assert_gap_empty(sub_c, sub_d)
    assert res.passed is False
    assert res.max_feature_count == 2
    assert ("tile=EPSG3414_i9_j9", 2) in res.offending

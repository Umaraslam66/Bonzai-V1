"""Sub-G T11 Step-0 (read-only): sub-C-minus-sub-D gap assertion + densest-N subset.

Feature counts read only the parquet footer
(``pq.ParquetFile(path).metadata.num_rows``) — no row data is materialized.
Densest-N selection is deterministic: sort by ``(-count, tile_name)``. This
module selects the "measurement run" subset and asserts the sub-C/sub-D tile gap
is genuinely empty (else a sub-D bug: it skipped a non-empty tile).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import pyarrow.parquet as pq


def tile_names(region_dir: Path) -> set[str]:
    return {p.name for p in region_dir.glob("tile=*") if p.is_dir()}


def feature_count(sub_c_region_dir: Path, tile: str) -> int:
    """sub-C features.parquet row count via footer metadata (no data read)."""
    return pq.ParquetFile(sub_c_region_dir / tile / "features.parquet").metadata.num_rows


@dataclass(frozen=True)
class GapAssertion:
    gap_tile_count: int
    max_feature_count: int
    passed: bool
    offending: list[tuple[str, int]]


def assert_gap_empty(sub_c_region_dir: Path, sub_d_region_dir: Path) -> GapAssertion:
    """Every tile in (sub-C minus sub-D) must have 0 features, else sub-D skipped a
    non-empty tile (a sub-D bug). Reports the max feature count across the gap so
    a pass distinguishes "binds + confirms upstream clean" from "uniformly empty".
    """
    gap = sorted(tile_names(sub_c_region_dir) - tile_names(sub_d_region_dir))
    counts = [(t, feature_count(sub_c_region_dir, t)) for t in gap]
    offending = [(t, c) for t, c in counts if c > 0]
    max_fc = max((c for _, c in counts), default=0)
    return GapAssertion(
        gap_tile_count=len(gap),
        max_feature_count=max_fc,
        passed=not offending,
        offending=offending,
    )


def select_densest(counts: dict[str, int], n: int) -> list[str]:
    """Top-n non-empty tiles by feature count; deterministic (-count, tile_name)."""
    non_empty = [(t, c) for t, c in counts.items() if c > 0]
    non_empty.sort(key=lambda tc: (-tc[1], tc[0]))
    return [t for t, _ in non_empty[:n]]


def _percentile(sorted_vals: list[int], p: float) -> int:
    if not sorted_vals:
        return 0
    idx = max(0, min(len(sorted_vals) - 1, math.ceil(p / 100.0 * len(sorted_vals)) - 1))
    return sorted_vals[idx]


def density_distribution(feature_counts: list[int]) -> dict:
    """min / median / p99 / max / total / n over the given feature counts."""
    if not feature_counts:
        return {"n": 0, "min": 0, "median": 0, "p99": 0, "max": 0, "total": 0}
    s = sorted(feature_counts)
    return {
        "n": len(s),
        "min": s[0],
        "median": median(s),
        "p99": _percentile(s, 99.0),
        "max": s[-1],
        "total": sum(s),
    }

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldFrequencyResult:
    field: str  # display label, e.g. "buildings.class"
    n_total: int  # total rows in the source table
    n_present: int  # rows where field is non-null (or list non-empty)
    counts: dict[str, int]  # category -> count
    is_list_field: bool  # True only for places.categories.alternate
    total_occurrences: int  # sum of counts; == n_present for scalar fields,
    # > n_present for list fields


@dataclass(frozen=True)
class FloorStrategy:
    name: str  # "Very lenient" | "Lenient" | "Moderate" | "Strict" | "Very strict"
    percentage: float | None  # e.g. 0.0003 = 0.03%; None for Very lenient
    hard_min: int  # absolute floor independent of N


@dataclass(frozen=True)
class CutBehaviorRow:
    strategy: FloorStrategy
    effective_floor: int  # max(percentage * N_present, hard_min)
    n_total_categories: int  # before cutting
    n_kept: int
    n_dropped: int
    coverage_retained_pct: float  # 0..100


@dataclass(frozen=True)
class ListLengthDistribution:
    field: str  # "places.categories.alternate"
    n_total_rows: int  # full theme row count (denominator for percentages)
    buckets: dict[str, tuple[int, float]]  # "0".."5+" -> (count, pct_of_n_total_rows)

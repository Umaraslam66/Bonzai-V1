from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc


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


# ---------------------------------------------------------------------------
# Frequency computation
# ---------------------------------------------------------------------------


def compute_field_frequencies(
    table: pa.Table,
    column_path: str,
    *,
    label: str | None = None,
    is_list_field: bool = False,
) -> FieldFrequencyResult:
    """Count categorical occurrences in *column_path* of *table*.

    column_path: in-table path. Dot syntax navigates into structs.
    label:       human-readable display name; defaults to column_path.
    is_list_field: if True, treat the resolved column as list<string>
                   and count each list element separately. n_present is
                   then the number of rows with a non-null, non-empty list.

    Raises ValueError if the column is missing, the struct path cannot be
    traversed, or is_list_field is set on a non-list column.
    """
    display = label if label is not None else column_path
    n_total = table.num_rows
    column = _resolve_column(table, column_path)
    if is_list_field:
        return _count_list_field(column, display, n_total)
    return _count_scalar_field(column, display, n_total)


def _resolve_column(table: pa.Table, column_path: str) -> pa.ChunkedArray:
    """Return the ChunkedArray at *column_path*, navigating into structs as needed.

    Uses ``pc.struct_field`` for each struct descent step so that null validity
    at the struct level is correctly propagated to the extracted child column.
    """
    head, _, tail = column_path.partition(".")
    if head not in table.column_names:
        raise ValueError(
            f"column {head!r} not found in table; available columns: {sorted(table.column_names)}"
        )
    column: pa.ChunkedArray = table.column(head)
    if not tail:
        return column
    # Navigate into a struct via remaining dot-separated keys.
    for key in tail.split("."):
        if not pa.types.is_struct(column.type):
            raise ValueError(
                f"cannot navigate into non-struct column at key {key!r} "
                f"(column type: {column.type})"
            )
        field_index = column.type.get_field_index(key)
        if field_index < 0:
            raise ValueError(
                f"struct field {key!r} not found; available fields: "
                f"{[column.type.field(i).name for i in range(column.type.num_fields)]}"
            )
        # pc.struct_field propagates struct-level nulls to the child array,
        # whereas chunk.field(i) does not — critical for nullable struct rows.
        column = pc.struct_field(column, key)
    return column


def _count_scalar_field(
    column: pa.ChunkedArray,
    display: str,
    n_total: int,
) -> FieldFrequencyResult:
    counter: Counter[str] = Counter()
    n_present = 0
    for value in column.to_pylist():
        if value is None:
            continue
        counter[str(value)] += 1
        n_present += 1
    return FieldFrequencyResult(
        field=display,
        n_total=n_total,
        n_present=n_present,
        counts=dict(counter),
        is_list_field=False,
        total_occurrences=n_present,
    )


def _count_list_field(
    column: pa.ChunkedArray,
    display: str,
    n_total: int,
) -> FieldFrequencyResult:
    if not pa.types.is_list(column.type) and not pa.types.is_large_list(column.type):
        raise ValueError(f"is_list_field=True but column type is {column.type}; expected list<...>")
    counter: Counter[str] = Counter()
    n_present = 0
    total_occurrences = 0
    for value in column.to_pylist():
        if value is None or len(value) == 0:
            continue
        n_present += 1
        for item in value:
            if item is None:
                continue
            counter[str(item)] += 1
            total_occurrences += 1
    return FieldFrequencyResult(
        field=display,
        n_total=n_total,
        n_present=n_present,
        counts=dict(counter),
        is_list_field=True,
        total_occurrences=total_occurrences,
    )

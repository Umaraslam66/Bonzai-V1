from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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


# ---------------------------------------------------------------------------
# List length distribution
# ---------------------------------------------------------------------------


_BUCKET_KEYS: tuple[str, ...] = ("0", "1", "2", "3", "4", "5+")


def compute_list_length_distribution(
    table: pa.Table,
    column_path: str,
    *,
    label: str | None = None,
) -> ListLengthDistribution:
    """Histogram of list lengths for a list-of-strings column.

    Buckets: "0", "1", "2", "3", "4", "5+" (where "5+" aggregates lengths >= 5).
    Null lists count as length 0. Denominator for percentages is the full
    table row count (n_total_rows), NOT the count of non-null rows.
    """
    display = label if label is not None else column_path
    column = _resolve_column(table, column_path)
    if not (pa.types.is_list(column.type) or pa.types.is_large_list(column.type)):
        raise ValueError(
            f"compute_list_length_distribution: expected list column, got {column.type}"
        )

    n_total_rows = table.num_rows
    counts: dict[str, int] = dict.fromkeys(_BUCKET_KEYS, 0)
    for value in column.to_pylist():
        length = 0 if value is None else len(value)
        key = str(length) if length < 5 else "5+"
        counts[key] += 1

    buckets: dict[str, tuple[int, float]] = {}
    for key in _BUCKET_KEYS:
        c = counts[key]
        pct = 100.0 * c / n_total_rows if n_total_rows > 0 else 0.0
        buckets[key] = (c, pct)

    return ListLengthDistribution(
        field=display,
        n_total_rows=n_total_rows,
        buckets=buckets,
    )


# ---------------------------------------------------------------------------
# Floor strategy application
# ---------------------------------------------------------------------------


def apply_floor_strategy(
    result: FieldFrequencyResult,
    strategy: FloorStrategy,
) -> CutBehaviorRow:
    """Apply a floor strategy and report the resulting cut behavior.

    Effective floor:  max(percentage * n_present, hard_min)  (percentage may be None)
    Coverage retained: sum(counts kept) / total_occurrences. Defined as 100% when
                      total_occurrences == 0 (nothing to lose).
    """
    if result.n_present < 0:
        raise ValueError(f"n_present must be non-negative; got {result.n_present}")

    if strategy.percentage is None:
        effective_floor = strategy.hard_min
    else:
        pct_floor = int(strategy.percentage * result.n_present)
        effective_floor = max(pct_floor, strategy.hard_min)

    kept_count = 0
    n_kept = 0
    for count in result.counts.values():
        if count >= effective_floor:
            kept_count += count
            n_kept += 1
    n_total_categories = len(result.counts)
    n_dropped = n_total_categories - n_kept

    if result.total_occurrences == 0:
        coverage_pct = 100.0
    else:
        coverage_pct = 100.0 * kept_count / result.total_occurrences

    return CutBehaviorRow(
        strategy=strategy,
        effective_floor=effective_floor,
        n_total_categories=n_total_categories,
        n_kept=n_kept,
        n_dropped=n_dropped,
        coverage_retained_pct=coverage_pct,
    )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

_HEADER_TEMPLATE = """<!-- Auto-generated by scripts/analyse_singapore_frequencies.py
     Commit:        {commit_sha}
     Overture:      {overture_release}
     Run UTC:       {run_iso}
     Re-run reason: {rerun_reason}
     Do not edit by hand — edit the script and re-run. -->
"""

_TITLE_AND_STATUS = """
# Phase 1 sub-B1 — Singapore frequency analysis

**Status: provisional. Floor strategies and per-field cuts derived from one region may shift when
Sweden is added. B2 should not freeze vocabulary on these numbers alone.**
"""

_METHODOLOGY = """
## 1. Methodology

This report characterises categorical-field distributions across nine fields of five Overture themes
for region {region_name} at release {overture_release}. The goal is to make Phase 1 vocabulary
decisions reviewable; B1 ships analysis only, no vocab YAML — that is B2's responsibility.

**Coverage definitions.** For scalar fields, `coverage` = fraction of rows whose value is non-null.
For `places.categories.alternate` (a list-of-strings column), `coverage` = fraction of rows whose
list is non-null *and* non-empty. The list-length distribution (in §3.8) uses the full theme row
count as the denominator, not the count of non-empty rows.

**Counting semantics.** Scalar fields contribute one count per non-null row.
For `categories.alternate`, each list element contributes one count, so `total_occurrences`
exceeds `n_present` and the
`% coverage retained` column uses `total_occurrences` as the denominator. The two denominators
differ; do not compare retention percentages across primary vs alternate naively.

**Floor strategies.** Effective floor per field is `max(percentage x N_present, hard_min)`. Five
named strategies span a ~100x pressure range (see §3 tables). Plots show horizontal threshold lines
at each strategy's effective floor for that field.

**Plot interpretation.** Log-log rank-frequency: x = rank (1 = most common), y = count. The shape
of the curve relative to threshold lines shows where natural cuts fall.

**Data source.** Overture release {overture_release}, fetched via sub-project A's loader. Cache
manifest: `{manifest_path}`.
"""

_IMPLICATIONS = """
## 4. Implications for B2 (provisional)

- **Missing-value handling.** B1 does not decide. Three options for B2:
  - emit `<unknown>` token for missing-class rows;
  - drop missing-class features from training entirely;
  - infer class from context (geometry, neighbouring features).
- **Per-field floor selection.** B2 picks one strategy per field from §3's lattice (or computes
  a hybrid). Strategies flagged as PRD-framing-only should not be picked unless intentional.
- **`places.categories.alternate` max-alternates cap.** Decision input is the list-length
  distribution in §3.8. B1 does not propose a cap.
- **All B2 decisions are provisional pending Sweden.** Re-run B1 against Sweden when sub-A's
  cold-fetch issue is resolved and a Swedish cache exists.
"""


def render_field_section(
    *,
    result: FieldFrequencyResult,
    cut_rows: list[CutBehaviorRow],
    plot_relative_path: str,
    list_length: ListLengthDistribution | None = None,
    binds_to_prd_framing_only: bool = False,
    section_number: str = "",
) -> str:
    """Render one ### 3.x field section. Pure string; no I/O."""
    header = f"### {section_number} {result.field}\n" if section_number else f"### {result.field}\n"
    lines: list[str] = [header]

    if binds_to_prd_framing_only:
        lines.append(
            "\n*Very strict floor binds to fewer than 4 categories on Singapore data; "
            "this row is shown for PRD §5 framing only and should not be selected as a "
            "Phase 1 cut.*\n"
        )

    coverage_pct = (100.0 * result.n_present / result.n_total) if result.n_total > 0 else 0.0
    lines.append(
        f"\n**Coverage:** {coverage_pct:.1f}%"
        f" ({result.n_present} / {result.n_total} rows present)\n"
    )

    if list_length is not None:
        lines.append("\n**List-length distribution** (denominator = all rows of theme):\n\n")
        lines.append("| List length | Count | % of all rows |\n")
        lines.append("|---|---:|---:|\n")
        for key in _BUCKET_KEYS:
            count, pct = list_length.buckets[key]
            lines.append(f"| {key} | {count:,} | {pct:.2f}% |\n")
        lines.append("\n")

    lines.append(f"\n![rank-frequency]({plot_relative_path})\n\n")

    lines.append(
        "| Strategy | Effective floor | Total categories | Kept | Dropped | % coverage retained |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for row in cut_rows:
        lines.append(
            f"| {row.strategy.name} | {row.effective_floor:,} | {row.n_total_categories:,} | "
            f"{row.n_kept:,} | {row.n_dropped:,} | {row.coverage_retained_pct:.2f}% |\n"
        )

    return "".join(lines)


def render_report(
    *,
    region_name: str,
    overture_release: str,
    manifest_path: Path,
    per_theme_sha256: dict[str, str],
    field_sections: list[str],
    coverage_summary_rows: list[tuple[str, int, int, float]],
    commit_sha: str,
    run_timestamp_utc: datetime,
    rerun_reason: str,
) -> str:
    """Assemble the full markdown report. Pure string; no I/O."""
    run_iso = run_timestamp_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    parts: list[str] = []

    parts.append(
        _HEADER_TEMPLATE.format(
            commit_sha=commit_sha,
            overture_release=overture_release,
            run_iso=run_iso,
            rerun_reason=rerun_reason,
        )
    )
    parts.append(_TITLE_AND_STATUS)
    parts.append(
        _METHODOLOGY.format(
            region_name=region_name,
            overture_release=overture_release,
            manifest_path=str(manifest_path),
        )
    )

    # § 2 — coverage summary
    parts.append("\n## 2. Coverage summary\n\n")
    parts.append("| Field | N_total | N_present | Coverage |\n|---|---:|---:|---:|\n")
    for field, n_total, n_present, coverage_pct in coverage_summary_rows:
        parts.append(f"| {field} | {n_total:,} | {n_present:,} | {coverage_pct:.2f}% |\n")

    # § 3 — per-field analyses
    parts.append("\n## 3. Field analyses\n\n")
    for section in field_sections:
        parts.append(section)
        parts.append("\n")

    parts.append(_IMPLICATIONS)

    # § 5 — reproducibility
    parts.append("\n## 5. Reproducibility\n\n")
    parts.append(f"- Overture release: `{overture_release}`\n")
    parts.append(f"- Cache manifest: `{manifest_path}`\n")
    parts.append("- Per-theme cache sha256s:\n")
    for theme, sha in sorted(per_theme_sha256.items()):
        parts.append(f"  - {theme}: `{sha}`\n")
    parts.append(f"- Code commit: `{commit_sha}`\n")
    parts.append(f"- Run timestamp (UTC): `{run_iso}`\n")
    parts.append(f"- Re-run reason: `{rerun_reason}`\n")
    parts.append("- Generated by: `scripts/analyse_singapore_frequencies.py`\n")

    return "".join(parts)

from __future__ import annotations

import pyarrow as pa

from cfm.data.overture.errors import OvertureSchemaMismatch

# The five themes we load in Phase 1.
EXPECTED_THEMES: tuple[str, ...] = (
    "buildings",
    "places",
    "transportation",
    "base",
    "divisions",
)

# Curated column schemas — the minimum set of columns we use per theme.
# Real Overture parquet may have many more columns; we tolerate extras.
# If Overture removes any of these columns, validate_schema raises.
#
# Geometry is stored as WKB (Well-Known Binary) bytes in Overture parquet.
# `class` and `subtype` are simple string categories.
# `categories` (places) is a struct with main + alternate fields; for Phase 1
#   we just require the column to exist — B1 will inspect its structure.

THEME_SCHEMAS: dict[str, pa.Schema] = {
    "buildings": pa.schema(
        [
            ("id", pa.string()),
            ("geometry", pa.binary()),
            ("class", pa.string()),
            ("height", pa.float64()),
            ("num_floors", pa.int32()),
        ]
    ),
    "places": pa.schema(
        [
            ("id", pa.string()),
            ("geometry", pa.binary()),
            ("categories", pa.string()),
        ]
    ),
    "transportation": pa.schema(
        [
            ("id", pa.string()),
            ("geometry", pa.binary()),
            ("class", pa.string()),
            ("subtype", pa.string()),
        ]
    ),
    "base": pa.schema(
        [
            ("id", pa.string()),
            ("geometry", pa.binary()),
            ("subtype", pa.string()),
        ]
    ),
    "divisions": pa.schema(
        [
            ("id", pa.string()),
            ("geometry", pa.binary()),
            ("country", pa.string()),
            ("subtype", pa.string()),
        ]
    ),
}


def validate_schema(table: pa.Table, *, theme: str) -> None:
    """Raise OvertureSchemaMismatch if `table` is missing any required column.

    Extras are tolerated. Column dtypes are not strictly checked beyond presence
    (real Overture parquet's nested dtypes are too varied to lock in Phase 1).
    """
    if theme not in THEME_SCHEMAS:
        raise OvertureSchemaMismatch(f"unknown theme {theme!r}; expected one of {EXPECTED_THEMES}")
    expected_columns = set(THEME_SCHEMAS[theme].names)
    actual_columns = set(table.column_names)
    missing = expected_columns - actual_columns
    if missing:
        raise OvertureSchemaMismatch(
            f"theme={theme!r} missing required columns: {sorted(missing)}; "
            f"has {sorted(actual_columns)}"
        )

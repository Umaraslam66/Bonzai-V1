"""Boundary-contract parquet writer.

Emits exactly 144 rows per tile: 112 internal_edge + 32 external_edge, sorted
by (slot_kind, slot_index). Schema is pinned via ``pa.schema(...)`` so
PyArrow type inference cannot drift (mirrors sub-D's _MACRO_CORE_SCHEMA
pattern at ``src/cfm/data/sub_d/io.py:41``). The neutral
``cfm.data.io.write_parquet`` helper is reused for byte-deterministic
serialisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Final

import pyarrow as pa

from cfm.data.io import write_parquet


class SlotKind(IntEnum):
    INTERNAL_EDGE = 1
    EXTERNAL_EDGE = 2


EXPECTED_INTERNAL_ROWS: Final[int] = 112
EXPECTED_EXTERNAL_ROWS: Final[int] = 32
EXPECTED_TOTAL_ROWS: Final[int] = EXPECTED_INTERNAL_ROWS + EXPECTED_EXTERNAL_ROWS


# Pinned schema: explicit pa.schema with nullable flags. `boundary_class_enum`
# is the only nullable column (null = "BOUNDARY_NOT_APPLICABLE" / non-active
# rows); every other column is non-null. Pinning prevents PyArrow type
# inference from drifting between writes and keeps byte-determinism robust to
# input-shape variation.
_BOUNDARY_CONTRACT_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("slot_kind", pa.int8(), nullable=False),
        pa.field("slot_index", pa.int16(), nullable=False),
        pa.field("lower_cell_i", pa.int8(), nullable=False),
        pa.field("lower_cell_j", pa.int8(), nullable=False),
        pa.field("axis", pa.int8(), nullable=False),
        pa.field("scope_marker", pa.int8(), nullable=False),
        pa.field("boundary_class_enum", pa.int16(), nullable=True),
    ]
)


@dataclass(frozen=True)
class BoundaryContractRow:
    slot_kind: SlotKind
    slot_index: int
    lower_cell_i: int
    lower_cell_j: int
    axis: int
    scope_marker: int
    boundary_class_enum: int | None


def write_boundary_contract(
    out_path: Path,
    rows: list[BoundaryContractRow],
) -> Path:
    """Write rows to `out_path` as a canonically sorted parquet file.

    Raises ValueError if the row count is not 144 or the split is not
    112 internal + 32 external. The writer's own sort by
    ``(slot_kind, slot_index)`` is load-bearing — it must not assume the
    input list is already sorted (see ``test_write_sorts_unordered_input``).
    """
    if len(rows) != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            f"expected {EXPECTED_TOTAL_ROWS} rows (112 internal + 32 external), got {len(rows)}"
        )
    n_internal = sum(1 for r in rows if r.slot_kind is SlotKind.INTERNAL_EDGE)
    n_external = sum(1 for r in rows if r.slot_kind is SlotKind.EXTERNAL_EDGE)
    if (n_internal, n_external) != (EXPECTED_INTERNAL_ROWS, EXPECTED_EXTERNAL_ROWS):
        raise ValueError(f"row split must be (112, 32), got ({n_internal}, {n_external})")

    sorted_rows = sorted(rows, key=lambda r: (int(r.slot_kind), r.slot_index))

    columns = {
        "slot_kind": [int(r.slot_kind) for r in sorted_rows],
        "slot_index": [r.slot_index for r in sorted_rows],
        "lower_cell_i": [r.lower_cell_i for r in sorted_rows],
        "lower_cell_j": [r.lower_cell_j for r in sorted_rows],
        "axis": [r.axis for r in sorted_rows],
        "scope_marker": [r.scope_marker for r in sorted_rows],
        "boundary_class_enum": [r.boundary_class_enum for r in sorted_rows],
    }
    table = pa.Table.from_pydict(columns, schema=_BOUNDARY_CONTRACT_SCHEMA)
    write_parquet(table, out_path)
    return out_path

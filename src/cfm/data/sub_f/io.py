"""Sub-F per-tile cells.parquet schema + writer.

Pinned `pa.schema` with explicit nullable flags per sub-E precedent
(`src/cfm/data/sub_e/writer.py` `_BOUNDARY_CONTRACT_SCHEMA`). Routes through
`cfm.data.io.write_parquet` for byte-deterministic output via
PARQUET_WRITE_KWARGS.

Spec references:
- sec 4.2 column types + nullability
- sec 4.3 row ordering: sorted by (cell_i, cell_j) row-major
- sec 4.7 inline-validator invariants: 64 rows, no duplicates, slot_index
  derivation check
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pyarrow as pa

from cfm.data.io import write_parquet

EXPECTED_ROWS_PER_TILE: Final[int] = 64  # 8x8 cell grid per sub-D lattice

CELLS_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("cell_i", pa.int8(), nullable=False),
        pa.field("cell_j", pa.int8(), nullable=False),
        pa.field("cell_slot_index", pa.int8(), nullable=False),
        pa.field("token_sequence", pa.list_(pa.int16()), nullable=False),
        pa.field("feature_count", pa.int16(), nullable=False),
        pa.field("provenance_sha256", pa.string(), nullable=False),
    ]
)


@dataclass(frozen=True)
class CellRow:
    """One row of cells.parquet."""

    cell_i: int
    cell_j: int
    cell_slot_index: int
    token_sequence: list[int]
    feature_count: int
    provenance_sha256: str


def write_cells_parquet(out_path: Path, rows: list[CellRow]) -> Path:
    """Write rows to cells.parquet, sorted by (cell_i, cell_j) row-major.

    Inline invariants (raise ValueError if violated):
      - len(rows) == 64
      - No duplicate (cell_i, cell_j)
      - cell_slot_index == cell_i * 8 + cell_j for every row
      - provenance_sha256 is 64-char lowercase hex (deferred to inline validator
        in Task 9 -- kept loose here so tests can use synthetic 'a'*64).

    Output path: caller chooses (typically
    data/processed/sub_f/<release>/<region>/tile=.../cells.parquet).
    """
    if len(rows) != EXPECTED_ROWS_PER_TILE:
        raise ValueError(f"expected {EXPECTED_ROWS_PER_TILE} rows, got {len(rows)}")

    seen: set[tuple[int, int]] = set()
    for r in rows:
        key = (r.cell_i, r.cell_j)
        if key in seen:
            raise ValueError(f"duplicate cell {key}")
        seen.add(key)
        expected_idx = r.cell_i * 8 + r.cell_j
        if r.cell_slot_index != expected_idx:
            raise ValueError(
                f"cell_slot_index {r.cell_slot_index} != cell_i*8+cell_j "
                f"= {expected_idx} for cell ({r.cell_i}, {r.cell_j})"
            )

    sorted_rows = sorted(rows, key=lambda r: (r.cell_i, r.cell_j))
    table = pa.Table.from_pydict(
        {
            "cell_i": [r.cell_i for r in sorted_rows],
            "cell_j": [r.cell_j for r in sorted_rows],
            "cell_slot_index": [r.cell_slot_index for r in sorted_rows],
            "token_sequence": [r.token_sequence for r in sorted_rows],
            "feature_count": [r.feature_count for r in sorted_rows],
            "provenance_sha256": [r.provenance_sha256 for r in sorted_rows],
        },
        schema=CELLS_SCHEMA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(table, out_path)
    return out_path

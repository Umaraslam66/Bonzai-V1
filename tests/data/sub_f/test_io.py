"""Sub-F cells.parquet schema + writer tests."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cfm.data.sub_f.io import (
    CELLS_SCHEMA,
    EXPECTED_ROWS_PER_TILE,
    CellRow,
    write_cells_parquet,
)


def _make_64_rows() -> list[CellRow]:
    """Construct 64 well-formed rows for an 8x8 cell grid in row-major order."""
    return [
        CellRow(
            cell_i=i,
            cell_j=j,
            cell_slot_index=i * 8 + j,
            token_sequence=[],
            feature_count=0,
            provenance_sha256="a" * 64,
        )
        for i in range(8)
        for j in range(8)
    ]


def test_cells_schema_field_types_pinned():
    """Per spec sec 4.2: int8 / int16 / list<int16> / string with explicit nullable=False."""
    fields = {f.name: f for f in CELLS_SCHEMA}
    assert fields["cell_i"].type == pa.int8()
    assert fields["cell_j"].type == pa.int8()
    assert fields["cell_slot_index"].type == pa.int8()
    assert fields["token_sequence"].type == pa.list_(pa.int16())
    assert fields["feature_count"].type == pa.int16()
    assert fields["provenance_sha256"].type == pa.string()
    for name, f in fields.items():
        assert f.nullable is False, f"{name} must be nullable=False per pinned schema"


def test_write_cells_parquet_round_trips(tmp_path: Path):
    """64 rows written -> read back with identical column types + values."""
    rows = _make_64_rows()
    path = tmp_path / "cells.parquet"
    write_cells_parquet(path, rows)

    table = pq.ParquetFile(path).read()
    assert table.num_rows == EXPECTED_ROWS_PER_TILE
    assert table.schema == CELLS_SCHEMA, "parquet schema must match pinned schema bit-for-bit"

    cell_i_col = table.column("cell_i").to_pylist()
    cell_j_col = table.column("cell_j").to_pylist()
    # Row-major sort: (cell_i, cell_j) ascending.
    expected_pairs = [(i, j) for i in range(8) for j in range(8)]
    assert list(zip(cell_i_col, cell_j_col, strict=True)) == expected_pairs


def test_write_cells_parquet_rejects_wrong_row_count(tmp_path: Path):
    """write_cells_parquet must error on != EXPECTED_ROWS_PER_TILE."""
    rows = _make_64_rows()[:63]
    with pytest.raises(ValueError, match=r"expected 64"):
        write_cells_parquet(tmp_path / "cells.parquet", rows)


def test_write_cells_parquet_rejects_duplicate_cell(tmp_path: Path):
    """Two rows for the same (cell_i, cell_j) must error -- invariant per spec sec 4.7."""
    rows = _make_64_rows()
    rows[1] = CellRow(  # duplicate (0,0)
        cell_i=0,
        cell_j=0,
        cell_slot_index=0,
        token_sequence=[],
        feature_count=0,
        provenance_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match=r"duplicate cell"):
        write_cells_parquet(tmp_path / "cells.parquet", rows)


def test_write_cells_parquet_rejects_slot_index_mismatch(tmp_path: Path):
    """cell_slot_index must equal cell_i * 8 + cell_j per spec sec 4.7."""
    rows = _make_64_rows()
    bad = CellRow(
        cell_i=2,
        cell_j=3,
        cell_slot_index=99,  # wrong: should be 19
        token_sequence=[],
        feature_count=0,
        provenance_sha256="c" * 64,
    )
    rows[2 * 8 + 3] = bad
    with pytest.raises(ValueError, match=r"cell_slot_index"):
        write_cells_parquet(tmp_path / "cells.parquet", rows)

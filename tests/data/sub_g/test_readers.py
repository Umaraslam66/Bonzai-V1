from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from cfm.data.io import write_parquet
from cfm.data.sub_g.readers import read_sub_e_contract_rows

# Mirror sub_e/writer.py:38-48 exactly.
_SCHEMA = pa.schema(
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


def test_reads_raw_rows_and_marks_active_emission(tmp_path: Path):
    cols = {
        "slot_kind": [1, 1, 2],
        "slot_index": [0, 1, 2],
        "lower_cell_i": [0, 0, 7],
        "lower_cell_j": [0, 1, 7],
        "axis": [0, 1, 0],
        "scope_marker": [0, 0, 1],  # 2 active, 1 non-active
        "boundary_class_enum": [2, 1, None],  # MAJOR, NONE, NULL
    }
    p = tmp_path / "boundary_contract.parquet"
    write_parquet(pa.Table.from_pydict(cols, schema=_SCHEMA), p)
    rows = read_sub_e_contract_rows(p)
    assert len(rows) == 3
    # active-emission = scope_marker==0 AND boundary_class_enum in {2,3}
    emitting = [r for r in rows if r.is_emitting()]
    assert len(emitting) == 1
    assert emitting[0].boundary_class_enum == 2  # MAJOR_ROAD
    assert emitting[0].class_label() == "MAJOR_ROAD"
    # the NONE (enum=1) active row is not emitting
    none_row = next(r for r in rows if r.boundary_class_enum == 1)
    assert not none_row.is_emitting()
    assert none_row.class_label() is None

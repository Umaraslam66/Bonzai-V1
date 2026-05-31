"""Sub-G's INDEPENDENT upstream readers.

Deliberately does NOT import ``sub_f.boundary_contract.load_boundary_contract``:
that reader applies the same join+class interpretation the encoder uses, so
reusing it would make seam 2 circular (design rule 2 / Decision 3b). Here we
read the raw 7-column sub-E parquet and expose the active-emission predicate
directly.

BoundaryClass enum (sub_e/derivation.py:19-23): NONE=1, MAJOR_ROAD=2,
MINOR_ROAD=3. Active-emission (could yield a bref) = scope_marker==0 AND
boundary_class_enum in {2,3}.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

_EMITTING_ENUMS = frozenset({2, 3})  # MAJOR_ROAD, MINOR_ROAD
_CLASS_LABEL = {2: "MAJOR_ROAD", 3: "MINOR_ROAD"}


@dataclass(frozen=True)
class SubEContractRow:
    slot_kind: int
    slot_index: int
    lower_cell_i: int
    lower_cell_j: int
    axis: int
    scope_marker: int
    boundary_class_enum: int | None

    def is_emitting(self) -> bool:
        return self.scope_marker == 0 and self.boundary_class_enum in _EMITTING_ENUMS

    def class_label(self) -> str | None:
        if self.boundary_class_enum in _EMITTING_ENUMS:
            return _CLASS_LABEL[self.boundary_class_enum]
        return None


def read_sub_e_contract_rows(path: Path) -> list[SubEContractRow]:
    """Read all 144 raw rows (no class interpretation beyond enum mapping).

    Uses ``pq.ParquetFile(path).read()`` per
    ``feedback_pyarrow_hive_partition_inference``.
    """
    tbl = pq.ParquetFile(path).read()
    cols = {n: tbl.column(n).to_pylist() for n in tbl.column_names}
    return [
        SubEContractRow(
            slot_kind=int(cols["slot_kind"][i]),
            slot_index=int(cols["slot_index"][i]),
            lower_cell_i=int(cols["lower_cell_i"][i]),
            lower_cell_j=int(cols["lower_cell_j"][i]),
            axis=int(cols["axis"][i]),
            scope_marker=int(cols["scope_marker"][i]),
            boundary_class_enum=(
                None
                if cols["boundary_class_enum"][i] is None
                else int(cols["boundary_class_enum"][i])
            ),
        )
        for i in range(tbl.num_rows)
    ]


def read_sub_f_cells(path: Path) -> dict[tuple[int, int], list[int]]:
    """(cell_i, cell_j) -> token_sequence (list[int]) from sub-F cells.parquet."""
    tbl = pq.ParquetFile(path).read()
    cols = {n: tbl.column(n).to_pylist() for n in tbl.column_names}
    return {
        (int(cols["cell_i"][i]), int(cols["cell_j"][i])): [
            int(t) for t in cols["token_sequence"][i]
        ]
        for i in range(tbl.num_rows)
    }


def read_sub_c_features_by_cell(path: Path) -> dict[tuple[int, int], list[dict]]:
    """(cell_i, cell_j) -> list of feature dicts in sub-C ROW ORDER.

    Row order is preserved because encode_tile preserves it
    (pipeline_writer.py:86). Each dict: {feature_class:int,
    source_feature_id:str, geometry:bytes(WKB), class_raw:str|None}.
    """
    tbl = pq.ParquetFile(path).read()
    cols = {n: tbl.column(n).to_pylist() for n in tbl.column_names}
    out: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for i in range(tbl.num_rows):
        key = (int(cols["cell_i"][i]), int(cols["cell_j"][i]))
        out[key].append(
            {
                "feature_class": int(cols["feature_class"][i]),
                "source_feature_id": cols["source_feature_id"][i],
                "geometry": cols["geometry"][i],
                "class_raw": cols["class_raw"][i],
            }
        )
    return dict(out)

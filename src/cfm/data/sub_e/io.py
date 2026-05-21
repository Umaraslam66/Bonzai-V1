"""Sub-E input readers.

Reads sub-C `crossings.parquet` + `features.parquet` and sub-D
`macro_core.parquet`. All reads use `pyarrow.parquet.ParquetFile(path).read()`
to avoid Hive partition inference on tile=... directories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class SubCCrossingRow:
    lower_cell_i: int
    lower_cell_j: int
    axis: int
    source_feature_id: str


@dataclass(frozen=True)
class SubCFeatureRow:
    source_feature_id: str
    # int8 enum per sub-C contract (cfm.data.sub_c.io.py:178 +
    # sub_c/enums.py:22 FEATURE_CLASS: {0: "road", 1: "building", 2: "poi",
    # 3: "base"}). Earlier draft typed this as `str` and the pipeline
    # filtered `feature_class == "road"` which silently never matched real
    # int8 data — feature filter became a no-op, every active edge with
    # crossings classified as MINOR_ROAD via default bucket, MAJOR_ROAD
    # never appeared. Caught by Task 14's writer-regression-guard test.
    # See memory feedback_external_source_of_truth_gate.md for the
    # sixth-gate discipline this trap mandated.
    feature_class: int
    class_raw: str | None


@dataclass(frozen=True)
class SubDMacroCoreRow:
    slot_kind: int
    slot_index: int
    cell_i: int | None
    cell_j: int | None
    lower_cell_i: int | None
    lower_cell_j: int | None
    axis: int | None
    scope: int
    zoning_class: int | None
    cell_density_bucket: int | None
    road_skeleton_class: int | None


def _read_table(path: Path) -> pa.Table:
    return pq.ParquetFile(path).read()


def read_sub_c_crossings(path: Path) -> list[SubCCrossingRow]:
    tbl = _read_table(path)
    cols = {name: tbl.column(name).to_pylist() for name in tbl.column_names}
    return [
        SubCCrossingRow(
            lower_cell_i=cols["lower_cell_i"][i],
            lower_cell_j=cols["lower_cell_j"][i],
            axis=cols["axis"][i],
            source_feature_id=cols["source_feature_id"][i],
        )
        for i in range(tbl.num_rows)
    ]


def read_sub_c_features(path: Path) -> list[SubCFeatureRow]:
    tbl = _read_table(path)
    cols = {name: tbl.column(name).to_pylist() for name in tbl.column_names}
    return [
        SubCFeatureRow(
            source_feature_id=cols["source_feature_id"][i],
            feature_class=cols["feature_class"][i],
            class_raw=cols["class_raw"][i],
        )
        for i in range(tbl.num_rows)
    ]


def read_sub_d_macro_core(path: Path) -> list[SubDMacroCoreRow]:
    tbl = _read_table(path)
    cols = {name: tbl.column(name).to_pylist() for name in tbl.column_names}
    return [
        SubDMacroCoreRow(
            slot_kind=cols["slot_kind"][i],
            slot_index=cols["slot_index"][i],
            cell_i=cols["cell_i"][i],
            cell_j=cols["cell_j"][i],
            lower_cell_i=cols["lower_cell_i"][i],
            lower_cell_j=cols["lower_cell_j"][i],
            axis=cols["axis"][i],
            scope=cols["scope"][i],
            zoning_class=cols["zoning_class"][i],
            cell_density_bucket=cols["cell_density_bucket"][i],
            road_skeleton_class=cols["road_skeleton_class"][i],
        )
        for i in range(tbl.num_rows)
    ]


def require_sub_d_success_marker(sub_d_region_dir: Path) -> None:
    """Gate sub-E on sub-D's `_SUCCESS` marker.

    Raises FileNotFoundError if the marker is absent. Sub-E does not start
    derivation against a sub-D region whose validator has not closed.
    """
    marker = sub_d_region_dir / "_SUCCESS"
    if not marker.exists():
        raise FileNotFoundError(
            f"sub-D _SUCCESS marker missing at {marker}; sub-E refuses to start"
        )

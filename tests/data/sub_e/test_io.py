from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cfm.data.sub_e.io import (
    SubCCrossingRow,
    read_sub_c_crossings,
    read_sub_c_features,
    read_sub_d_macro_core,
    require_sub_d_success_marker,
)


def _write_synthetic_crossings(path: Path) -> None:
    table = pa.table(
        {
            "lower_cell_i": pa.array([0, 1], type=pa.int8()),
            "lower_cell_j": pa.array([0, 2], type=pa.int8()),
            "axis": pa.array([0, 1], type=pa.int8()),
            "source_feature_id": pa.array(["F1", "F2"], type=pa.string()),
        }
    )
    pq.write_table(table, path)


def _write_synthetic_features(path: Path) -> None:
    table = pa.table(
        {
            "source_feature_id": pa.array(["F1", "F2"], type=pa.string()),
            "feature_class": pa.array(["road", "road"], type=pa.string()),
            "class_raw": pa.array(["primary", "residential"], type=pa.string()),
        }
    )
    pq.write_table(table, path)


def _write_synthetic_macro_core(path: Path) -> None:
    table = pa.table(
        {
            "slot_kind": pa.array([0, 1], type=pa.int8()),
            "slot_index": pa.array([0, 0], type=pa.int16()),
            "cell_i": pa.array([0, None], type=pa.int8()),
            "cell_j": pa.array([0, None], type=pa.int8()),
            "lower_cell_i": pa.array([None, 0], type=pa.int8()),
            "lower_cell_j": pa.array([None, 0], type=pa.int8()),
            "axis": pa.array([None, 0], type=pa.int8()),
            "scope": pa.array([0, 0], type=pa.int8()),
            "zoning_class": pa.array([1, None], type=pa.int16()),
            "cell_density_bucket": pa.array([2, None], type=pa.int16()),
            "road_skeleton_class": pa.array([None, 1], type=pa.int16()),
        }
    )
    pq.write_table(table, path)


def test_read_sub_c_crossings_returns_typed_rows(tmp_path: Path) -> None:
    p = tmp_path / "crossings.parquet"
    _write_synthetic_crossings(p)
    rows = read_sub_c_crossings(p)
    assert len(rows) == 2
    assert isinstance(rows[0], SubCCrossingRow)
    assert rows[0].lower_cell_i == 0
    assert rows[0].lower_cell_j == 0
    assert rows[0].axis == 0
    assert rows[0].source_feature_id == "F1"


def test_read_sub_c_features_returns_typed_rows(tmp_path: Path) -> None:
    p = tmp_path / "features.parquet"
    _write_synthetic_features(p)
    rows = read_sub_c_features(p)
    by_id = {r.source_feature_id: r for r in rows}
    assert by_id["F1"].class_raw == "primary"
    assert by_id["F2"].class_raw == "residential"


def test_read_sub_d_macro_core_returns_typed_rows(tmp_path: Path) -> None:
    p = tmp_path / "macro_core.parquet"
    _write_synthetic_macro_core(p)
    rows = read_sub_d_macro_core(p)
    assert len(rows) == 2
    cell_rows = [r for r in rows if r.slot_kind == 0]
    edge_rows = [r for r in rows if r.slot_kind == 1]
    assert len(cell_rows) == 1
    assert len(edge_rows) == 1
    assert cell_rows[0].zoning_class == 1
    assert edge_rows[0].axis == 0


def test_require_sub_d_success_marker_passes_when_present(tmp_path: Path) -> None:
    region = tmp_path / "sub_d_region"
    region.mkdir()
    (region / "_SUCCESS").touch()
    require_sub_d_success_marker(region)  # should not raise


def test_require_sub_d_success_marker_raises_when_absent(tmp_path: Path) -> None:
    region = tmp_path / "sub_d_region"
    region.mkdir()
    with pytest.raises(FileNotFoundError, match="_SUCCESS"):
        require_sub_d_success_marker(region)

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
    # feature_class is int8 per sub-C contract (sub_c/io.py:44 +
    # sub_c/enums.py:22 FEATURE_CLASS: {0: "road", ...}). Encode "road"
    # symbolically via sub-C's encode_enum so this fixture stays tied
    # to sub-C's enum source rather than the magic number 0.
    from cfm.data.sub_c.enums import FEATURE_CLASS, encode_enum

    road_code = encode_enum(FEATURE_CLASS, "road")
    table = pa.table(
        {
            "source_feature_id": pa.array(["F1", "F2"], type=pa.string()),
            "feature_class": pa.array([road_code, road_code], type=pa.int8()),
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


def test_sub_c_features_schema_field_types_match_sub_e_reader_dataclass() -> None:
    """Sixth-gate cross-reference: sub-E's SubCFeatureRow field types match
    sub-C's actual parquet schema as ground truth.

    Hand-enumerated from sub-C's source (NOT derived from sub-E's reader):
    - source_feature_id: pa.string() — sub_c/io.py:42 schema field
    - feature_class: pa.int8() — sub_c/io.py:44 schema field
    - class_raw: pa.string() (nullable) — sub_c/io.py:52 schema field

    Sub-E's SubCFeatureRow dataclass (sub_e/io.py:26-37) must agree with
    these types. The earlier feature_class: str declaration silently made
    sub-E's `f.feature_class == "road"` filter a no-op against real int8
    data — caught by Task 14 writer-regression-guard. Memory entry
    feedback_external_source_of_truth_gate.md mandates this kind of
    cross-reference test for every new abstraction over an existing
    module's contract. Hand-enumerate expected types from upstream
    source; do not derive from the new abstraction.
    """
    import dataclasses

    from cfm.data.sub_e.io import SubCFeatureRow

    # Hand-enumerated per sub-C io.py:42-52.
    expected_field_types: dict[str, type] = {
        "source_feature_id": str,
        "feature_class": int,
        "class_raw": str,  # str | None on the dataclass; type-erasure here checks the str half
    }

    actual_fields = {f.name: f.type for f in dataclasses.fields(SubCFeatureRow)}
    assert set(actual_fields) == set(expected_field_types), (
        f"sub-E SubCFeatureRow fields {set(actual_fields)} disagree with "
        f"hand-enumerated sub-C contract fields {set(expected_field_types)}"
    )
    # Coarse type check (dataclass field.type is the annotation string in
    # __future__.annotations mode, so we check by stringification).
    actual_field_type_strs = {n: str(t) for n, t in actual_fields.items()}
    # feature_class must be `int`, NOT `str`. This is the canonical guard.
    assert "int" in actual_field_type_strs["feature_class"], (
        f"sub-E SubCFeatureRow.feature_class annotation is "
        f"{actual_field_type_strs['feature_class']!r}; sub-C schema requires int "
        f"(int8 enum per sub_c/io.py:44 + sub_c/enums.py:22 FEATURE_CLASS)."
    )
    # source_feature_id and class_raw should be string-shaped.
    assert "str" in actual_field_type_strs["source_feature_id"]
    assert "str" in actual_field_type_strs["class_raw"]

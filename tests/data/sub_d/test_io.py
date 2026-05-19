"""Tests for sub-D macro_core + derivation_evidence parquet writers (Task 9).

Schemas pinned to spec §11.2 (macro_core) and §11.3 (derivation_evidence).
Sort keys: macro_core by ``(slot_kind, slot_index)``;
derivation_evidence by ``(slot_kind, slot_index, metric_namespace, metric_name)``.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cfm.data.sub_d.enums import (
    FeatureClass,
    MetricNamespace,
    Scope,
    SlotKind,
)
from cfm.data.sub_d.io import (
    DerivationEvidenceRow,
    MacroCoreRow,
    read_derivation_evidence_parquet,
    read_macro_core_parquet,
    write_derivation_evidence_parquet,
    write_macro_core_parquet,
)


# ---------------------------------------------------------------------------
# Schema tests (spec §11.2, §11.3)
# ---------------------------------------------------------------------------


def test_macro_core_schema_matches_spec_11_2(tmp_path: Path):
    """The 11-field schema is pinned via pa.schema(); types match spec §11.2."""
    row = MacroCoreRow(
        slot_kind=SlotKind.CELL,
        slot_index=0,
        cell_i=0,
        cell_j=0,
        lower_cell_i=None,
        lower_cell_j=None,
        axis=None,
        scope=Scope.ACTIVE,
        zoning_class=0,
        cell_density_bucket=0,
        road_skeleton_class=None,
    )
    path = tmp_path / "macro_core.parquet"
    write_macro_core_parquet([row], path)
    table = pq.ParquetFile(path).read()

    expected_fields = {
        "slot_kind": pa.int8(),
        "slot_index": pa.int16(),
        "cell_i": pa.int8(),
        "cell_j": pa.int8(),
        "lower_cell_i": pa.int8(),
        "lower_cell_j": pa.int8(),
        "axis": pa.int8(),
        "scope": pa.int8(),
        "zoning_class": pa.int16(),
        "cell_density_bucket": pa.int16(),
        "road_skeleton_class": pa.int16(),
    }
    assert set(table.schema.names) == set(expected_fields.keys())
    for name, expected_type in expected_fields.items():
        field = table.schema.field(name)
        assert field.type == expected_type, (
            f"{name}: expected {expected_type}, got {field.type}"
        )


def test_derivation_evidence_schema_matches_spec_11_3(tmp_path: Path):
    """The 10-field schema is pinned via pa.schema(); types match spec §11.3."""
    row = DerivationEvidenceRow(
        slot_kind=SlotKind.CELL,
        slot_index=0,
        metric_namespace=MetricNamespace.ZONING,
        metric_name="feature_count_road",
        value=5,
        derivation_version="1.0",
    )
    path = tmp_path / "derivation_evidence.parquet"
    write_derivation_evidence_parquet([row], path)
    table = pq.ParquetFile(path).read()

    expected_fields = {
        "slot_kind": pa.int8(),
        "slot_index": pa.int16(),
        "metric_namespace": pa.int8(),
        "metric_name": pa.string(),
        "value_type": pa.int8(),
        "value_float": pa.float64(),
        "value_int": pa.int64(),
        "value_string": pa.string(),
        "value_bool": pa.bool_(),
        "derivation_version": pa.string(),
    }
    assert set(table.schema.names) == set(expected_fields.keys())
    for name, expected_type in expected_fields.items():
        field = table.schema.field(name)
        assert field.type == expected_type, (
            f"{name}: expected {expected_type}, got {field.type}"
        )


# ---------------------------------------------------------------------------
# Sort-order tests (canonical keys)
# ---------------------------------------------------------------------------


def test_macro_core_writer_sorts_by_slot_kind_slot_index(tmp_path: Path):
    # Submit rows in scrambled order across all three slot_kinds.
    rows = [
        MacroCoreRow(SlotKind.EXTERNAL_EDGE, 5, None, None, -1, 5, 0, Scope.FULLY_MASKED, None, None, None),
        MacroCoreRow(SlotKind.CELL, 63, 7, 7, None, None, None, Scope.ACTIVE, 0, 1, None),
        MacroCoreRow(SlotKind.INTERNAL_EDGE, 0, None, None, 0, 0, 0, Scope.ACTIVE, None, None, 0),
        MacroCoreRow(SlotKind.CELL, 0, 0, 0, None, None, None, Scope.ACTIVE, 0, 0, None),
        MacroCoreRow(SlotKind.INTERNAL_EDGE, 111, None, None, 7, 6, 1, Scope.SCOPE_BOUNDARY, None, None, None),
    ]
    path = tmp_path / "macro_core.parquet"
    write_macro_core_parquet(rows, path)
    table = pq.ParquetFile(path).read()

    keys = list(zip(table["slot_kind"].to_pylist(), table["slot_index"].to_pylist()))
    expected = sorted([
        (int(SlotKind.EXTERNAL_EDGE), 5),
        (int(SlotKind.CELL), 63),
        (int(SlotKind.INTERNAL_EDGE), 0),
        (int(SlotKind.CELL), 0),
        (int(SlotKind.INTERNAL_EDGE), 111),
    ])
    assert keys == expected
    # macro_core never carries SlotKind.TILE rows (those live only in
    # derivation_evidence.parquet per spec §11.2).
    assert int(SlotKind.TILE) not in {k[0] for k in keys}


def test_derivation_evidence_writer_sorts_by_canonical_key(tmp_path: Path):
    # Scrambled across all four key fields.
    rows = [
        DerivationEvidenceRow(
            SlotKind.INTERNAL_EDGE, 10, MetricNamespace.ROAD_SKELETON, "road_crossing_count", 3, "1.0"
        ),
        DerivationEvidenceRow(
            SlotKind.CELL, 0, MetricNamespace.ZONING, "feature_count_building", 5, "1.0"
        ),
        DerivationEvidenceRow(
            SlotKind.CELL, 0, MetricNamespace.ZONING, "feature_count_road", 2, "1.0"
        ),
        DerivationEvidenceRow(
            SlotKind.CELL, 0, MetricNamespace.CELL_DENSITY, "building_footprint_ratio", 0.1, "1.0"
        ),
        DerivationEvidenceRow(
            SlotKind.TILE, 0, MetricNamespace.TILE_POPULATION_DENSITY, "p75_building_footprint_ratio", 0.05, "1.0"
        ),
    ]
    path = tmp_path / "derivation_evidence.parquet"
    write_derivation_evidence_parquet(rows, path)
    table = pq.ParquetFile(path).read()

    keys = list(zip(
        table["slot_kind"].to_pylist(),
        table["slot_index"].to_pylist(),
        table["metric_namespace"].to_pylist(),
        table["metric_name"].to_pylist(),
    ))
    expected = sorted([
        (int(SlotKind.INTERNAL_EDGE), 10, int(MetricNamespace.ROAD_SKELETON), "road_crossing_count"),
        (int(SlotKind.CELL), 0, int(MetricNamespace.ZONING), "feature_count_building"),
        (int(SlotKind.CELL), 0, int(MetricNamespace.ZONING), "feature_count_road"),
        (int(SlotKind.CELL), 0, int(MetricNamespace.CELL_DENSITY), "building_footprint_ratio"),
        (int(SlotKind.TILE), 0, int(MetricNamespace.TILE_POPULATION_DENSITY), "p75_building_footprint_ratio"),
    ])
    assert keys == expected


# ---------------------------------------------------------------------------
# Determinism (byte-identical for same input)
# ---------------------------------------------------------------------------


def test_writers_are_byte_identical_for_same_rows(tmp_path: Path):
    macro_rows = [
        MacroCoreRow(SlotKind.CELL, i, i // 8, i % 8, None, None, None, Scope.ACTIVE, 0, 0, None)
        for i in range(8)
    ]
    evid_rows = [
        DerivationEvidenceRow(
            SlotKind.CELL,
            i,
            MetricNamespace.ZONING,
            f"feature_count_{cls.name.lower()}",
            i + j,
            "1.0",
        )
        for i in range(4)
        for j, cls in enumerate(
            (FeatureClass.ROAD, FeatureClass.BUILDING, FeatureClass.POI, FeatureClass.BASE)
        )
    ]
    # Float + int + string + bool mix exercises value_type dispatch.
    evid_rows.append(
        DerivationEvidenceRow(
            SlotKind.CELL, 0, MetricNamespace.CELL_DENSITY, "building_footprint_ratio", 0.5, "1.0"
        )
    )
    evid_rows.append(
        DerivationEvidenceRow(
            SlotKind.TILE, 0, MetricNamespace.TILE_POPULATION_DENSITY, "as_string", "high", "1.0"
        )
    )
    evid_rows.append(
        DerivationEvidenceRow(
            SlotKind.CELL, 1, MetricNamespace.ZONING, "as_bool", True, "1.0"
        )
    )

    a = tmp_path / "run_a"
    b = tmp_path / "run_b"
    a.mkdir()
    b.mkdir()
    write_macro_core_parquet(macro_rows, a / "macro_core.parquet")
    write_macro_core_parquet(macro_rows, b / "macro_core.parquet")
    write_derivation_evidence_parquet(evid_rows, a / "derivation_evidence.parquet")
    write_derivation_evidence_parquet(evid_rows, b / "derivation_evidence.parquet")

    assert (a / "macro_core.parquet").read_bytes() == (b / "macro_core.parquet").read_bytes()
    assert (
        (a / "derivation_evidence.parquet").read_bytes()
        == (b / "derivation_evidence.parquet").read_bytes()
    )


# ---------------------------------------------------------------------------
# Value-type dispatch (bool BEFORE int — Python isinstance trap)
# ---------------------------------------------------------------------------


def test_derivation_evidence_value_type_dispatches_bool_before_int(tmp_path: Path):
    """Pin the bool/int dispatch ordering. ``isinstance(True, int) is True``
    in Python, so a naive int-first dispatch would serialise ``True`` as
    ``value_int=1`` instead of ``value_bool=True``. This test catches that.
    """
    rows = [
        DerivationEvidenceRow(SlotKind.CELL, 0, MetricNamespace.ZONING, "as_bool_t", True, "1.0"),
        DerivationEvidenceRow(SlotKind.CELL, 0, MetricNamespace.ZONING, "as_bool_f", False, "1.0"),
        DerivationEvidenceRow(SlotKind.CELL, 0, MetricNamespace.ZONING, "as_int_1", 1, "1.0"),
        DerivationEvidenceRow(SlotKind.CELL, 0, MetricNamespace.ZONING, "as_int_0", 0, "1.0"),
        DerivationEvidenceRow(SlotKind.CELL, 0, MetricNamespace.ZONING, "as_float", 1.0, "1.0"),
        DerivationEvidenceRow(SlotKind.CELL, 0, MetricNamespace.ZONING, "as_string", "x", "1.0"),
    ]
    path = tmp_path / "ev.parquet"
    write_derivation_evidence_parquet(rows, path)
    table = pq.ParquetFile(path).read()

    by_name = {n: i for i, n in enumerate(table["metric_name"].to_pylist())}
    value_types = table["value_type"].to_pylist()
    value_bool = table["value_bool"].to_pylist()
    value_int = table["value_int"].to_pylist()
    value_float = table["value_float"].to_pylist()
    value_string = table["value_string"].to_pylist()

    # spec §11.3 value_type enum: 0=float64, 1=int64, 2=string, 3=bool.
    assert value_types[by_name["as_bool_t"]] == 3
    assert value_bool[by_name["as_bool_t"]] is True
    assert value_int[by_name["as_bool_t"]] is None

    assert value_types[by_name["as_bool_f"]] == 3
    assert value_bool[by_name["as_bool_f"]] is False

    assert value_types[by_name["as_int_1"]] == 1
    assert value_int[by_name["as_int_1"]] == 1
    assert value_bool[by_name["as_int_1"]] is None

    assert value_types[by_name["as_float"]] == 0
    assert value_float[by_name["as_float"]] == pytest.approx(1.0)

    assert value_types[by_name["as_string"]] == 2
    assert value_string[by_name["as_string"]] == "x"


def test_read_helpers_round_trip(tmp_path: Path):
    """read_*_parquet returns the rows as dataclasses, byte-stable."""
    macro_rows = [
        MacroCoreRow(SlotKind.CELL, 0, 0, 0, None, None, None, Scope.ACTIVE, 0, 0, None),
        MacroCoreRow(
            SlotKind.INTERNAL_EDGE, 0, None, None, 0, 0, 0, Scope.ACTIVE, None, None, 0
        ),
    ]
    evid_rows = [
        DerivationEvidenceRow(
            SlotKind.CELL, 0, MetricNamespace.ZONING, "feature_count_road", 5, "1.0"
        ),
    ]
    macro_path = tmp_path / "macro_core.parquet"
    evid_path = tmp_path / "derivation_evidence.parquet"
    write_macro_core_parquet(macro_rows, macro_path)
    write_derivation_evidence_parquet(evid_rows, evid_path)

    read_macro = read_macro_core_parquet(macro_path)
    read_evid = read_derivation_evidence_parquet(evid_path)

    # Order matches the canonical sort (already in our inputs).
    assert [r.slot_index for r in read_macro] == [0, 0]
    assert [r.slot_kind for r in read_macro] == [SlotKind.CELL, SlotKind.INTERNAL_EDGE]
    assert read_evid[0].metric_name == "feature_count_road"
    assert read_evid[0].value == 5

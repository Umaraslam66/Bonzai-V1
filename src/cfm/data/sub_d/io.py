"""Per-tile sub-D artifact writers (spec §11.2, §11.3) — Task 9.

Schemas are pinned via ``pa.schema(...)`` so PyArrow type inference can
never silently widen an int8 column to int32 across versions. Sort keys
are documented in the spec and enforced before write:

- macro_core.parquet: ``(slot_kind, slot_index)``
- derivation_evidence.parquet:
  ``(slot_kind, slot_index, metric_namespace, metric_name)``

Determinism: writers route through the neutral ``cfm.data.io.write_parquet``
helper which pins ``PARQUET_WRITE_KWARGS`` (compression, row-group size,
data-page size, version) for byte-stable output.

Value-type dispatch for derivation_evidence rows (the value column union):

- ``bool``  -> value_type=3, value_bool. Checked BEFORE int because
  ``isinstance(True, int) is True`` in Python.
- ``int``   -> value_type=1, value_int.
- ``float`` -> value_type=0, value_float.
- ``str``   -> value_type=2, value_string.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cfm.data.io import write_parquet
from cfm.data.sub_d.enums import MetricNamespace, Scope, SlotKind


# ---------------------------------------------------------------------------
# macro_core.parquet (spec §11.2)
# ---------------------------------------------------------------------------


_MACRO_CORE_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("slot_kind", pa.int8()),
        pa.field("slot_index", pa.int16()),
        pa.field("cell_i", pa.int8()),
        pa.field("cell_j", pa.int8()),
        pa.field("lower_cell_i", pa.int8()),
        pa.field("lower_cell_j", pa.int8()),
        pa.field("axis", pa.int8()),
        pa.field("scope", pa.int8()),
        pa.field("zoning_class", pa.int16()),
        pa.field("cell_density_bucket", pa.int16()),
        pa.field("road_skeleton_class", pa.int16()),
    ]
)


@dataclass(frozen=True)
class MacroCoreRow:
    """One row of macro_core.parquet (spec §11.2).

    ``cell_i``/``cell_j`` are populated only for cell slots; edge slots use
    ``lower_cell_i``/``lower_cell_j``/``axis``. Target columns
    (``zoning_class``, ``cell_density_bucket``, ``road_skeleton_class``)
    are populated only on active cell/edge rows per spec §11.2 validation
    rules; inactive or masked slots leave them ``None``.
    """

    slot_kind: SlotKind
    slot_index: int
    cell_i: int | None
    cell_j: int | None
    lower_cell_i: int | None
    lower_cell_j: int | None
    axis: int | None
    scope: Scope
    zoning_class: int | None
    cell_density_bucket: int | None
    road_skeleton_class: int | None


def write_macro_core_parquet(rows: list[MacroCoreRow], path: Path) -> None:
    """Write macro_core.parquet with the pinned schema and canonical sort key."""
    sorted_rows = sorted(rows, key=lambda r: (int(r.slot_kind), int(r.slot_index)))
    columns: dict[str, list] = {
        "slot_kind": [int(r.slot_kind) for r in sorted_rows],
        "slot_index": [int(r.slot_index) for r in sorted_rows],
        "cell_i": [r.cell_i for r in sorted_rows],
        "cell_j": [r.cell_j for r in sorted_rows],
        "lower_cell_i": [r.lower_cell_i for r in sorted_rows],
        "lower_cell_j": [r.lower_cell_j for r in sorted_rows],
        "axis": [r.axis for r in sorted_rows],
        "scope": [int(r.scope) for r in sorted_rows],
        "zoning_class": [r.zoning_class for r in sorted_rows],
        "cell_density_bucket": [r.cell_density_bucket for r in sorted_rows],
        "road_skeleton_class": [r.road_skeleton_class for r in sorted_rows],
    }
    table = pa.Table.from_pydict(columns, schema=_MACRO_CORE_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(table, path)


def read_macro_core_parquet(path: Path) -> list[MacroCoreRow]:
    """Read macro_core.parquet rows back as ``MacroCoreRow`` dataclasses."""
    table = pq.ParquetFile(path).read()
    columns = {name: table[name].to_pylist() for name in table.schema.names}
    rows: list[MacroCoreRow] = []
    for i in range(table.num_rows):
        rows.append(
            MacroCoreRow(
                slot_kind=SlotKind(columns["slot_kind"][i]),
                slot_index=int(columns["slot_index"][i]),
                cell_i=columns["cell_i"][i],
                cell_j=columns["cell_j"][i],
                lower_cell_i=columns["lower_cell_i"][i],
                lower_cell_j=columns["lower_cell_j"][i],
                axis=columns["axis"][i],
                scope=Scope(columns["scope"][i]),
                zoning_class=columns["zoning_class"][i],
                cell_density_bucket=columns["cell_density_bucket"][i],
                road_skeleton_class=columns["road_skeleton_class"][i],
            )
        )
    return rows


# ---------------------------------------------------------------------------
# derivation_evidence.parquet (spec §11.3)
# ---------------------------------------------------------------------------


_DERIVATION_EVIDENCE_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("slot_kind", pa.int8()),
        pa.field("slot_index", pa.int16()),
        pa.field("metric_namespace", pa.int8()),
        pa.field("metric_name", pa.string()),
        pa.field("value_type", pa.int8()),
        pa.field("value_float", pa.float64()),
        pa.field("value_int", pa.int64()),
        pa.field("value_string", pa.string()),
        pa.field("value_bool", pa.bool_()),
        pa.field("derivation_version", pa.string()),
    ]
)


@dataclass(frozen=True)
class DerivationEvidenceRow:
    """One row of derivation_evidence.parquet (spec §11.3).

    ``value`` is a Python union over the four supported types; the writer
    dispatches to the correct ``value_*`` column and stamps ``value_type``.
    """

    slot_kind: SlotKind
    slot_index: int
    metric_namespace: MetricNamespace
    metric_name: str
    value: float | int | str | bool
    derivation_version: str


def _dispatch_value(value: float | int | str | bool) -> tuple[int, float | None, int | None, str | None, bool | None]:
    """Return ``(value_type, value_float, value_int, value_string, value_bool)``.

    ``bool`` is checked BEFORE ``int`` because ``isinstance(True, int)`` is
    ``True`` in Python — a naive int-first dispatch would serialise booleans
    as ``value_int`` and lose their type.
    """
    if isinstance(value, bool):
        return 3, None, None, None, bool(value)
    if isinstance(value, int):
        return 1, None, int(value), None, None
    if isinstance(value, float):
        return 0, float(value), None, None, None
    if isinstance(value, str):
        return 2, None, None, str(value), None
    raise TypeError(
        f"derivation_evidence value of type {type(value).__name__} is not supported; "
        "expected one of bool, int, float, str"
    )


def write_derivation_evidence_parquet(
    rows: list[DerivationEvidenceRow], path: Path
) -> None:
    """Write derivation_evidence.parquet with the pinned schema and sort key.

    Canonical sort key per spec §11.3:
    ``(slot_kind, slot_index, metric_namespace, metric_name)``.
    """
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            int(r.slot_kind),
            int(r.slot_index),
            int(r.metric_namespace),
            r.metric_name,
        ),
    )
    columns: dict[str, list] = {
        "slot_kind": [int(r.slot_kind) for r in sorted_rows],
        "slot_index": [int(r.slot_index) for r in sorted_rows],
        "metric_namespace": [int(r.metric_namespace) for r in sorted_rows],
        "metric_name": [r.metric_name for r in sorted_rows],
        "value_type": [],
        "value_float": [],
        "value_int": [],
        "value_string": [],
        "value_bool": [],
        "derivation_version": [r.derivation_version for r in sorted_rows],
    }
    for r in sorted_rows:
        vt, vf, vi, vs, vb = _dispatch_value(r.value)
        columns["value_type"].append(vt)
        columns["value_float"].append(vf)
        columns["value_int"].append(vi)
        columns["value_string"].append(vs)
        columns["value_bool"].append(vb)

    table = pa.Table.from_pydict(columns, schema=_DERIVATION_EVIDENCE_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(table, path)


def read_derivation_evidence_parquet(path: Path) -> list[DerivationEvidenceRow]:
    """Read derivation_evidence.parquet rows back as ``DerivationEvidenceRow``."""
    table = pq.ParquetFile(path).read()
    columns = {name: table[name].to_pylist() for name in table.schema.names}
    rows: list[DerivationEvidenceRow] = []
    for i in range(table.num_rows):
        value_type = int(columns["value_type"][i])
        if value_type == 0:
            value: float | int | str | bool = float(columns["value_float"][i])
        elif value_type == 1:
            value = int(columns["value_int"][i])
        elif value_type == 2:
            value = str(columns["value_string"][i])
        elif value_type == 3:
            value = bool(columns["value_bool"][i])
        else:
            raise ValueError(f"unknown value_type={value_type} at row {i}")
        rows.append(
            DerivationEvidenceRow(
                slot_kind=SlotKind(columns["slot_kind"][i]),
                slot_index=int(columns["slot_index"][i]),
                metric_namespace=MetricNamespace(columns["metric_namespace"][i]),
                metric_name=str(columns["metric_name"][i]),
                value=value,
                derivation_version=str(columns["derivation_version"][i]),
            )
        )
    return rows

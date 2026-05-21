from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from cfm.data.sub_e.derivation import BoundaryClass
from cfm.data.sub_e.writer import (
    BoundaryContractRow,
    SlotKind,
    write_boundary_contract,
)


def _make_full_lattice_rows() -> list[BoundaryContractRow]:
    """Construct a synthetic full-lattice 144-row set with all-active scope."""
    rows: list[BoundaryContractRow] = []
    for idx in range(112):
        rows.append(
            BoundaryContractRow(
                slot_kind=SlotKind.INTERNAL_EDGE,
                slot_index=idx,
                lower_cell_i=idx % 8,
                lower_cell_j=idx // 8 % 8,
                axis=idx % 2,
                scope_marker=0,  # active
                boundary_class_enum=int(BoundaryClass.NONE),
            )
        )
    for idx in range(32):
        rows.append(
            BoundaryContractRow(
                slot_kind=SlotKind.EXTERNAL_EDGE,
                slot_index=idx,
                lower_cell_i=idx % 8,
                lower_cell_j=idx // 8 % 8,
                axis=idx % 2,
                scope_marker=3,  # external_deferred
                boundary_class_enum=None,
            )
        )
    return rows


def test_write_produces_144_rows_sorted_canonical(tmp_path: Path) -> None:
    out_path = tmp_path / "boundary_contract.parquet"
    write_boundary_contract(out_path, _make_full_lattice_rows())
    tbl = pq.ParquetFile(out_path).read()
    assert tbl.num_rows == 144
    slot_kinds = tbl.column("slot_kind").to_pylist()
    slot_indices = tbl.column("slot_index").to_pylist()
    sorted_pairs = sorted(zip(slot_kinds, slot_indices, strict=True))
    assert list(zip(slot_kinds, slot_indices, strict=True)) == sorted_pairs


def test_internal_row_count_is_112_external_is_32(tmp_path: Path) -> None:
    out_path = tmp_path / "boundary_contract.parquet"
    write_boundary_contract(out_path, _make_full_lattice_rows())
    tbl = pq.ParquetFile(out_path).read()
    slot_kinds = tbl.column("slot_kind").to_pylist()
    assert slot_kinds.count(int(SlotKind.INTERNAL_EDGE)) == 112
    assert slot_kinds.count(int(SlotKind.EXTERNAL_EDGE)) == 32


def test_boundary_class_enum_nullable(tmp_path: Path) -> None:
    out_path = tmp_path / "boundary_contract.parquet"
    write_boundary_contract(out_path, _make_full_lattice_rows())
    tbl = pq.ParquetFile(out_path).read()
    values = tbl.column("boundary_class_enum").to_pylist()
    # 112 active internal rows have value NONE=1; 32 external rows are null.
    assert values.count(None) == 32
    assert values.count(1) == 112


def test_write_rejects_wrong_row_count(tmp_path: Path) -> None:
    import pytest

    out_path = tmp_path / "boundary_contract.parquet"
    with pytest.raises(ValueError, match="144"):
        write_boundary_contract(out_path, _make_full_lattice_rows()[:100])


def test_write_sorts_unordered_input(tmp_path: Path) -> None:
    """Canonical sort key is enforced at write-time, not assumed from input
    order. Pin: the writer's own sort is load-bearing for byte-determinism
    across upstream code paths that might emit rows in any order.
    """
    import random

    rows = _make_full_lattice_rows()
    rng = random.Random(0xBEEF)
    shuffled = rows.copy()
    rng.shuffle(shuffled)
    assert shuffled != rows, "test setup: shuffled order must differ from canonical"

    out_path = tmp_path / "boundary_contract.parquet"
    write_boundary_contract(out_path, shuffled)
    tbl = pq.ParquetFile(out_path).read()
    slot_kinds = tbl.column("slot_kind").to_pylist()
    slot_indices = tbl.column("slot_index").to_pylist()
    pairs = list(zip(slot_kinds, slot_indices, strict=True))
    assert pairs == sorted(pairs), "writer must sort by (slot_kind, slot_index)"


def test_write_is_byte_deterministic_on_rerun(tmp_path: Path) -> None:
    """Same-process determinism: two writes of identical inputs produce
    byte-identical parquet files. Catches schema/sort/dict-ordering drift
    earliest — without waiting for Task 7's validator or Task 14's
    integration test.
    """
    import hashlib

    rows = _make_full_lattice_rows()
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    write_boundary_contract(a, rows)
    write_boundary_contract(b, rows)
    h_a = hashlib.sha256(a.read_bytes()).hexdigest()
    h_b = hashlib.sha256(b.read_bytes()).hexdigest()
    assert h_a == h_b, "same-process determinism required"


def test_write_is_byte_deterministic_under_input_shuffling(tmp_path: Path) -> None:
    """Byte-determinism survives input-order perturbation. Two writes of the
    SAME logical row set but in different input orders must produce the
    same bytes — because the writer's canonical sort dominates input order.
    """
    import hashlib
    import random

    rows = _make_full_lattice_rows()
    rng = random.Random(0xC0FFEE)
    perm = rows.copy()
    rng.shuffle(perm)

    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    write_boundary_contract(a, rows)
    write_boundary_contract(b, perm)
    h_a = hashlib.sha256(a.read_bytes()).hexdigest()
    h_b = hashlib.sha256(b.read_bytes()).hexdigest()
    assert h_a == h_b, "writer's canonical sort must dominate input order"

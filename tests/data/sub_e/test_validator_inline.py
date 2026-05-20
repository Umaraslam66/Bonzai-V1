from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cfm.data.sub_e.derivation import BoundaryClass
from cfm.data.sub_e.validator_inline import (
    InlineValidationError,
    validate_boundary_contract,
)
from cfm.data.sub_e.versions import BOUNDARY_DERIVATION_VERSION
from cfm.data.sub_e.writer import (
    BoundaryContractRow,
    SlotKind,
    write_boundary_contract,
)


def _valid_rows() -> list[BoundaryContractRow]:
    rows: list[BoundaryContractRow] = []
    for idx in range(112):
        rows.append(
            BoundaryContractRow(
                slot_kind=SlotKind.INTERNAL_EDGE,
                slot_index=idx,
                lower_cell_i=idx % 8,
                lower_cell_j=idx // 8 % 8,
                axis=idx % 2,
                scope_marker=0,
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
                scope_marker=3,
                boundary_class_enum=None,
            )
        )
    return rows


def _write(tmp_path: Path, rows: list[BoundaryContractRow]) -> Path:
    p = tmp_path / "boundary_contract.parquet"
    # Writer enforces row count; bypass when needed via direct table writing
    # is NOT permitted here — we exercise inline validator on writer output.
    write_boundary_contract(p, rows)
    return p


def _validate(
    p: Path,
    *,
    expected: str = BOUNDARY_DERIVATION_VERSION,
    provenance: str = BOUNDARY_DERIVATION_VERSION,
    lever_3_collapse: bool = False,
) -> None:
    """Test helper: passes both required version kwargs to the validator.

    Defaults to the locked v1.0 for both, so tests that don't exercise
    invariant #8 stay terse. Tests that exercise #8 pass mismatched values
    explicitly.
    """
    validate_boundary_contract(
        p,
        expected_derivation_version=expected,
        provenance_derivation_version=provenance,
        lever_3_collapse=lever_3_collapse,
    )


def test_valid_lattice_passes_all_invariants(tmp_path: Path) -> None:
    p = _write(tmp_path, _valid_rows())
    _validate(p)  # should not raise


def test_invariant_3_class_non_null_iff_scope_active(tmp_path: Path) -> None:
    rows = _valid_rows()
    # Set an external (scope_marker=3) row to have a non-null class — violates #3.
    rows[112] = replace(rows[112], boundary_class_enum=int(BoundaryClass.MAJOR_ROAD))
    p = _write(tmp_path, rows)
    with pytest.raises(InlineValidationError, match="non-null iff scope_marker == 0"):
        _validate(p)


def test_invariant_4_active_class_membership(tmp_path: Path) -> None:
    rows = _valid_rows()
    # Use BOUNDARY_NOT_APPLICABLE (0) on an active row — sentinel is dataloader-side
    # only; on-disk active rows must be in {NONE=1, MAJOR_ROAD=2, MINOR_ROAD=3}.
    rows[0] = replace(rows[0], boundary_class_enum=int(BoundaryClass.BOUNDARY_NOT_APPLICABLE))
    p = _write(tmp_path, rows)
    with pytest.raises(InlineValidationError, match="active class membership"):
        _validate(p)


def test_invariant_5_scope_marker_membership(tmp_path: Path) -> None:
    rows = _valid_rows()
    # Mutate scope_marker=9 AND null the boundary_class_enum so invariants
    # #3/#4 are not in violation on this row — but invariant #5 (membership)
    # fires regardless because it's structurally prior under the
    # membership-before-semantic loop order.
    rows[0] = replace(rows[0], scope_marker=9, boundary_class_enum=None)
    p = _write(tmp_path, rows)
    with pytest.raises(InlineValidationError, match="scope_marker membership"):
        _validate(p)


def test_invariant_6_slot_index_range(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows[0] = replace(rows[0], slot_index=999)
    p = _write(tmp_path, rows)
    with pytest.raises(InlineValidationError, match="slot_index range"):
        _validate(p)


def test_invariant_7_axis_membership(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows[0] = replace(rows[0], axis=2)  # AXIS = {0, 1}
    p = _write(tmp_path, rows)
    with pytest.raises(InlineValidationError, match="axis membership"):
        _validate(p)


def test_invariant_8_derivation_version_match(tmp_path: Path) -> None:
    """Provenance mismatch with expected raises. Both kwargs are required
    on the validator; the test passes an explicit mismatch."""
    p = _write(tmp_path, _valid_rows())
    with pytest.raises(InlineValidationError, match="boundary_derivation_version"):
        _validate(p, expected="9.9", provenance=BOUNDARY_DERIVATION_VERSION)


# Invariants #1 (row count) and #2 (sort key) are enforced by the writer
# itself (Task 6); a malformed parquet that bypasses the writer would still
# trigger them. Cover them by reading the parquet, mutating, and re-writing
# raw via pyarrow:


def test_invariant_1_total_row_count(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    p = _write(tmp_path, _valid_rows())
    tbl = pq.ParquetFile(p).read()
    # Slice to 100 rows — bypasses writer.
    bad = tbl.slice(0, 100)
    pq.write_table(bad, p)
    with pytest.raises(InlineValidationError, match="144"):
        _validate(p)


def test_invariant_2_sort_key(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    p = _write(tmp_path, _valid_rows())
    tbl = pq.ParquetFile(p).read()
    # Reverse — sort key violated.
    bad = tbl.slice(0, tbl.num_rows).take(list(range(tbl.num_rows - 1, -1, -1)))
    pq.write_table(bad, p)
    with pytest.raises(InlineValidationError, match="sort key"):
        _validate(p)


def test_lever_3_collapse_passes_with_uniform_null(tmp_path: Path) -> None:
    """Under lever-3, all boundary_class_enum values null even on active rows."""
    rows = _valid_rows()
    rows = [replace(r, boundary_class_enum=None) for r in rows]
    p = _write(tmp_path, rows)
    _validate(p, lever_3_collapse=True)  # should not raise


def test_lever_3_collapse_rejects_any_non_null(tmp_path: Path) -> None:
    """Under lever-3, even a single non-null boundary_class_enum is a violation."""
    rows = _valid_rows()
    rows = [replace(r, boundary_class_enum=None) for r in rows]
    rows[0] = replace(rows[0], boundary_class_enum=int(BoundaryClass.NONE))
    p = _write(tmp_path, rows)
    with pytest.raises(InlineValidationError, match="lever-3"):
        _validate(p, lever_3_collapse=True)


def test_invariant_9_slotkind_cross_enum_byte_equivalence() -> None:
    """Invariant #9 (Task-6 carry-forward): sub-E writer's SlotKind enum
    integer values must match sub-D's SlotKind byte-for-byte at INTERNAL_EDGE
    and EXTERNAL_EDGE. Two separate IntEnum classes maintain wire
    compatibility manually; this test is the regression guard against
    silent drift if either side gains a member or reorders values.
    """
    from cfm.data.sub_d.enums import SlotKind as SubDSlotKind

    assert int(SlotKind.INTERNAL_EDGE) == int(SubDSlotKind.INTERNAL_EDGE) == 1
    assert int(SlotKind.EXTERNAL_EDGE) == int(SubDSlotKind.EXTERNAL_EDGE) == 2


def test_invariant_9_drift_simulation_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Invariant #9 fires at validator runtime when wire values diverge.

    Simulates drift by monkey-patching sub-D's `SlotKind` to a divergent
    IntEnum. The validator must raise InlineValidationError; the validator's
    lazy import of `cfm.data.sub_d.enums.SlotKind` is what makes this test
    able to observe the drift.
    """
    from enum import IntEnum

    from cfm.data.sub_d import enums as sub_d_enums

    class _DriftedSlotKind(IntEnum):
        INTERNAL_EDGE = 99  # divergent from sub-E writer's INTERNAL_EDGE=1
        EXTERNAL_EDGE = 2

    monkeypatch.setattr(sub_d_enums, "SlotKind", _DriftedSlotKind)
    p = _write(tmp_path, _valid_rows())
    with pytest.raises(InlineValidationError, match="SlotKind"):
        _validate(p)

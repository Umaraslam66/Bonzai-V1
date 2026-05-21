"""Sub-E per-tile inline validator.

Implements the 8 invariants from spec §10.1 plus 1 sub-E-local invariant #9
(Task-6 carry-forward) asserting cross-enum byte-equivalence between sub-E
writer's SlotKind and sub-D's SlotKind. Invariant #9 is mode-independent
(applies under both default and lever_3_collapse modes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pyarrow.parquet as pq

from cfm.data.sub_e.derivation import BoundaryClass
from cfm.data.sub_e.writer import (
    EXPECTED_EXTERNAL_ROWS,
    EXPECTED_INTERNAL_ROWS,
    EXPECTED_TOTAL_ROWS,
    SlotKind,
)

_ACTIVE_CLASS_IDS: Final[frozenset[int]] = frozenset(
    {
        int(BoundaryClass.NONE),
        int(BoundaryClass.MAJOR_ROAD),
        int(BoundaryClass.MINOR_ROAD),
    }
)

_SCOPE_MARKER_VALUES: Final[frozenset[int]] = frozenset({0, 1, 2, 3})
_AXIS_VALUES: Final[frozenset[int]] = frozenset({0, 1})


class InlineValidationError(ValueError):
    """Raised when a sub-E boundary_contract.parquet fails any inline invariant."""


def _assert_slotkind_byte_equivalence_with_sub_d() -> None:
    """Invariant #9 (Task-6 carry-forward, sub-E-local — not in original 8).

    Sub-D's `SlotKind` (`src/cfm/data/sub_d/enums.py`) and sub-E writer's
    `SlotKind` (`src/cfm/data/sub_e/writer.py`) are two separate IntEnum
    classes that share wire values `INTERNAL_EDGE=1` and `EXTERNAL_EDGE=2`.
    If either side gains a member or reorders values without coordinating,
    the cross-table foreign-key path corrupts silently. This guard catches
    drift at validation time. The import is intentionally lazy so test
    monkey-patches against `cfm.data.sub_d.enums.SlotKind` are observable.
    """
    from cfm.data.sub_d.enums import SlotKind as SubDSlotKind

    if int(SlotKind.INTERNAL_EDGE) != int(SubDSlotKind.INTERNAL_EDGE):
        raise InlineValidationError(
            "sub-E and sub-D SlotKind.INTERNAL_EDGE wire values diverged "
            f"(sub-E={int(SlotKind.INTERNAL_EDGE)}, "
            f"sub-D={int(SubDSlotKind.INTERNAL_EDGE)}); invariant #9 violated"
        )
    if int(SlotKind.EXTERNAL_EDGE) != int(SubDSlotKind.EXTERNAL_EDGE):
        raise InlineValidationError(
            "sub-E and sub-D SlotKind.EXTERNAL_EDGE wire values diverged "
            f"(sub-E={int(SlotKind.EXTERNAL_EDGE)}, "
            f"sub-D={int(SubDSlotKind.EXTERNAL_EDGE)}); invariant #9 violated"
        )


def validate_boundary_contract(
    path: Path,
    *,
    expected_derivation_version: str,
    provenance_derivation_version: str,
    lever_3_collapse: bool = False,
) -> None:
    """Validate one boundary_contract.parquet. Raises InlineValidationError.

    Both `expected_derivation_version` (what the pipeline expects to see for
    this run) and `provenance_derivation_version` (what was actually recorded
    in the sibling provenance.yaml) are **required**. There is no on-disk
    source for the actual value inside the parquet itself — Task 10's
    pipeline orchestrator owns the responsibility of loading the actual
    value from the sibling `provenance.yaml` and passing it here. This
    keeps invariant #8 inline (per spec §10.1 categorisation) while pushing
    the I/O responsibility to the caller.

    Under `lever_3_collapse=True`, invariants #3 (non-null iff active) and
    #4 (active class membership) are replaced by a single uniform-null check
    (every row's boundary_class_enum is null). Other invariants still apply,
    including #9 (cross-enum byte-equivalence — mode-independent).

    Loop order: invariant #9 first (structural enum drift), then file-level
    invariants #1 + #2 (count + sort), then per-row invariants in
    **membership-before-semantic** order — #5/#6/#7 (structural enum/range
    membership) before #3/#4 (semantic non-null relationship + active class
    membership). Membership invariants are structurally prior; semantic
    relationships rely on values being in-range first. Catching #5/#6/#7
    early also yields more useful error messages on real-data violations.
    """
    # Invariant 9 (Task-6 carry-forward): structural enum drift is the
    # earliest possible failure mode; check before any file-level invariant.
    _assert_slotkind_byte_equivalence_with_sub_d()

    tbl = pq.ParquetFile(path).read()
    slot_kinds = tbl.column("slot_kind").to_pylist()
    slot_indices = tbl.column("slot_index").to_pylist()
    scope_markers = tbl.column("scope_marker").to_pylist()
    boundary_classes = tbl.column("boundary_class_enum").to_pylist()
    axes = tbl.column("axis").to_pylist()

    # Invariant 1: row count.
    if tbl.num_rows != EXPECTED_TOTAL_ROWS:
        raise InlineValidationError(
            f"row count must be {EXPECTED_TOTAL_ROWS} (112 + 32), got {tbl.num_rows}"
        )
    n_internal = sum(1 for k in slot_kinds if k == int(SlotKind.INTERNAL_EDGE))
    n_external = sum(1 for k in slot_kinds if k == int(SlotKind.EXTERNAL_EDGE))
    if (n_internal, n_external) != (
        EXPECTED_INTERNAL_ROWS,
        EXPECTED_EXTERNAL_ROWS,
    ):
        raise InlineValidationError(
            f"slot_kind split must be (112, 32), got ({n_internal}, {n_external})"
        )

    # Invariant 2: canonical sort key (slot_kind, slot_index).
    pairs = list(zip(slot_kinds, slot_indices, strict=True))
    if pairs != sorted(pairs):
        raise InlineValidationError("rows not sorted by canonical sort key (slot_kind, slot_index)")

    # Lever-3 mode collapses invariants #3 + #4 into a single uniform-null
    # check across all rows. Run this before the per-row membership/semantic
    # loop so a mode violation surfaces as a clear "lever-3" message.
    if lever_3_collapse:
        for i, cls in enumerate(boundary_classes):
            if cls is not None:
                raise InlineValidationError(
                    f"row {i}: lever-3 mode requires boundary_class_enum is null "
                    f"in every row (got {cls})"
                )

    # Per-row invariants in membership-before-semantic order: structural
    # validity (#5, #6, #7) before semantic relationships (#3, #4).
    # Membership invariants are independent of derivation; semantic
    # invariants assume the per-row values are already in valid ranges.
    for i, (sk, si, scope, cls, axis) in enumerate(
        zip(
            slot_kinds,
            slot_indices,
            scope_markers,
            boundary_classes,
            axes,
            strict=True,
        )
    ):
        # 5: scope_marker membership.
        if scope not in _SCOPE_MARKER_VALUES:
            raise InlineValidationError(
                f"row {i}: scope_marker membership violated (scope={scope})"
            )
        # 6: slot_index range per slot_kind.
        if sk == int(SlotKind.INTERNAL_EDGE) and not (0 <= si < 112):
            raise InlineValidationError(f"row {i}: slot_index range violated (internal, idx={si})")
        if sk == int(SlotKind.EXTERNAL_EDGE) and not (0 <= si < 32):
            raise InlineValidationError(f"row {i}: slot_index range violated (external, idx={si})")
        # 7: axis membership.
        if axis not in _AXIS_VALUES:
            raise InlineValidationError(f"row {i}: axis membership violated (axis={axis})")

        if not lever_3_collapse:
            # 3: boundary_class_enum non-null iff scope_marker == 0.
            is_active = scope == 0
            if is_active and cls is None:
                raise InlineValidationError(
                    f"row {i}: boundary_class_enum non-null iff scope_marker == 0 "
                    f"(scope=active, class=null)"
                )
            if (not is_active) and cls is not None:
                raise InlineValidationError(
                    f"row {i}: boundary_class_enum non-null iff scope_marker == 0 "
                    f"(scope={scope}, class={cls})"
                )
            # 4: active class membership (sentinel 0 forbidden on-disk).
            if is_active and cls not in _ACTIVE_CLASS_IDS:
                raise InlineValidationError(
                    f"row {i}: active class membership violated "
                    f"(class={cls} not in {sorted(_ACTIVE_CLASS_IDS)})"
                )

    # Invariant 8: provenance derivation version matches expected.
    if provenance_derivation_version != expected_derivation_version:
        raise InlineValidationError(
            f"boundary_derivation_version mismatch: expected "
            f"{expected_derivation_version}, got {provenance_derivation_version}"
        )

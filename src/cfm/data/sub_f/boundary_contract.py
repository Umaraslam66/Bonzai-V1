"""Sub-F BP7 sub-E boundary-contract reader.

Consumes sub-E `boundary_contract.parquet` and emits a per-cell per-edge
class map that the encoder uses to decide Case A/B/C/D and to look up
the correct <bref_DIR_CLASS> token.

SOURCE-DERIVED CONTRACT (NOT inferred):
Every contract fact below is read directly from sub-E's locked source
modules. See `_SUB_E_CONTRACT` constant for the complete file:line
citation list. The only residual verification debt is empirical
(real-data numeric ratios for T3c stage-4) and integration-end-to-end
(first real-data flow through the reader confirms source-derived
constants weren't misread). Schema, dtypes, enum values, sentinel
encoding, hierarchy, join key shape: all source-derived.

FAIL-LOUD AT PARSE BOUNDARY:
The reader validates every sub-E parquet against source-derived
constants and raises `SubEContractViolation` on any mismatch. This is
NOT defense against "wrong inference" - it is the named signal that
sub-E source has drifted (e.g., a future sub-E version bump touched
the schema). When that signal fires, the consumer needs updating.

See the audit trail at `_SUB_E_CONTRACT`; see
`reports/2026-05-23-phase-1-sub-F-close-checklist.md` for the residual
absent-data obligation (first real sub-E read should succeed; spot-check
T3c stage-4 ratio when sub-E lands).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq

from cfm.data.sub_e.derivation import BoundaryClass
from cfm.data.sub_e.rotation import EdgeKind
from cfm.data.sub_e.writer import (
    EXPECTED_EXTERNAL_ROWS,
    EXPECTED_INTERNAL_ROWS,
    EXPECTED_TOTAL_ROWS,
    SlotKind,
)
from cfm.data.sub_f.rotation import cell_edge_directions

# Sub-E BoundaryClass enum mapping (sourced from derivation.py:19-23).
_CLASS_LABEL_BY_ENUM: Final[dict[int, str]] = {
    int(BoundaryClass.NONE): "NONE",
    int(BoundaryClass.MAJOR_ROAD): "MAJOR_ROAD",
    int(BoundaryClass.MINOR_ROAD): "MINOR_ROAD",
}

# Only these two classes emit a <bref> token (per spec §3.7 BP7 lock).
_EMITTING_CLASSES: Final[frozenset[str]] = frozenset({"MAJOR_ROAD", "MINOR_ROAD"})


# ---- SOURCE-DERIVED CONTRACT (audit trail) ------------------------------

_SUB_E_CONTRACT: Final[str] = """
SOURCE-DERIVED sub-E boundary_contract.parquet contract. Every fact below
is read from sub-E's locked source modules (not inferred from spec):

  1. Schema (7 columns): src/cfm/data/sub_e/writer.py:38-48
     _BOUNDARY_CONTRACT_SCHEMA. Matched bit-for-bit by _EXPECTED_SCHEMA
     below. The only nullable column is boundary_class_enum (int16);
     all 6 others are non-null with dtypes pinned per writer.

  2. Row count: 144 per tile (112 INTERNAL + 32 EXTERNAL). Source:
     src/cfm/data/sub_e/writer.py:28-30 EXPECTED_TOTAL_ROWS. Reader
     validates total + split.

  3. Sort order: (slot_kind, slot_index). Source:
     src/cfm/data/sub_e/writer.py:82. Reader does not require this for
     correctness (it builds a dict join), but it's part of the contract.

  4. SlotKind enum: INTERNAL_EDGE=1, EXTERNAL_EDGE=2. Source:
     src/cfm/data/sub_e/writer.py:23-25.

  5. BoundaryClass enum: BOUNDARY_NOT_APPLICABLE=0 (dataloader-only,
     never on-disk per derivation.py:20 docstring), NONE=1,
     MAJOR_ROAD=2, MINOR_ROAD=3. Source:
     src/cfm/data/sub_e/derivation.py:19-23.

  6. NULL-vs-enum sentinel invariant (LOAD-BEARING): boundary_class_enum
     non-null iff scope_marker == 0. Source:
     src/cfm/data/sub_e/validator_inline.py:169-178. This is sub-E's
     on-disk encoding of BOUNDARY_NOT_APPLICABLE: a non-active row
     (scope_marker != 0) carries boundary_class_enum=NULL; an active
     row (scope_marker == 0) carries one of {NONE, MAJOR, MINOR}.

  7. EdgeIdTuple shape: (lower_cell_i, lower_cell_j, axis, kind). Source:
     src/cfm/data/sub_e/rotation.py:57-58.

  8. Join key (cell x edge -> parquet row): (slot_kind, lower_cell_i,
     lower_cell_j, axis) where slot_kind is derived from EdgeKind
     (INTERNAL->1, EXTERNAL->2). Internal edges appear once on-disk
     even though they're shared between two cells; the join lookup
     surfaces the same row for both cells' views.

  9. Hierarchy (MAJOR > MINOR > NONE): src/cfm/data/sub_e/derivation.py:27-31.
     Unknown class_raw values fall through to MINOR (derivation.py:85).
     `motorway` is NOT in `configs/macro_plan/v1/boundary_vocab.yaml`
     MAJOR_ROAD list {primary, trunk, secondary} -> falls through to
     MINOR_ROAD per the fallthrough rule. Recorded for spec §13.1
     cascade #9 traceability.

 10. Lever-3 collapse mode: all boundary_class_enum NULL. Source:
     src/cfm/data/sub_e/pipeline.py:8 + validator_inline.py:87, 136.
     If sub-F is asked to consume a lever-3 tile, every row is NULL
     and zero BP7 tokens are emitted - valid behavior.

If sub-E ever bumps a version (spec §6 SOURCE axis) and the schema or
sentinel rules drift, the SubEContractViolation surface below fires
on first real read; that's the named signal to update this consumer.
"""


# ---- Source-derived parse-boundary defense constants --------------------

# Bit-for-bit match against src/cfm/data/sub_e/writer.py:38-48
# _BOUNDARY_CONTRACT_SCHEMA. If sub-E changes the schema, this constant
# must change with it (and ideally the change is caught by the lock-and-
# guards-travel-together discipline at sub-E's commit time).
_EXPECTED_SCHEMA: Final[pa.Schema] = pa.schema(
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

_VALID_SLOT_KINDS: Final[frozenset[int]] = frozenset(
    {int(SlotKind.INTERNAL_EDGE), int(SlotKind.EXTERNAL_EDGE)}
)

# BOUNDARY_NOT_APPLICABLE (enum=0) is forbidden as a non-null on-disk
# value per derivation.py:20 docstring; valid non-null enums are
# {NONE, MAJOR_ROAD, MINOR_ROAD}.
_VALID_BOUNDARY_CLASS_ENUMS: Final[frozenset[int]] = frozenset(
    {int(BoundaryClass.NONE), int(BoundaryClass.MAJOR_ROAD), int(BoundaryClass.MINOR_ROAD)}
)


class SubEContractViolation(ValueError):
    """Raised when real sub-E parquet violates the source-derived contract.

    Sub-F-v1's BP7 consumer is built against sub-E's locked source modules
    (see _SUB_E_CONTRACT for the file:line audit trail). This exception is
    the named signal that sub-E source has DRIFTED - e.g., a future sub-E
    version bump touched the schema or sentinel rules without updating
    this consumer. When this fires on first real-data contact, update the
    consumer (and propagate the lock-and-guards discipline back into
    sub-E's commit).

    See `_SUB_E_CONTRACT` module constant for the full contract list; see
    `reports/2026-05-23-phase-1-sub-F-close-checklist.md` for the
    residual real-data obligation (first real sub-E read should succeed;
    spot-check T3c stage-4 ratio when sub-E lands).
    """


def resolve_bref_tag(direction: str, class_label: str) -> str | None:
    """Build the BP7 token tag for (direction, class). Returns None if the
    class is non-emitting (NONE).

    Per spec §3.7 BP7 lock: 8 active tokens {N,E,S,W} x {MAJOR,MINOR}.
    """
    if class_label not in _EMITTING_CLASSES:
        return None
    if direction not in ("N", "E", "S", "W"):
        raise ValueError(f"resolve_bref_tag: unsupported direction {direction!r}")
    short = "MAJOR" if class_label == "MAJOR_ROAD" else "MINOR"
    return f"<bref_{direction}_{short}>"


def load_boundary_contract(
    parquet_path: Path,
) -> dict[tuple[int, int], dict[str, str]]:
    """Read a sub-E boundary_contract.parquet and emit a per-cell map.

    Output shape:
        { (cell_i, cell_j): { "N": class_label, "E": ..., "S": ..., "W": ... } }

    For each cell x edge, the reader looks up the parquet row whose
    (slot_kind, lower_cell_i, lower_cell_j, axis) matches the
    EdgeIdTuple from cell_edge_directions. The reported class_label is:
      - "MAJOR_ROAD" or "MINOR_ROAD" if the row is active (scope_marker=0)
        with the matching enum
      - "NONE" if the row is active with enum=NONE OR the row is non-active
        (boundary_class_enum=NULL, encoding sub-E's BOUNDARY_NOT_APPLICABLE
        per validator_inline.py:169-178 invariant)

    Per `feedback_pyarrow_hive_partition_inference`: uses
    `pq.ParquetFile(path).read()` - never bare `pq.read_table()` on the
    parent directory.

    Defensive parse-boundary assertions (any failure -> SubEContractViolation):
      - Schema exact match against _EXPECTED_SCHEMA
        (writer.py:38-48 _BOUNDARY_CONTRACT_SCHEMA)
      - Row count == EXPECTED_TOTAL_ROWS (144); split (112, 32)
      - slot_kind in _VALID_SLOT_KINDS
      - NULL-vs-enum invariant: boundary_class_enum non-null iff scope_marker == 0
        (validator_inline.py:169-178)
      - When non-null: boundary_class_enum in _VALID_BOUNDARY_CLASS_ENUMS
        (BOUNDARY_NOT_APPLICABLE=0 forbidden as non-null per derivation.py:20)
    """
    table = pq.ParquetFile(parquet_path).read()

    # Parse-boundary defense (1): exact schema match.
    if table.schema != _EXPECTED_SCHEMA:
        raise SubEContractViolation(
            f"sub-E parquet schema mismatch at {parquet_path}.\n"
            f"  expected: {_EXPECTED_SCHEMA}\n"
            f"  got:      {table.schema}\n"
            f"See _SUB_E_CONTRACT (module docstring) item 1 + "
            f"reports/2026-05-23-phase-1-sub-F-close-checklist.md."
        )

    # Parse-boundary defense (2): row count + split.
    if table.num_rows != EXPECTED_TOTAL_ROWS:
        raise SubEContractViolation(
            f"sub-E parquet row count at {parquet_path}: "
            f"got {table.num_rows}, expected {EXPECTED_TOTAL_ROWS} "
            f"({EXPECTED_INTERNAL_ROWS} INTERNAL + {EXPECTED_EXTERNAL_ROWS} EXTERNAL). "
            f"See _SUB_E_CONTRACT item 2."
        )

    rows = table.to_pylist()
    n_internal = sum(1 for r in rows if int(r["slot_kind"]) == int(SlotKind.INTERNAL_EDGE))
    n_external = sum(1 for r in rows if int(r["slot_kind"]) == int(SlotKind.EXTERNAL_EDGE))
    if (n_internal, n_external) != (EXPECTED_INTERNAL_ROWS, EXPECTED_EXTERNAL_ROWS):
        raise SubEContractViolation(
            f"sub-E parquet split at {parquet_path}: got ({n_internal}, "
            f"{n_external}), expected ({EXPECTED_INTERNAL_ROWS}, "
            f"{EXPECTED_EXTERNAL_ROWS}). See _SUB_E_CONTRACT item 2."
        )

    # Build the join lookup keyed by (slot_kind, lower_cell_i, lower_cell_j, axis).
    # Each parquet row maps to a class_label per the NULL-vs-enum invariant.
    join_lookup: dict[tuple[int, int, int, int], str] = {}
    for row_idx, r in enumerate(rows):
        sk = int(r["slot_kind"])
        li = int(r["lower_cell_i"])
        lj = int(r["lower_cell_j"])
        ax = int(r["axis"])
        sm = int(r["scope_marker"])
        cls_enum_raw = r["boundary_class_enum"]  # int | None

        # Parse-boundary defense (3): slot_kind in valid set.
        if sk not in _VALID_SLOT_KINDS:
            raise SubEContractViolation(
                f"sub-E parquet row {row_idx} at {parquet_path}: slot_kind={sk} "
                f"not in {sorted(_VALID_SLOT_KINDS)} (INTERNAL_EDGE=1, "
                f"EXTERNAL_EDGE=2). See _SUB_E_CONTRACT item 4."
            )

        # Parse-boundary defense (4): NULL-vs-enum invariant.
        # validator_inline.py:169-178: boundary_class_enum non-null iff scope_marker == 0.
        is_active = sm == 0
        is_null = cls_enum_raw is None
        if is_active and is_null:
            raise SubEContractViolation(
                f"sub-E parquet row {row_idx} at {parquet_path}: scope_marker=0 "
                f"(active) but boundary_class_enum is NULL - violates "
                f"validator_inline.py:169-178 invariant (non-null iff "
                f"scope_marker == 0). See _SUB_E_CONTRACT item 6."
            )
        if (not is_active) and (not is_null):
            raise SubEContractViolation(
                f"sub-E parquet row {row_idx} at {parquet_path}: scope_marker={sm} "
                f"(non-active) but boundary_class_enum={cls_enum_raw} is non-null - "
                f"violates validator_inline.py:169-178 invariant. See _SUB_E_CONTRACT item 6."
            )

        if is_null:
            # Non-active row: sub-E's encoding of BOUNDARY_NOT_APPLICABLE per
            # derivation.py:20 dataloader-only docstring; reader surfaces as NONE
            # (encoder emits no BP7 token).
            cls_label = "NONE"
        else:
            cls_enum = int(cls_enum_raw)
            # Parse-boundary defense (5): on-disk enum in valid set.
            if cls_enum not in _VALID_BOUNDARY_CLASS_ENUMS:
                raise SubEContractViolation(
                    f"sub-E parquet row {row_idx} at {parquet_path}: "
                    f"boundary_class_enum={cls_enum} not in "
                    f"{sorted(_VALID_BOUNDARY_CLASS_ENUMS)} "
                    f"(NONE=1, MAJOR_ROAD=2, MINOR_ROAD=3). "
                    f"BOUNDARY_NOT_APPLICABLE (enum=0) is forbidden as a "
                    f"non-null on-disk value per derivation.py:20 docstring. "
                    f"See _SUB_E_CONTRACT items 5+6."
                )
            cls_label = _CLASS_LABEL_BY_ENUM[cls_enum]
        join_lookup[(sk, li, lj, ax)] = cls_label

    # Walk every cell x direction; build the per-cell map via the join
    # lookup. EdgeIdTuple is (lower_cell_i, lower_cell_j, axis, kind); the
    # join key is (slot_kind, lower_cell_i, lower_cell_j, axis) where
    # slot_kind is derived from EdgeKind.
    contract: dict[tuple[int, int], dict[str, str]] = {}
    for cell_i in range(8):
        for cell_j in range(8):
            edge_ids = cell_edge_directions(cell_i, cell_j)
            cell_edges: dict[str, str] = {}
            for direction in ("N", "E", "S", "W"):
                lower_i, lower_j, axis, kind = edge_ids[direction]
                slot_kind = (
                    int(SlotKind.INTERNAL_EDGE)
                    if kind is EdgeKind.INTERNAL
                    else int(SlotKind.EXTERNAL_EDGE)
                )
                key = (slot_kind, int(lower_i), int(lower_j), int(axis))
                # Default to NONE if the lookup misses (e.g., truncated tile
                # in test fixtures); real sub-E always emits all 144 rows
                # per the row-count defense above, so the .get() default
                # is only reached in well-formed tiles where every edge has
                # a row.
                cell_edges[direction] = join_lookup.get(key, "NONE")
            contract[(cell_i, cell_j)] = cell_edges

    return contract

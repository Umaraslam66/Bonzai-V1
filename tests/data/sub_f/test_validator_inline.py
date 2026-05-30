"""Tests for sub-F inline validator (Task 9).

Adversarial test suite — per `gate_must_distinguish_regimes` discipline,
a validator test that only passes valid input proves nothing. The dominant
value is in the NEGATIVE tests: one per invariant, each isolating exactly
one violated rule and asserting the specific error substring.

Rules under test (spec §4.7):
  1. Schema conformance
  2. Row count (exactly 64)
  3. Empty-cell invariant: feature_count==0 ⟺ len(token_sequence)==0
  4. Derivation: cell_slot_index == cell_i*8 + cell_j
  5. Token-ID membership in actual sparse vocab (NOT merely numeric bounds)
  6. provenance_sha256 format: 64-char lowercase hex

T10 scope (NOT tested here): BP7 four-test composite, grammar well-formedness,
cross-tile sha uniqueness, all-64-cells-present across region, version manifest.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cfm.data.sub_f.io import CELLS_SCHEMA, CellRow, write_cells_parquet
from cfm.data.sub_f.validator_inline import InlineValidationError, validate_inline

# ---------------------------------------------------------------------------
# Helper: 64 fully-valid rows (empty cells — no tokens — so token-range rule
# never fires here; each negative isolates its own violation).
# ---------------------------------------------------------------------------

_GOOD_SHA = "a" * 64  # 64 lowercase hex chars — valid sha256 format


def _good_rows() -> list[CellRow]:
    """Return 64 valid CellRows (all empty cells) covering the 8x8 grid."""
    return [
        CellRow(
            cell_i=i,
            cell_j=j,
            cell_slot_index=i * 8 + j,
            token_sequence=[],
            feature_count=0,
            provenance_sha256=_GOOD_SHA,
        )
        for i in range(8)
        for j in range(8)
    ]


# ---------------------------------------------------------------------------
# Rule 1 helpers: write parquet with a deliberately wrong schema, bypassing
# write_cells_parquet (which would enforce CELLS_SCHEMA). Mirrors the
# precedent in tests/data/sub_f/test_boundary_contract.py::
# test_load_raises_on_schema_mismatch_extra_column.
# ---------------------------------------------------------------------------


def _write_wrong_schema_parquet(
    path: Path, *, extra_column: bool = False, wrong_dtype: bool = False
) -> None:
    """Write a parquet file whose schema differs from CELLS_SCHEMA."""
    rows = _good_rows()
    if extra_column:
        wrong_schema = pa.schema(
            [
                pa.field("cell_i", pa.int8(), nullable=False),
                pa.field("cell_j", pa.int8(), nullable=False),
                pa.field("cell_slot_index", pa.int8(), nullable=False),
                pa.field("token_sequence", pa.list_(pa.int16()), nullable=False),
                pa.field("feature_count", pa.int16(), nullable=False),
                pa.field("provenance_sha256", pa.string(), nullable=False),
                pa.field("drift_column", pa.string(), nullable=False),  # extra
            ]
        )
        pylist = [
            {
                "cell_i": r.cell_i,
                "cell_j": r.cell_j,
                "cell_slot_index": r.cell_slot_index,
                "token_sequence": r.token_sequence,
                "feature_count": r.feature_count,
                "provenance_sha256": r.provenance_sha256,
                "drift_column": "x",
            }
            for r in rows
        ]
    elif wrong_dtype:
        # int32 where int8 expected for cell_i
        wrong_schema = pa.schema(
            [
                pa.field("cell_i", pa.int32(), nullable=False),  # wrong dtype
                pa.field("cell_j", pa.int8(), nullable=False),
                pa.field("cell_slot_index", pa.int8(), nullable=False),
                pa.field("token_sequence", pa.list_(pa.int16()), nullable=False),
                pa.field("feature_count", pa.int16(), nullable=False),
                pa.field("provenance_sha256", pa.string(), nullable=False),
            ]
        )
        pylist = [
            {
                "cell_i": r.cell_i,
                "cell_j": r.cell_j,
                "cell_slot_index": r.cell_slot_index,
                "token_sequence": r.token_sequence,
                "feature_count": r.feature_count,
                "provenance_sha256": r.provenance_sha256,
            }
            for r in rows
        ]
    else:
        raise ValueError("must specify extra_column or wrong_dtype")

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(pylist, schema=wrong_schema)
    pq.write_table(table, path)


def _write_row_count_parquet(path: Path, n_rows: int) -> None:
    """Write a parquet with n_rows rows (bypasses write_cells_parquet's 64-row guard)."""
    # Use first n_rows of the good-rows grid (or wrap if more)
    all_rows = _good_rows()
    rows = all_rows[:n_rows]
    pylist = [
        {
            "cell_i": r.cell_i,
            "cell_j": r.cell_j,
            "cell_slot_index": r.cell_slot_index,
            "token_sequence": r.token_sequence,
            "feature_count": r.feature_count,
            "provenance_sha256": r.provenance_sha256,
        }
        for r in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(pylist, schema=CELLS_SCHEMA)
    pq.write_table(table, path)


def _write_derivation_violation_parquet(path: Path) -> None:
    """Write parquet where row 0 has cell_slot_index != cell_i*8+cell_j.

    Bypasses write_cells_parquet's derivation guard by writing directly.
    """
    rows = _good_rows()
    pylist = [
        {
            "cell_i": r.cell_i,
            "cell_j": r.cell_j,
            "cell_slot_index": r.cell_slot_index,
            "token_sequence": r.token_sequence,
            "feature_count": r.feature_count,
            "provenance_sha256": r.provenance_sha256,
        }
        for r in rows
    ]
    # Break derivation for row 0 (cell_i=0, cell_j=0 → expected index=0, set to 99)
    pylist[0]["cell_slot_index"] = 99
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(pylist, schema=CELLS_SCHEMA)
    pq.write_table(table, path)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_validator_accepts_good_parquet(tmp_path: Path) -> None:
    """Happy path: 64 valid empty-cell rows — validator must not raise."""
    path = tmp_path / "cells.parquet"
    write_cells_parquet(path, _good_rows())
    validate_inline(path)  # no exception


def test_validator_accepts_non_empty_cells(tmp_path: Path) -> None:
    """Happy path: cells with tokens — feature_count>0 and token_sequence non-empty.

    Uses a real on-disk vocab id so the token-membership check also passes.
    """
    from cfm.data.sub_f.vocab import load_sub_f_vocab

    vocab = load_sub_f_vocab()
    real_token_id = vocab[0].token_id  # e.g. id=0, a BP1 semantic slot

    rows = _good_rows()
    # Replace cell (0,0) with a non-empty cell using a valid token
    rows[0] = CellRow(
        cell_i=0,
        cell_j=0,
        cell_slot_index=0,
        token_sequence=[real_token_id],
        feature_count=1,
        provenance_sha256=_GOOD_SHA,
    )
    path = tmp_path / "cells.parquet"
    write_cells_parquet(path, rows)
    validate_inline(path)  # no exception


# ---------------------------------------------------------------------------
# Rule 1: Schema conformance
# ---------------------------------------------------------------------------


def test_validator_rejects_extra_column(tmp_path: Path) -> None:
    """Rule 1 — schema: extra column → raises InlineValidationError matching 'schema'."""
    path = tmp_path / "cells.parquet"
    _write_wrong_schema_parquet(path, extra_column=True)
    with pytest.raises(InlineValidationError, match=r"schema"):
        validate_inline(path)


def test_validator_rejects_wrong_dtype(tmp_path: Path) -> None:
    """Rule 1 — schema: int32 where int8 expected → raises matching 'schema'."""
    path = tmp_path / "cells.parquet"
    _write_wrong_schema_parquet(path, wrong_dtype=True)
    with pytest.raises(InlineValidationError, match=r"schema"):
        validate_inline(path)


# ---------------------------------------------------------------------------
# Rule 2: Row count
# ---------------------------------------------------------------------------


def test_validator_rejects_too_few_rows(tmp_path: Path) -> None:
    """Rule 2 — row count: 60 rows (not 64) → raises matching 'row count' / 'expected 64'."""
    path = tmp_path / "cells.parquet"
    _write_row_count_parquet(path, n_rows=60)
    with pytest.raises(InlineValidationError, match=r"row count|expected 64"):
        validate_inline(path)


def test_validator_rejects_too_many_rows(tmp_path: Path) -> None:
    """Rule 2 — row count: 0 rows → raises matching 'row count' / 'expected 64'."""
    path = tmp_path / "cells.parquet"
    _write_row_count_parquet(path, n_rows=0)
    with pytest.raises(InlineValidationError, match=r"row count|expected 64"):
        validate_inline(path)


# ---------------------------------------------------------------------------
# Rule 3: Empty-cell invariant
# ---------------------------------------------------------------------------


def test_validator_rejects_tokens_with_zero_feature_count(tmp_path: Path) -> None:
    """Rule 3 — empty-cell: feature_count=0 but token_sequence non-empty → raises 'empty-cell'.

    NOTE: token_sequence is non-empty; we use a valid token id (no token-range
    co-violation) so only the empty-cell rule fires. feature_count IS 0 to
    create the mismatch.
    """
    from cfm.data.sub_f.vocab import load_sub_f_vocab

    vocab = load_sub_f_vocab()
    real_token_id = vocab[0].token_id

    rows = _good_rows()
    rows[0] = CellRow(
        cell_i=0,
        cell_j=0,
        cell_slot_index=0,
        token_sequence=[real_token_id],  # non-empty
        feature_count=0,  # mismatch: says 0 features but has tokens
        provenance_sha256=_GOOD_SHA,
    )
    path = tmp_path / "cells.parquet"
    write_cells_parquet(path, rows)
    with pytest.raises(InlineValidationError, match=r"empty-cell"):
        validate_inline(path)


def test_validator_rejects_feature_count_with_empty_sequence(tmp_path: Path) -> None:
    """Rule 3 — empty-cell (converse): feature_count>0 + empty sequence → raises 'empty-cell'."""
    rows = _good_rows()
    rows[0] = CellRow(
        cell_i=0,
        cell_j=0,
        cell_slot_index=0,
        token_sequence=[],  # empty sequence
        feature_count=3,  # mismatch: claims features but no tokens
        provenance_sha256=_GOOD_SHA,
    )
    path = tmp_path / "cells.parquet"
    write_cells_parquet(path, rows)
    with pytest.raises(InlineValidationError, match=r"empty-cell"):
        validate_inline(path)


# ---------------------------------------------------------------------------
# Rule 4: Derivation
# ---------------------------------------------------------------------------


def test_validator_rejects_derivation_violation(tmp_path: Path) -> None:
    """Rule 4 — derivation: cell_slot_index != cell_i*8+cell_j → raises 'derivation'.

    Written directly via pyarrow to bypass write_cells_parquet's own guard.
    """
    path = tmp_path / "cells.parquet"
    _write_derivation_violation_parquet(path)
    with pytest.raises(InlineValidationError, match=r"derivation"):
        validate_inline(path)


# ---------------------------------------------------------------------------
# Rule 5: Token-ID membership (the STRICT sparse-vocab interpretation)
# ---------------------------------------------------------------------------


def test_validator_rejects_token_out_of_namespace(tmp_path: Path) -> None:
    """Rule 5 — token: id=9999 (> max vocab id 1507) → raises matching 'token'.

    This is the straightforward out-of-bounds case.
    """
    rows = _good_rows()
    rows[0] = CellRow(
        cell_i=0,
        cell_j=0,
        cell_slot_index=0,
        token_sequence=[9999],
        feature_count=1,  # consistent with non-empty sequence
        provenance_sha256=_GOOD_SHA,
    )
    path = tmp_path / "cells.parquet"
    write_cells_parquet(path, rows)
    with pytest.raises(InlineValidationError, match=r"token"):
        validate_inline(path)


def test_validator_rejects_token_in_namespace_gap(tmp_path: Path) -> None:
    """Rule 5 — token: id=1000 is within [0,1599] but NOT in the vocab (gap) → raises 'token'.

    This proves the validator uses vocab MEMBERSHIP not numeric bounds.
    """
    from cfm.data.sub_f.vocab import load_sub_f_vocab

    vocab_ids = frozenset(s.token_id for s in load_sub_f_vocab())
    assert 1000 not in vocab_ids, "1000 must be a gap id for this test to be meaningful"

    rows = _good_rows()
    rows[0] = CellRow(
        cell_i=0,
        cell_j=0,
        cell_slot_index=0,
        token_sequence=[1000],
        feature_count=1,
        provenance_sha256=_GOOD_SHA,
    )
    path = tmp_path / "cells.parquet"
    write_cells_parquet(path, rows)
    with pytest.raises(InlineValidationError, match=r"token"):
        validate_inline(path)


def test_validator_rejects_retired_direction_token(tmp_path: Path) -> None:
    """Rule 5 — token: id=420 is in retired direction range 396-443 (Halt-2 relocation).

    420 is within [0,1599] but is NOT in the vocab after the Halt-2 N=360
    re-lock (direction block relocated from 396-443 to 511-870). This test
    proves the validator enforces the Halt-2 change, not just a numeric bound.

    The only legitimate appearance of 396/420/443 here is as a REJECTED input
    (a negative test fixture). 420 must NOT be a valid vocab id — confirmed in
    Step 0 of the task brief.
    """
    from cfm.data.sub_f.vocab import load_sub_f_vocab

    vocab_ids = frozenset(s.token_id for s in load_sub_f_vocab())
    # Hard assertion: if 420 somehow got back into the vocab, this test is
    # pointing at a real regression (Halt-2 relocation didn't land). STOP.
    assert 420 not in vocab_ids, (
        "HALT: token id 420 is in the vocab — retired direction range 396-443 "
        "must not be present after Halt-2 N=360 relocation. Lock regression."
    )

    rows = _good_rows()
    rows[0] = CellRow(
        cell_i=0,
        cell_j=0,
        cell_slot_index=0,
        token_sequence=[420],  # retired direction token — must be rejected
        feature_count=1,
        provenance_sha256=_GOOD_SHA,
    )
    path = tmp_path / "cells.parquet"
    write_cells_parquet(path, rows)
    with pytest.raises(InlineValidationError, match=r"token"):
        validate_inline(path)


# ---------------------------------------------------------------------------
# Rule 6: provenance_sha256 format
# ---------------------------------------------------------------------------


def test_validator_rejects_sha_wrong_length(tmp_path: Path) -> None:
    """Rule 6 — sha: 32-char hex (too short) → raises matching 'sha|provenance'."""
    rows = _good_rows()
    rows[0] = CellRow(
        cell_i=0,
        cell_j=0,
        cell_slot_index=0,
        token_sequence=[],
        feature_count=0,
        provenance_sha256="a" * 32,  # 32 chars, not 64
    )
    path = tmp_path / "cells.parquet"
    write_cells_parquet(path, rows)
    with pytest.raises(InlineValidationError, match=r"sha|provenance"):
        validate_inline(path)


def test_validator_rejects_sha_non_hex(tmp_path: Path) -> None:
    """Rule 6 — sha: uppercase / non-hex chars → raises matching 'sha|provenance'."""
    rows = _good_rows()
    rows[0] = CellRow(
        cell_i=0,
        cell_j=0,
        cell_slot_index=0,
        token_sequence=[],
        feature_count=0,
        provenance_sha256="Z" * 64,  # not lowercase hex
    )
    path = tmp_path / "cells.parquet"
    write_cells_parquet(path, rows)
    with pytest.raises(InlineValidationError, match=r"sha|provenance"):
        validate_inline(path)

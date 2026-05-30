"""Sub-F per-cell inline validator.

Per spec §4.7 inline checks (ALL six):
1. Schema conformance — table.schema == CELLS_SCHEMA (all declared columns at
   declared types; any extra/missing column or wrong dtype raises).
2. Row count — exactly EXPECTED_ROWS_PER_TILE (64) rows.
3. Empty-cell invariant — feature_count == 0 ⟺ len(token_sequence) == 0.
4. Derivation (load-bearing) — cell_slot_index == cell_i * 8 + cell_j.
5. Token-ID membership — every id in every token_sequence must be a member of
   the actual sparse sub-F vocab (NOT merely 0 <= id <= 1599). Tokens in
   namespace gaps or the retired direction range 396-443 (Halt-2 relocation,
   2026-05-29) are explicitly rejected.
6. provenance_sha256 format — 64-char lowercase hex ^[0-9a-f]{64}$.

Checks are ordered schema → row-count → per-row, so structural failures surface
before per-row iteration.

T10 scope (NOT here): BP7 four-test composite, grammar well-formedness,
cross-tile sha uniqueness, all-64-cells-present across region, version
manifest consistency. Those belong in validator_cross_tile.py (Task 10).
"""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow.parquet as pq

from cfm.data.sub_f.io import CELLS_SCHEMA, EXPECTED_ROWS_PER_TILE
from cfm.data.sub_f.vocab import load_sub_f_vocab

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class InlineValidationError(ValueError):
    """Raised when a sub-F cells.parquet fails any inline invariant."""


def validate_inline(parquet_path: Path) -> None:
    """Validate a cells.parquet against all spec §4.7 inline invariants.

    Reads via pq.ParquetFile(path).read() per feedback_pyarrow_hive_partition_inference
    (avoids spurious `tile` column injection from pq.read_table on Hive-partitioned dirs).

    Raises InlineValidationError on the FIRST violation found, with a message
    containing a rule-specific substring (schema / row count / empty-cell /
    derivation / token / sha) so callers can identify which rule fired.
    """
    table = pq.ParquetFile(parquet_path).read()

    # ------------------------------------------------------------------
    # Check 1: Schema conformance
    # Must run BEFORE row-count + per-row checks — wrong schema means
    # column names / types are undefined; iterating rows would be unsafe.
    # ------------------------------------------------------------------
    actual_schema = table.schema
    # Remove metadata (e.g. pandas metadata) before comparing — only
    # column names + types + nullability matter.
    actual_no_meta = actual_schema.remove_metadata()
    expected_no_meta = CELLS_SCHEMA.remove_metadata()
    if actual_no_meta != expected_no_meta:
        raise InlineValidationError(
            f"schema mismatch: expected {expected_no_meta}, got {actual_no_meta}"
        )

    # ------------------------------------------------------------------
    # Check 2: Row count
    # ------------------------------------------------------------------
    n_rows = len(table)
    if n_rows != EXPECTED_ROWS_PER_TILE:
        raise InlineValidationError(f"row count: expected {EXPECTED_ROWS_PER_TILE}, got {n_rows}")

    # ------------------------------------------------------------------
    # Per-row checks 3-6
    # Build vocab membership set once (lru_cache on load_sub_f_vocab means
    # no re-parse cost on repeated calls within same process).
    # ------------------------------------------------------------------
    vocab_ids: frozenset[int] = frozenset(s.token_id for s in load_sub_f_vocab())

    rows = table.to_pylist()
    for r in rows:
        ci, cj = r["cell_i"], r["cell_j"]

        # Check 3: Empty-cell invariant
        has_tokens = len(r["token_sequence"]) > 0
        has_features = r["feature_count"] > 0
        if has_tokens != has_features:
            raise InlineValidationError(
                f"empty-cell invariant violated at ({ci},{cj}): "
                f"feature_count={r['feature_count']} "
                f"token_sequence_len={len(r['token_sequence'])}"
            )

        # Check 4: Derivation (load-bearing)
        expected_slot = ci * 8 + cj
        if r["cell_slot_index"] != expected_slot:
            raise InlineValidationError(
                f"derivation check failed at ({ci},{cj}): "
                f"cell_slot_index={r['cell_slot_index']} != cell_i*8+cell_j={expected_slot}"
            )

        # Check 5: Token-ID membership (sparse vocab; NOT just numeric bounds)
        for tok_id in r["token_sequence"]:
            if tok_id not in vocab_ids:
                raise InlineValidationError(
                    f"token ID {tok_id} not in sub-F vocab at ({ci},{cj}) — "
                    "check for namespace gaps or retired direction range 396-443"
                )

        # Check 6: provenance_sha256 format
        if not _SHA256_RE.match(r["provenance_sha256"]):
            raise InlineValidationError(
                f"provenance_sha256 format invalid at ({ci},{cj}): "
                f"expected 64-char lowercase hex, got {r['provenance_sha256'][:20]!r}"
            )

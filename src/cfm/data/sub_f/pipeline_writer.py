"""Sub-F per-tile encode orchestrator.

Glues T8.1-T8.7 together:
  features.parquet (sub-C) + boundary_contract.parquet (sub-E)
    -> per-cell token sequences
    -> cells.parquet (via write_cells_parquet)

Provenance + region manifest write are out of scope for T8.8 — Task 11
(pipeline orchestrator) wires the per-tile encode + provenance + manifest +
SUCCESS marker.

BP7 emission — UNVERIFIED against real sub-E parquet; integration test
test_encode_tile_against_real_sub_e_singapore is @pytest.mark.skip pending
sub-E cache regeneration. See close-checklist.
"""

from __future__ import annotations

import hashlib
import struct
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from shapely.wkb import loads as wkb_loads

from cfm.data.sub_f.boundary_contract import load_boundary_contract
from cfm.data.sub_f.encoder import encode_cell
from cfm.data.sub_f.io import CellRow, write_cells_parquet

# Sub-C feature_class -> primary key for semantic_tag construction.
# Sub-F-v1 scope: highway (0) + building (1) get first-class semantic tags.
# POI (2) and base (3) use BP4 fallback per cascade #4 deferral.
_FEATURE_CLASS_TO_KEY: dict[int, str] = {
    0: "highway",
    1: "building",
    2: "amenity",  # sub-C feature_class=2 lumps poi categories; BP4 fallback
    3: "natural",  # sub-C feature_class=3 lumps base; BP4 fallback
}


def _semantic_tag_from_row(row: dict) -> str:
    """Build "key=value" from sub-C row.

    Reads row keys:
      - ``feature_class`` (int): expected values 0=highway, 1=building,
        2=poi (maps to "amenity"), 3=base (maps to "natural"). Any other
        value raises ``ValueError``.
      - ``class_raw`` (str | None): the raw OSM class tag value; may be
        ``None`` or a sentinel; treated as empty string in either case.
    """
    fc = int(row["feature_class"])
    key = _FEATURE_CLASS_TO_KEY.get(fc)
    if key is None:
        raise ValueError(
            f"_semantic_tag_from_row: unknown feature_class {fc!r} "
            f"(expected 0=highway, 1=building, 2=poi, 3=base)"
        )
    value = row.get("class_raw") or ""
    return f"{key}={value}"


def encode_tile(
    sub_c_features_parquet: Path,
    sub_e_boundary_contract_parquet: Path,
    out_cells_parquet: Path,
) -> Path:
    """Encode one tile's features into a cells.parquet.

    Reads:
      - sub-C features.parquet (cell-keyed feature rows)
      - sub-E boundary_contract.parquet (per-edge class map)

    Writes:
      - cells.parquet with 64 rows (8x8 grid; empty cells get tokens=[]).

    Per `feedback_pyarrow_hive_partition_inference`: both reads use
    `pq.ParquetFile(path).read()`.
    """
    features_table = pq.ParquetFile(sub_c_features_parquet).read()
    contract = load_boundary_contract(sub_e_boundary_contract_parquet)

    # Group sub-C features by (cell_i, cell_j) preserving sub-C row order
    # per §3.3 + Halt 5 spec §5.2 row "OSM feature iteration order".
    per_cell_features: dict[tuple[int, int], list[tuple[object, str]]] = defaultdict(list)
    for r in features_table.to_pylist():
        cell_key = (int(r["cell_i"]), int(r["cell_j"]))
        geom = wkb_loads(r["geometry"])
        semantic_tag = _semantic_tag_from_row(r)
        per_cell_features[cell_key].append((geom, semantic_tag))

    rows: list[CellRow] = []
    for cell_i in range(8):
        for cell_j in range(8):
            features = per_cell_features.get((cell_i, cell_j), [])
            cell_edges = contract.get((cell_i, cell_j), {})
            encoded = encode_cell(features=features, cell_edges=cell_edges)
            # Per spec §6 sha-anchor: provenance_sha256 computed at provenance
            # write time (Task 11). For T8.8 we use a stub sha so cells.parquet
            # is well-formed; Task 11 overwrites.
            # Token IDs exceed 255 so we cannot use bytes() directly; pack each
            # token as big-endian uint16 (2 bytes) for a stable byte string.
            # DECISION: big-endian uint16 per token is the simplest lossless
            # encoding; this stub sha is replaced wholesale by Task 11.
            token_bytes = (
                struct.pack(f">{len(encoded.tokens)}H", *encoded.tokens) if encoded.tokens else b""
            )
            stub_sha = hashlib.sha256(token_bytes + bytes([cell_i, cell_j])).hexdigest()
            rows.append(
                CellRow(
                    cell_i=cell_i,
                    cell_j=cell_j,
                    cell_slot_index=cell_i * 8 + cell_j,
                    token_sequence=encoded.tokens,
                    feature_count=encoded.feature_count,
                    provenance_sha256=stub_sha,
                )
            )

    return write_cells_parquet(out_cells_parquet, rows)

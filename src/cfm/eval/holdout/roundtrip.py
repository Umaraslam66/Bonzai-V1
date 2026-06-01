"""Round-tripped-real geometry: decode sub-F's already-emitted tokens.

sub-F's cells.parquet IS the encoded real tiles. Round-tripped-real geometry =
decode_feature(block) over each feature block. We reuse sub-F's decoder and
sub-G's feature splitter so the round-trip is byte-identical to what sub-G's
check_decodability validated (one source). We never re-encode.

Each decoded block is paired with its cell's cell_density_bucket so the shared
bref-rate (§2) and the reference distributions can stratify at cell granularity
(D's stratification key - density is an aggregate, see labels.py).
"""

from __future__ import annotations

from typing import Any

from cfm.data.sub_f.decoder import decode_feature
from cfm.data.sub_g.seam_decodability import split_cell_into_features


def decode_region_blocks(
    tokens_by_cell: dict[tuple[int, int], list[int]],
    cell_density_by_cell: dict[tuple[int, int], int],
) -> tuple[list[list[int]], list[dict[str, Any]], list[int]]:
    """Return aligned (blocks, decoded_geoms, strata) for one tile/region.

    A cell with no recorded cell_density_bucket is skipped (not bucketed as 0).
    """
    blocks: list[list[int]] = []
    geoms: list[dict[str, Any]] = []
    strata: list[int] = []
    for cell, token_sequence in sorted(tokens_by_cell.items()):
        stratum = cell_density_by_cell.get(cell)
        if stratum is None:
            continue
        for block in split_cell_into_features(token_sequence):
            blocks.append(block)
            geoms.append(decode_feature(block))
            strata.append(int(stratum))
    return blocks, geoms, strata

from __future__ import annotations

from cfm.eval.holdout import roundtrip

# A Case-A feature block (no bref): <feature>=509 ... <feature_end>=510.
# Verified 2026-06-01 to decode to a 2-distinct-vertex LineString via decode_feature.
_SIMPLE_BLOCK = [509, 41, 300, 323, 363, 369, 1, 50, 510]


def test_decode_blocks_pairs_each_block_with_its_cell_stratum():
    tokens_by_cell = {(0, 0): _SIMPLE_BLOCK, (1, 0): _SIMPLE_BLOCK}
    cell_density_by_cell = {(0, 0): 3, (1, 0): 0}

    blocks, geoms, strata = roundtrip.decode_region_blocks(tokens_by_cell, cell_density_by_cell)

    assert len(blocks) == len(geoms) == len(strata) == 2
    # geoms are GeoJSON dicts (decode_feature output), blocks retain token provenance:
    assert all(isinstance(g, dict) and "type" in g for g in geoms)
    assert all(b[0] == 509 and b[-1] == 510 for b in blocks)
    assert sorted(strata) == [0, 3]


def test_decode_skips_cells_without_a_density_bucket():
    # A cell with no recorded cell_density_bucket (masked/non-active) is dropped from
    # the stratified stream rather than silently bucketed as 0.
    blocks, geoms, strata = roundtrip.decode_region_blocks(
        {(2, 2): _SIMPLE_BLOCK}, cell_density_by_cell={}
    )
    assert blocks == [] and geoms == [] and strata == []

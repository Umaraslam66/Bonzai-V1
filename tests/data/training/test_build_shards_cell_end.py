"""Tooth 2 (cell-EOS, CPU): build_shards_in_memory injects <cell_end>=260 at the
end of every NON-EMPTY cell, exactly once, never interior, and NEVER on an empty
cell (the empty-guard). Also asserts every non-empty cell ends (..., 510, 260) —
so Tooth 3's synthetic 510->260 loss fixture is REPRESENTATIVE of real cell-ends,
not assumed.

This is the real proof of spec change B: the synthetic golden fixture in
test_shard_cache.py bypasses build_shards_in_memory (hand-built shards), so the
260 byte-change is guarded HERE, against the real builder, not by the golden.
"""

from __future__ import annotations

from cfm.data.sub_f.vocab import CELL_END_TOKEN_ID
from cfm.data.training import build_shards as BS
from cfm.eval.holdout.labels import MorphologyStratum, TileLabels

_RELEASE = "2026-04-15.0"
_REGION = "singapore"


def _fake_labels(tile_dir, *, tile_i, tile_j):
    return TileLabels(
        tile_i=tile_i,
        tile_j=tile_j,
        population_density_bucket=0,
        cell_density_buckets=(),
        morphology_stratum=MorphologyStratum(dominant_zoning_class=0, modal_road_skeleton_class=1),
        coastal_inland_river=0,
        admin_region=None,
        sub_c_morphology_class=None,
    )


def _patch_builder(monkeypatch, cells_by_id):
    """Stub every path-consuming read so build_shards_in_memory runs on CPU with no
    on-disk tiles; character_stats are irrelevant here (-> {}, every cell absent)."""
    monkeypatch.setattr(BS, "epsg_label_for_region", lambda region: "EPSG3414")
    monkeypatch.setattr(BS, "read_tile_labels", _fake_labels)
    monkeypatch.setattr(BS, "_cell_density_by_cell", lambda tile_dir: {})
    monkeypatch.setattr(BS, "derive_character_stats", lambda p: {})
    monkeypatch.setattr(BS, "read_sub_f_cells", lambda p: dict(cells_by_id))


def test_build_appends_cell_end_at_nonempty_cell_end_only(monkeypatch):
    # (0,0): two feature blocks, ends in 510 (a real cell-end shape).
    # (1,0): a single feature block, ends in 510.
    # (2,0): EMPTY cell -> must stay () (empty-guard), no 260.
    _patch_builder(
        monkeypatch,
        {
            (0, 0): [509, 1, 2, 510, 509, 3, 510],
            (1, 0): [509, 4, 510],
            (2, 0): [],
        },
    )

    [shard] = BS.build_shards_in_memory(_RELEASE, _REGION, tile_ids=[(0, 0)])
    by_cell = {(c.cell_i, c.cell_j): c.tokens for c in shard.cells}

    for cid in ((0, 0), (1, 0)):
        toks = by_cell[cid]
        assert toks[-1] == CELL_END_TOKEN_ID, f"{cid}: non-empty cell must end in 260"
        assert toks[-2] == 510, f"{cid}: non-empty cell must end (..., 510, 260)"
        assert toks.count(CELL_END_TOKEN_ID) == 1, f"{cid}: exactly one 260"
        assert CELL_END_TOKEN_ID not in toks[:-1], f"{cid}: 260 never interior"

    # empty-guard: an empty cell keeps () — a blanket append would have made it (260,)
    assert by_cell[(2, 0)] == (), "empty cell must stay () (no 260)"


def test_build_cell_end_is_deterministic(monkeypatch):
    """The 260 injection is pure: same input -> byte-identical cell tokens twice."""
    cells = {(0, 0): [509, 7, 510], (1, 0): []}
    _patch_builder(monkeypatch, cells)
    [a] = BS.build_shards_in_memory(_RELEASE, _REGION, tile_ids=[(0, 0)])
    _patch_builder(monkeypatch, cells)
    [b] = BS.build_shards_in_memory(_RELEASE, _REGION, tile_ids=[(0, 0)])
    assert tuple(c.tokens for c in a.cells) == tuple(c.tokens for c in b.cells)

from __future__ import annotations

import ast
import inspect

from cfm.eval.holdout import selector
from cfm.eval.holdout.labels import MorphologyStratum, TileLabels


def _tile(i: int, j: int, *, pdb: int, cells: list[int], zon: int, sk: int) -> TileLabels:
    return TileLabels(
        tile_i=i,
        tile_j=j,
        population_density_bucket=pdb,
        cell_density_buckets=tuple(cells),
        morphology_stratum=MorphologyStratum(
            dominant_zoning_class=zon, modal_road_skeleton_class=sk
        ),
        coastal_inland_river=1,
        admin_region="Central Region",
        sub_c_morphology_class="Asian-megacity",
    )


def test_selection_is_deterministic_and_tie_breaks_lexicographically():
    tiles = [
        _tile(2, 1, pdb=0, cells=[0, 0], zon=0, sk=0),
        _tile(1, 1, pdb=0, cells=[0, 0], zon=0, sk=0),
    ]
    quotas = {(0, (0, 0)): 1}
    r1 = selector.select_holdout_tiles(tiles, quotas, cell_density_floor={0: 2})
    r2 = selector.select_holdout_tiles(tiles, quotas, cell_density_floor={0: 2})
    assert r1.selected == r2.selected
    assert r1.selected == [(1, 1)]  # lexicographically-smallest fills the quota


def test_GUARD_underpowered_stratum_surfaces_not_silently_dropped():
    """#11 failed by SILENTLY skipping the sparse side. The fresh selector must
    FAIL LOUD: a cell-density stratum whose selected cells fall below its floor is
    reported as UNDERPOWERED, never omitted-and-called-success."""
    tiles = [_tile(1, 1, pdb=0, cells=[0, 0], zon=0, sk=0)]
    quotas = {(0, (0, 0)): 1}
    res = selector.select_holdout_tiles(tiles, quotas, cell_density_floor={0: 2, 3: 5})
    assert res.selected == [(1, 1)]
    assert 3 in res.underpowered_cell_density_strata  # surfaced, not dropped
    sf = res.underpowered_cell_density_strata[3]
    assert sf.available < sf.floor


def test_GUARD_unfillable_tile_quota_surfaces_as_underpowered():
    tiles = [_tile(1, 1, pdb=0, cells=[0], zon=0, sk=0)]
    quotas = {(0, (0, 0)): 3}  # quota 3 but only 1 tile in the stratum
    res = selector.select_holdout_tiles(tiles, quotas, cell_density_floor={0: 1})
    assert (0, (0, 0)) in res.underpowered_tile_strata
    assert res.underpowered_tile_strata[(0, (0, 0))].available == 1


def test_consumes_labels_only_no_rederivation():
    # The selector must consume sub-D-derived TileLabels (one source) and never import
    # a sub-D derivation module (evidence / frequency_analysis). AST import-surface
    # check (robust; not a substring scan that would trip on docstrings).
    tree = ast.parse(inspect.getsource(selector))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any("evidence" in m or "frequency_analysis" in m for m in imported), (
        f"selector must not import sub-D derivation code; imports={imported}"
    )
    assert "cfm.eval.holdout.labels" in imported  # consumes the one-source labels

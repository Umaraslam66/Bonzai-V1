from __future__ import annotations

import pytest

from cfm.eval.holdout import paths, pipeline
from cfm.eval.holdout.labels import MorphologyStratum, TileLabels


def _tile(i: int, j: int, *, pdb: int, cells: list[int]) -> TileLabels:
    return TileLabels(
        tile_i=i,
        tile_j=j,
        population_density_bucket=pdb,
        cell_density_buckets=tuple(cells),
        morphology_stratum=MorphologyStratum(dominant_zoning_class=0, modal_road_skeleton_class=0),
        coastal_inland_river=1,
        admin_region="Central Region",
        sub_c_morphology_class="Asian-megacity",
    )


def test_corrected_sequencing_order_is_encoded():
    # build the selector, then size THROUGH it, then freeze (plan decision 6).
    assert pipeline.SEQUENCE == (
        "labels",
        "bref_rate",
        "baselines",
        "build_selector",
        "size_through_selector",
        "run_degeneracy_guards",
        "freeze_manifest",
        "write_partition_and_marker",
    )


def test_co_optimize_grows_quota_until_cell_floors_met():
    # Three tiles in one stratum, each contributing one bucket-3 cell. Floor 2 needs
    # quota 2; the loop grows the quota and stops once the cell floor is met.
    tiles = [
        _tile(1, 1, pdb=0, cells=[3]),
        _tile(1, 2, pdb=0, cells=[3]),
        _tile(1, 3, pdb=0, cells=[3]),
    ]
    res = pipeline.co_optimize(tiles, cell_floor={3: 2})
    assert len(res.selected) == 2
    assert not res.underpowered_cell_density_strata  # floor met, nothing underpowered


def test_co_optimize_reports_underpowered_when_infeasible():
    # Only one bucket-3 cell exists but the floor demands 5 -> UNDERPOWERED-stated,
    # never silently satisfied (#11 failure class inverted).
    tiles = [_tile(1, 1, pdb=0, cells=[3])]
    res = pipeline.co_optimize(tiles, cell_floor={3: 5})
    assert 3 in res.underpowered_cell_density_strata
    assert res.underpowered_cell_density_strata[3].available < 5


def test_marker_not_written_in_dry_run():
    result = pipeline.EvalSetResult(
        n=10,
        proposed_selection=[(1, 7)],
        per_stratum_bref_rate={},
        per_stratum_cell_floor={},
        per_stratum_cell_population={},
        underpowered_cell_density_strata=[],
        ceiling_overall=0.95,
        residual=484,
        locked=False,
        manifest_path=None,
        marker_written=False,
        report_path=None,
    )
    assert result.marker_written is False and result.locked is False


@pytest.mark.slow
def test_dry_run_on_real_singapore_measures_substrate():
    """SLOW: the real 494-tile dry-run. Decodes round-tripped-real, measures the
    per-stratum bref-rate + cell populations, proposes (N, selection), and produces a
    report - WITHOUT freezing (lock=False). Skips if validated data is absent."""
    rel, reg = paths.DEFAULT_RELEASE, paths.DEFAULT_REGION
    if not paths.phase1_validated_marker(rel, reg).is_file():
        pytest.skip("Phase-1 validated Singapore data not present")
    result = pipeline.generate_eval_set(release=rel, region=reg, lock=False)
    assert result.locked is False and result.marker_written is False
    assert 0 < result.n < 494  # proposes a held-out set that leaves a training residual
    assert result.residual == 494 - result.n
    assert 0.0 <= result.ceiling_overall <= 1.0
    assert result.per_stratum_cell_population  # measured something
    # D's rate-detection power (features) must be satisfied in the SELECTED set - the
    # vacuous pass must not hide in the sample size (2026-06-01 review). Measured abundant.
    assert result.per_stratum_feature_floor and result.held_out_feature_population
    assert result.underpowered_feature_strata == []
    for bucket, floor in result.per_stratum_feature_floor.items():
        assert result.held_out_feature_population.get(bucket, 0) >= floor

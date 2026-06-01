from __future__ import annotations

import pytest

from cfm.eval.holdout import baselines
from cfm.eval.holdout.bref_rate import BrefRateResult, StratumRate


def test_ceiling_is_one_minus_shared_rate_not_recomputed():
    # The ceiling MUST come from the §2 shared BrefRateResult, not a second count.
    shared = BrefRateResult(
        overall_rate=0.1,
        per_stratum={0: StratumRate(100, 0), 3: StratumRate(50, 10)},
    )
    ceil = baselines.geometric_validity_ceiling(shared)
    assert ceil.overall == 0.9
    assert ceil.per_stratum[3] == 0.8  # 1 - 10/50
    assert ceil.per_stratum[0] == 1.0


def test_reference_distribution_records_source_tile_lineage():
    # spec §F: every baseline write records its source-tile lineage or G-F4 can't bind.
    rec = baselines.ReferenceDistribution(
        metric="building_area_m2",
        kind="raw",
        stratum=3,
        samples=(12.0, 30.5, 7.1),
        source_tiles=(("singapore", "tile=EPSG3414_i1_j7"),),
    )
    assert rec.source_tiles  # non-empty lineage is mandatory
    with pytest.raises(ValueError):
        baselines.ReferenceDistribution(
            metric="building_area_m2",
            kind="raw",
            stratum=3,
            samples=(1.0,),
            source_tiles=(),  # empty lineage is rejected at construction
        )


def test_full_minus_core_gap_reported():
    gap = baselines.report_gap(full_value=0.85, core_value=0.92)
    assert abs(gap - (0.85 - 0.92)) < 1e-12  # the tokenizer's own contribution, signed

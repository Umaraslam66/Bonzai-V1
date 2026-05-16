from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from cfm.data.frequency import (
    CutBehaviorRow,
    FieldFrequencyResult,
    FloorStrategy,
    ListLengthDistribution,
)


def test_field_frequency_result_is_frozen() -> None:
    r = FieldFrequencyResult(
        field="buildings.class",
        n_total=10,
        n_present=8,
        counts={"residential": 5, "commercial": 3},
        is_list_field=False,
        total_occurrences=8,
    )
    with pytest.raises(FrozenInstanceError):
        r.n_total = 99  # type: ignore[misc]


def test_floor_strategy_percentage_can_be_none() -> None:
    s = FloorStrategy(name="Very lenient", percentage=None, hard_min=10)
    assert s.percentage is None
    assert s.hard_min == 10


def test_cut_behavior_row_holds_strategy_and_metrics() -> None:
    s = FloorStrategy(name="Lenient", percentage=0.0003, hard_min=30)
    row = CutBehaviorRow(
        strategy=s,
        effective_floor=30,
        n_total_categories=20,
        n_kept=15,
        n_dropped=5,
        coverage_retained_pct=99.4,
    )
    assert row.strategy.name == "Lenient"
    assert row.effective_floor == 30


def test_list_length_distribution_buckets_keys() -> None:
    d = ListLengthDistribution(
        field="places.categories.alternate",
        n_total_rows=100,
        buckets={
            "0": (40, 40.0),
            "1": (30, 30.0),
            "2": (15, 15.0),
            "3": (10, 10.0),
            "4": (3, 3.0),
            "5+": (2, 2.0),
        },
    )
    assert set(d.buckets.keys()) == {"0", "1", "2", "3", "4", "5+"}

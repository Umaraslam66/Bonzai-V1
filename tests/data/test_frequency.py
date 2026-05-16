from __future__ import annotations

from dataclasses import FrozenInstanceError

import pyarrow as pa
import pytest

from cfm.data.frequency import (
    CutBehaviorRow,
    FieldFrequencyResult,
    FloorStrategy,
    ListLengthDistribution,
    compute_field_frequencies,
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


# ---------------------------------------------------------------------------
# compute_field_frequencies — scalar branch
# ---------------------------------------------------------------------------


def _scalar_table(values: list[str | None]) -> pa.Table:
    return pa.table({"class": pa.array(values, type=pa.string())})


def test_compute_field_frequencies_basic() -> None:
    table = _scalar_table(["residential", "residential", "commercial", "residential", None])
    result = compute_field_frequencies(table, "class", label="buildings.class")
    assert result.field == "buildings.class"
    assert result.n_total == 5
    assert result.n_present == 4
    assert result.counts == {"residential": 3, "commercial": 1}
    assert result.is_list_field is False
    assert result.total_occurrences == 4


def test_compute_field_frequencies_label_defaults_to_column_path() -> None:
    table = _scalar_table(["a", "b"])
    result = compute_field_frequencies(table, "class")
    assert result.field == "class"


def test_compute_field_frequencies_all_null() -> None:
    table = _scalar_table([None, None, None])
    result = compute_field_frequencies(table, "class")
    assert result.n_total == 3
    assert result.n_present == 0
    assert result.counts == {}
    assert result.total_occurrences == 0


def test_compute_field_frequencies_single_value() -> None:
    table = _scalar_table(["only_one", "only_one", "only_one"])
    result = compute_field_frequencies(table, "class")
    assert result.n_present == 3
    assert result.counts == {"only_one": 3}


def test_compute_field_frequencies_missing_column_raises() -> None:
    table = _scalar_table(["a"])
    with pytest.raises(ValueError, match="not found"):
        compute_field_frequencies(table, "nonexistent")


# ---------------------------------------------------------------------------
# compute_field_frequencies — struct + list branch
# ---------------------------------------------------------------------------


def _struct_with_list_table(
    primaries: list[str | None],
    alternates: list[list[str] | None],
) -> pa.Table:
    """Mimics real Overture places.categories: struct<primary: string, alternate: list<string>>."""
    struct_type = pa.struct(
        [
            ("primary", pa.string()),
            ("alternate", pa.list_(pa.string())),
        ]
    )
    structs = []
    for p, a in zip(primaries, alternates, strict=True):
        if p is None and a is None:
            structs.append(None)
        else:
            structs.append({"primary": p, "alternate": a if a is not None else []})
    arr = pa.array(structs, type=struct_type)
    return pa.table({"categories": arr})


def test_compute_field_frequencies_struct_primary() -> None:
    table = _struct_with_list_table(
        primaries=["restaurant", "cafe", "restaurant", None],
        alternates=[[], [], [], None],
    )
    result = compute_field_frequencies(
        table, "categories.primary", label="places.categories.primary"
    )
    assert result.field == "places.categories.primary"
    assert result.n_present == 3
    assert result.counts == {"restaurant": 2, "cafe": 1}
    assert result.is_list_field is False
    assert result.total_occurrences == 3


def test_compute_field_frequencies_list_field_basic() -> None:
    table = _struct_with_list_table(
        primaries=["restaurant", "cafe", "shop", "shop"],
        alternates=[["vegan", "italian"], ["vegan"], [], ["thrift", "secondhand", "vintage"]],
    )
    result = compute_field_frequencies(
        table, "categories.alternate", label="places.categories.alternate", is_list_field=True
    )
    assert result.field == "places.categories.alternate"
    assert result.n_total == 4
    assert result.n_present == 3  # row with [] does not count as present
    assert result.counts == {"vegan": 2, "italian": 1, "thrift": 1, "secondhand": 1, "vintage": 1}
    assert result.total_occurrences == 6  # 2 + 1 + 0 + 3
    assert result.is_list_field is True


def test_compute_field_frequencies_list_field_all_empty() -> None:
    table = _struct_with_list_table(
        primaries=["a", "b", "c"],
        alternates=[[], [], None],
    )
    result = compute_field_frequencies(table, "categories.alternate", is_list_field=True)
    assert result.n_present == 0
    assert result.counts == {}
    assert result.total_occurrences == 0


def test_compute_field_frequencies_list_flag_on_scalar_column_raises() -> None:
    table = _scalar_table(["a", "b"])
    with pytest.raises(ValueError, match="expected list"):
        compute_field_frequencies(table, "class", is_list_field=True)


def test_compute_field_frequencies_struct_path_into_scalar_raises() -> None:
    table = _scalar_table(["a"])
    with pytest.raises(ValueError, match="cannot navigate into non-struct"):
        compute_field_frequencies(table, "class.foo")


def test_compute_field_frequencies_missing_struct_key_raises() -> None:
    table = _struct_with_list_table(primaries=["a"], alternates=[[]])
    with pytest.raises(ValueError, match="struct field 'nonexistent' not found"):
        compute_field_frequencies(table, "categories.nonexistent")

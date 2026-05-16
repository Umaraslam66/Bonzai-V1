from __future__ import annotations

import pytest

from cfm.data.frequency import FieldFrequencyResult
from cfm.data.vocab_derivation import SectionMetadata, apply_floor_to_kept_set


def _valid_metadata(**overrides):
    base = dict(
        source_field="buildings.class",
        source_fields=None,
        floor_strategy="Moderate",
        floor_value=100,
        coverage_retained_pct=98.83,
        coverage_singapore_pct=22.13,
        total_kept=23,
        is_provisional=True,
        decision_basis="marginal-cost elbow + building distinctiveness",
        notes="placeholder notes",
        denominator_type=None,
        alternate_only_provenance=None,
    )
    base.update(overrides)
    return base


def test_section_metadata_rejects_both_source_field_and_source_fields_set():
    with pytest.raises(ValueError, match="exactly one of source_field, source_fields"):
        SectionMetadata(
            **_valid_metadata(
                source_field="buildings.class",
                source_fields=("places.categories.primary",),
            )
        )


def test_section_metadata_rejects_neither_source_field_nor_source_fields_set():
    with pytest.raises(ValueError, match="exactly one of source_field, source_fields"):
        SectionMetadata(
            **_valid_metadata(
                source_field=None,
                source_fields=None,
            )
        )


def test_section_metadata_accepts_source_field_only():
    SectionMetadata(**_valid_metadata())


def test_section_metadata_accepts_source_fields_only():
    SectionMetadata(
        **_valid_metadata(
            source_field=None,
            source_fields=("places.categories.primary", "places.categories.alternate"),
        )
    )


def _make_result(counts: dict[str, int], *, is_list_field: bool = False) -> FieldFrequencyResult:
    return FieldFrequencyResult(
        field="test.field",
        n_total=sum(counts.values()) + 10,
        n_present=sum(counts.values()),
        counts=counts,
        is_list_field=is_list_field,
        total_occurrences=sum(counts.values()),
    )


def test_apply_floor_returns_kept_set_sorted_by_count_name():
    result = _make_result({"alpha": 500, "beta": 100, "gamma": 200, "delta": 50})
    kept = apply_floor_to_kept_set(result, floor_value=100)
    # gamma(200) before beta(100) by count; alpha(500) leads.
    assert kept == [("alpha", 500), ("gamma", 200), ("beta", 100)]


def test_apply_floor_ties_broken_alphabetically():
    result = _make_result({"banana": 100, "apple": 100, "cherry": 100, "low": 50})
    kept = apply_floor_to_kept_set(result, floor_value=100)
    assert kept == [("apple", 100), ("banana", 100), ("cherry", 100)]


def test_apply_floor_filters_below_threshold():
    result = _make_result({"a": 99, "b": 100, "c": 101})
    kept = apply_floor_to_kept_set(result, floor_value=100)
    assert kept == [("c", 101), ("b", 100)]


def test_apply_floor_returns_empty_when_nothing_meets_floor():
    result = _make_result({"a": 5, "b": 9})
    kept = apply_floor_to_kept_set(result, floor_value=100)
    assert kept == []

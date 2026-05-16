from __future__ import annotations

import pytest

from cfm.data.frequency import FieldFrequencyResult
from cfm.data.vocab_derivation import (
    SectionMetadata,
    apply_floor_to_kept_set,
    compute_alternate_only_provenance,
    derive_poi_union,
    derive_section,
)


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


def test_compute_alternate_only_provenance_set_difference():
    primary_kept = {"restaurant", "school", "park"}
    alternate_kept = {"restaurant", "vape_shop", "tobacco_shop"}
    result = compute_alternate_only_provenance(primary_kept, alternate_kept)
    assert result == ("tobacco_shop", "vape_shop")  # alphabetical


def test_compute_alternate_only_provenance_empty_when_alternate_subset_of_primary():
    primary_kept = {"a", "b", "c"}
    alternate_kept = {"a", "b"}
    assert compute_alternate_only_provenance(primary_kept, alternate_kept) == ()


def test_derive_section_includes_unknown_first_when_policy_emit_unknown_token():
    result = _make_result({"residential": 5000, "commercial": 300, "industrial": 150})
    section = derive_section(
        section_name="building",
        prefix="B_",
        field_result=result,
        floor_value=100,
        missing_policy="emit_unknown_token",
        coverage_singapore_pct=22.13,
        decision_basis="marginal-cost elbow + building distinctiveness",
        notes="placeholder",
        is_provisional=True,
    )
    assert section.tokens[0] == "B_unknown"
    assert section.tokens[1:] == ("B_residential", "B_commercial", "B_industrial")
    assert section.metadata.total_kept == 4  # 3 kept + 1 unknown
    assert section.metadata.source_field == "test.field"


def test_derive_section_omits_unknown_when_policy_drop_row():
    result = _make_result({"motorway": 5000, "primary": 300})
    section = derive_section(
        section_name="road",
        prefix="R_",
        field_result=result,
        floor_value=100,
        missing_policy="drop_row",
        coverage_singapore_pct=99.98,
        decision_basis="pedestrian-infrastructure distinctiveness",
        notes="placeholder",
        is_provisional=True,
    )
    assert all(not t.endswith("_unknown") for t in section.tokens)
    assert section.metadata.total_kept == 2


def test_derive_section_omits_unknown_when_policy_n_a():
    result = _make_result({"water": 5000, "park": 300})
    section = derive_section(
        section_name="base",
        prefix="BASE_",
        field_result=result,
        floor_value=100,
        missing_policy="n_a",
        coverage_singapore_pct=100.0,
        decision_basis="append-only safety on small-N field",
        notes="placeholder",
        is_provisional=True,
    )
    assert all(not t.endswith("_unknown") for t in section.tokens)


def test_derive_section_metadata_fields_populated():
    result = _make_result({"a": 1000, "b": 500})
    section = derive_section(
        section_name="building",
        prefix="B_",
        field_result=result,
        floor_value=100,
        missing_policy="drop_row",
        coverage_singapore_pct=42.0,
        decision_basis="basis",
        notes="notes",
        is_provisional=False,
    )
    md = section.metadata
    # floor_value=100 maps to Moderate in our locked decisions
    assert md.floor_strategy == "Moderate"
    # Coverage retained: 100% (both above floor)
    assert md.coverage_retained_pct == pytest.approx(100.0)
    assert md.coverage_singapore_pct == pytest.approx(42.0)
    assert md.is_provisional is False
    assert md.decision_basis == "basis"
    assert md.notes == "notes"


def test_derive_poi_union_combines_primary_and_alternate_kept_sets():
    primary = _make_result({"restaurant": 5000, "school": 1000, "park": 200})
    alternate = _make_result(
        {"restaurant": 200, "vape_shop": 150, "tobacco_shop": 120}, is_list_field=True
    )
    section = derive_poi_union(
        primary_result=primary,
        alternate_result=alternate,
        floor_value_primary=145,
        floor_value_alternate=109,
        missing_policy="emit_unknown_token",
        primary_coverage_singapore_pct=97.41,
        alternate_coverage_singapore_pct=73.45,
        decision_basis="union for semantic-equivalence",
        notes="placeholder",
        is_provisional=True,
    )

    # POI_unknown first; then primary-kept in (-count, name) order;
    # then alternate-only-kept in (-count, name) order.
    assert section.tokens[0] == "POI_unknown"
    # Primary kept: restaurant(5000), school(1000), park(200) all >= 145.
    assert section.tokens[1:4] == ("POI_restaurant", "POI_school", "POI_park")
    # Alternate-only kept: vape_shop(150), tobacco_shop(120) >= 109; restaurant overlaps primary.
    assert section.tokens[4:] == ("POI_vape_shop", "POI_tobacco_shop")


def test_derive_poi_union_provenance_set_difference_recorded():
    primary = _make_result({"restaurant": 5000, "school": 1000})
    alternate = _make_result(
        {"restaurant": 200, "vape_shop": 150, "tobacco_shop": 120}, is_list_field=True
    )
    section = derive_poi_union(
        primary_result=primary,
        alternate_result=alternate,
        floor_value_primary=145,
        floor_value_alternate=109,
        missing_policy="emit_unknown_token",
        primary_coverage_singapore_pct=97.41,
        alternate_coverage_singapore_pct=73.45,
        decision_basis="union",
        notes="notes",
        is_provisional=True,
    )
    assert section.metadata.alternate_only_provenance == ("tobacco_shop", "vape_shop")
    assert section.metadata.source_fields == (
        "places.categories.primary",
        "places.categories.alternate",
    )
    assert section.metadata.source_field is None
    assert section.metadata.denominator_type == "occurrences"


def test_derive_poi_union_no_duplicates_in_token_list():
    # restaurant appears in BOTH primary and alternate; should be in tokens exactly once.
    primary = _make_result({"restaurant": 5000})
    alternate = _make_result({"restaurant": 200, "cafe": 150}, is_list_field=True)
    section = derive_poi_union(
        primary_result=primary,
        alternate_result=alternate,
        floor_value_primary=145,
        floor_value_alternate=109,
        missing_policy="emit_unknown_token",
        primary_coverage_singapore_pct=97.41,
        alternate_coverage_singapore_pct=73.45,
        decision_basis="union",
        notes="notes",
        is_provisional=True,
    )
    name_counts = {name: section.tokens.count(name) for name in set(section.tokens)}
    assert all(count == 1 for count in name_counts.values()), name_counts

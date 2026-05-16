from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cfm.data.frequency import FieldFrequencyResult
from cfm.data.vocab_derivation import (
    SectionMetadata,
    apply_floor_to_kept_set,
    canonicalize_yaml,
    compute_alternate_only_provenance,
    compute_yaml_sha256,
    derive_phase1_policy,
    derive_phase1_vocab,
    derive_poi_union,
    derive_section,
    policy_to_dict,
    vocab_to_dict,
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


def test_canonicalize_yaml_byte_deterministic():
    data = {"b": 2, "a": 1, "nested": {"y": "value", "x": 10}}
    out1 = canonicalize_yaml(data)
    out2 = canonicalize_yaml(data)
    assert out1 == out2
    assert isinstance(out1, str)
    # Keys must be sorted: 'a' before 'b' at top level; 'x' before 'y' in nested.
    assert out1.index("a:") < out1.index("b:")
    assert out1.index("x:") < out1.index("y:")


def test_compute_yaml_sha256_excludes_self_field():
    data_a = {"vocab_sha256": "AAA", "a": 1, "b": 2}
    data_b = {"vocab_sha256": "BBB", "a": 1, "b": 2}
    # The sha256 field is excluded from the hash, so identical content
    # under different sha256 placeholders should hash identically.
    assert compute_yaml_sha256(data_a) == compute_yaml_sha256(data_b)


def test_compute_yaml_sha256_changes_when_content_changes():
    data_a = {"vocab_sha256": "<placeholder>", "a": 1}
    data_b = {"vocab_sha256": "<placeholder>", "a": 2}
    assert compute_yaml_sha256(data_a) != compute_yaml_sha256(data_b)


def test_compute_yaml_sha256_returns_hex_digest():
    data = {"vocab_sha256": "<placeholder>", "a": 1}
    h = compute_yaml_sha256(data)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def _phase1_inputs_minimal():
    """Build minimal FieldFrequencyResult set covering all 5 fields."""
    return {
        "buildings.class": FieldFrequencyResult(
            field="buildings.class",
            n_total=339_972,
            n_present=75_240,
            counts={"residential": 50_000, "commercial": 20_000, "industrial": 5_000},
            is_list_field=False,
            total_occurrences=75_000,
        ),
        "transportation.class": FieldFrequencyResult(
            field="transportation.class",
            n_total=202_334,
            n_present=202_292,
            counts={"motorway": 100_000, "primary": 80_000, "secondary": 22_000},
            is_list_field=False,
            total_occurrences=202_000,
        ),
        "base.class": FieldFrequencyResult(
            field="base.class",
            n_total=8_636,
            n_present=8_636,
            counts={"water": 4_000, "park": 3_000, "forest": 1_000},
            is_list_field=False,
            total_occurrences=8_000,
        ),
        "places.categories.primary": FieldFrequencyResult(
            field="places.categories.primary",
            n_total=149_657,
            n_present=145_774,
            counts={"restaurant": 50_000, "school": 5_000, "park": 200},
            is_list_field=False,
            total_occurrences=55_200,
        ),
        "places.categories.alternate": FieldFrequencyResult(
            field="places.categories.alternate",
            n_total=149_657,
            n_present=109_929,
            counts={"restaurant": 200, "vape_shop": 150, "tobacco_shop": 120},
            is_list_field=True,
            total_occurrences=470,
        ),
    }


def test_derive_phase1_vocab_assembles_four_feature_class_sections():
    inputs = _phase1_inputs_minimal()
    vocab = derive_phase1_vocab(
        field_results=inputs,
        overture_release="2026-04-15.0",
        source_report_path="reports/test.md",
        commit_sha="a" * 40,
        run_timestamp_utc=datetime(2026, 5, 16, 15, 25, 43, tzinfo=UTC),
    )
    section_names = [s.section_name for s in vocab.sections]
    assert section_names == ["road", "building", "poi", "base"]
    # Building should have B_unknown at index 0; road should not have R_unknown.
    building = next(s for s in vocab.sections if s.section_name == "building")
    road = next(s for s in vocab.sections if s.section_name == "road")
    assert building.tokens[0] == "B_unknown"
    assert all(not t.endswith("_unknown") for t in road.tokens)


def test_derive_phase1_vocab_metadata_fields_set():
    inputs = _phase1_inputs_minimal()
    vocab = derive_phase1_vocab(
        field_results=inputs,
        overture_release="2026-04-15.0",
        source_report_path="reports/test.md",
        commit_sha="b" * 40,
        run_timestamp_utc=datetime(2026, 5, 16, 15, 25, 43, tzinfo=UTC),
    )
    assert vocab.schema_version == "1.0"
    assert vocab.phase == 1
    assert vocab.vocab_version == "1.0"
    assert vocab.generated_at_commit == "b" * 40
    assert vocab.generated_utc == "2026-05-16T15:25:43Z"
    assert vocab.generated_from["overture_release"] == "2026-04-15.0"
    assert vocab.generated_from["regions"] == ["singapore"]


def test_derive_phase1_policy_field_set_matches_expected():
    inputs = _phase1_inputs_minimal()
    policy = derive_phase1_policy(
        field_results=inputs,
        overture_release="2026-04-15.0",
        source_report_path="reports/test.md",
        commit_sha="c" * 40,
        run_timestamp_utc=datetime(2026, 5, 16, 15, 25, 43, tzinfo=UTC),
    )
    expected_fields = {
        "buildings.class",
        "transportation.class",
        "base.class",
        "places.categories.primary",
        "places.categories.alternate",
    }
    actual_fields = {p.field for p in policy.field_policies}
    assert actual_fields == expected_fields


def test_derive_phase1_policy_enum_values_per_field():
    inputs = _phase1_inputs_minimal()
    policy = derive_phase1_policy(
        field_results=inputs,
        overture_release="2026-04-15.0",
        source_report_path="reports/test.md",
        commit_sha="d" * 40,
        run_timestamp_utc=datetime(2026, 5, 16, 15, 25, 43, tzinfo=UTC),
    )
    by_field = {p.field: p for p in policy.field_policies}
    assert by_field["buildings.class"].type == "emit_unknown_token"
    assert by_field["transportation.class"].type == "drop_row"
    assert by_field["base.class"].type == "n_a"
    assert by_field["places.categories.primary"].type == "emit_unknown_token"
    assert by_field["places.categories.alternate"].type == "n_a"

    # list_cap policy on alternate.
    assert len(policy.list_field_caps) == 1
    cap = policy.list_field_caps[0]
    assert cap.field == "places.categories.alternate"
    assert cap.cap_value == 2
    assert cap.cap_application == "tokenizer_time"
    assert cap.storage_policy == "preserve_all"


def test_vocab_to_dict_round_trips_sections():
    inputs = _phase1_inputs_minimal()
    vocab = derive_phase1_vocab(
        field_results=inputs,
        overture_release="2026-04-15.0",
        source_report_path="r.md",
        commit_sha="0" * 40,
        run_timestamp_utc=datetime(2026, 5, 16, tzinfo=UTC),
    )
    d = vocab_to_dict(vocab)
    assert d["schema_version"] == "1.0"
    assert d["phase"] == 1
    assert "feature_class" in d
    assert set(d["feature_class"].keys()) == {"road", "building", "poi", "base"}
    assert d["feature_class"]["building"]["tokens"][0] == "B_unknown"


def test_policy_to_dict_uses_unified_policies_dict():
    inputs = _phase1_inputs_minimal()
    policy = derive_phase1_policy(
        field_results=inputs,
        overture_release="2026-04-15.0",
        source_report_path="r.md",
        commit_sha="0" * 40,
        run_timestamp_utc=datetime(2026, 5, 16, tzinfo=UTC),
    )
    d = policy_to_dict(policy)
    assert "fields" in d
    # places.categories.alternate has both missing_value (n_a) and list_cap.
    alt = d["fields"]["places.categories.alternate"]
    assert "policies" in alt
    assert alt["policies"]["missing_value"]["type"] == "n_a"
    assert alt["policies"]["list_cap"]["cap_value"] == 2
    # buildings.class has only missing_value.
    assert "list_cap" not in d["fields"]["buildings.class"]["policies"]
    assert (
        d["fields"]["buildings.class"]["policies"]["missing_value"]["type"] == "emit_unknown_token"
    )

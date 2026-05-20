from __future__ import annotations

from cfm.data.sub_e.derivation import (
    BoundaryClass,
    derive_boundary_class,
    load_class_grouping_map,
)


def test_load_class_grouping_map_from_vocab_yaml() -> None:
    mapping = load_class_grouping_map()
    assert mapping["primary"] is BoundaryClass.MAJOR_ROAD
    assert mapping["secondary"] is BoundaryClass.MAJOR_ROAD
    assert mapping["trunk"] is BoundaryClass.MAJOR_ROAD
    assert mapping["residential"] is BoundaryClass.MINOR_ROAD
    assert mapping["service"] is BoundaryClass.MINOR_ROAD
    assert mapping["footway"] is BoundaryClass.MINOR_ROAD


def test_empty_crossings_returns_none() -> None:
    result = derive_boundary_class(class_raws=[])
    assert result is BoundaryClass.NONE


def test_single_primary_crossing_returns_major() -> None:
    result = derive_boundary_class(class_raws=["primary"])
    assert result is BoundaryClass.MAJOR_ROAD


def test_single_residential_crossing_returns_minor() -> None:
    result = derive_boundary_class(class_raws=["residential"])
    assert result is BoundaryClass.MINOR_ROAD


def test_hierarchy_wins_primary_beats_residential() -> None:
    result = derive_boundary_class(class_raws=["residential", "primary"])
    assert result is BoundaryClass.MAJOR_ROAD


def test_hierarchy_wins_three_minor_one_major() -> None:
    result = derive_boundary_class(class_raws=["footway", "residential", "service", "secondary"])
    assert result is BoundaryClass.MAJOR_ROAD


def test_default_bucket_unknown_class_raw_demotes_to_minor() -> None:
    """Overture rare values not in the named 10 → MINOR_ROAD."""
    result = derive_boundary_class(class_raws=["proposed"])
    assert result is BoundaryClass.MINOR_ROAD


def test_default_bucket_null_class_raw_treats_as_minor() -> None:
    result = derive_boundary_class(class_raws=[None])
    assert result is BoundaryClass.MINOR_ROAD


def test_default_bucket_does_not_promote_to_major() -> None:
    """Mixing an unknown with a primary still resolves to MAJOR via the
    primary, but an unknown alone never resolves to MAJOR. Demonstrate the
    asymmetry."""
    assert derive_boundary_class(class_raws=["proposed"]) is BoundaryClass.MINOR_ROAD
    assert derive_boundary_class(class_raws=["proposed", "primary"]) is BoundaryClass.MAJOR_ROAD


def test_boundary_class_enum_values_match_vocab_ids() -> None:
    assert BoundaryClass.BOUNDARY_NOT_APPLICABLE.value == 0
    assert BoundaryClass.NONE.value == 1
    assert BoundaryClass.MAJOR_ROAD.value == 2
    assert BoundaryClass.MINOR_ROAD.value == 3


def test_strict_matching_case_variant_falls_to_default_bucket() -> None:
    """Case variants are not byte-equal to the lowercase Overture values
    and must fall through to the MINOR_ROAD default bucket. Pins the
    strict-equality convention locked under boundary_derivation_version 1.0
    (spec §5.1).
    """
    assert derive_boundary_class(class_raws=["Primary"]) is BoundaryClass.MINOR_ROAD
    assert derive_boundary_class(class_raws=["PRIMARY"]) is BoundaryClass.MINOR_ROAD


def test_strict_matching_whitespace_variant_falls_to_default_bucket() -> None:
    """Leading/trailing/internal whitespace prevents byte-equality lookup
    and the value falls through to the MINOR_ROAD default bucket. Pins the
    strict-equality convention locked under boundary_derivation_version 1.0
    (spec §5.1).
    """
    assert derive_boundary_class(class_raws=[" primary"]) is BoundaryClass.MINOR_ROAD
    assert derive_boundary_class(class_raws=["primary "]) is BoundaryClass.MINOR_ROAD
    assert derive_boundary_class(class_raws=[" primary "]) is BoundaryClass.MINOR_ROAD


def test_strict_matching_empty_string_falls_to_default_bucket() -> None:
    """Empty string is not in the class_grouping_map and falls to default."""
    assert derive_boundary_class(class_raws=[""]) is BoundaryClass.MINOR_ROAD

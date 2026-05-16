from __future__ import annotations

import pytest

from cfm.data.vocab_derivation import SectionMetadata


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

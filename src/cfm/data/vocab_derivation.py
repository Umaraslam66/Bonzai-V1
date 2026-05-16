from __future__ import annotations

from dataclasses import dataclass

from cfm.data.frequency import FieldFrequencyResult


@dataclass(frozen=True)
class SectionMetadata:
    """Metadata block emitted alongside a feature_class section's token list.

    Exactly one of source_field / source_fields must be set. The POI section
    uses source_fields (because it derives from primary AND alternate columns);
    all other sections use source_field.
    """

    source_field: str | None
    source_fields: tuple[str, ...] | None
    floor_strategy: str  # "Moderate" | "Strict"
    floor_value: int
    coverage_retained_pct: float  # 0..100; among present (non-null) source rows
    coverage_singapore_pct: float  # 0..100; fraction of source rows with non-null value
    total_kept: int
    is_provisional: bool
    decision_basis: str
    notes: str
    denominator_type: str | None  # "occurrences" | "rows" | "rows_with_non_empty"
    alternate_only_provenance: tuple[str, ...] | None  # poi section only

    def __post_init__(self) -> None:
        has_one = self.source_field is not None
        has_many = self.source_fields is not None
        if has_one == has_many:  # both set OR both unset
            raise ValueError(
                "SectionMetadata: exactly one of source_field, source_fields must be set; "
                f"got source_field={self.source_field!r}, source_fields={self.source_fields!r}"
            )


@dataclass(frozen=True)
class FieldPolicy:
    field: str
    type: str  # "emit_unknown_token" | "drop_row" | "n_a"
    rationale: str
    is_provisional: bool


@dataclass(frozen=True)
class ListFieldCap:
    field: str
    cap_value: int
    cap_application: str  # "tokenizer_time" | "storage_time" | "not_applied"
    storage_policy: str  # "preserve_all" | "truncate"
    dead_token_fraction_upper_bound: float
    caveat: str
    is_provisional: bool


@dataclass(frozen=True)
class SectionDerivation:
    section_name: str  # "road" | "building" | "poi" | "base"
    prefix: str  # "R_" | "B_" | "POI_" | "BASE_"
    tokens: tuple[str, ...]
    metadata: SectionMetadata


@dataclass(frozen=True)
class Phase1Vocab:
    schema_version: str
    phase: int
    vocab_version: str
    generated_at_commit: str
    generated_utc: str
    generated_from: dict
    sections: tuple[SectionDerivation, ...]


@dataclass(frozen=True)
class Phase1Policy:
    schema_version: str
    phase: int
    policy_version: str
    generated_at_commit: str
    generated_utc: str
    generated_from: dict
    field_policies: tuple[FieldPolicy, ...]
    list_field_caps: tuple[ListFieldCap, ...]


def apply_floor_to_kept_set(
    result: FieldFrequencyResult,
    floor_value: int,
) -> list[tuple[str, int]]:
    """Return kept categories sorted deterministically by (-count, name).

    A category is kept iff its count is >= floor_value. Tie-breaking is
    alphabetical by name. Matches B1's library sort tuple so derivations
    are reproducible against the rank-frequency report.
    """
    return sorted(
        ((name, count) for name, count in result.counts.items() if count >= floor_value),
        key=lambda item: (-item[1], item[0]),
    )

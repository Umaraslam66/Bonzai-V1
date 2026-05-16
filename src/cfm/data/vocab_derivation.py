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


# Floor-value → strategy-name lookup. The locked floors per the B2 spec §7
# are the only valid floor values; this mapping documents the relationship.
_FLOOR_TO_STRATEGY = {
    202: "Moderate",  # transportation.class
    100: "Moderate",  # buildings.class
    145: "Moderate",  # places.categories.primary
    109: "Moderate",  # places.categories.alternate
    300: "Strict",  # base.class
}


def compute_alternate_only_provenance(
    primary_kept: set[str],
    alternate_kept: set[str],
) -> tuple[str, ...]:
    """Return (alternate_kept \\ primary_kept) sorted alphabetically.

    These are POI categories that survive Moderate on the alternate column
    but not the primary column. Under the cap=2 tokenizer-time policy, some
    of these may be dead tokens (alternate-position-3+); estimated upper
    bound on the dead-token fraction is 1.78% — see policy YAML caveat.
    """
    return tuple(sorted(alternate_kept - primary_kept))


def derive_section(
    *,
    section_name: str,
    prefix: str,
    field_result: FieldFrequencyResult,
    floor_value: int,
    missing_policy: str,
    coverage_singapore_pct: float,
    decision_basis: str,
    notes: str,
    is_provisional: bool,
    denominator_type: str | None = None,
) -> SectionDerivation:
    """Build a SectionDerivation for one non-POI feature_class section.

    The POI section requires the union of two field results; use
    `derive_poi_union` instead.

    Prepends `<prefix>unknown` at index 0 iff missing_policy == "emit_unknown_token".
    """
    kept = apply_floor_to_kept_set(field_result, floor_value)
    prefixed_tokens = [f"{prefix}{name}" for name, _ in kept]

    if missing_policy == "emit_unknown_token":
        tokens = (f"{prefix}unknown", *prefixed_tokens)
    elif missing_policy in ("drop_row", "n_a"):
        tokens = tuple(prefixed_tokens)
    else:
        raise ValueError(f"unknown missing_policy: {missing_policy!r}")

    total_occurrences_kept = sum(count for _, count in kept)
    if field_result.total_occurrences == 0:
        coverage_retained_pct = 100.0
    else:
        coverage_retained_pct = 100.0 * total_occurrences_kept / field_result.total_occurrences

    floor_strategy = _FLOOR_TO_STRATEGY.get(floor_value, f"Custom(floor={floor_value})")

    metadata = SectionMetadata(
        source_field=field_result.field,
        source_fields=None,
        floor_strategy=floor_strategy,
        floor_value=floor_value,
        coverage_retained_pct=round(coverage_retained_pct, 2),
        coverage_singapore_pct=round(coverage_singapore_pct, 2),
        total_kept=len(tokens),
        is_provisional=is_provisional,
        decision_basis=decision_basis,
        notes=notes,
        denominator_type=denominator_type,
        alternate_only_provenance=None,
    )

    return SectionDerivation(
        section_name=section_name,
        prefix=prefix,
        tokens=tokens,
        metadata=metadata,
    )

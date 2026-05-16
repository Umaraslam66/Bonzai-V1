from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime

import yaml

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


def derive_poi_union(
    *,
    primary_result: FieldFrequencyResult,
    alternate_result: FieldFrequencyResult,
    floor_value_primary: int,
    floor_value_alternate: int,
    missing_policy: str,
    primary_coverage_singapore_pct: float,
    alternate_coverage_singapore_pct: float,
    decision_basis: str,
    notes: str,
    is_provisional: bool,
) -> SectionDerivation:
    r"""Build the POI section as the union of primary and alternate Moderate cuts.

    Ordering of the token list:
      [<prefix>unknown if emit_unknown_token]
      + primary-kept by (-count, name)
      + alternate-only-kept by (-count, name)  (= alternate_kept \ primary_kept)

    Provenance metadata records the alternate-only set so consumers can
    detect tokens that may be dead under the current encoder.
    """
    prefix = "POI_"

    primary_kept = apply_floor_to_kept_set(primary_result, floor_value_primary)
    alternate_kept = apply_floor_to_kept_set(alternate_result, floor_value_alternate)

    primary_names = {name for name, _ in primary_kept}
    alternate_names = {name for name, _ in alternate_kept}
    alternate_only = alternate_names - primary_names

    # Preserve (-count, name) order from alternate_kept for alternate-only entries.
    alternate_only_ordered = [
        (name, count) for name, count in alternate_kept if name in alternate_only
    ]

    primary_tokens = [f"{prefix}{name}" for name, _ in primary_kept]
    alternate_only_tokens = [f"{prefix}{name}" for name, _ in alternate_only_ordered]

    if missing_policy == "emit_unknown_token":
        tokens = (f"{prefix}unknown", *primary_tokens, *alternate_only_tokens)
    elif missing_policy in ("drop_row", "n_a"):
        tokens = tuple(primary_tokens + alternate_only_tokens)
    else:
        raise ValueError(f"unknown missing_policy: {missing_policy!r}")

    primary_total_kept = sum(count for _, count in primary_kept)
    if primary_result.total_occurrences == 0:
        primary_coverage_retained = 100.0
    else:
        primary_coverage_retained = 100.0 * primary_total_kept / primary_result.total_occurrences

    metadata = SectionMetadata(
        source_field=None,
        source_fields=("places.categories.primary", "places.categories.alternate"),
        floor_strategy="Moderate",
        floor_value=floor_value_primary,  # primary's floor; the spec records both via notes
        coverage_retained_pct=round(primary_coverage_retained, 2),
        coverage_singapore_pct=round(primary_coverage_singapore_pct, 2),
        total_kept=len(tokens),
        is_provisional=is_provisional,
        decision_basis=decision_basis,
        notes=notes,
        denominator_type="occurrences",
        alternate_only_provenance=compute_alternate_only_provenance(primary_names, alternate_names),
    )

    return SectionDerivation(
        section_name="poi",
        prefix=prefix,
        tokens=tokens,
        metadata=metadata,
    )


# Sha-fields excluded from canonicalisation when computing self-hash.
_SELF_HASH_FIELDS = frozenset({"vocab_sha256", "policy_sha256"})


def canonicalize_yaml(data: dict) -> str:
    """Serialise *data* to a byte-deterministic YAML string.

    Sorts dict keys at every level. Uses block style (no flow) for nested
    structures so diffs read cleanly. Newline at end of file. The same input
    produces the same output bytes across runs.
    """
    return yaml.safe_dump(
        data,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
        indent=2,
        width=4096,  # avoid wrapping; we want stable lines
    )


def compute_yaml_sha256(data: dict) -> str:
    """Return sha256 hex digest of canonicalised *data* with self-hash fields stripped.

    Strips `vocab_sha256` / `policy_sha256` before hashing so the embedded
    sha256 doesn't participate in its own computation (cyclic dependency).
    """
    stripped = deepcopy(data)
    for field in _SELF_HASH_FIELDS:
        stripped.pop(field, None)
    canonical = canonicalize_yaml(stripped)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Locked decisions per the B2 spec §7 + §8.
# Centralised here so the derivation reads as a transparent application of
# the spec, not as scattered constants.
_LOCKED_FLOOR_VALUES = {
    "transportation.class": 202,
    "buildings.class": 100,
    "places.categories.primary": 145,
    "places.categories.alternate": 109,
    "base.class": 300,
}

_LOCKED_MISSING_POLICIES = {
    "buildings.class": (
        "emit_unknown_token",
        (
            "78.0% missing on Singapore; dropping forfeits the bulk of building "
            "data; append-only safety."
        ),
        True,
    ),
    "transportation.class": (
        "drop_row",
        "0.02% missing (42 rows); too few to warrant a token slot.",
        False,
    ),
    "base.class": (
        "n_a",
        "100% coverage on Singapore; no missing rows.",
        False,
    ),
    "places.categories.primary": (
        "emit_unknown_token",
        "2.59% missing (3,883 rows); geometric info valid; consistency with buildings.class.",
        True,
    ),
    "places.categories.alternate": (
        "n_a",
        "List field; empty list is 'no secondary categories', not missing data.",
        False,
    ),
}

_DECISION_BASIS = {
    "road": "pedestrian-infrastructure distinctiveness over Strict's scaling-math",
    "building": "marginal-cost elbow + building distinctiveness",
    "poi": "marginal-cost elbow on both columns; union for semantic-equivalence",
    "base": "append-only safety on small-N field",
}

_NOTES = {
    "road": (
        "Closer call vs Strict; kept Moderate for pedestrian-infrastructure "
        "distinction (cycleway/footway band). Floor=202 SG rows scales to 4K-20K "
        "global at 5%-1% Singapore share; low end below PRD §5's 10,000-global-"
        "instance learnability threshold. B1' Sweden re-run is required for "
        "de-provisioning; revisit Moderate->Strict if Sweden's pedestrian counts "
        "don't lift these above 10K globally."
    ),
    "building": (
        "Singapore coverage 22.13% (78% missing); B_unknown included. "
        "Floor=100 SG rows scales to 2K-10K global at 5%-1% Singapore share; "
        "low end below PRD §5's 10,000-global-instance learnability threshold. "
        "B1' Sweden re-run is required for de-provisioning; revisit "
        "Moderate->Lenient (13 cheap cats) at the same time."
    ),
    "poi": (
        "Union of primary-Moderate-kept U alternate-Moderate-kept. "
        "POI_unknown included for primary missing-value handling. "
        "Denominator: alternate counts use occurrences-among-rows-with-alternates. "
        "Cap=2 at tokenizer time means alternate-only-position-3+ categories may "
        "be dead under current encoder; estimated <=1.78% of POI tokens. "
        "Floors of 145 (primary) and 109 (alternate) SG rows scale to 2.9K-14.5K "
        "and 2.18K-10.9K global respectively at 5%-1% Singapore share; both low "
        "ends below PRD §5's 10K learnability threshold. B1' Sweden re-run is "
        "required for de-provisioning."
    ),
    "base": (
        "Small-N field (8,636 Singapore rows). Append-only safety dominated "
        "marginal-cost-of-cut. Floor=300 SG rows scales to 6K-30K global at "
        "5%-1% Singapore share; low end below PRD §5's 10K learnability "
        "threshold but in the marginal-but-learnable band. B1' Sweden re-run "
        "is required for de-provisioning; should specifically check whether "
        "the 7 dropped Lenient->Strict categories deserve appending."
    ),
}

_LIST_CAP_CAVEAT = (
    "Moderate-cut survival counted alternates at all positions. Under cap=2 at "
    "tokenizer time, categories appearing only at position 3+ have allocated "
    "token IDs but will never be emitted. Estimated dead-token fraction <=1.78%. "
    "B1' Sweden re-run can optionally re-compute frequencies under a position≤2 "
    "filter to refine the kept set."
)


def derive_phase1_vocab(
    *,
    field_results: dict,
    overture_release: str,
    source_report_path: str,
    commit_sha: str,
    run_timestamp_utc: datetime,
    schema_version: str = "1.0",
    phase: int = 1,
    vocab_version: str = "1.0",
) -> Phase1Vocab:
    """Assemble the full Phase 1 vocab from B1 field results. Pure; no I/O."""

    # Per spec §9, section order in the YAML follows the feature_class outline
    # used by the tokenizer's _flatten: road, building, poi, base.
    road = derive_section(
        section_name="road",
        prefix="R_",
        field_result=field_results["transportation.class"],
        floor_value=_LOCKED_FLOOR_VALUES["transportation.class"],
        missing_policy=_LOCKED_MISSING_POLICIES["transportation.class"][0],
        coverage_singapore_pct=_coverage_pct(field_results["transportation.class"]),
        decision_basis=_DECISION_BASIS["road"],
        notes=_NOTES["road"],
        is_provisional=True,
    )
    building = derive_section(
        section_name="building",
        prefix="B_",
        field_result=field_results["buildings.class"],
        floor_value=_LOCKED_FLOOR_VALUES["buildings.class"],
        missing_policy=_LOCKED_MISSING_POLICIES["buildings.class"][0],
        coverage_singapore_pct=_coverage_pct(field_results["buildings.class"]),
        decision_basis=_DECISION_BASIS["building"],
        notes=_NOTES["building"],
        is_provisional=True,
    )
    poi = derive_poi_union(
        primary_result=field_results["places.categories.primary"],
        alternate_result=field_results["places.categories.alternate"],
        floor_value_primary=_LOCKED_FLOOR_VALUES["places.categories.primary"],
        floor_value_alternate=_LOCKED_FLOOR_VALUES["places.categories.alternate"],
        missing_policy=_LOCKED_MISSING_POLICIES["places.categories.primary"][0],
        primary_coverage_singapore_pct=_coverage_pct(field_results["places.categories.primary"]),
        alternate_coverage_singapore_pct=_coverage_pct(
            field_results["places.categories.alternate"]
        ),
        decision_basis=_DECISION_BASIS["poi"],
        notes=_NOTES["poi"],
        is_provisional=True,
    )
    base = derive_section(
        section_name="base",
        prefix="BASE_",
        field_result=field_results["base.class"],
        floor_value=_LOCKED_FLOOR_VALUES["base.class"],
        missing_policy=_LOCKED_MISSING_POLICIES["base.class"][0],
        coverage_singapore_pct=_coverage_pct(field_results["base.class"]),
        decision_basis=_DECISION_BASIS["base"],
        notes=_NOTES["base"],
        is_provisional=True,
    )

    return Phase1Vocab(
        schema_version=schema_version,
        phase=phase,
        vocab_version=vocab_version,
        generated_at_commit=commit_sha,
        generated_utc=run_timestamp_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        generated_from={
            "overture_release": overture_release,
            "regions": ["singapore"],
            "source_report": source_report_path,
        },
        sections=(road, building, poi, base),
    )


def derive_phase1_policy(
    *,
    field_results: dict,
    overture_release: str,
    source_report_path: str,
    commit_sha: str,
    run_timestamp_utc: datetime,
    schema_version: str = "1.0",
    phase: int = 1,
    policy_version: str = "1.0",
) -> Phase1Policy:
    """Assemble the full Phase 1 policy from B1 field results. Pure; no I/O."""

    field_policies = tuple(
        FieldPolicy(
            field=field,
            type=policy_type,
            rationale=rationale,
            is_provisional=is_provisional,
        )
        for field, (policy_type, rationale, is_provisional) in _LOCKED_MISSING_POLICIES.items()
    )

    list_field_caps = (
        ListFieldCap(
            field="places.categories.alternate",
            cap_value=2,
            cap_application="tokenizer_time",
            storage_policy="preserve_all",
            dead_token_fraction_upper_bound=0.0178,
            caveat=_LIST_CAP_CAVEAT,
            is_provisional=True,
        ),
    )

    return Phase1Policy(
        schema_version=schema_version,
        phase=phase,
        policy_version=policy_version,
        generated_at_commit=commit_sha,
        generated_utc=run_timestamp_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        generated_from={
            "overture_release": overture_release,
            "regions": ["singapore"],
            "source_report": source_report_path,
        },
        field_policies=field_policies,
        list_field_caps=list_field_caps,
    )


def _coverage_pct(result: FieldFrequencyResult) -> float:
    if result.n_total == 0:
        return 0.0
    return 100.0 * result.n_present / result.n_total


def vocab_to_dict(vocab: Phase1Vocab) -> dict:
    """Convert a Phase1Vocab to a dict ready for YAML serialisation.

    The structure matches the §9 exemplar in the B2 spec. The vocab_sha256
    field is added later by the CLI (after this dict is computed) since
    sha256 needs the canonicalised form of the dict-without-sha256.
    """
    feature_class = {}
    for section in vocab.sections:
        feature_class[section.section_name] = _section_to_dict(section)

    # Phase 0 control/hierarchy/anchor/move pulled from vocab_phase0.yaml's outline.
    return {
        "schema_version": vocab.schema_version,
        "phase": vocab.phase,
        "vocab_version": vocab.vocab_version,
        "generated_at_commit": vocab.generated_at_commit,
        "generated_utc": vocab.generated_utc,
        "generated_from": vocab.generated_from,
        "phase_links": {
            "prev_phase_file": "configs/tokenizer/vocab_phase0.yaml",
        },
        "control": [
            "PAD",
            "BOS",
            "EOS",
            "CELL",
            "END_CELL",
            "FEATURE_START",
            "FEATURE_END",
            "EXIT",
            "POINT",
            "LINE",
            "POLYGON",
        ],
        "hierarchy": ["MACRO", "END_MACRO", "MICRO", "END_MICRO"],
        "feature_class": feature_class,
        "anchor": {"axis_count": 250},
        "move": {
            "directions": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
            "steps_m": [1, 2, 4, 8, 16, 32],
        },
    }


def _section_to_dict(section: SectionDerivation) -> dict:
    md = section.metadata
    out: dict = {}
    if md.source_field is not None:
        out["source_field"] = md.source_field
    if md.source_fields is not None:
        out["source_fields"] = list(md.source_fields)
    out["floor_strategy"] = md.floor_strategy
    out["floor_value"] = md.floor_value
    out["coverage_retained_pct"] = md.coverage_retained_pct
    out["coverage_singapore_pct"] = md.coverage_singapore_pct
    out["total_kept"] = md.total_kept
    out["is_provisional"] = md.is_provisional
    out["decision_basis"] = md.decision_basis
    out["notes"] = md.notes
    if md.denominator_type is not None:
        out["denominator_type"] = md.denominator_type
    if md.alternate_only_provenance is not None:
        out["alternate_only_provenance"] = list(md.alternate_only_provenance)
    out["tokens"] = list(section.tokens)
    return out


def policy_to_dict(policy: Phase1Policy) -> dict:
    """Convert a Phase1Policy to dict matching the §10 exemplar.

    Per-field shape: { policies: { missing_value: {...}, [list_cap: {...}] } }.
    """
    fields: dict = {}
    for fp in policy.field_policies:
        fields[fp.field] = {
            "policies": {
                "missing_value": {
                    "type": fp.type,
                    "rationale": fp.rationale,
                    "is_provisional": fp.is_provisional,
                }
            }
        }
    for cap in policy.list_field_caps:
        fields[cap.field]["policies"]["list_cap"] = {
            "cap_value": cap.cap_value,
            "cap_application": cap.cap_application,
            "storage_policy": cap.storage_policy,
            "dead_token_fraction_upper_bound": cap.dead_token_fraction_upper_bound,
            "caveat": cap.caveat,
            "is_provisional": cap.is_provisional,
        }

    return {
        "schema_version": policy.schema_version,
        "phase": policy.phase,
        "policy_version": policy.policy_version,
        "generated_at_commit": policy.generated_at_commit,
        "generated_utc": policy.generated_utc,
        "generated_from": policy.generated_from,
        "fields": fields,
    }

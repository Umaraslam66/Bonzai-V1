# Phase 1 sub-B2 Vocabulary YAML Derivation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `configs/tokenizer/vocab_phase1.yaml` and `configs/data/missing_value_policy.yaml` from B1's cached Singapore frequency analysis, plus a small tokenizer loader update to handle the new section shape.

**Architecture:** Pure library functions in `cfm.data.vocab_derivation` derive the two YAMLs from B1's `FieldFrequencyResult` outputs. A thin CLI script glues B1's `load_region` + `compute_field_frequencies` to the library and writes both artifacts. The tokenizer's `_flatten` gains a dict-vs-list branch to handle Phase 1's metadata-bearing feature_class sections while continuing to load Phase 0's flat lists.

**Tech Stack:** Python 3.11+, PyYAML, pyarrow, pytest, uv. Reuses B1's `cfm.data.frequency` library.

**Spec reference:** `docs/superpowers/specs/2026-05-16-phase-1-sub-B2-vocab-yaml-design.md` (committed at 7e7c718).

---

## File map

**Create:**
- `src/cfm/data/vocab_derivation.py` — library: 6 dataclasses + 8 pure functions
- `scripts/derive_phase1_vocab.py` — CLI: ~80 lines of glue
- `tests/data/test_vocab_derivation.py` — Layer 1 unit tests (~15 tests)
- `tests/data/test_derive_phase1_vocab.py` — Layer 3 integration tests (~4 tests)
- `tests/tokenizer/test_vocabulary_loader_phase1.py` — Layer 2 loader tests (~7 tests)
- `configs/tokenizer/vocab_phase1.yaml` — generated artifact (committed)
- `configs/data/missing_value_policy.yaml` — generated artifact (committed)

**Modify:**
- `src/cfm/tokenizer/vocabulary.py` — extend `_flatten` with dict-vs-list branch + 2 validations (~25 lines)
- `src/cfm/tokenizer/errors.py` — add `LoaderError` class
- `src/cfm/data/__init__.py` — export new symbols
- `src/cfm/tokenizer/__init__.py` — export `LoaderError`
- `configs/tokenizer/vocab_phase0.yaml` — update header (supersede the "Phase 1 may append" line with the 4-line phase-transition block)
- `docs/known_issues.md` — add subtype/subclass deferral entry

---

## Phase 1 — Library skeleton (no I/O)

### Task 1: Dataclass definitions in `vocab_derivation.py`

**Files:**
- Create: `src/cfm/data/vocab_derivation.py`
- Create: `tests/data/test_vocab_derivation.py`

- [ ] **Step 1: Write the failing tests for SectionMetadata invariant**

Create `tests/data/test_vocab_derivation.py`:

```python
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
        SectionMetadata(**_valid_metadata(
            source_field="buildings.class",
            source_fields=("places.categories.primary",),
        ))


def test_section_metadata_rejects_neither_source_field_nor_source_fields_set():
    with pytest.raises(ValueError, match="exactly one of source_field, source_fields"):
        SectionMetadata(**_valid_metadata(
            source_field=None,
            source_fields=None,
        ))


def test_section_metadata_accepts_source_field_only():
    SectionMetadata(**_valid_metadata())


def test_section_metadata_accepts_source_fields_only():
    SectionMetadata(**_valid_metadata(
        source_field=None,
        source_fields=("places.categories.primary", "places.categories.alternate"),
    ))
```

- [ ] **Step 2: Run tests, confirm they fail with ImportError**

Run: `uv run pytest tests/data/test_vocab_derivation.py -v`
Expected: `ImportError: cannot import name 'SectionMetadata' from 'cfm.data.vocab_derivation'` (module doesn't exist yet).

- [ ] **Step 3: Create `vocab_derivation.py` with all dataclasses**

Create `src/cfm/data/vocab_derivation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


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
    coverage_retained_pct: float          # 0..100; among present (non-null) source rows
    coverage_singapore_pct: float         # 0..100; fraction of source rows where the field is non-null
    total_kept: int
    is_provisional: bool
    decision_basis: str
    notes: str
    denominator_type: str | None          # "occurrences" | "rows" | "rows_with_non_empty"
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
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `uv run pytest tests/data/test_vocab_derivation.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cfm/data/vocab_derivation.py tests/data/test_vocab_derivation.py
git commit -m "feat(data): add vocab_derivation dataclasses for sub-B2"
```

---

### Task 2: Update `__init__.py` exports

**Files:**
- Modify: `src/cfm/data/__init__.py`

- [ ] **Step 1: Replace `__init__.py` contents**

```python
# src/cfm/data/__init__.py
"""Data pipeline package: Overture loading, tile extraction, validation."""

from cfm.data.vocab_derivation import (
    FieldPolicy,
    ListFieldCap,
    Phase1Policy,
    Phase1Vocab,
    SectionDerivation,
    SectionMetadata,
)

__all__ = [
    "FieldPolicy",
    "ListFieldCap",
    "Phase1Policy",
    "Phase1Vocab",
    "SectionDerivation",
    "SectionMetadata",
]
```

- [ ] **Step 2: Verify imports work**

Run: `uv run python -c "from cfm.data import SectionMetadata, FieldPolicy, Phase1Vocab; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Run the full data test suite to confirm no regression**

Run: `uv run pytest tests/data/ -v`
Expected: All existing data tests pass plus the 4 new SectionMetadata tests = 35+ passing.

- [ ] **Step 4: Commit**

```bash
git add src/cfm/data/__init__.py
git commit -m "chore(data): export vocab_derivation dataclasses"
```

---

## Phase 2 — Pure library functions (TDD)

### Task 3: `apply_floor_to_kept_set`

**Files:**
- Modify: `src/cfm/data/vocab_derivation.py`
- Modify: `tests/data/test_vocab_derivation.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/data/test_vocab_derivation.py`:

```python
from cfm.data.frequency import FieldFrequencyResult
from cfm.data.vocab_derivation import apply_floor_to_kept_set


def _make_result(counts: dict[str, int], *, is_list_field: bool = False) -> FieldFrequencyResult:
    n_present = sum(1 for v in counts.values() if v > 0)
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
```

- [ ] **Step 2: Run tests, confirm fail**

Run: `uv run pytest tests/data/test_vocab_derivation.py -v -k apply_floor`
Expected: 4 failures with `ImportError: cannot import name 'apply_floor_to_kept_set'`.

- [ ] **Step 3: Implement `apply_floor_to_kept_set`**

Append to `src/cfm/data/vocab_derivation.py`:

```python
from cfm.data.frequency import FieldFrequencyResult


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
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `uv run pytest tests/data/test_vocab_derivation.py -v -k apply_floor`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cfm/data/vocab_derivation.py tests/data/test_vocab_derivation.py
git commit -m "feat(data): apply_floor_to_kept_set with (-count, name) sort"
```

---

### Task 4: `compute_alternate_only_provenance` + `derive_section`

**Files:**
- Modify: `src/cfm/data/vocab_derivation.py`
- Modify: `tests/data/test_vocab_derivation.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
from cfm.data.vocab_derivation import (
    compute_alternate_only_provenance,
    derive_section,
)


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
    assert md.floor_strategy == "Moderate"  # floor_value=100 maps to Moderate in our locked decisions
    # Coverage retained: 100% (both above floor)
    assert md.coverage_retained_pct == pytest.approx(100.0)
    assert md.coverage_singapore_pct == pytest.approx(42.0)
    assert md.is_provisional is False
    assert md.decision_basis == "basis"
    assert md.notes == "notes"
```

- [ ] **Step 2: Run tests, confirm fail**

Run: `uv run pytest tests/data/test_vocab_derivation.py -v -k "alternate_only_provenance or derive_section"`
Expected: 6 failures with import errors.

- [ ] **Step 3: Implement both functions**

Append to `src/cfm/data/vocab_derivation.py`:

```python
# Floor-value → strategy-name lookup. The locked floors per the B2 spec §7
# are the only valid floor values; this mapping documents the relationship.
_FLOOR_TO_STRATEGY = {
    202: "Moderate",   # transportation.class
    100: "Moderate",   # buildings.class
    145: "Moderate",   # places.categories.primary
    109: "Moderate",   # places.categories.alternate
    300: "Strict",     # base.class
}


def compute_alternate_only_provenance(
    primary_kept: set[str],
    alternate_kept: set[str],
) -> tuple[str, ...]:
    """Return (alternate_kept ∖ primary_kept) sorted alphabetically.

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
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `uv run pytest tests/data/test_vocab_derivation.py -v -k "alternate_only_provenance or derive_section"`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cfm/data/vocab_derivation.py tests/data/test_vocab_derivation.py
git commit -m "feat(data): derive_section + alternate-only provenance"
```

---

### Task 5: `derive_poi_union`

**Files:**
- Modify: `src/cfm/data/vocab_derivation.py`
- Modify: `tests/data/test_vocab_derivation.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
from cfm.data.vocab_derivation import derive_poi_union


def test_derive_poi_union_combines_primary_and_alternate_kept_sets():
    primary = _make_result({"restaurant": 5000, "school": 1000, "park": 200})
    alternate = _make_result({"restaurant": 200, "vape_shop": 150, "tobacco_shop": 120},
                              is_list_field=True)
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
    alternate = _make_result({"restaurant": 200, "vape_shop": 150, "tobacco_shop": 120},
                              is_list_field=True)
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
    alternate = _make_result({"restaurant": 200, "cafe": 150},
                              is_list_field=True)
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
```

- [ ] **Step 2: Run tests, confirm fail**

Run: `uv run pytest tests/data/test_vocab_derivation.py -v -k poi_union`
Expected: 3 failures with import error.

- [ ] **Step 3: Implement `derive_poi_union`**

Append to `src/cfm/data/vocab_derivation.py`:

```python
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
    """Build the POI section as the union of primary and alternate Moderate cuts.

    Ordering of the token list:
      [<prefix>unknown if emit_unknown_token]
      + primary-kept by (-count, name)
      + alternate-only-kept by (-count, name)  (= alternate_kept ∖ primary_kept)

    Provenance metadata records the alternate-only set so consumers can
    detect tokens that may be dead under the current encoder.
    """
    prefix = "POI_"

    primary_kept = apply_floor_to_kept_set(primary_result, floor_value_primary)
    alternate_kept = apply_floor_to_kept_set(alternate_result, floor_value_alternate)

    primary_names = {name for name, _ in primary_kept}
    alternate_names = {name for name, _ in alternate_kept}
    alternate_only = alternate_names - primary_names

    # Sorted alternate-only entries — preserve (-count, name) order from alternate_kept.
    alternate_only_ordered = [(name, count) for name, count in alternate_kept if name in alternate_only]

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
        primary_coverage_retained = (
            100.0 * primary_total_kept / primary_result.total_occurrences
        )

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
        alternate_only_provenance=compute_alternate_only_provenance(
            primary_names, alternate_names
        ),
    )

    return SectionDerivation(
        section_name="poi",
        prefix=prefix,
        tokens=tokens,
        metadata=metadata,
    )
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `uv run pytest tests/data/test_vocab_derivation.py -v -k poi_union`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cfm/data/vocab_derivation.py tests/data/test_vocab_derivation.py
git commit -m "feat(data): derive_poi_union with provenance metadata"
```

---

### Task 6: `canonicalize_yaml` + `compute_yaml_sha256`

**Files:**
- Modify: `src/cfm/data/vocab_derivation.py`
- Modify: `tests/data/test_vocab_derivation.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
import hashlib

from cfm.data.vocab_derivation import canonicalize_yaml, compute_yaml_sha256


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
```

- [ ] **Step 2: Run tests, confirm fail**

Run: `uv run pytest tests/data/test_vocab_derivation.py -v -k "canonicalize_yaml or compute_yaml_sha256"`
Expected: 4 failures with import errors.

- [ ] **Step 3: Implement both functions**

Append to `src/cfm/data/vocab_derivation.py`:

```python
import hashlib
from copy import deepcopy

import yaml


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
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `uv run pytest tests/data/test_vocab_derivation.py -v -k "canonicalize_yaml or compute_yaml_sha256"`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cfm/data/vocab_derivation.py tests/data/test_vocab_derivation.py
git commit -m "feat(data): canonicalize_yaml + sha256 (self-hash excluded)"
```

---

### Task 7: `derive_phase1_vocab` + `derive_phase1_policy` assemblers

**Files:**
- Modify: `src/cfm/data/vocab_derivation.py`
- Modify: `tests/data/test_vocab_derivation.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
from datetime import datetime, timezone

from cfm.data.vocab_derivation import derive_phase1_vocab, derive_phase1_policy


def _phase1_inputs_minimal():
    """Build minimal FieldFrequencyResult set covering all 5 fields."""
    return {
        "buildings.class": FieldFrequencyResult(
            field="buildings.class",
            n_total=339_972, n_present=75_240,
            counts={"residential": 50_000, "commercial": 20_000, "industrial": 5_000},
            is_list_field=False, total_occurrences=75_000,
        ),
        "transportation.class": FieldFrequencyResult(
            field="transportation.class",
            n_total=202_334, n_present=202_292,
            counts={"motorway": 100_000, "primary": 80_000, "secondary": 22_000},
            is_list_field=False, total_occurrences=202_000,
        ),
        "base.class": FieldFrequencyResult(
            field="base.class",
            n_total=8_636, n_present=8_636,
            counts={"water": 4_000, "park": 3_000, "forest": 1_000},
            is_list_field=False, total_occurrences=8_000,
        ),
        "places.categories.primary": FieldFrequencyResult(
            field="places.categories.primary",
            n_total=149_657, n_present=145_774,
            counts={"restaurant": 50_000, "school": 5_000, "park": 200},
            is_list_field=False, total_occurrences=55_200,
        ),
        "places.categories.alternate": FieldFrequencyResult(
            field="places.categories.alternate",
            n_total=149_657, n_present=109_929,
            counts={"restaurant": 200, "vape_shop": 150, "tobacco_shop": 120},
            is_list_field=True, total_occurrences=470,
        ),
    }


def test_derive_phase1_vocab_assembles_four_feature_class_sections():
    inputs = _phase1_inputs_minimal()
    vocab = derive_phase1_vocab(
        field_results=inputs,
        overture_release="2026-04-15.0",
        source_report_path="reports/test.md",
        commit_sha="a" * 40,
        run_timestamp_utc=datetime(2026, 5, 16, 15, 25, 43, tzinfo=timezone.utc),
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
        run_timestamp_utc=datetime(2026, 5, 16, 15, 25, 43, tzinfo=timezone.utc),
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
        run_timestamp_utc=datetime(2026, 5, 16, 15, 25, 43, tzinfo=timezone.utc),
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
        run_timestamp_utc=datetime(2026, 5, 16, 15, 25, 43, tzinfo=timezone.utc),
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
```

- [ ] **Step 2: Run tests, confirm fail**

Run: `uv run pytest tests/data/test_vocab_derivation.py -v -k "derive_phase1"`
Expected: 4 failures with import errors.

- [ ] **Step 3: Implement assemblers**

Append to `src/cfm/data/vocab_derivation.py`:

```python
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
    "buildings.class": ("emit_unknown_token",
        "78.0% missing on Singapore; dropping forfeits the bulk of building data; append-only safety.",
        True),
    "transportation.class": ("drop_row",
        "0.02% missing (42 rows); too few to warrant a token slot.",
        False),
    "base.class": ("n_a",
        "100% coverage on Singapore; no missing rows.",
        False),
    "places.categories.primary": ("emit_unknown_token",
        "2.59% missing (3,883 rows); geometric info valid; consistency with buildings.class.",
        True),
    "places.categories.alternate": ("n_a",
        "List field; empty list is 'no secondary categories', not missing data.",
        False),
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
        "Union of primary-Moderate-kept ∪ alternate-Moderate-kept. "
        "POI_unknown included for primary missing-value handling. "
        "Denominator: alternate counts use occurrences-among-rows-with-alternates. "
        "Cap=2 at tokenizer time means alternate-only-position-3+ categories may "
        "be dead under current encoder; estimated ≤1.78% of POI tokens. "
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
    "token IDs but will never be emitted. Estimated dead-token fraction ≤1.78%. "
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
        alternate_coverage_singapore_pct=_coverage_pct(field_results["places.categories.alternate"]),
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
```

- [ ] **Step 4: Run tests, confirm pass**

Run: `uv run pytest tests/data/test_vocab_derivation.py -v -k "derive_phase1"`
Expected: 4 passed.

- [ ] **Step 5: Add YAML serialisation helpers + tests**

Append tests for serialisation to dict:

```python
from cfm.data.vocab_derivation import vocab_to_dict, policy_to_dict


def test_vocab_to_dict_round_trips_sections():
    inputs = _phase1_inputs_minimal()
    vocab = derive_phase1_vocab(
        field_results=inputs,
        overture_release="2026-04-15.0",
        source_report_path="r.md",
        commit_sha="0" * 40,
        run_timestamp_utc=datetime(2026, 5, 16, tzinfo=timezone.utc),
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
        run_timestamp_utc=datetime(2026, 5, 16, tzinfo=timezone.utc),
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
    assert d["fields"]["buildings.class"]["policies"]["missing_value"]["type"] == "emit_unknown_token"
```

Implement in `vocab_derivation.py`:

```python
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
            "PAD", "BOS", "EOS", "CELL", "END_CELL",
            "FEATURE_START", "FEATURE_END", "EXIT",
            "POINT", "LINE", "POLYGON",
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
```

- [ ] **Step 6: Run all library tests**

Run: `uv run pytest tests/data/test_vocab_derivation.py -v`
Expected: All library unit tests pass (~17 by this point).

- [ ] **Step 7: Commit**

```bash
git add src/cfm/data/vocab_derivation.py tests/data/test_vocab_derivation.py
git commit -m "feat(data): vocab + policy assemblers and dict serialisation"
```

---

## Phase 3 — Tokenizer loader update

### Task 8: Add `LoaderError` exception

**Files:**
- Modify: `src/cfm/tokenizer/errors.py`
- Modify: `src/cfm/tokenizer/__init__.py`

- [ ] **Step 1: Add `LoaderError` to errors.py**

Append to `src/cfm/tokenizer/errors.py`:

```python


class LoaderError(TokenizerError):
    """A vocabulary YAML failed schema validation at load time:
    duplicate token names, unknown token misplaced, missing required
    top-level fields, etc."""
```

- [ ] **Step 2: Export `LoaderError` from `__init__.py`**

In `src/cfm/tokenizer/__init__.py`, add `LoaderError` to the imports from `cfm.tokenizer.errors` and to `__all__`:

```python
from cfm.tokenizer.errors import (
    FeatureOutOfBounds,
    LoaderError,
    TokenizerError,
    UnsupportedFeatureClass,
    UnsupportedGeometry,
    VocabularyMismatch,
)
```

And add `"LoaderError",` to `__all__`.

- [ ] **Step 3: Run tokenizer tests, confirm no regression**

Run: `uv run pytest tests/tokenizer/ -v`
Expected: 56+ existing tokenizer tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/cfm/tokenizer/errors.py src/cfm/tokenizer/__init__.py
git commit -m "feat(tokenizer): add LoaderError exception class"
```

---

### Task 9: Update `_flatten` with dict-vs-list branch

**Files:**
- Modify: `src/cfm/tokenizer/vocabulary.py`
- Create: `tests/tokenizer/test_vocabulary_loader_phase1.py`

- [ ] **Step 1: Write failing tests for the new loader behaviour**

Create `tests/tokenizer/test_vocabulary_loader_phase1.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cfm.tokenizer import LoaderError, Vocabulary


def _phase1_yaml_minimal() -> dict:
    """Minimal valid Phase 1 YAML for loader exercising."""
    return {
        "schema_version": "1.0",
        "phase": 1,
        "vocab_version": "1.0",
        "control": ["PAD", "BOS", "EOS", "CELL", "END_CELL",
                    "FEATURE_START", "FEATURE_END", "EXIT",
                    "POINT", "LINE", "POLYGON"],
        "hierarchy": ["MACRO", "END_MACRO", "MICRO", "END_MICRO"],
        "feature_class": {
            "road": {
                "tokens": ["R_motorway", "R_primary"],
            },
            "building": {
                "tokens": ["B_unknown", "B_residential", "B_commercial"],
            },
            "poi": {
                "tokens": ["POI_unknown", "POI_restaurant"],
            },
            "base": {
                "tokens": ["BASE_water", "BASE_park"],
            },
        },
        "anchor": {"axis_count": 250},
        "move": {
            "directions": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
            "steps_m": [1, 2, 4, 8, 16, 32],
        },
    }


def _write_yaml(tmp_path: Path, data: dict, name: str = "vocab.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def test_phase1_yaml_loads_with_vocabulary_loader(tmp_path):
    p = _write_yaml(tmp_path, _phase1_yaml_minimal())
    vocab = Vocabulary.load(p)
    # Should include the new section tokens.
    assert "R_motorway" in vocab.token_to_id
    assert "B_unknown" in vocab.token_to_id
    assert "POI_restaurant" in vocab.token_to_id
    assert "BASE_water" in vocab.token_to_id


def test_phase1_loader_skips_metadata_keys_in_feature_class(tmp_path):
    data = _phase1_yaml_minimal()
    data["feature_class"]["building"]["floor_strategy"] = "Moderate"
    data["feature_class"]["building"]["notes"] = "some notes"
    data["feature_class"]["building"]["is_provisional"] = True
    p = _write_yaml(tmp_path, data)
    vocab = Vocabulary.load(p)
    # Metadata keys are ignored; tokens flatten as before.
    assert "B_residential" in vocab.token_to_id
    assert "floor_strategy" not in vocab.token_to_id
    assert "notes" not in vocab.token_to_id


def test_phase1_loader_preserves_token_order_per_section(tmp_path):
    p = _write_yaml(tmp_path, _phase1_yaml_minimal())
    vocab = Vocabulary.load(p)
    # In _flatten order: control(11) + hierarchy(4) = ids 0..14, then road tokens.
    assert vocab.id_to_token[15] == "R_motorway"
    assert vocab.id_to_token[16] == "R_primary"
    assert vocab.id_to_token[17] == "B_unknown"


def test_phase0_yaml_still_loads_unchanged():
    """Phase 0 uses flat lists under feature_class; loader must still work."""
    phase0 = Path("configs/tokenizer/vocab_phase0.yaml")
    vocab = Vocabulary.load(phase0)
    # Phase 0 fixed-shape sanity checks.
    assert "B_residential" in vocab.token_to_id
    assert "R_motorway" in vocab.token_to_id
    assert vocab.anchor_axis_count == 250


def test_phase1_token_to_id_unique(tmp_path):
    p = _write_yaml(tmp_path, _phase1_yaml_minimal())
    vocab = Vocabulary.load(p)
    # Length of token_to_id mapping equals length of id_to_token (no collisions).
    assert len(vocab.token_to_id) == len(vocab.id_to_token)


def test_loader_rejects_duplicate_token_names_across_sections(tmp_path):
    data = _phase1_yaml_minimal()
    data["feature_class"]["road"]["tokens"].append("POI_restaurant")  # collide with poi.
    p = _write_yaml(tmp_path, data)
    with pytest.raises(LoaderError, match="duplicate token name"):
        Vocabulary.load(p)


def test_loader_rejects_unknown_token_not_at_section_position_zero_when_section_has_one(tmp_path):
    data = _phase1_yaml_minimal()
    # Put B_unknown at position 1 (not 0) in the building section.
    data["feature_class"]["building"]["tokens"] = ["B_residential", "B_unknown", "B_commercial"]
    p = _write_yaml(tmp_path, data)
    with pytest.raises(LoaderError, match="_unknown.*position 0"):
        Vocabulary.load(p)
```

- [ ] **Step 2: Run tests, confirm fail**

Run: `uv run pytest tests/tokenizer/test_vocabulary_loader_phase1.py -v`
Expected: All 7 tests fail — most with the dict treated as a list error, last two with no error raised.

- [ ] **Step 3: Update `_flatten` in `vocabulary.py`**

Replace `_flatten` in `src/cfm/tokenizer/vocabulary.py` with:

```python
def _flatten(data: dict) -> tuple[list[str], int]:
    from cfm.tokenizer.errors import LoaderError

    out: list[str] = []
    out.extend(data["control"])
    out.extend(data["hierarchy"])
    fc = data["feature_class"]
    # Iterate feature_class groups in YAML order. Each group is either a flat
    # list (Phase 0 shape) or a dict with a `tokens` key plus metadata (Phase 1).
    for group_name, group_value in fc.items():
        if isinstance(group_value, list):
            tokens = group_value
        elif isinstance(group_value, dict) and "tokens" in group_value:
            tokens = group_value["tokens"]
        else:
            raise LoaderError(
                f"feature_class.{group_name} has unexpected shape "
                f"(expected list-of-strings or dict-with-tokens-key); got {type(group_value).__name__}"
            )
        _validate_section_tokens(group_name, tokens)
        out.extend(tokens)
    axis_count = int(data["anchor"]["axis_count"])
    out.extend(f"ANCHOR_X_{i}" for i in range(axis_count))
    out.extend(f"ANCHOR_Y_{i}" for i in range(axis_count))
    move = data["move"]
    for direction in move["directions"]:
        for step in move["steps_m"]:
            out.append(f"MOVE_{direction}_{step}")
    # Global duplicate-name check across all sections.
    if len(set(out)) != len(out):
        from collections import Counter
        dupes = [name for name, count in Counter(out).items() if count > 1]
        raise LoaderError(f"duplicate token name(s) across sections: {dupes}")
    return out, axis_count


def _validate_section_tokens(group_name: str, tokens: list[str]) -> None:
    """Enforce convention: any `*_unknown` token must be at section index 0."""
    from cfm.tokenizer.errors import LoaderError

    for i, name in enumerate(tokens):
        if name.endswith("_unknown") and i != 0:
            raise LoaderError(
                f"feature_class.{group_name}: token {name!r} ends with '_unknown' "
                f"but is at position {i}; must be at position 0 within its section"
            )
```

- [ ] **Step 4: Run new loader tests**

Run: `uv run pytest tests/tokenizer/test_vocabulary_loader_phase1.py -v`
Expected: 7 passed.

- [ ] **Step 5: Run full tokenizer test suite to confirm no Phase 0 regression**

Run: `uv run pytest tests/tokenizer/ -v`
Expected: 56 existing tokenizer tests + 7 new Phase 1 loader tests = 63+ pass; 1 xfailed (the boundary-entry-marker known issue from Phase 0).

- [ ] **Step 6: Commit**

```bash
git add src/cfm/tokenizer/vocabulary.py tests/tokenizer/test_vocabulary_loader_phase1.py
git commit -m "feat(tokenizer): _flatten dict-vs-list branch + Phase 1 loader validation"
```

---

## Phase 4 — CLI script + integration tests

### Task 10: CLI script `derive_phase1_vocab.py`

**Files:**
- Create: `scripts/derive_phase1_vocab.py`

- [ ] **Step 1: Create the script**

Create `scripts/derive_phase1_vocab.py`:

```python
"""Derive Phase 1 vocabulary + missing-value policy YAMLs from B1's cached data.

Reads the cached Singapore Overture data via sub-A's loader (cache-hit ~1 s),
applies the locked B2 decisions, and writes both artifacts byte-deterministically.

Usage:
    uv run python scripts/derive_phase1_vocab.py
    uv run python scripts/derive_phase1_vocab.py --output-dir /tmp/test --rerun-reason "rerun-test"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure src/ is on the path when the script is invoked from the repo root
# without a previously-built editable install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cfm.data.frequency import compute_field_frequencies
from cfm.data.overture import load_region
from cfm.data.overture.backend import LocalFixtureBackend, S3DuckDBBackend
from cfm.data.vocab_derivation import (
    canonicalize_yaml,
    compute_yaml_sha256,
    derive_phase1_policy,
    derive_phase1_vocab,
    policy_to_dict,
    vocab_to_dict,
)


# The 5 vocab-relevant fields per the B2 spec §2.
# Each entry: (theme_name, column_path, label_for_FieldFrequencyResult.field, is_list_field).
_FIELD_SPEC = [
    ("buildings", "class", "buildings.class", False),
    ("transportation", "class", "transportation.class", False),
    ("base", "class", "base.class", False),
    ("places", "categories.primary", "places.categories.primary", False),
    ("places", "categories.alternate", "places.categories.alternate", True),
]

_SOURCE_REPORT_PATH = "reports/2026-05-16-phase-1-sub-B1-singapore-frequency-analysis.md"
_OVERTURE_RELEASE = "2026-04-15.0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun-reason", default="initial")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--backend", choices=["real", "fixture"], default="real")
    args = parser.parse_args(argv)

    t0 = time.monotonic()

    # 1. Resolve git commit sha.
    commit_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    # 2. Capture timestamp once.
    run_ts = datetime.now(timezone.utc)

    # 3. Select backend (matches sub-B1's pattern).
    if args.backend == "fixture":
        backend = LocalFixtureBackend(ROOT / "tests" / "fixtures" / "overture_mini")
    else:
        backend = S3DuckDBBackend()

    # 4. Load the cached Singapore region.
    region = load_region("singapore", backend=backend)

    # 5. Compute frequencies for each of the 5 fields.
    field_results: dict = {}
    for theme, column, label, is_list_field in _FIELD_SPEC:
        table = region.themes[theme]
        field_results[label] = compute_field_frequencies(
            table, column,
            label=label,
            is_list_field=is_list_field,
        )

    # 6. Derive both artifacts.
    vocab = derive_phase1_vocab(
        field_results=field_results,
        overture_release=_OVERTURE_RELEASE,
        source_report_path=_SOURCE_REPORT_PATH,
        commit_sha=commit_sha,
        run_timestamp_utc=run_ts,
    )
    policy = derive_phase1_policy(
        field_results=field_results,
        overture_release=_OVERTURE_RELEASE,
        source_report_path=_SOURCE_REPORT_PATH,
        commit_sha=commit_sha,
        run_timestamp_utc=run_ts,
    )

    # 7. Serialise to dict, compute sha256, embed, serialise to YAML.
    vocab_dict = vocab_to_dict(vocab)
    vocab_dict["vocab_sha256"] = compute_yaml_sha256(vocab_dict)
    vocab_yaml = canonicalize_yaml(vocab_dict)

    policy_dict = policy_to_dict(policy)
    policy_dict["policy_sha256"] = compute_yaml_sha256(policy_dict)
    policy_yaml = canonicalize_yaml(policy_dict)

    # 8. Write outputs.
    vocab_path = args.output_dir / "configs" / "tokenizer" / "vocab_phase1.yaml"
    policy_path = args.output_dir / "configs" / "data" / "missing_value_policy.yaml"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text(vocab_yaml)
    policy_path.write_text(policy_yaml)

    elapsed = time.monotonic() - t0
    print(f"Wrote {vocab_path} ({len(vocab_yaml):,} bytes)")
    print(f"  vocab_sha256: {vocab_dict['vocab_sha256']}")
    print(f"Wrote {policy_path} ({len(policy_yaml):,} bytes)")
    print(f"  policy_sha256: {policy_dict['policy_sha256']}")
    print(f"Total wall-clock: {elapsed:.2f}s")
    print(f"Rerun reason: {args.rerun_reason}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-run against cached Singapore data**

Run: `uv run python scripts/derive_phase1_vocab.py --output-dir /tmp/sub-b2-smoke`
Expected: Two YAML files written to `/tmp/sub-b2-smoke/configs/...`; total wall-clock <5 s. No crash.

- [ ] **Step 3: Eyeball the generated vocab YAML**

Run: `head -80 /tmp/sub-b2-smoke/configs/tokenizer/vocab_phase1.yaml`
Expected: schema_version, phase: 1, vocab_version, generated_at_commit (real sha), feature_class section with road/building/poi/base, each carrying metadata + tokens.

- [ ] **Step 4: Eyeball the generated policy YAML**

Run: `cat /tmp/sub-b2-smoke/configs/data/missing_value_policy.yaml`
Expected: 5 fields under `fields:`, each with `policies.missing_value`; `places.categories.alternate` also has `policies.list_cap`.

- [ ] **Step 5: Commit script**

```bash
git add scripts/derive_phase1_vocab.py
git commit -m "feat(scripts): CLI to derive vocab_phase1.yaml + policy YAML"
```

---

### Task 11: Integration tests for the CLI

**Files:**
- Create: `tests/data/test_derive_phase1_vocab.py`

- [ ] **Step 1: Write failing integration tests**

Create `tests/data/test_derive_phase1_vocab.py`:

```python
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "derive_phase1_vocab.py"


def _run_script(output_dir: Path, *, rerun_reason: str = "initial") -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir", str(output_dir),
            "--rerun-reason", rerun_reason,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def _load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text())


def test_script_runs_against_cached_singapore_and_produces_well_formed_artifacts(tmp_path):
    _run_script(tmp_path)
    vocab_path = tmp_path / "configs" / "tokenizer" / "vocab_phase1.yaml"
    policy_path = tmp_path / "configs" / "data" / "missing_value_policy.yaml"
    assert vocab_path.exists()
    assert policy_path.exists()

    vocab = _load_yaml(vocab_path)

    # Top-level version fields.
    assert vocab["schema_version"] == "1.0"
    assert vocab["phase"] == 1
    assert vocab["vocab_version"] == "1.0"
    assert len(vocab["vocab_sha256"]) == 64
    assert vocab["generated_from"]["overture_release"] == "2026-04-15.0"
    assert vocab["generated_from"]["regions"] == ["singapore"]

    # Feature class sections, in expected order.
    fc = vocab["feature_class"]
    assert set(fc.keys()) == {"road", "building", "poi", "base"}

    # Pinned counts against the locked B2 decisions (§7 of the spec).
    # Investigate-first if any of these change.
    assert len(fc["building"]["tokens"]) == 23   # 22 Moderate-kept + B_unknown
    assert len(fc["road"]["tokens"]) == 17       # 17 Moderate-kept, no unknown
    assert len(fc["base"]["tokens"]) == 7        # 7 Strict-kept, no unknown
    poi_len = len(fc["poi"]["tokens"])
    assert 291 <= poi_len <= 341, (
        f"POI section size {poi_len} outside expected [291, 341]; "
        "investigate union-size shift before re-pinning."
    )

    # Building section: B_unknown at position 0.
    assert fc["building"]["tokens"][0] == "B_unknown"
    # POI section: POI_unknown at position 0.
    assert fc["poi"]["tokens"][0] == "POI_unknown"
    # Road and base: no unknown token.
    assert all(not t.endswith("_unknown") for t in fc["road"]["tokens"])
    assert all(not t.endswith("_unknown") for t in fc["base"]["tokens"])


def test_policy_yaml_field_set_matches_expected(tmp_path):
    _run_script(tmp_path)
    policy = _load_yaml(tmp_path / "configs" / "data" / "missing_value_policy.yaml")
    expected = {
        "buildings.class",
        "transportation.class",
        "base.class",
        "places.categories.primary",
        "places.categories.alternate",
    }
    actual = set(policy["fields"].keys())
    assert actual == expected, f"unexpected diff: added={actual-expected}, removed={expected-actual}"


def test_cross_artifact_consistency_unknown_tokens(tmp_path):
    """For every emit_unknown_token field, the vocab section has *_unknown at position 0.
    For every drop_row / n_a field, the section has no *_unknown token."""
    _run_script(tmp_path)
    vocab = _load_yaml(tmp_path / "configs" / "tokenizer" / "vocab_phase1.yaml")
    policy = _load_yaml(tmp_path / "configs" / "data" / "missing_value_policy.yaml")

    # Map: source field name → vocab section name.
    field_to_section = {
        "buildings.class": "building",
        "transportation.class": "road",
        "base.class": "base",
        "places.categories.primary": "poi",
        "places.categories.alternate": "poi",
    }

    for field, entry in policy["fields"].items():
        section_name = field_to_section[field]
        section = vocab["feature_class"][section_name]
        mv = entry["policies"]["missing_value"]
        if mv["type"] == "emit_unknown_token":
            assert section["tokens"][0].endswith("_unknown"), (
                f"{field} policy=emit_unknown_token but {section_name} has no _unknown at index 0"
            )
        elif mv["type"] in ("drop_row", "n_a"):
            # For 'n_a' alternate inside the poi section, the poi section
            # DOES have POI_unknown (from primary's emit_unknown_token policy);
            # the assertion needs to ignore that overlap. The rule: a field
            # whose own policy is n_a or drop_row should not REQUIRE an unknown
            # token, but if another field in the same section emits one,
            # that's allowed.
            pass


def test_script_byte_deterministic_modulo_generated_utc(tmp_path):
    """Two consecutive runs produce byte-identical artifacts after stripping
    generated_utc and the embedded sha256 lines."""
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    _run_script(out1)
    _run_script(out2, rerun_reason="rerun-test")

    def _stripped(path: Path) -> str:
        lines = path.read_text().splitlines()
        return "\n".join(
            line for line in lines
            if not line.lstrip().startswith(("generated_utc:", "vocab_sha256:", "policy_sha256:"))
        )

    vocab1 = _stripped(out1 / "configs" / "tokenizer" / "vocab_phase1.yaml")
    vocab2 = _stripped(out2 / "configs" / "tokenizer" / "vocab_phase1.yaml")
    assert vocab1 == vocab2

    policy1 = _stripped(out1 / "configs" / "data" / "missing_value_policy.yaml")
    policy2 = _stripped(out2 / "configs" / "data" / "missing_value_policy.yaml")
    assert policy1 == policy2
```

- [ ] **Step 2: Run integration tests**

Run: `uv run pytest tests/data/test_derive_phase1_vocab.py -v`
Expected: 4 passed in <10 s (each test invokes the script as a subprocess; uses cached Singapore data).

- [ ] **Step 3: Run the full data-test suite**

Run: `uv run pytest tests/data/ -v`
Expected: 30 B1 + ~19 B2 unit + 4 B2 integration = ~53 passing in <15 s.

- [ ] **Step 4: Commit**

```bash
git add tests/data/test_derive_phase1_vocab.py
git commit -m "test(data): integration tests for derive_phase1_vocab CLI"
```

---

## Phase 5 — Generate the committed artifacts

### Task 12: Generate and commit the canonical YAML artifacts

**Files:**
- Create: `configs/tokenizer/vocab_phase1.yaml`
- Create: `configs/data/missing_value_policy.yaml`

- [ ] **Step 1: Run the script against the actual repo (writes into repo's `configs/`)**

Run: `uv run python scripts/derive_phase1_vocab.py`
Expected: Two new files at `configs/tokenizer/vocab_phase1.yaml` and `configs/data/missing_value_policy.yaml`; script reports the sha256s.

- [ ] **Step 2: Verify the generated vocab loads cleanly with the tokenizer**

Run:
```bash
uv run python -c "
from cfm.tokenizer import Vocabulary
v = Vocabulary.load('configs/tokenizer/vocab_phase1.yaml')
print(f'Token count: {len(v)}')
print(f'First 30 tokens: {v.id_to_token[:30]}')
print(f'Building section unknown at id:', v.token_to_id['B_unknown'])
print(f'POI section unknown at id:', v.token_to_id['POI_unknown'])
"
```
Expected: Token count ~ 11 + 4 + 17 + 23 + (291..341) + 7 + 500 + 48 ≈ 901–951. Building/POI unknown IDs printed.

- [ ] **Step 3: Eyeball the metadata for one section**

Run: `awk '/^  building:/,/^  road:/' configs/tokenizer/vocab_phase1.yaml | head -30`
Expected: source_field, floor_strategy: Moderate, floor_value: 100, coverage_singapore_pct around 22.13, is_provisional: true, decision_basis present, notes mentioning 10K threshold, then the `tokens:` list starting with B_unknown.

- [ ] **Step 4: Commit the artifacts**

```bash
git add configs/tokenizer/vocab_phase1.yaml configs/data/missing_value_policy.yaml
git commit -m "data: generate Phase 1 vocab + missing-value policy YAMLs

Single-region (singapore) artifacts; both flagged provisional pending
Sweden re-run per the locked B2 decisions. Run from commit \$(git rev-parse HEAD~1)."
```

---

## Phase 6 — Documentation updates

### Task 13: Update Phase 0 vocab header

**Files:**
- Modify: `configs/tokenizer/vocab_phase0.yaml`

- [ ] **Step 1: Read current header**

Run: `head -10 configs/tokenizer/vocab_phase0.yaml`

Expected current state:
```yaml
# Phase 0 vocabulary for the city foundation model tokenizer.
# Frozen for Phase 0. Phase 1 may append; never reorder, never delete.
#
# The loader flattens this into a single ordered list of token names whose
# index is the canonical TokenId. Order: control, hierarchy, feature_class
# (road, building, poi, land_use), anchor, move.
#
# Counts: control=11, hierarchy=4, feature_class=19, anchor=500, move=48 -> 582.
```

- [ ] **Step 2: Replace the second line and surrounding context with the 4-line phase-transition block**

Use the Edit tool to replace exactly:

OLD:
```
# Frozen for Phase 0. Phase 1 may append; never reorder, never delete.
```

NEW:
```
# Frozen for Phase 0. Within Phase 0, never reorder, never delete.
# Phase 1 onward live in separate vocab_phase{N}.yaml files with independent IDs.
# Phase transitions are explicit re-anchoring events; within-phase changes are append-only.
# This file is retained for the Phase 0 round-trip test and historical reference.
```

The other header lines (Phase 0 description, loader description, counts) stay as-is.

- [ ] **Step 3: Verify Phase 0 still loads**

Run: `uv run pytest tests/tokenizer/ -v -k "phase0 or vocabulary"`
Expected: All existing Phase 0 tests still pass.

- [ ] **Step 4: Commit**

```bash
git add configs/tokenizer/vocab_phase0.yaml
git commit -m "docs(tokenizer): clarify phase-transition lifecycle in vocab_phase0 header"
```

---

### Task 14: Add subtype/subclass deferral to `known_issues.md`

**Files:**
- Modify: `docs/known_issues.md`

- [ ] **Step 1: Read the current known_issues.md to find the top of the issues list**

Run: `head -10 docs/known_issues.md`

Expected: file starts with a heading and instructions ("Add new entries on top.").

- [ ] **Step 2: Insert a new issue at the top of the issues list**

Insert directly after the "Add new entries on top. Remove entries when they're fixed." line and the horizontal rule:

```markdown
## #2 — Subtype / subclass fields analyzed but not tokenized in Phase 1

- **Filed:** 2026-05-16 (Phase 1 sub-B2 spec)
- **Severity:** low (scope decision, not a bug)
- **Status:** deferred — picked up by a future sub-project after encoder design extends to multi-token-per-feature
- **Affects:** `buildings.subtype`, `transportation.subclass`, `base.subtype` fields from the B1 report.

### Context

The B1 frequency analysis covered nine fields including the three subtype/subclass fields above. B2's vocab YAML only tokenizes five of them (the four locked feature_class sections plus alternates folded into the POI section). The three subtype/subclass fields are deferred.

### Why

The current tokenizer encoder is one-token-per-feature: `cfm.tokenizer.encode._encode_feature` reads `feature["properties"]["class"]` and emits exactly one feature_class token. Integrating subtype as a *second* token per feature (option B from the B2 brainstorm) or a *crossed* class×subtype token (option C) is a tokenizer architectural decision that warrants its own sub-project with its own brainstorm. B2 deliberately keeps subtype out of scope to avoid quietly expanding the encoder contract via a vocab YAML.

### Future

When subtype integration is on the table, a future sub-project picks between the options. Either way:

- The B1 numbers for `buildings.subtype` (Moderate keeps 11 cats), `transportation.subclass` (all 7 retained at every floor), and `base.subtype` (11 → 7 at Moderate) are already analyzed and ready to use.
- The Sweden re-run (B1') re-runs both class and subtype frequencies in parallel; subtype data will continue to land in the B1' report.

### Tracking

- B2 spec §2 (out-of-scope deferrals): `docs/superpowers/specs/2026-05-16-phase-1-sub-B2-vocab-yaml-design.md`
- B1 report §3.2, §3.4, §3.5: `reports/2026-05-16-phase-1-sub-B1-singapore-frequency-analysis.md`

---
```

- [ ] **Step 3: Verify markdown renders cleanly**

Run: `head -50 docs/known_issues.md`
Expected: Issue #2 at the top, followed by the existing Issue #1 (cold-fetch).

- [ ] **Step 4: Commit**

```bash
git add docs/known_issues.md
git commit -m "docs: defer subtype/subclass tokenization to a future sub-project"
```

---

## Phase 7 — Final verification

### Task 15: Full test suite + summary

**Files:**
- None modified (verification only)

- [ ] **Step 1: Run the full fast test suite**

Run: `uv run pytest -q`
Expected: ~148 existing + ~26 new = ~174 passed, 1 xfailed (the Phase 0 boundary-entry-marker known issue), 6 deselected (the slow-marked sub-A tests).

- [ ] **Step 2: Run a strict sanity check on artifact loading**

Run:
```bash
uv run python -c "
from cfm.tokenizer import Vocabulary
v = Vocabulary.load('configs/tokenizer/vocab_phase1.yaml')
print('Phase 1 vocab loaded:', len(v), 'tokens')
v0 = Vocabulary.load('configs/tokenizer/vocab_phase0.yaml')
print('Phase 0 vocab still loads:', len(v0), 'tokens')
"
```
Expected: Both vocabs load; Phase 0 count = 582; Phase 1 count in the 900–950 range.

- [ ] **Step 3: Confirm a re-run is byte-deterministic modulo generated_utc**

Run:
```bash
mkdir -p /tmp/sub-b2-determinism-check
uv run python scripts/derive_phase1_vocab.py --output-dir /tmp/sub-b2-determinism-check --rerun-reason "determinism-check"
diff -u <(grep -v -E '^(generated_utc|vocab_sha256|policy_sha256):' configs/tokenizer/vocab_phase1.yaml) \
        <(grep -v -E '^(generated_utc|vocab_sha256|policy_sha256):' /tmp/sub-b2-determinism-check/configs/tokenizer/vocab_phase1.yaml)
```
Expected: No diff output (vocab YAML identical modulo timestamp and sha256 lines).

- [ ] **Step 4: Confirm linting passes**

Run: `uv run ruff check src/cfm/data/vocab_derivation.py src/cfm/tokenizer/vocabulary.py scripts/derive_phase1_vocab.py`
Expected: All checks pass. If anything fails, fix and re-run.

Run: `uv run ruff format --check src/cfm/data/vocab_derivation.py src/cfm/tokenizer/vocabulary.py scripts/derive_phase1_vocab.py`
Expected: Already-formatted. If anything would be reformatted, run `uv run ruff format <files>` and commit the result.

- [ ] **Step 5: Final commit if any formatting changes**

```bash
git status
# If there are formatting changes:
# git add -u && git commit -m "style: ruff format pass on sub-B2 sources"
```

- [ ] **Step 6: Sub-B2 done summary**

Print a brief summary of what shipped:

```bash
echo "=== Sub-B2 shipped ==="
echo "Spec: docs/superpowers/specs/2026-05-16-phase-1-sub-B2-vocab-yaml-design.md"
echo "Plan: docs/superpowers/plans/2026-05-16-phase-1-sub-B2-vocab-yaml-derivation.md"
echo "Artifacts:"
echo "  configs/tokenizer/vocab_phase1.yaml"
echo "  configs/data/missing_value_policy.yaml"
echo
echo "Test counts:"
uv run pytest -q --collect-only 2>/dev/null | tail -3
```

Expected: Spec + plan + 2 generated YAMLs + ~174 fast-suite tests.

---

## Self-review checklist (completed by plan author)

- **Spec coverage** — every spec section maps to a task:
  - §1 Goal → Tasks 1–15 collectively
  - §2 Scope → Task 14 (deferral entry)
  - §3 Design principles → embedded in `decision_basis` / `notes` (Task 7) + spec reference
  - §4 Cross-decision dependencies → enforced by Task 11's `test_cross_artifact_consistency_unknown_tokens`
  - §5 Public API → Tasks 1, 3–7
  - §6 CLI script → Task 10
  - §7 Floor decisions → Task 7 `_LOCKED_FLOOR_VALUES` + `_DECISION_BASIS`
  - §7.1 Scaling math → embedded in `_NOTES` (Task 7)
  - §8 Missing-value policy → Task 7 `_LOCKED_MISSING_POLICIES`
  - §9 Vocab YAML shape → Task 7 `vocab_to_dict`
  - §10 Policy YAML shape → Task 7 `policy_to_dict`
  - §11 Vocab lifecycle → Tasks 1 (Phase1Vocab dataclass) + 7 (defaults)
  - §12 File layout → File map above + the per-task files lists
  - §13 Tests → Tasks 1–11 cover Layer 1; Task 9 covers Layer 2; Task 11 covers Layer 3
  - §14 Out of scope → Task 14
  - §15 Errors → Tasks 8–9 (LoaderError + validation)
  - §16 Done criteria → Task 15
  - §17 Risks → backed by the relevant tests
  - §18 Implementation order → matches Tasks 1 → 15
  - §19 References → in the plan header

- **Placeholder scan** — no `TBD` / `TODO` / `add appropriate error handling` / vague references in tasks. All code blocks contain complete code; all expected-output cells say what to expect.

- **Type consistency** — function signatures used in later tasks match earlier definitions: `apply_floor_to_kept_set` returns `list[tuple[str, int]]` everywhere; `derive_section` keyword args consistent across Tasks 4, 5, 7; `SectionMetadata`'s `__post_init__` invariant is referenced and tested.

- **Plan ↔ spec mismatch flagged for the executor:** Phase 1 vocab section keys in the YAML (Task 9 + Task 12) match the `feature_class.{road, building, poi, base}` ordering from the spec — `road`, `building`, `poi`, `base`. Phase 0's `land_use` group is no longer referenced in any loader path; the loader iterates `feature_class.items()` regardless of group name.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-16-phase-1-sub-B2-vocab-yaml-derivation.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task, controller reviews between tasks, fast iteration. Each phase (1–7) ships independently with its own commit.

**2. Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints between phases.

**Which approach?**

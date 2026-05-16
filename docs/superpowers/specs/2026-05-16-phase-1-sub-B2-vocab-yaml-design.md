# Phase 1 sub-project B2 — vocabulary YAML derivation design

- **Date:** 2026-05-16
- **Phase:** 1, sub-project B2 (vocabulary YAML from B1 frequency analysis)
- **Status:** Draft, pending user review
- **Owner:** umar

## 1. Goal

Derive the canonical Phase 1 vocabulary file `configs/tokenizer/vocab_phase1.yaml` from B1's Singapore frequency analysis report, alongside a sibling `configs/data/missing_value_policy.yaml` that records the per-field missing-value handling decision and the list-field cap policy. B2 ships two YAML artifacts plus the library that derives them; downstream consumers (sub-C tile extraction, future encoder versions, the model embedding layer, model checkpoint validators) read these as the single source of truth for Phase 1 tokens and the policies attached to them.

Both YAMLs are auto-generated, byte-reproducible from a code commit + B1's cached parquets, and append-only within the phase. Phase 0 → Phase 1 is a phase transition (re-anchoring permitted); a future Phase 1 → Sweden-extended is append-only (minor version bump).

## 2. Scope (in / out)

**In scope for B2:**

- A library `cfm.data.vocab_derivation` with pure functions for floor application to a kept set, section derivation, POI-union with provenance, YAML canonicalisation, and sha256 computation.
- A thin CLI script `scripts/derive_phase1_vocab.py` that loads B1's cached Singapore data, applies the locked decisions, and writes both YAMLs with embedded version metadata.
- The `vocab_phase1.yaml` deliverable covering 5 vocabulary-relevant fields:
  - `buildings.class` → `feature_class.building` section
  - `transportation.class` → `feature_class.road` section
  - `places.categories.primary` + `places.categories.alternate` → `feature_class.poi` section (union)
  - `base.class` → `feature_class.base` section (renamed from Phase 0's `land_use`)
- The `missing_value_policy.yaml` sibling artifact recording the policy decisions for all 5 fields plus the `places.categories.alternate` list-cap policy.
- A small update (~25 lines including new validation logic) to `src/cfm/tokenizer/vocabulary.py::_flatten` to handle the dict-vs-list branch for feature_class sections (Phase 1 sections wrap tokens in a `tokens: [...]` key with metadata siblings; Phase 0 sections remain flat).
- A 4-line header update to `configs/tokenizer/vocab_phase0.yaml` clarifying the phase-transition relationship.
- A `docs/known_issues.md` entry deferring subtype / subclass tokenization to a future sub-project.
- Pytest unit tests on the library, schema-validity + negative tests on the loader update, and one shape-only integration test that runs the script against B1's cache.

**Explicitly out of vocab scope** (analyzed in B1 but not a vocab field):

- **`divisions.country`** is intentionally not a vocab token. Its semantic role is conditioning (PRD §8: "country and admin region" lives in the conditioning vocabulary at the start of every training sequence, not in the geometric token vocabulary). Future readers seeing B1 §3.9 analyze it should not look for a corresponding feature_class section here. Country becomes a conditioning enum, designed separately.

**Out of scope for B2** (deferred):

- **Subtype / subclass tokenization.** `buildings.subtype`, `transportation.subclass`, `base.subtype` analyses live in the B1 report as scouting context. Their integration into the vocab requires an encoder design change (current encoder is one-token-per-feature; subtype integration would change that contract). Belongs to a future sub-project with its own brainstorm.
- **Class inference from context.** B1 §4 enumerated this as a missing-value-handling option; B2's policy YAML does not select it. A future labeling sub-project may revisit.
- **B1 re-run with combined primary+alternate counts.** Statistically cleaner POI floor derivation, but expands scope into B1 territory. Deferred to B1' Sweden re-run.
- **B1 re-run with position≤2 filter on alternates.** Refines the alternate kept set under the cap=2 policy. Deferred to B1'.
- **End-to-end encode/decode round-trip with Phase 1 vocab against hand-crafted GeoJSON.** Phase 0 round-trip tests stay against vocab_phase0.yaml. Phase 1 round-trip belongs to sub-C or a future sub-project.
- **Sub-C tile-extraction logic.** Sub-C reads `missing_value_policy.yaml` and applies the policies; B2 produces the policy artifact but does not implement the consumer.
- **Sweden frequency analysis.** Follows when sub-A's cold-fetch issue is resolved. B2 is structured so a Sweden re-run is an append-only minor version bump (vocab_version 1.0 → 1.1).

## 3. Design principles

Three principles emerged across the brainstorm and apply to every B2 decision plus future Phase 1 evolution. They are documented here once so future contributors who weren't in the brainstorm can apply them directly.

### 3.1 Cheap to keep, impossible to recover

Vocab YAML changes are append-only inside a phase. Adding a token allocates a new ID at the end and leaves existing IDs untouched, so existing model checkpoints remain compatible. Removing a token would require reordering or deletion, which invalidates every checkpoint that has weights at the deleted ID — an operation Phase 0's lock explicitly forbids.

The asymmetry matters at every decision point where data flows in only one direction:

- **Vocab token inclusion** (topic 1, 5 fields): keep categories now that might be valuable later; we can always tell the encoder to skip a token, but introducing one later is expensive — it forces a minor vocab_version bump and propagates to every consumer (sub-C, tokenizer, model checkpoint validator).
- **`<unknown>` token inclusion** (topic 2, 2 fields): including `B_unknown` and `POI_unknown` now is one slot per field; deciding we need them later means a new minor version every consumer must adopt.
- **Storage policy for `places.categories.alternate`** (topic 3): sub-C preserves all alternates in storage; truncation happens at tokenization where it's reversible. Once tile extraction is committed, the data is fixed; tokenizer policy can change.

The principle drove three decisions across two topics and is the dominant safety constraint when any single decision creates an irreversible data-loss path.

### 3.2 Append-only within a phase; re-anchoring permitted at phase transitions

Within a vocab phase (e.g., Phase 1), once locked, no reordering or deletion is permitted. New categories (e.g., from a Sweden re-run) are appended at the end of their section, get IDs after the previous-highest, and produce a minor version bump (`vocab_version: 1.0` → `1.1`).

At phase transitions (Phase 0 → Phase 1, or future Phase 1 → Phase 2), IDs may regenerate from scratch. A phase transition is an explicit retraining boundary; old checkpoints are not expected to load on the new vocab. Phase transitions are rare and warrant the re-anchoring cost.

This principle is what permits Phase 1 to be a standalone file (`vocab_phase1.yaml`) rather than a strict extension of `vocab_phase0.yaml`, and what permits the group renaming described in §3.3.

### 3.3 Phase transitions may rename slots for empirical accuracy; structural shape is preserved

A phase transition retains the high-level schema shape (control / hierarchy / feature_class / anchor / move sections, same hierarchy of subsections) but may rename a subsection to reflect what the data actually contains. Phase 0's `feature_class.land_use` was a guess at zoning-style content; the empirical reality from Overture's `base` theme is landscape/landcover. Phase 1 renames the slot to `feature_class.base` and updates the prefix from `L_` to `BASE_`.

Names that require commentary to make sense should be renamed at the next phase transition. Names that are accurate should be preserved.

## 4. Cross-decision dependencies

The locked decisions are not independent. Several pairs co-depend, and the artifact must enforce the joint constraints. The B2 implementation and the cross-artifact consistency test in §13 verify these.

- **Topic 3 (cap=2 at tokenizer time) ↔ Topic 4 (storage_policy: preserve_all).** The cap is reversible only because sub-C preserves all alternates in storage. If sub-C truncates at extraction, the cap value cannot be raised later without re-extraction. The policy YAML records both fields; sub-C must honor both. A single-source-of-truth invariant: cap_application=tokenizer_time iff storage_policy=preserve_all.

- **Topic 2 (emit_unknown_token policy) ↔ Topic 4 (`<unknown>`-first ordering).** Fields with `missing_value.type: emit_unknown_token` in the policy YAML must have a corresponding `*_unknown` token at position 0 of their vocab section. Fields with `missing_value.type: drop_row` or `n_a` must not. The cross-artifact consistency test enforces this and would fail loudly if one artifact were edited without updating the other.

- **Topic 1 (Moderate POI floor applied to both primary and alternate) ↔ Topic 4 (union POI section).** The POI section's `tokens` list is the union of two Moderate-cut kept sets. The integration test re-derives the union from the underlying counts and asserts byte equality against the YAML.

- **Topic 4 (`alternate_only_provenance` metadata) ↔ Topic 1 (which categories ended up in the union).** The provenance set is the set difference (alternate_kept ∖ primary_kept), computed as a single set-difference call. The integration test asserts this set is consistent with the union.

- **Topic 4 (vocab_sha256) ↔ Topic 4 (model checkpoint validator).** The sha256 is computed over canonicalized YAML content, with the sha256 field itself excluded from the canonicalization to break the cyclic dependency. The model checkpoint validator at inference time recomputes the sha256 of the loaded vocab file and compares against the stored checkpoint metadata. Mismatch = hard error.

- **Topic 4 (loader dict-vs-list branch) ↔ Phase 0 backward compatibility.** The updated `_flatten` must continue to load `vocab_phase0.yaml` unchanged (Phase 0 sections are flat lists). A schema-validity test enforces this; if the loader regresses on Phase 0, the Phase 0 round-trip test suite fails as a side effect.

- **Topic 1 (Strict for base.class, append-only deciding) ↔ Design principle §3.1.** base.class is the only field where append-only safety dominated marginal-cost-of-cut. The `decision_basis` metadata field captures this explicitly so future readers see the principle that drove the cut, not just the result.

- **The two `is_provisional` flags live in different artifacts and have different scope.** The vocab YAML's section-level `is_provisional` flag means "the floor strategy / kept-set may shift after Sweden lands." The policy YAML's per-field `is_provisional` flag means "the missing-value policy may shift after Sweden lands." A field can be provisional in one artifact but not the other (e.g., `base.class` vocab section is provisional; `base.class` missing-value policy is `n_a` and not provisional because base has no missing rows). Tooling that filters by `is_provisional` must specify which artifact.

## 5. Public API (library)

```python
from cfm.data.vocab_derivation import (
    SectionMetadata,
    FieldPolicy,
    ListFieldCap,
    SectionDerivation,
    Phase1Vocab,
    Phase1Policy,
    apply_floor_to_kept_set,
    derive_section,
    derive_poi_union,
    compute_alternate_only_provenance,
    derive_phase1_vocab,
    derive_phase1_policy,
    canonicalize_yaml,
    compute_yaml_sha256,
)
```

Key dataclasses (frozen):

```python
@dataclass(frozen=True)
class SectionMetadata:
    source_field: str | None              # e.g. "buildings.class" (None when source_fields is set)
    source_fields: tuple[str, ...] | None # for poi section only
    floor_strategy: str                   # "Moderate" | "Strict"
    floor_value: int                      # effective floor in source rows
    coverage_retained_pct: float          # 0..100; among present (non-null) source rows
    coverage_singapore_pct: float         # 0..100; fraction of source rows where the field is non-null
    total_kept: int
    is_provisional: bool
    decision_basis: str                   # compressed principle string
    notes: str                            # multi-line prose
    denominator_type: str | None          # "occurrences" | "rows" | "rows_with_non_empty"
    alternate_only_provenance: tuple[str, ...] | None  # poi section only

    def __post_init__(self) -> None:
        # Discriminated union: exactly one of {source_field, source_fields} must be set.
        # The POI section uses source_fields; all others use source_field.
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
    type: str                             # "emit_unknown_token" | "drop_row" | "n_a"
    rationale: str
    is_provisional: bool

@dataclass(frozen=True)
class ListFieldCap:
    field: str
    cap_value: int
    cap_application: str                  # "tokenizer_time" | "storage_time" | "not_applied"
    storage_policy: str                   # "preserve_all" | "truncate"
    dead_token_fraction_upper_bound: float
    caveat: str
    is_provisional: bool

@dataclass(frozen=True)
class SectionDerivation:
    section_name: str                     # "road" | "building" | "poi" | "base"
    prefix: str                           # "R_" | "B_" | "POI_" | "BASE_"
    tokens: tuple[str, ...]               # ordered final token names (with optional <unknown> at index 0)
    metadata: SectionMetadata
```

Functions:

```python
def apply_floor_to_kept_set(
    result: FieldFrequencyResult,
    floor_value: int,
) -> list[tuple[str, int]]:
    """Returns kept categories sorted deterministically by (-count, name)."""

def derive_section(
    *,
    section_name: str,
    prefix: str,
    field_result: FieldFrequencyResult,
    floor_value: int,
    missing_policy: str,
    decision_basis: str,
    notes: str,
    is_provisional: bool,
    extra_metadata: dict | None = None,
) -> SectionDerivation:
    """Builds a SectionDerivation. Prepends prefix to every category name.
    Prepends <prefix>unknown at index 0 iff missing_policy == 'emit_unknown_token'.
    """

def derive_poi_union(
    *,
    primary_result: FieldFrequencyResult,
    alternate_result: FieldFrequencyResult,
    floor_value_primary: int,
    floor_value_alternate: int,
    missing_policy: str,
) -> SectionDerivation:
    """Union of two Moderate-cut kept sets. Computes alternate_only_provenance.
    Ordering: <unknown> (if emit_unknown_token), then primary-kept by (-count, name),
    then alternate-only-kept by (-count, name) appended at the end."""

def compute_alternate_only_provenance(
    primary_kept: set[str],
    alternate_kept: set[str],
) -> tuple[str, ...]:
    """Set difference, sorted alphabetically."""

def derive_phase1_vocab(
    region: Region,
    *,
    overture_release: str,
    source_report_path: str,
    commit_sha: str,
    run_timestamp_utc: datetime,
    schema_version: str = "1.0",
    phase: int = 1,
    vocab_version: str = "1.0",
) -> Phase1Vocab:
    """Assembles the full vocab YAML structure. Pure; no I/O."""

def derive_phase1_policy(
    *,
    field_results: dict[str, FieldFrequencyResult],
    commit_sha: str,
    run_timestamp_utc: datetime,
    overture_release: str,
    source_report_path: str,
    schema_version: str = "1.0",
    phase: int = 1,
    policy_version: str = "1.0",
) -> Phase1Policy:
    """Assembles the full policy YAML structure. Pure; no I/O."""

def canonicalize_yaml(data: dict) -> str:
    """Deterministic YAML serialisation. Sorted dict keys; consistent flow style;
    no machine-generated comments. The output is byte-stable across runs."""

def compute_yaml_sha256(data: dict) -> str:
    """sha256 of canonicalised content. The `vocab_sha256` / `policy_sha256` field
    is excluded from the canonicalisation before hashing to break the cyclic
    dependency."""
```

## 6. CLI script — `scripts/derive_phase1_vocab.py`

Thin glue, ~80 lines:

1. Parse `--rerun-reason <str>` (default `"initial"`), `--output-dir <path>` (default repo root), `--backend {real,fixture}` (default `real`).
2. Resolve `commit_sha = git rev-parse HEAD`; capture `datetime.now(timezone.utc)` once.
3. `region = load_region("singapore", backend=...)`.
4. For each of the 5 fields, run `compute_field_frequencies` from B1's library (`cfm.data.frequency`).
5. Build the vocab via `derive_phase1_vocab(...)`. Compute and embed sha256.
6. Build the policy via `derive_phase1_policy(...)`. Compute and embed sha256.
7. Write `configs/tokenizer/vocab_phase1.yaml` and `configs/data/missing_value_policy.yaml`.
8. Print summary to stdout: paths, sha256s, total wall-clock.

The script does not invoke a re-fetch. Cache present → ~1 s end-to-end. Cache missing → falls through to sub-A's cold path (8.2 h known issue).

## 7. Floor decisions — locked per field

| Section | Source field(s) | Strategy | Floor (SG rows) | Tokens kept | Coverage retained | Decision basis |
|---|---|---|---:|---:|---:|---|
| road | transportation.class | Moderate | 202 | 17 | 99.93% | pedestrian-infrastructure distinctiveness over Strict's scaling-math |
| building | buildings.class | Moderate | 100 | 22 + B_unknown | 98.83% | marginal-cost elbow + building distinctiveness; revisit Moderate→Lenient post-Sweden |
| poi | places.categories.primary ∪ places.categories.alternate | Moderate each | 145 / 109 | ~290–340 + POI_unknown | 83.10% (primary) / 88.36% (alternate) | marginal-cost elbow on both; union for semantic-equivalence |
| base | base.class | Strict | 300 | 7 | 95.31% | append-only safety on small-N field |

All four sections carry `is_provisional: true`. The next subsection explains why: every locked floor's low-end scaling sits below PRD §5's 10,000-global-instance learnability threshold.

### 7.1 Scaling math against PRD §5's 10,000-global-instance threshold

PRD §5 sets the learnability bar at **10,000 global instances per kept category** — categories below this are at risk of being too rare for the model to learn reliably. Singapore alone supplies a fraction of eventual training data; the rule of thumb during this brainstorm assumed Singapore would constitute **1–5% of the full corpus** when Sweden, Sri Lanka, and other regions are added. Multiplying each section's floor by the inverse of that range gives the projected global instance count per kept category:

| Section | Floor (SG rows) | Global at 5% SG share | Global at 1% SG share | Both ends clear 10K? |
|---|---:|---:|---:|---|
| road (transportation.class) | 202 | 4,040 | 20,200 | **low end below 10K** |
| building (buildings.class) | 100 | 2,000 | 10,000 | **low end below 10K** |
| poi (places.primary) | 145 | 2,900 | 14,500 | **low end below 10K** |
| poi (places.alternate) | 109 | 2,180 | 10,900 | **low end below 10K** |
| base (base.class) | 300 | 6,000 | 30,000 | **low end below 10K** |

Every locked floor's low-end scaling lands **below** PRD §5's 10K threshold; the high end (1% Singapore share) clears it everywhere except `building` (exactly at) and `places.alternate` (marginally over). This is exactly what `is_provisional: true` is hedging against: Singapore alone provides marginal-but-learnable evidence per the band the user defined in topic 1; Sweden is the validator that should lift these counts safely above the threshold. The Sweden re-run (B1') is **required for de-provisioning**, not optional — see §14.

Each section's `decision_basis` and `notes` fields reference this scaling-math context explicitly so future readers can trace why each floor was locked under uncertainty.

## 8. Missing-value policy — locked per field

| Field | Policy | Rationale | Provisional |
|---|---|---|---|
| buildings.class | `emit_unknown_token` | 78.0% missing; dropping forfeits the bulk of building data; append-only safety. | yes |
| transportation.class | `drop_row` | 0.02% missing (42 rows); too few to warrant a token slot. | no |
| base.class | `n_a` | 100% coverage on Singapore. | no |
| places.categories.primary | `emit_unknown_token` | 2.59% missing (3,883 rows); geometric info valid; consistency with buildings.class. | yes |
| places.categories.alternate | `n_a` | List field; empty list is "no secondary categories", not missing data. | no |

`places.categories.alternate` list-cap policy: `cap_value: 2`, `cap_application: tokenizer_time`, `storage_policy: preserve_all`, `dead_token_fraction_upper_bound: 0.0178`, `is_provisional: true`.

Two new vocab tokens are introduced by these policies: `B_unknown` (buildings section), `POI_unknown` (poi section).

## 9. Vocab YAML shape — exemplar

```yaml
schema_version: 1.0
phase: 1
vocab_version: 1.0
generated_at_commit: <40-char sha>
generated_utc: 2026-05-DD
vocab_sha256: <sha of canonicalised content; this field is excluded from the hash>
generated_from:
  overture_release: 2026-04-15.0
  regions: [singapore]
  source_report: reports/2026-05-16-phase-1-sub-B1-singapore-frequency-analysis.md
phase_links:
  prev_phase_file: configs/tokenizer/vocab_phase0.yaml  # informational only; not loaded

control:
  - PAD
  - BOS
  - EOS
  - CELL
  - END_CELL
  - FEATURE_START
  - FEATURE_END
  - EXIT
  - POINT
  - LINE
  - POLYGON

hierarchy:
  - MACRO
  - END_MACRO
  - MICRO
  - END_MICRO

feature_class:
  building:
    source_field: buildings.class
    floor_strategy: Moderate
    floor_value: 100
    coverage_retained_pct: 98.83   # among present (non-null) rows; 75,240 row denominator
    coverage_singapore_pct: 22.13  # fraction of source rows where buildings.class is non-null
    total_kept: 23
    is_provisional: true
    decision_basis: "marginal-cost elbow + building distinctiveness"
    notes: |
      Singapore coverage 22.13% (78% missing); B_unknown included.
      Floor=100 SG rows scales to 2K–10K global at 5%–1% Singapore share; low
      end below PRD §5's 10,000-global-instance learnability threshold. B1'
      Sweden re-run is required for de-provisioning; revisit Moderate→Lenient
      (13 cheap cats) at the same time.
    tokens:
      - B_unknown
      - B_residential
      - B_commercial
      # ... 20 more in (-count, name) order

  road:
    source_field: transportation.class
    floor_strategy: Moderate
    floor_value: 202
    coverage_retained_pct: 99.93   # among present (non-null) rows
    coverage_singapore_pct: 99.98  # fraction of source rows where transportation.class is non-null
    total_kept: 17
    is_provisional: true
    decision_basis: "pedestrian-infrastructure distinctiveness over Strict's scaling-math"
    notes: |
      Closer call vs Strict; kept Moderate for pedestrian-infrastructure
      distinction (cycleway/footway band). Floor=202 SG rows scales to 4K–20K
      global at 5%–1% Singapore share; low end below PRD §5's 10,000-global-
      instance learnability threshold. B1' Sweden re-run is required for de-
      provisioning; revisit Moderate→Strict if Sweden's pedestrian counts
      don't lift these above 10K globally.
    tokens:
      - R_motorway
      - R_primary
      # ... 15 more

  poi:
    source_fields: [places.categories.primary, places.categories.alternate]
    floor_strategy: Moderate
    primary_floor_value: 145
    primary_coverage_retained_pct: 83.10   # among present primary rows
    primary_coverage_singapore_pct: 97.41  # fraction of places rows with non-null primary
    primary_total_kept: 221
    alternate_floor_value: 109
    alternate_coverage_retained_pct: 88.36 # among rows with non-empty alternate
    alternate_coverage_singapore_pct: 73.45 # fraction of places rows with non-empty alternate list
    alternate_total_kept: 267
    union_total_kept: <290..340>
    denominator_type: occurrences   # for alternate counts; primary uses rows
    alternate_only_provenance:
      - POI_<example_alt_only_category>
      # ... categories surviving alternate-Moderate but not primary-Moderate
    dead_token_fraction_upper_bound: 0.0178
    is_provisional: true
    decision_basis: "marginal-cost elbow on both columns; union for semantic-equivalence"
    notes: |
      Union of primary-Moderate-kept ∪ alternate-Moderate-kept.
      POI_unknown included for primary missing-value handling.
      Denominator: alternate counts use occurrences-among-rows-with-alternates.
      Cap=2 at tokenizer time means alternate-only-position-3+ categories may
      be dead under current encoder; estimated ≤1.78% of POI tokens.
      Floors of 145 (primary) and 109 (alternate) SG rows scale to 2.9K–14.5K
      and 2.18K–10.9K global respectively at 5%–1% Singapore share; both low
      ends below PRD §5's 10,000-global-instance learnability threshold. B1'
      Sweden re-run is required for de-provisioning.
    tokens:
      - POI_unknown
      - POI_restaurant
      # ... primary-kept in (-count, name) order
      # ... then alternate-only-kept in (-count, name) order

  base:
    source_field: base.class
    floor_strategy: Strict
    floor_value: 300
    coverage_retained_pct: 95.31   # among present rows (=all rows; field has 100% coverage)
    coverage_singapore_pct: 100.0  # fraction of source rows where base.class is non-null
    total_kept: 7
    is_provisional: true
    decision_basis: "append-only safety on small-N field"
    notes: |
      Small-N field (8,636 Singapore rows). Append-only safety dominated
      marginal-cost-of-cut. Floor=300 SG rows scales to 6K–30K global at
      5%–1% Singapore share; low end below PRD §5's 10,000-global-instance
      learnability threshold but in the marginal-but-learnable band. B1'
      Sweden re-run is required for de-provisioning; should specifically
      check whether the 7 dropped Lenient→Strict categories deserve
      appending.
    tokens:
      - BASE_water
      - BASE_park
      # ... 5 more

anchor:
  axis_count: 250

move:
  directions: [N, NE, E, SE, S, SW, W, NW]
  steps_m: [1, 2, 4, 8, 16, 32]
```

**Header update for `configs/tokenizer/vocab_phase0.yaml`.** Supersedes the existing "Frozen for Phase 0. Phase 1 may append; never reorder, never delete." line. The loader-description and token-count comments that follow stay as-is.

```yaml
# Frozen for Phase 0. Within Phase 0, never reorder, never delete.
# Phase 1 onward live in separate vocab_phase{N}.yaml files with independent IDs.
# Phase transitions are explicit re-anchoring events; within-phase changes are append-only.
# This file is retained for the Phase 0 round-trip test and historical reference.
```

## 10. Policy YAML shape — exemplar

```yaml
schema_version: 1.0
phase: 1
policy_version: 1.0
generated_at_commit: <40-char sha>
generated_utc: 2026-05-DD
policy_sha256: <sha>
generated_from:
  overture_release: 2026-04-15.0
  regions: [singapore]
  source_report: reports/2026-05-16-phase-1-sub-B1-singapore-frequency-analysis.md

# Per-field policies. Unified `policies` dict keyed by policy_type for extensibility.
fields:
  buildings.class:
    policies:
      missing_value:
        type: emit_unknown_token
        rationale: "78.0% missing on Singapore; dropping forfeits the bulk of building data; append-only safety."
        is_provisional: true

  transportation.class:
    policies:
      missing_value:
        type: drop_row
        rationale: "0.02% missing (42 rows); too few to warrant a token slot."
        is_provisional: false

  base.class:
    policies:
      missing_value:
        type: n_a
        rationale: "100% coverage on Singapore; no missing rows."
        is_provisional: false

  places.categories.primary:
    policies:
      missing_value:
        type: emit_unknown_token
        rationale: "2.59% missing (3,883 rows); geometric info valid; consistency with buildings.class."
        is_provisional: true

  places.categories.alternate:
    policies:
      missing_value:
        type: n_a
        rationale: "List field; empty list is 'no secondary categories', not missing data."
        is_provisional: false
      list_cap:
        cap_value: 2
        cap_application: tokenizer_time
        storage_policy: preserve_all
        dead_token_fraction_upper_bound: 0.0178
        caveat: |
          Moderate-cut survival (topic 1) counted alternates at all positions.
          Under cap=2 at tokenizer time, categories appearing only at position
          3+ have allocated token IDs but will never be emitted. Estimated
          dead-token fraction ≤1.78%. B1' Sweden re-run can optionally
          re-compute frequencies under a position≤2 filter to refine the
          kept set.
        is_provisional: true
```

**Enums:**

- `missing_value.type` ∈ {`emit_unknown_token`, `drop_row`, `n_a`}
- `list_cap.cap_application` ∈ {`tokenizer_time`, `storage_time`, `not_applied`}
- `list_cap.storage_policy` ∈ {`preserve_all`, `truncate`}

## 11. Vocab lifecycle

Three orthogonal version dimensions, each with its own semantics and bump cadence:

| Dimension | Bump trigger | Compatibility implication |
|---|---|---|
| `schema_version` | Change to metadata schema shape (new fields, enum changes) | Loader must support all known schema_version values, or fail loudly. |
| `phase` (integer) | Phase transition (Phase 0 → 1 → 2) | Re-anchoring permitted; checkpoints do not cross phase boundaries. |
| `vocab_version` (semver-style, phase-local) | See below | Within-phase append/patch progression; resets to `1.0` at every phase transition. |

`vocab_version` is **phase-local**: it tracks append/patch progression within a single phase. The `phase` field is the cross-phase identifier; `vocab_version` does not encode phase transitions. At a phase transition, both `phase` increments and `vocab_version` resets to `1.0`.

`vocab_version` semantics within a phase:

- **minor bump** (`1.0` → `1.1`): append-only category addition (e.g., Sweden lands and we extend `feature_class.*.tokens` lists). Prefix-compatible with checkpoints from any earlier minor version in the same phase.
- **patch bump** (`1.0.0` → `1.0.1`): metadata-only correction (no token list change, e.g., typo in `notes`). Fully checkpoint-compatible.

**Model checkpoint contract:**

Each checkpoint dumps `{vocab_path, vocab_sha256, schema_version, phase, vocab_version}` at training start. The inference-time validator:

- `vocab_sha256` mismatch → hard error.
- `phase` mismatch → hard error (this is the cross-phase guard; checkpoints never cross phase boundaries).
- `vocab_version` mismatch within the same phase → "compatible-but-extended" status iff the new vocab is prefix-compatible (first N IDs identical to the checkpoint's expectation); else hard error.

The prefix-compatibility check is a runtime operation on the loaded vocabs at inference time, not a B2 test.

## 12. Module and file layout

```
src/cfm/data/
└── vocab_derivation.py                          # library — dataclasses + pure functions

src/cfm/tokenizer/
├── vocabulary.py                                 # +25 lines for dict-vs-list branch + new validation
└── errors.py                                     # +1 LoaderError class

scripts/
└── derive_phase1_vocab.py                       # CLI

tests/data/
├── test_vocab_derivation.py                    # library unit tests
└── test_derive_phase1_vocab.py                 # integration test

tests/tokenizer/
└── test_vocabulary_loader_phase1.py            # loader update tests

configs/tokenizer/
├── vocab_phase0.yaml                            # +4-line header update
└── vocab_phase1.yaml                            # NEW

configs/data/
└── missing_value_policy.yaml                   # NEW

docs/
└── known_issues.md                              # +1 entry: subtype/subclass deferral
```

`src/cfm/data/__init__.py` and `src/cfm/tokenizer/__init__.py` get new exports.

## 13. Tests

**Layer 1 — Unit tests on library functions** (`test_vocab_derivation.py`, fast suite):

- `test_apply_floor_returns_kept_set_sorted_by_count_name`
- `test_apply_floor_ties_broken_alphabetically`
- `test_derive_section_includes_unknown_first_when_policy_emit_unknown_token`
- `test_derive_section_omits_unknown_when_policy_drop_row_or_na`
- `test_derive_section_metadata_fields_populated`
- `test_compute_union_provenance_set_difference`
- `test_compute_union_token_list_dedupes`
- `test_yaml_emitter_byte_deterministic`
- `test_yaml_emitter_includes_required_top_level_fields`
- `test_vocab_sha256_self_consistent`  — recompute over canonicalised content (sha256 field excluded); assert match
- `test_policy_yaml_schema_uses_unified_policies_dict`
- `test_policy_yaml_enums_valid`
- `test_dead_token_fraction_upper_bound_is_float`
- `test_section_metadata_rejects_both_source_field_and_source_fields_set` (negative; `__post_init__` raises ValueError)
- `test_section_metadata_rejects_neither_source_field_nor_source_fields_set` (negative; `__post_init__` raises ValueError)

**Layer 2 — Schema-validity + loader tests** (`test_vocabulary_loader_phase1.py`, fast suite):

Positive tests:
- `test_phase1_yaml_loads_with_vocabulary_loader`
- `test_phase1_loader_skips_metadata_keys_in_feature_class`
- `test_phase1_loader_preserves_token_order_per_section`
- `test_phase0_yaml_still_loads_unchanged`
- `test_phase1_token_to_id_unique`

Negative tests:
- `test_loader_rejects_duplicate_token_names_across_sections`
- `test_loader_rejects_unknown_token_not_at_section_position_zero_when_section_has_one`

**Layer 3 — Integration test** (`test_derive_phase1_vocab.py`, fast suite, shape-only):

- `test_script_runs_against_cached_singapore_and_produces_well_formed_artifacts`
  - Both YAMLs at expected paths under `tmp_path`.
  - Vocab YAML has 4 feature_class sections (`road`, `building`, `poi`, `base`) + control / hierarchy / anchor / move.
  - **Section-count pinning is against the currently-locked decisions in §7.** Deviations are investigated, not silently re-pinned. Expected counts:
    - `building.tokens`: 23 (22 Moderate-kept + B_unknown)
    - `road.tokens`: 17 (17 Moderate-kept; no unknown under drop_row policy)
    - `base.tokens`: 7 (7 Strict-kept; no unknown under n_a policy)
    - `poi.tokens`: ∈ [291, 341] (primary∪alternate Moderate-kept union + POI_unknown; exact size depends on category-name overlap not visible from B1's report text)
  - All four counts are pinned only against the current locked decisions; a B1' Sweden re-run that legitimately changes any of them must update this test deliberately, not bump the number silently.
  - Policy YAML has 5 field entries with valid enum values.
  - Both YAMLs have valid versioning fields and self-consistent sha256.

- `test_policy_yaml_field_set_matches_expected` — assert the set of keys in `policies["fields"]` equals the expected 5 fields exactly. Failure produces a diff listing added/removed fields rather than just a count mismatch (test-failure-as-documentation).

- `test_cross_artifact_consistency_unknown_tokens` — for every field in the policy YAML with `missing_value.type: emit_unknown_token`, assert the corresponding vocab section contains a `*_unknown` token at position 0. For every field with `drop_row` or `n_a`, assert no `*_unknown` token in the corresponding vocab section. Catches the failure mode of editing one artifact without updating the other.

- `test_script_byte_deterministic_modulo_generated_utc` — invoke the script twice into separate `tmp_path` directories, then assert the two output files are byte-identical after stripping `generated_utc` and `vocab_sha256` / `policy_sha256` lines. (The sha256 line changes only if the rest of the content changes, so after stripping `generated_utc`, the sha256s should also match — but stripping both is defensive.) Backstops the §16 done-criterion that re-runs are byte-deterministic.

**Fixture strategy:** library unit tests use hand-crafted `FieldFrequencyResult` instances. Integration test uses the cached Singapore B1 data via `load_region("singapore")` (cache-hit ~1 s); shape-only assertions to avoid coupling tests to empirical category names.

**Determinism:** all YAMLs byte-deterministic. A second `--rerun-reason "rerun-test"` run produces byte-identical files except `generated_utc` and any field that legitimately depends on the run timestamp.

**Estimated total:** ~15 unit + ~7 loader + ~4 integration tests = ~26 named tests. Fast suite runs in <5 s.

## 14. Out of scope — deferrals

| Item | Reason | Picked up by |
|---|---|---|
| Subtype / subclass tokenization | Requires encoder design change (one-token-per-feature contract) | Future sub-project; `known_issues.md` entry added. |
| Class inference from context | Separate labelling pipeline | Future sub-project if ever needed. |
| Combined primary+alternate counts floor for POI | Statistically cleaner but expands scope into B1 | B1' Sweden re-run (optional refinement; current union is correct enough to ship). |
| Position≤2 filter for alternate counts | Refines kept set under cap=2 | B1' Sweden re-run (optional refinement). |
| Phase 1 encode/decode round-trip with hand-crafted GeoJSON | Couples test to empirical category names | Sub-C or future sub-project. |
| Cross-version invariant tests against synthetic v1.1 | Runtime checkpoint validator handles this | Model training infra (Phase 2+). |
| Real Overture data fetch | Sub-A's cold-fetch deferred | Fix sub-A's cold-fetch before adding Sweden. |
| Revisit Moderate→Lenient on `buildings.class` (13 cheap cats) | All four sections carry `is_provisional: true`; Singapore-only scaling math has every floor's low end below PRD §5's 10K threshold (§7.1) | **B1' Sweden re-run — required for de-provisioning.** |
| Revisit Moderate→Strict on `transportation.class` (pedestrian band) | Same as above | **B1' Sweden re-run — required for de-provisioning.** |
| Revisit Strict→Lenient (7 cats) on `base.class` | Same as above | **B1' Sweden re-run — required for de-provisioning.** |
| Revisit POI Moderate floor (both columns) | Same as above | **B1' Sweden re-run — required for de-provisioning.** |

## 15. Errors

B2 raises standard Python exceptions; one new exception class is added to the tokenizer for loader validation:

- `ValueError` from library functions if input dataclass invariants are violated (negative count, unknown floor strategy name, etc.).
- `ValueError` from `derive_phase1_vocab` / `derive_phase1_policy` if a required field's `FieldFrequencyResult` is missing.
- `RuntimeError` from the CLI if `git rev-parse HEAD` fails — abort rather than write a vocab with a bogus commit sha.
- `LoaderError` (new in `cfm.tokenizer.errors`) when `Vocabulary.load` detects:
  - duplicate token name across sections;
  - a `*_unknown`-suffixed token at a position other than 0 within its section (when the section has one);
  - missing required top-level fields (schema_version, phase, vocab_version).

The loader detects `*_unknown` tokens by `endswith("_unknown")` — convention, documented in the loader code.

## 16. Done criteria

B2 is done when:

- `uv run pytest` passes (the full fast suite, ~148 existing + ~26 new = ~174 tests).
- `uv run python scripts/derive_phase1_vocab.py` produces `configs/tokenizer/vocab_phase1.yaml` and `configs/data/missing_value_policy.yaml` in ≤ 5 seconds wall-clock.
- Both YAMLs have valid sha256 self-consistency and all required versioning fields.
- `vocab_phase0.yaml` has the updated 4-line header.
- `docs/known_issues.md` has the subtype/subclass-deferral entry.
- The cross-artifact consistency test passes (no `<unknown>`-policy mismatch between policy YAML and vocab YAML).
- A second invocation of the script produces byte-identical artifacts (except `generated_utc`).
- The user has reviewed both artifacts and this spec and approved.

## 17. Risks specific to B2

- **POI union size uncertainty.** The exact union size depends on category-name overlap not visible from B1's report text. The integration test allows a range [291, 341]; a number outside this range indicates either a B1 data shift or a library bug. Mitigation: range-based assertion; investigate failures rather than re-pinning blindly.
- **Loader regression on Phase 0.** The `_flatten` change adds a dict-vs-list branch. A bug could break Phase 0's round-trip tests. Mitigation: explicit `test_phase0_yaml_still_loads_unchanged` test; Phase 0's existing 56 tokenizer tests run unchanged in CI.
- **`generated_utc` causing non-byte-identical re-runs.** Same issue B1 hit. The script captures the timestamp once at start and writes it into both YAMLs. Determinism tests exclude this field; strict byte-identity tests would fail across runs. Documented behaviour.
- **B1's `LocalFixtureBackend` cache-hits real Singapore data.** Same wart documented in the end-of-sub-B1 handoff (issue #4). B2's integration test inherits this; the test name "integration against fixture" is mildly misleading but the shape-only assertions are correct.
- **Sweden timing.** If sub-A's cold-fetch fix lands before B2, the spec needs a minor update (regions list grows) but the design is unchanged. If B2 ships first and Sweden lands later, vocab_version bumps from 1.0 to 1.1 as an append-only operation. Either ordering works.
- **YAML library determinism.** Python's `yaml` package can produce non-byte-identical output across versions / flow-style toggles. The `canonicalize_yaml` function pins flow style and key ordering explicitly; if drift still occurs across PyYAML patch versions, document the exception in the methodology and don't burn cycles on it (same caveat B1 made for matplotlib PNG metadata).

## 18. Implementation order (advisory)

1. Library skeleton (`vocab_derivation.py` with dataclass definitions and pure functions, no I/O).
2. Library unit tests (Layer 1), TDD on each function.
3. Loader update (`vocabulary.py` dict-vs-list branch + validation) + Layer 2 tests, TDD.
4. CLI script (`derive_phase1_vocab.py`) + Layer 3 integration test.
5. Generate the artifacts once locally; eyeball; commit if shape looks right.
6. Update `vocab_phase0.yaml` header (4-line change).
7. Add `docs/known_issues.md` entry for the subtype/subclass deferral.
8. Final full-suite test run.

## 19. References

- Sub-B1 spec: `docs/superpowers/specs/2026-05-16-phase-1-sub-B1-singapore-frequency-analysis-design.md`
- Sub-B1 report: `reports/2026-05-16-phase-1-sub-B1-singapore-frequency-analysis.md`
- End-of-sub-B1 handoff: `docs/handoffs/2026-05-16-end-of-sub-B1.md`
- Phase 0 vocab: `configs/tokenizer/vocab_phase0.yaml`
- Phase 0 tokenizer: `src/cfm/tokenizer/`
- B1 library: `src/cfm/data/frequency.py`
- PRD: `PRD.md`
- Auto-memory feedback files: `feedback_marginal_cost_of_cut.md`, `feedback_append_only_vocab_safety.md`

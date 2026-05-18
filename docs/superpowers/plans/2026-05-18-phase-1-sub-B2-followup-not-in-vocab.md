# Phase 1 sub-B2 Follow-up — `not_in_vocab` axis extension

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for inline checkpoint-driven execution (this plan is small enough that subagent-driven dispatch is overkill). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend B2's `_LOCKED_MISSING_POLICIES` from a single-axis `(type, rationale, is_provisional)` 3-tuple per field to a two-axis `{missing_value: PolicyAxis, not_in_vocab: PolicyAxis}` shape. Regenerate `configs/data/missing_value_policy.yaml` so sub-C can read both axes per the four-case rule.

**Architecture:** The design is locked in **sub-C spec §10.2** (`docs/superpowers/specs/2026-05-17-phase-1-sub-C-tile-extraction-design.md`). This plan does NOT re-design; it executes the migration. A new `PolicyAxis` dataclass holds `(type, rationale, is_provisional)`; `FieldPolicy` becomes `{field, missing_value: PolicyAxis, not_in_vocab: PolicyAxis}`. The YAML structure adds `policies.not_in_vocab` sibling to `policies.missing_value`. Sub-C's Task 0 gate (sub-C plan) verifies this follow-up has landed before sub-C implementation can begin.

**Tech Stack:** Python 3.11+, dataclasses, PyYAML, pytest, uv. Touches `src/cfm/data/vocab_derivation.py` + tests + `configs/data/missing_value_policy.yaml` + B2 spec.

**Design source of truth:** `docs/superpowers/specs/2026-05-17-phase-1-sub-C-tile-extraction-design.md` §10.2 (four-case `{missing_value, not_in_vocab}` table). The migration values come directly from that table — no re-decision in this plan.

**`is_provisional` semantic for the `not_in_vocab` axis (LOCKED):** `not_in_vocab.is_provisional` tracks the **underlying floor's provisionality**, NOT the policy-decision-itself provisionality. Reasoning: `not_in_vocab` is *defined by* which categories made the Phase 1 floor; if the floor is provisional (subject to Sweden-re-run de-provisioning per B2 spec §7.1 scaling math), then which categories trigger `not_in_vocab` is provisional too. All four Phase 1 vocab sections carry `is_provisional: true` per B2 spec §7 (the low-end scaling on every floor sits below PRD §5's 10K-global-instance threshold, pending Sweden), so all five `not_in_vocab.is_provisional` values are `True` in this plan. The `missing_value.is_provisional` values stay as-is — they reflect missing-rate stability across regions (a different axis), which is correctly per-field heterogeneous.

---

## Branch discipline

- All work on branch `phase-1-sub-B2-followup-not-in-vocab` (off `main`).
- Task-by-task commits with conventional-commit prefixes.
- Local-first; no PR flow; no push to remote.
- Merge to `main` via `git merge --no-ff` at end (Task F6), matching the sub-A/B1/B2/sub-C pattern.
- Inline execution in this session; no subagent dispatch.

---

## Test discipline

If a test fails because real data violates an assumed invariant, STOP and escalate (per `feedback_test_weakening_to_pass.md`). The migration is structural: we expect existing tests to fail because the dict shape and YAML shape change; updating those tests to reflect the new schema is correct. Tests that fail in *unexpected* ways are bugs in the migration.

---

## File map

**Modify:**
- `src/cfm/data/vocab_derivation.py` — add `PolicyAxis` dataclass; refactor `FieldPolicy`; extend `_LOCKED_MISSING_POLICIES` to four-case schema; update `derive_phase1_vocab` + `derive_phase1_policy` + `policy_to_dict` consumers.
- `tests/data/test_vocab_derivation.py` — update 3 affected tests; add new tests for `not_in_vocab` axis.
- `tests/data/test_derive_phase1_vocab.py` — update 1 affected test (`test_policy_yaml_field_set_matches_expected`).
- `configs/data/missing_value_policy.yaml` — regenerate via `scripts/derive_phase1_vocab.py`.
- `docs/superpowers/specs/2026-05-16-phase-1-sub-B2-vocab-yaml-design.md` — update §8 (Missing-value policy table) + §10 (Policy YAML exemplar).

**Note:** No new files. Pure extension of existing B2 artifacts.

---

## Four-case schema reference (from sub-C spec §10.2)

| Field | `missing_value.type` | `not_in_vocab.type` |
|---|---|---|
| `buildings.class` | `emit_unknown_token` | `emit_unknown_token` |
| `transportation.class` | `drop_row` | `drop_row` |
| `base.class` | `n_a` | `drop_row` |
| `places.categories.primary` | `emit_unknown_token` | `emit_unknown_token` |
| `places.categories.alternate` | `n_a` | `drop_element` |

Rationales for the new `not_in_vocab` axis values are listed under Task F2 Step 3 below.

---

## Phase F — B2 follow-up

### Task F1: Branch + clean-state verify

**Files:** none modified (verification only)

**Dependencies:** none

**Complexity:** trivial

- [ ] **Step 1: Verify clean working tree on `main`**

Run:
```bash
git status
git rev-parse --abbrev-ref HEAD
```
Expected: branch is `main`; no uncommitted changes. If anything else, abort and reconcile (the recent sub-C spec/plan commits should already be on main; if they're not, that's a sub-C-cycle issue to resolve first).

- [ ] **Step 2: Confirm fast-suite passes on baseline**

Run: `uv run pytest -q`
Expected: `187 passed, 6 deselected, 1 xfailed` (B2 merge baseline).

- [ ] **Step 3: Create B2 follow-up branch**

Run:
```bash
git checkout -b phase-1-sub-B2-followup-not-in-vocab
git rev-parse --abbrev-ref HEAD
```
Expected: branch is `phase-1-sub-B2-followup-not-in-vocab`.

No commit needed (no files modified).

---

### Task F2: Extend dataclass + dict + consumers atomically

**Files:**
- Modify: `src/cfm/data/vocab_derivation.py`

**Spec sections:** sub-C spec §10.2 (four-case table)

**Dependencies:** F1

**Complexity:** medium (atomic edit across 4 functions to avoid broken intermediate state)

> **Rationale for atomicity:** the dataclass shape (`FieldPolicy`), the source-of-truth dict (`_LOCKED_MISSING_POLICIES`), and three consumers (`derive_phase1_vocab`, `derive_phase1_policy`, `policy_to_dict`) all depend on each other's shape. Splitting across commits would leave the module non-importable between commits. Single commit is correct.

- [ ] **Step 1: Add `PolicyAxis` dataclass and refactor `FieldPolicy`**

In `src/cfm/data/vocab_derivation.py`, add `PolicyAxis` immediately before `FieldPolicy` (around line 45) and replace `FieldPolicy`:

```python
@dataclass(frozen=True)
class PolicyAxis:
    """One axis of a field-level policy: missing_value OR not_in_vocab.

    Per sub-C spec §10.2: each field carries TWO axes — what to do with NULL
    values (missing_value) and what to do with present-but-not-in-Phase-1-vocab
    values (not_in_vocab). The four-case table in sub-C spec §10.2 defines the
    valid (missing_value.type, not_in_vocab.type) combinations per field.
    """

    type: str  # "emit_unknown_token" | "drop_row" | "n_a" | "drop_element"
    rationale: str
    is_provisional: bool


@dataclass(frozen=True)
class FieldPolicy:
    """Per-field policy carrying both axes (sub-C spec §10.2 four-case schema)."""

    field: str
    missing_value: PolicyAxis
    not_in_vocab: PolicyAxis
```

- [ ] **Step 2: Update `_LOCKED_MISSING_POLICIES` to four-case schema**

Replace the current dict (lines ~326-355) with:

```python
_LOCKED_MISSING_POLICIES: dict[str, dict[str, tuple[str, str, bool]]] = {
    "buildings.class": {
        "missing_value": (
            "emit_unknown_token",
            (
                "78.0% missing on Singapore; dropping forfeits the bulk of building "
                "data; append-only safety."
            ),
            True,
        ),
        "not_in_vocab": (
            "emit_unknown_token",
            (
                "Below-Moderate-floor classes (~1.17% of present rows) map to "
                "B__UNK__ at tokenize time. is_provisional=True tracks the Moderate-100 "
                "floor's provisionality (B2 spec §7 + §7.1 scaling math: low-end below "
                "PRD §5's 10K-global-instance threshold; Sweden re-run required for "
                "de-provisioning). Phase 1.1 vocab expansion (e.g., Sweden) can promote "
                "originally-rare classes to their own tokens without re-extracting tiles "
                "per sub-C spec §4.1 cost-asymmetry."
            ),
            True,
        ),
    },
    "transportation.class": {
        "missing_value": (
            "drop_row",
            "0.02% missing (42 rows); too few to warrant a token slot.",
            False,
        ),
        "not_in_vocab": (
            "drop_row",
            (
                "Below-Moderate-floor classes (~0.07% of present rows) are dropped at "
                "sub-C raw level alongside NULL rows. is_provisional=True tracks the "
                "Moderate-202 floor's provisionality (B2 spec §7: kept Moderate over "
                "Strict for pedestrian-infrastructure distinction at the cusp of "
                "learnability; Sweden re-run required to confirm pedestrian counts "
                "lift these above 10K globally)."
            ),
            True,
        ),
    },
    "base.class": {
        "missing_value": (
            "n_a",
            "100% coverage on Singapore; no missing rows.",
            False,
        ),
        "not_in_vocab": (
            "drop_row",
            (
                "Sub-C drops below-Strict-300-floor base rows at the policy step "
                "(~4.69% of base rows below Strict). Sea-defining rows (class IN "
                "{ocean,strait,bay}; ~35 SG rows below floor) are correctly dropped "
                "from feature emission here — sea polygons are masks, not features; "
                "sea-mask uses pre-policy derive_sea_polygons view per sub-C spec "
                "§6 + §9.1. is_provisional=True tracks the Strict-300 floor's "
                "provisionality (B2 spec §7.1: low-end below PRD §5's 10K threshold; "
                "Sweden re-run should check whether the 7 dropped Lenient→Strict "
                "categories deserve appending)."
            ),
            True,
        ),
    },
    "places.categories.primary": {
        "missing_value": (
            "emit_unknown_token",
            "2.59% missing (3,883 rows); geometric info valid; consistency with buildings.class.",
            True,
        ),
        "not_in_vocab": (
            "emit_unknown_token",
            (
                "Below-Moderate-floor primary categories map to POI__UNK__ at tokenize "
                "time. is_provisional=True tracks the Moderate-145 floor's provisionality "
                "(B2 spec §7.1: 2.9K-14.5K global at 5%-1% SG share; low-end below PRD "
                "§5's 10K threshold; Sweden re-run required for de-provisioning). Same "
                "Phase 1.1 expansion benefit as buildings.class."
            ),
            True,
        ),
    },
    "places.categories.alternate": {
        "missing_value": (
            "n_a",
            "List field; empty list is 'no secondary categories', not missing data.",
            False,
        ),
        "not_in_vocab": (
            "drop_element",
            (
                "List field; tokenizer at encode time filters not-in-vocab elements "
                "per-element (NOT per-row drop). Sub-C stores full alternate list raw "
                "per storage_policy=preserve_all. is_provisional=True tracks the "
                "Moderate-109 floor's provisionality (B2 spec §7.1: 2.18K-10.9K global "
                "at 5%-1% SG share; low-end below PRD §5's 10K threshold; Sweden re-run "
                "required for de-provisioning + optional position≤2 filter refinement)."
            ),
            True,
        ),
    },
}
```

- [ ] **Step 3: Update `derive_phase1_vocab` consumers**

The `derive_phase1_vocab` function (around line 410) uses `_LOCKED_MISSING_POLICIES[field][0]` to extract the policy type. Update each of the four call sites (`missing_policy=_LOCKED_MISSING_POLICIES["<field>"][0]` → `missing_policy=_LOCKED_MISSING_POLICIES["<field>"]["missing_value"][0]`):

```python
# Before:
missing_policy=_LOCKED_MISSING_POLICIES["transportation.class"][0],
# After:
missing_policy=_LOCKED_MISSING_POLICIES["transportation.class"]["missing_value"][0],
```

Apply to all four occurrences (transportation, buildings, places.primary, base). `derive_phase1_vocab`'s behavior is unchanged — it still keys vocab-section construction off the `missing_value` axis only (vocab token emission is missing-value-policy-driven, not not-in-vocab-policy-driven; the latter affects tokenizer behavior at encode time, not vocab YAML emission).

- [ ] **Step 4: Update `derive_phase1_policy` to build FieldPolicy with both axes**

Replace the existing comprehension (lines ~501-509):

```python
field_policies = tuple(
    FieldPolicy(
        field=field,
        missing_value=PolicyAxis(
            type=axes["missing_value"][0],
            rationale=axes["missing_value"][1],
            is_provisional=axes["missing_value"][2],
        ),
        not_in_vocab=PolicyAxis(
            type=axes["not_in_vocab"][0],
            rationale=axes["not_in_vocab"][1],
            is_provisional=axes["not_in_vocab"][2],
        ),
    )
    for field, axes in _LOCKED_MISSING_POLICIES.items()
)
```

- [ ] **Step 5: Update `policy_to_dict` to emit both axes**

Replace the existing `policy_to_dict` (around line 613) field-policy-emission section:

```python
def policy_to_dict(policy: Phase1Policy) -> dict:
    """Convert a Phase1Policy to dict matching sub-C spec §10.2 four-case schema.

    Per-field shape:
      { policies:
          { missing_value: {type, rationale, is_provisional},
            not_in_vocab: {type, rationale, is_provisional},
            [list_cap: {...}] }
      }
    """
    fields: dict = {}
    for fp in policy.field_policies:
        fields[fp.field] = {
            "policies": {
                "missing_value": {
                    "type": fp.missing_value.type,
                    "rationale": fp.missing_value.rationale,
                    "is_provisional": fp.missing_value.is_provisional,
                },
                "not_in_vocab": {
                    "type": fp.not_in_vocab.type,
                    "rationale": fp.not_in_vocab.rationale,
                    "is_provisional": fp.not_in_vocab.is_provisional,
                },
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

- [ ] **Step 6: Update `cfm.data.__init__` to export `PolicyAxis`**

Modify `src/cfm/data/__init__.py` to add `PolicyAxis` to its re-exports alongside `FieldPolicy` (if `FieldPolicy` is exported; if not, no action needed).

- [ ] **Step 7: Verify module imports cleanly**

Run:
```bash
uv run python -c "
from cfm.data.vocab_derivation import (
    PolicyAxis, FieldPolicy, _LOCKED_MISSING_POLICIES,
    derive_phase1_policy, policy_to_dict,
)
sample = _LOCKED_MISSING_POLICIES['buildings.class']
assert 'missing_value' in sample and 'not_in_vocab' in sample
assert sample['missing_value'][0] == 'emit_unknown_token'
assert sample['not_in_vocab'][0] == 'emit_unknown_token'
print('OK: four-case schema imports clean')
"
```
Expected: `OK: four-case schema imports clean`.

- [ ] **Step 8: Commit**

```bash
git add src/cfm/data/vocab_derivation.py src/cfm/data/__init__.py
git commit -m "$(cat <<'EOF'
refactor(vocab_derivation): extend missing_value_policy to four-case schema

Per sub-C spec §10.2 four-case table: each field now carries TWO policy axes
(missing_value AND not_in_vocab) instead of just missing_value. Introduces
PolicyAxis dataclass; refactors FieldPolicy to {field, missing_value,
not_in_vocab}; extends _LOCKED_MISSING_POLICIES; updates derive_phase1_vocab
(still keys off missing_value only — vocab emission is unchanged),
derive_phase1_policy (builds both axes), and policy_to_dict (emits both axes
under per-field policies dict).

PREREQUISITE for sub-C: sub-C Task 0 gate verifies this schema is in place
before sub-C implementation can begin.

Atomic single-commit migration; dataclass + dict + 3 consumers all change
together to avoid broken intermediate state.
EOF
)"
```

---

### Task F3: Update + extend tests

**Files:**
- Modify: `tests/data/test_vocab_derivation.py`
- Modify: `tests/data/test_derive_phase1_vocab.py`

**Dependencies:** F2

**Complexity:** small–medium (4 existing tests update + ~4 new tests for not_in_vocab axis)

- [ ] **Step 1: Update `test_derive_phase1_policy_field_set_matches_expected`**

Locate this test (around `tests/data/test_vocab_derivation.py:388`). The current assertion checks `policy.field_policies` field-names match expected. Field name list is unchanged; the test continues to pass as-is. **Verify**:

Run: `uv run pytest tests/data/test_vocab_derivation.py::test_derive_phase1_policy_field_set_matches_expected -v`
Expected: pass (no change needed).

- [ ] **Step 2: Update `test_derive_phase1_policy_enum_values_per_field`**

This test (around `tests/data/test_vocab_derivation.py:408`) likely asserts `fp.type` per field. Update to assert `fp.missing_value.type` (and add asserts for `fp.not_in_vocab.type` per the four-case table):

```python
def test_derive_phase1_policy_enum_values_per_field():
    # ... (existing setup)
    policy = derive_phase1_policy(...)
    by_field = {fp.field: fp for fp in policy.field_policies}

    # missing_value axis (existing assertions, updated path):
    assert by_field["buildings.class"].missing_value.type == "emit_unknown_token"
    assert by_field["transportation.class"].missing_value.type == "drop_row"
    assert by_field["base.class"].missing_value.type == "n_a"
    assert by_field["places.categories.primary"].missing_value.type == "emit_unknown_token"
    assert by_field["places.categories.alternate"].missing_value.type == "n_a"

    # not_in_vocab axis (NEW assertions per sub-C spec §10.2 four-case table):
    assert by_field["buildings.class"].not_in_vocab.type == "emit_unknown_token"
    assert by_field["transportation.class"].not_in_vocab.type == "drop_row"
    assert by_field["base.class"].not_in_vocab.type == "drop_row"
    assert by_field["places.categories.primary"].not_in_vocab.type == "emit_unknown_token"
    assert by_field["places.categories.alternate"].not_in_vocab.type == "drop_element"
```

- [ ] **Step 3: Update `test_policy_to_dict_uses_unified_policies_dict`**

Around `tests/data/test_vocab_derivation.py:450`. The test asserts the YAML structure has `policies.missing_value` per field; extend to assert `policies.not_in_vocab` is also present:

```python
def test_policy_to_dict_uses_unified_policies_dict():
    # ... (existing setup builds Phase1Policy → dict)
    d = policy_to_dict(policy)
    buildings = d["fields"]["buildings.class"]
    assert "policies" in buildings
    assert "missing_value" in buildings["policies"]  # existing
    assert "not_in_vocab" in buildings["policies"]   # NEW
    # Both axes have the canonical PolicyAxis sub-fields
    for axis_name in ("missing_value", "not_in_vocab"):
        axis = buildings["policies"][axis_name]
        assert "type" in axis
        assert "rationale" in axis
        assert "is_provisional" in axis
```

- [ ] **Step 4: Update `test_policy_yaml_field_set_matches_expected`** (in `tests/data/test_derive_phase1_vocab.py:81`)

This test loads the regenerated YAML and asserts shape. Update to also assert each field has both axes:

```python
def test_policy_yaml_field_set_matches_expected(tmp_path):
    # ... (existing CLI invocation → tmp_path)
    policy = _load_yaml(tmp_path / "configs" / "data" / "missing_value_policy.yaml")
    expected_fields = {
        "buildings.class",
        "transportation.class",
        "base.class",
        "places.categories.primary",
        "places.categories.alternate",
    }
    assert set(policy["fields"].keys()) == expected_fields
    # Each field has both axes (NEW assertion):
    for field_name in expected_fields:
        field_block = policy["fields"][field_name]
        assert "policies" in field_block
        assert "missing_value" in field_block["policies"]
        assert "not_in_vocab" in field_block["policies"]
```

- [ ] **Step 5: Add new test asserting the four-case table values match**

Append to `tests/data/test_vocab_derivation.py`:

```python
def test_locked_missing_policies_four_case_table_matches_sub_c_spec_10_2():
    """Verify _LOCKED_MISSING_POLICIES matches sub-C spec §10.2 four-case table.
    Spec: docs/superpowers/specs/2026-05-17-phase-1-sub-C-tile-extraction-design.md §10.2
    """
    from cfm.data.vocab_derivation import _LOCKED_MISSING_POLICIES
    expected_table = {
        "buildings.class": ("emit_unknown_token", "emit_unknown_token"),
        "transportation.class": ("drop_row", "drop_row"),
        "base.class": ("n_a", "drop_row"),
        "places.categories.primary": ("emit_unknown_token", "emit_unknown_token"),
        "places.categories.alternate": ("n_a", "drop_element"),
    }
    for field, (expected_mv, expected_niv) in expected_table.items():
        axes = _LOCKED_MISSING_POLICIES[field]
        assert axes["missing_value"][0] == expected_mv, (
            f"{field}.missing_value: expected {expected_mv}, got {axes['missing_value'][0]}"
        )
        assert axes["not_in_vocab"][0] == expected_niv, (
            f"{field}.not_in_vocab: expected {expected_niv}, got {axes['not_in_vocab'][0]}"
        )
```

- [ ] **Step 6: Run full test suite, iterate until green**

Run: `uv run pytest -q`
Expected: 188 passed (187 baseline + 1 new `test_locked_missing_policies_four_case_table_matches_sub_c_spec_10_2`), 1 xfailed, 6 deselected. Watch for failures in tests that touched FieldPolicy field positions (legacy 3-tuple access patterns like `policy.field_policies[0].type` — update to `.missing_value.type`).

- [ ] **Step 7: Commit**

```bash
git add tests/data/test_vocab_derivation.py tests/data/test_derive_phase1_vocab.py
git commit -m "$(cat <<'EOF'
test(vocab_derivation): four-case schema assertions for FieldPolicy

Updates 3 existing tests (test_derive_phase1_policy_enum_values_per_field,
test_policy_to_dict_uses_unified_policies_dict, test_policy_yaml_field_set_matches_expected)
to reflect the new {field, missing_value, not_in_vocab} FieldPolicy shape.
Adds test_locked_missing_policies_four_case_table_matches_sub_c_spec_10_2
as a cross-reference assertion: the dict values match the canonical table
in sub-C spec §10.2 verbatim. Future spec/code drift caught immediately.
EOF
)"
```

---

### Task F4: Regenerate `missing_value_policy.yaml`

**Files:**
- Modify: `configs/data/missing_value_policy.yaml` (regenerated)

**Dependencies:** F2, F3

**Complexity:** trivial

- [ ] **Step 1: Run the derivation script**

Run:
```bash
uv run python scripts/derive_phase1_vocab.py --rerun-reason "B2-followup: add not_in_vocab axis"
```
Expected: completes in ~1s (cache-hit); prints paths to regenerated YAMLs.

- [ ] **Step 2: Verify YAML shape**

Run:
```bash
uv run python -c "
import yaml
with open('configs/data/missing_value_policy.yaml') as f:
    p = yaml.safe_load(f)
for fname, fblock in p['fields'].items():
    pol = fblock['policies']
    assert 'missing_value' in pol, f'{fname}: missing missing_value axis'
    assert 'not_in_vocab' in pol, f'{fname}: missing not_in_vocab axis'
    print(f'{fname}: missing_value={pol[\"missing_value\"][\"type\"]}, not_in_vocab={pol[\"not_in_vocab\"][\"type\"]}')
"
```
Expected (per sub-C spec §10.2 four-case table):
```
base.class: missing_value=n_a, not_in_vocab=drop_row
buildings.class: missing_value=emit_unknown_token, not_in_vocab=emit_unknown_token
places.categories.alternate: missing_value=n_a, not_in_vocab=drop_element
places.categories.primary: missing_value=emit_unknown_token, not_in_vocab=emit_unknown_token
transportation.class: missing_value=drop_row, not_in_vocab=drop_row
```

- [ ] **Step 3: Confirm vocab_phase1.yaml is byte-identical** (no spurious changes from the regeneration)

Run:
```bash
git diff configs/tokenizer/vocab_phase1.yaml
```
Expected: only `generated_at_commit` + `generated_utc` + `vocab_sha256` differ (the timestamp/commit fields update on every run; the rest of the vocab content is unchanged because the four-case schema doesn't affect vocab emission).

- [ ] **Step 4: Commit**

```bash
git add configs/data/missing_value_policy.yaml configs/tokenizer/vocab_phase1.yaml
git commit -m "$(cat <<'EOF'
data: regenerate missing_value_policy.yaml with four-case schema

policies.<field> now carries both missing_value and not_in_vocab axes per
sub-C spec §10.2 four-case table. vocab_phase1.yaml is unchanged in content
(metadata fields refresh as expected on regeneration: generated_at_commit,
generated_utc, vocab_sha256).
EOF
)"
```

---

### Task F5: Update B2 spec §8 + §10

**Files:**
- Modify: `docs/superpowers/specs/2026-05-16-phase-1-sub-B2-vocab-yaml-design.md`

**Dependencies:** F4

**Complexity:** small (documentation update; no design change — the four-case design lives in sub-C spec §10.2)

- [ ] **Step 1: Update B2 spec §8 (Missing-value policy table)**

Locate the §8 "Missing-value policy — locked per field" table. Add a `not_in_vocab` column to mirror sub-C spec §10.2's four-case table. Cite sub-C spec §10.2 as the canonical source. Example:

```markdown
| Field | Policy (missing_value) | Policy (not_in_vocab) | Rationale | Provisional |
|---|---|---|---|---|
| buildings.class | `emit_unknown_token` | `emit_unknown_token` | 78.0% missing... | yes |
| transportation.class | `drop_row` | `drop_row` | 0.02% missing... | no |
| ...
```

Add a paragraph noting: "The `not_in_vocab` axis was added in the B2 follow-up (2026-05-18) to support sub-C's four-case policy application per sub-C spec §10.2. The four-case rule is authoritative there; this table reflects it."

- [ ] **Step 2: Update B2 spec §10 (Policy YAML shape exemplar)**

Replace the YAML exemplar's per-field `policies` block to include both axes:

```yaml
fields:
  buildings.class:
    policies:
      missing_value:
        type: emit_unknown_token
        rationale: "78.0% missing on Singapore; ..."
        is_provisional: true
      not_in_vocab:
        type: emit_unknown_token
        rationale: "Symmetric extension of missing_value; ..."
        is_provisional: true
  # ... (other fields similarly)
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-05-16-phase-1-sub-B2-vocab-yaml-design.md
git commit -m "$(cat <<'EOF'
docs(sub_b2): document not_in_vocab axis in §8 + §10 (sub-C-driven follow-up)

§8 missing-value policy table gains a not_in_vocab column matching sub-C
spec §10.2's four-case table. §10 policy YAML shape exemplar gains the
not_in_vocab block. The four-case design is authoritative in sub-C spec
§10.2; this update brings B2 spec into agreement with the regenerated
artifact + extended _LOCKED_MISSING_POLICIES.
EOF
)"
```

---

### Task F6: Full suite + merge

**Files:** none modified (verification only); merge

**Dependencies:** F2–F5

**Complexity:** trivial

- [ ] **Step 1: Run full fast test suite**

Run: `uv run pytest -q`
Expected: all green (187 prior + ~1 new test for the four-case-table assertion; 1 xfailed).

- [ ] **Step 2: Lint + format check**

Run: `uv run ruff check src/cfm/data/vocab_derivation.py`
Expected: clean.

Run: `uv run ruff format --check src/cfm/data/vocab_derivation.py`
Expected: already-formatted.

- [ ] **Step 3: Verify sub-C Task 0 gate would now pass**

Run:
```bash
uv run python -c "
from cfm.data.vocab_derivation import _LOCKED_MISSING_POLICIES
sample = _LOCKED_MISSING_POLICIES['buildings.class']
assert isinstance(sample, dict)
assert 'missing_value' in sample and 'not_in_vocab' in sample
print('B2 follow-up OK: four-case schema present')
"
```
Expected: `B2 follow-up OK: four-case schema present`.

Run:
```bash
uv run python -c "
import yaml
with open('configs/data/missing_value_policy.yaml') as f:
    policy = yaml.safe_load(f)
buildings = policy['fields']['buildings.class']['policies']
assert 'missing_value' in buildings and 'not_in_vocab' in buildings
print('YAML OK: not_in_vocab axis present')
"
```
Expected: `YAML OK: not_in_vocab axis present`.

- [ ] **Step 4: Merge to main**

Run:
```bash
git checkout main
git merge --no-ff phase-1-sub-B2-followup-not-in-vocab -m "merge: B2 follow-up — _LOCKED_MISSING_POLICIES + missing_value_policy.yaml four-case schema (sub-C prereq)"
git log --oneline -5
```
Expected: merge commit at HEAD.

- [ ] **Step 5: Done summary**

Run:
```bash
echo "=== B2 follow-up shipped ==="
echo "Plan: docs/superpowers/plans/2026-05-18-phase-1-sub-B2-followup-not-in-vocab.md"
echo "Code: src/cfm/data/vocab_derivation.py (PolicyAxis + FieldPolicy refactor + _LOCKED_MISSING_POLICIES extension + 3 consumers)"
echo "Tests: tests/data/test_vocab_derivation.py + tests/data/test_derive_phase1_vocab.py"
echo "Regenerated: configs/data/missing_value_policy.yaml"
echo "B2 spec updated: §8 + §10"
echo
echo "Sub-C unblocked. Next: subagent-driven execution of"
echo "docs/superpowers/plans/2026-05-17-phase-1-sub-C-tile-extraction.md"
```

---

## Self-review checklist (completed by plan author)

- **Design source of truth:** sub-C spec §10.2 four-case table. This plan does NOT re-design; it executes. The Task F2 Step 2 dict values match the table verbatim, and Task F3 Step 5 adds an assertion test (`test_locked_missing_policies_four_case_table_matches_sub_c_spec_10_2`) that catches future drift.
- **Atomicity:** F2's single commit covers dataclass + dict + 3 consumers because they depend on each other's shape; splitting would leave the module non-importable between commits.
- **Test coverage:** 3 existing tests updated (enum_values, policy_to_dict, yaml_field_set), 1 new test added (four-case table assertion). The `test_derive_phase1_policy_field_set_matches_expected` test continues to pass without modification (field-name set is unchanged).
- **Spec coverage:** B2 spec §8 + §10 updated to document the new axis; cites sub-C spec §10.2 as canonical source.
- **Sub-C unblocking:** F6 Step 3 verifies the exact assertions that sub-C plan's Task 0 will run.
- **Branch discipline:** F1 creates the branch; F6 merges via `--no-ff`. Matches sub-A/B1/B2/sub-C pattern.
- **No placeholders:** every step contains the actual code or command.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-18-phase-1-sub-B2-followup-not-in-vocab.md`.**

Recommended execution: **Inline via `superpowers:executing-plans`** (the plan is 6 small tasks, fully within one session's scope; subagent-dispatch overhead isn't justified).

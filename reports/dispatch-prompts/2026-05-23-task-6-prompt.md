# Task 6 implementer dispatch prompt

**Status:** Draft v2; pending reviewer read-through / approval before dispatch.
**Target:** General-purpose subagent / Codex agent.
**Suggested model:** Sonnet-class.
**Branch:** `phase-1-sub-F-micro-tokenizer` (base includes Task 2 close commit `5428704`).

> The prompt below is the verbatim text to give the implementer agent. Everything between the `===` markers is the agent's prompt body.

===

Task: Sub-F Task 6 - BP6 six-axis version manifest + SOURCE semantic verification, Halt 6 surface only.

You are working in `/Users/umaraslam/Projects/Bonzai-OSM` on branch `phase-1-sub-F-micro-tokenizer`. You are not alone in the codebase: do not revert edits made by others; inspect current state and work with it. Do not push. Do not create a PR. Do not proceed past Halt 6 approval. Stop at Halt 6 with a WIP commit and a halt report.

## Preconditions

- Branch: `phase-1-sub-F-micro-tokenizer`.
- Task 1 is closed and `configs/sub_f/semantic_vocab.yaml` is `LOCKED`.
- Task 4 is closed and `configs/sub_f/unknown_family.yaml` plus BP1/BP4/dataloader portions of `configs/sub_f/sentinel_inventory.yaml` are `LOCKED`.
- Task 2 is closed at `5428704`; `configs/sub_f/encoding_primitives.yaml` is `LOCKED`; BP2 block `300..1499` is `LOCKED`; BP7 block `1500..1599` remains `PLACEHOLDER`.
- Do not push. Do not PR. Do not continue to Task 8 writer work.

## Non-negotiable discipline

- Halt-on-defect: if an audit or verification contradicts the plan/spec, STOP and report `BLOCKED`. Do not silently adapt the implementation.
- Verify-before-lock: Halt 6 is specifically a verification halt. Surface file:line evidence for upstream contracts before proposing reviewer locks.
- Cascade resolution: if sub-D or sub-A source contradicts the plan/spec, classify it under spec section 9.6.1 before changing behavior.
- No final `feat:` commit before reviewer approval. This task exits with `DONE_WITH_CONCERNS` at Halt 6 and a `wip:` commit.

## Pre-dispatch audits

### Audit step 1: confirm sub-D VersionNamespace enum shape

Run:

```bash
uv run python -c "from cfm.data.sub_d.versions import VersionNamespace; print([m.name for m in VersionNamespace]); print([m.value for m in VersionNamespace])"
```

Expected:

```text
['ARTIFACT_FORMAT', 'DATA_SHAPE', 'VOCAB', 'DERIVATION', 'VALIDATOR']
['artifact_format', 'data_shape', 'vocab', 'derivation', 'validator']
```

This confirms Task 6's extension mechanism is a single enum-add: add `SOURCE = "source"`.

If `SOURCE` already exists or the member set differs: STOP, report BLOCKED. The branch state no longer matches the approved Task 6 base.

### Audit step 2: confirm sub-D compare_version signature

Run:

```bash
uv run python -c "import inspect; from cfm.data.sub_d.versions import compare_version; print(inspect.signature(compare_version))"
```

Expected: `compare_version(namespace, expected, actual)` with one `VersionNamespace` and two `VersionRef` arguments.

If the helper is kwarg-based, string-only, or otherwise not the enum/VersionRef shape: STOP, report BLOCKED. That would reopen the Plan Revision 1 classification.

### Audit step 3: confirm SOURCE release pin source-of-truth

Run:

```bash
uv run python -c "import yaml; from pathlib import Path; d=yaml.safe_load(Path('configs/data/overture_release.yaml').read_text()); print(d)"
grep -n "single source of truth\|release identifier\|Currently pinned" docs/data/overture_pinning_policy.md
```

Expected:
- `configs/data/overture_release.yaml` has `release: "2026-04-15.0"` plus release date/schema fields.
- Pinning policy states `configs/data/overture_release.yaml` is the single source of truth and `release` is the release identifier.

If the pin file or policy is absent or ambiguous: STOP, report BLOCKED. Do not invent a SOURCE semantic.

### Audit step 4: confirm sub-E provenance/manifest hashing precedent

Run:

```bash
grep -n "SUB_E_EXCLUDED_FROM_SHA\|compute_sha256_excluding\|provenance_sha256" src/cfm/data/sub_e/provenance.py
grep -n "manifest_sha256\|SUB_E_EXCLUDED_FROM_SHA" src/cfm/data/sub_e/manifest.py
```

Expected:
- Sub-E uses `cfm.data.determinism.compute_sha256_excluding`.
- `SUB_E_EXCLUDED_FROM_SHA` excludes live-clock fields and final-segment `*_sha256` fields.
- Manifest self-integrity uses the same exclusion table.

If this precedent moved, inspect the new source and cite the new file:line. If no equivalent exists: STOP, report BLOCKED.

### Audit step 5: confirm ID namespace state before Task 6

Run:

```bash
uv run python -c "import yaml; d=yaml.safe_load(open('configs/sub_f/sentinel_inventory.yaml')); print(d['_status']); print(d['bp2_encoding_primitives']['status'], d['bp2_encoding_primitives']['start_id'], d['bp2_encoding_primitives']['end_id']); print(d['bp7_boundary_ref_placeholder']['status'], d['bp7_boundary_ref_placeholder']['start_id'], d['bp7_boundary_ref_placeholder']['end_id'])"
```

Expected:
- `_status` is `LOCKED`.
- BP2 is `LOCKED at Halt 2 approval`, range `300..1499`.
- BP7 remains `PLACEHOLDER`, range `1500..1599`.

If the ID namespace drifted: STOP, report BLOCKED.

### Audit step 6: confirm sub-C output version/hash surface for SOURCE decision

Run:

```bash
uv run python -c "import yaml; from pathlib import Path; p=Path('data/processed/sub_c/2026-04-15.0/singapore/manifest.yaml'); d=yaml.safe_load(p.read_text()); print('manifest_exists', p.exists()); print('sub_c_schema_version', d.get('sub_c_schema_version')); print('release', d.get('release')); print('commit_sha', d.get('initial_extraction', {}).get('commit_sha')); print('tile_count', len(d.get('tiles', []))); print('first_tile_keys', sorted(d.get('tiles', [{}])[0].keys()))"
grep -n "class RegionManifest\|sub_c_schema_version\|aggregate_tile_inventory\|compute_sha256_excluding" src/cfm/data/sub_c/manifest.py
```

Expected:
- Cached sub-C Singapore has `manifest.yaml`.
- Manifest exposes `sub_c_schema_version`, `release`, `initial_extraction.commit_sha`, and per-tile `provenance_sha256`.
- `src/cfm/data/sub_c/manifest.py` shows the manifest schema and tile provenance hash aggregation path.

This audit does not lock SOURCE semantics. It supplies evidence for the Halt 6 reviewer question: whether SOURCE should mean only the Overture release pin, or Overture release pin plus a direct sub-C output version/hash reference.

If cached sub-C has no region manifest or no stable tile provenance chain: STOP, report BLOCKED. If it lacks only an explicit top-level `manifest_sha256`, continue and surface that limitation.

## Implementation scope

Modify:
- `src/cfm/data/sub_d/versions.py`

Create:
- `src/cfm/data/sub_f/versions.py`
- `src/cfm/data/sub_f/provenance.py`
- `src/cfm/data/sub_f/manifest.py`
- `tests/data/sub_f/test_provenance.py`
- `tests/data/sub_f/test_manifest.py`
- `reports/2026-05-23-phase-1-sub-F-task-6-halt.md`

Do not create writer/orchestrator code. Do not alter BP1/BP2/BP4 locked vocab YAML except for read-only checks.

## Step 1: extend sub-D VersionNamespace by enum-add only

In `src/cfm/data/sub_d/versions.py`, add:

```python
SOURCE = "source"
```

to `VersionNamespace`.

Do not change `VersionRef`. Do not change `compare_version` signature or behavior. Existing sub-D/sub-E callers must remain backward-compatible.

After the edit, run:

```bash
uv run pytest tests/data/sub_d/ -v
```

If any sub-D test fails: STOP, report BLOCKED. Do not weaken sub-D tests.

## Step 2: create sub-F versions.py

Create `src/cfm/data/sub_f/versions.py` with:

- Six v1 constants:
  - `SUB_F_ARTIFACT_FORMAT_VERSION = "1.0"`
  - `SUB_F_SCHEMA_VERSION = "1.0"`
  - `SUB_F_VOCAB_VERSION = "1.0"`
  - `SUB_F_DERIVATION_VERSION = "1.0"`
  - `SUB_F_VALIDATOR_VERSION = "1.0"`
- `load_sub_f_source_version()` reading `configs/data/overture_release.yaml` at build time and returning the `release` field.
- `sub_f_version_manifest()` returning `dict[VersionNamespace, VersionRef]` for all six axes:
  - `ARTIFACT_FORMAT`
  - `DATA_SHAPE`
  - `VOCAB`
  - `DERIVATION`
  - `VALIDATOR`
  - `SOURCE`

Do not duplicate the SOURCE release as a separate hard-coded constant. The pin file is the source of truth.

## Step 3: create sub-F provenance.py

Create `src/cfm/data/sub_f/provenance.py`.

Use the existing helper:

```python
from cfm.data.determinism import compute_sha256_excluding as _compute_sha256_excluding
```

Do not hand-roll recursive SHA exclusion if the helper remains available.

Define `SUB_F_EXCLUDED_FROM_SHA` mirroring sub-E's grammar:

```python
SUB_F_EXCLUDED_FROM_SHA: dict[str, list[str]] = {
    "*": ["*_sha256"],
    "provenance.yaml": [
        "extraction.extracted_utc",
    ],
    "manifest.yaml": [
        "initial_extraction.started_utc",
        "initial_extraction.completed_utc",
    ],
}
```

Provide `provenance_sha256(data: dict) -> str` using `_compute_sha256_excluding(data, "provenance.yaml", SUB_F_EXCLUDED_FROM_SHA)`.

You may add small dataclasses only if they simplify the on-disk shape. Keep this task focused on version/provenance contract, not writer implementation.

## Step 4: create sub-F manifest.py

Create `src/cfm/data/sub_f/manifest.py`.

Provide:
- `build_region_manifest(region: str, release: str, tile_entries: list[dict], vocab_sources: dict[str, Any]) -> dict`
- `manifest_sha256(data: dict) -> str`

Task 6's manifest API is provisional because BP7 boundary-reference vocab locks later at Task 7. The six version axes can lock structurally at Task 6, but `vocab_sources` content is not complete until BP7 exists.

Required manifest fields:
- `region`
- `release`
- `sub_f_artifact_format_version`
- `sub_f_schema_version`
- `sub_f_vocab_version`
- `sub_f_derivation_version`
- `sub_f_validator_version`
- `sub_f_source_version`
- `vocab_sources_status`
- `vocab_sources`
- `tiles`
- `manifest_sha256`

Rules:
- `vocab_sources` is region-scope shared metadata, not per-tile provenance.
- At Task 6, `vocab_sources` must include exactly the three already-locked vocab/config sources:
  - BP1 semantic vocab: `configs/sub_f/semantic_vocab.yaml`
  - BP4 unknown family: `configs/sub_f/unknown_family.yaml`
  - BP2 encoding primitives: `configs/sub_f/encoding_primitives.yaml`
- Do not include BP7 boundary-reference vocab at Task 6; it does not exist yet.
- Add `vocab_sources_status: "partial_pending_bp7"` to every Task 6 manifest.
- Treat the Task 6 `manifest_sha256` as provisional over the partial manifest. Final assembly with complete `vocab_sources` and final `manifest_sha256` is deferred to sub-F close after Task 7 locks BP7.
- `sub_f_source_version` must equal the Overture release pin.
- `tiles` should be sorted deterministically if tile coordinates are present; otherwise preserve stable caller-supplied order only if there is no sortable key. State the behavior in docstrings/tests.
- `manifest_sha256` computes self-integrity over canonical YAML with `SUB_F_EXCLUDED_FROM_SHA` exclusions.

## Step 5: tests

Add focused tests in `tests/data/sub_f/test_provenance.py` and `tests/data/sub_f/test_manifest.py`.

Minimum test coverage:
- Sub-D `VersionNamespace` now includes exactly six members and `SOURCE.value == "source"`.
- Existing `compare_version(namespace, expected, actual)` works for all six axes, including SOURCE.
- `sub_f_version_manifest()` returns all six axes and each `VersionRef.namespace` matches its key.
- `load_sub_f_source_version()` equals `configs/data/overture_release.yaml["release"]`.
- Version constants are `"1.0"` for ARTIFACT_FORMAT, DATA_SHAPE, VOCAB, DERIVATION, VALIDATOR.
- `SUB_F_EXCLUDED_FROM_SHA` excludes `extraction.extracted_utc` from `provenance_sha256`.
- `SUB_F_EXCLUDED_FROM_SHA` excludes nested final-segment `*_sha256` fields from `provenance_sha256`.
- `provenance_sha256` changes on real semantic content changes.
- Region manifest has all six version fields.
- Region manifest includes `vocab_sources` at region scope and does not require `vocab_sources` inside tile entries.
- Task 6 `vocab_sources` includes exactly the three locked sources: semantic vocab, unknown family, and encoding primitives.
- Task 6 `vocab_sources` does not include boundary-reference vocab.
- Task 6 manifest carries `vocab_sources_status == "partial_pending_bp7"`.
- Region `manifest_sha256` excludes live-clock fields and final-segment `*_sha256` fields.
- Region `manifest_sha256` changes on real semantic content changes.
- Cross-axis coupling sanity:
  - SOURCE value changes when release input changes while DERIVATION remains `"1.0"`.
  - DERIVATION can be changed in a synthetic manifest dict while SOURCE remains the release pin.
- Sentinel inventory check confirms BP2 remains LOCKED and BP7 remains PLACEHOLDER.

## Step 6: Halt 6 report

Create `reports/2026-05-23-phase-1-sub-F-task-6-halt.md`.

Report must include:
- Status: `DONE_WITH_CONCERNS` pending Halt 6 reviewer approval, or `BLOCKED` with classification.
- Audit step outcomes 1-6.
- File:line citations for:
  - sub-D `VersionNamespace`
  - sub-D `compare_version`
  - Overture release pin
  - pinning policy SOURCE semantic
  - sub-E `SUB_E_EXCLUDED_FROM_SHA` / hashing precedent
  - sub-C manifest/version/hash surface
- Plan Revision 1+2 evidence: sub-D had five axes; sub-F adds SOURCE for six total axes.
- SOURCE semantic verification branch:
  - expected branch (a): source-data-pinning identifier.
  - If ambiguous, stop and classify before proceeding.
- SOURCE semantic reviewer question, with both readings surfaced:
  - (a) SOURCE = Overture release pin only; sub-F reads sub-C output, but sub-C lineage is tracked indirectly through cache/config/vocab hashes rather than a dedicated sub-C field.
  - (b) SOURCE = Overture release pin plus explicit sub-C output version/hash field in the sub-F manifest.
  - Audit and cite whether sub-C exposes a stable artifact to reference: region `manifest.yaml`, `sub_c_schema_version`, `initial_extraction.commit_sha`, per-tile `provenance_sha256`, explicit `manifest_sha256` if present, or a derived manifest content hash if no explicit hash exists.
  - Do not choose (a) or (b) unilaterally. Halt 6 reviewer locks this.
- `compare_version` extensibility outcome:
  - expected: enum-add, no signature change.
  - If not enum-add, classify as cascade before proceeding.
- Validator-axis decision surface:
  - propose v1 shared `SUB_F_VALIDATOR_VERSION` for inline + cross-tile validators.
  - explicitly defer `VALIDATOR_INLINE` / `VALIDATOR_CROSS_TILE` split to sub-F-v2 unless reviewer chooses otherwise.
- Region-vs-tile provenance scope:
  - `vocab_sources` lives in region manifest, not per-tile provenance.
- Manifest completeness:
  - Task 6 manifest is provisional and carries `vocab_sources_status: partial_pending_bp7`.
  - Task 6 `vocab_sources` covers BP1/BP2/BP4 only.
  - BP7 boundary-ref vocab source is intentionally absent until Task 7.
- Deferred-to-sub-F-close obligations:
  - Add BP7 boundary-ref vocab source to region manifest after Task 7 BP7 lock.
  - Recompute final complete `manifest_sha256` at sub-F close / Task 15 after the BP7 source is present.
  - Remove or update `vocab_sources_status` from `partial_pending_bp7` only when the manifest is content-complete.
- ID namespace confirmation:
  - BP1 `0..199` LOCKED.
  - BP4 `200..255` LOCKED.
  - dataloader sentinels `256..299` LOCKED / `on_disk: false`.
  - BP2 `300..1499` LOCKED at Halt 2.
  - BP7 `1500..1599` PLACEHOLDER pending Task 7.
- Section 10.5 telemetry:
  - implementer-time-to-data-surface.
  - any reviewer-time fields remain pending.

Reviewer approves:
- SOURCE-bump semantic.
- compare_version path.
- provenance scope schema.
- shared vs split VALIDATOR axis.
- post-N reserved block confirmation.

## Verification

Run:

```bash
uv run pytest tests/data/sub_d/ -v
uv run pytest tests/data/sub_f/test_provenance.py tests/data/sub_f/test_manifest.py -v
uv run pytest tests/data/sub_f/test_encoder.py tests/data/sub_f/test_vocab.py -v
git diff --check
```

If `uv` cache access is blocked by sandboxing, report that exactly; do not claim tests passed.

## Commit

Commit the halt surface with a WIP message:

```text
wip(sub_f): T6 pre-halt - 6-axis version manifest + SOURCE verification (Halt 6 pending)
```

Do not use a final `feat:` commit before reviewer approval.

Final status:
- `DONE_WITH_CONCERNS` if Halt 6 report and WIP commit are ready for reviewer approval.
- `BLOCKED` if any audit, upstream contract, or test outcome contradicts the plan.

Surface:
- commit SHA
- changed files
- verification results
- halt report content

===

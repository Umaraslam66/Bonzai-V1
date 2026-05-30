# Phase 1 Sub-F Task 6 Halt 6 Report

Status: DONE_WITH_CONCERNS pending Halt 6 reviewer approval.

Branch: `phase-1-sub-F-micro-tokenizer`

WIP commit: pending at report creation time.

## Scope

Implemented the Halt 6 surface only:

- Added sub-D `VersionNamespace.SOURCE` by enum-add.
- Added sub-F six-axis version helpers.
- Added sub-F provenance hash exclusion helper.
- Added provisional sub-F region manifest helper.
- Added focused tests for SOURCE composite identity, six-axis refs, hash exclusions, provisional vocab-source coverage, and ID namespace state.

No writer/orchestrator work was added. No BP7 vocab was created. No push or PR was performed.

## Audit Step Outcomes

1. sub-D enum shape before edit matched the approved five-axis base:
   names `['ARTIFACT_FORMAT', 'DATA_SHAPE', 'VOCAB', 'DERIVATION', 'VALIDATOR']`,
   values `['artifact_format', 'data_shape', 'vocab', 'derivation', 'validator']`.
   After edit, SOURCE is present at `src/cfm/data/sub_d/versions.py:18-26`.

2. sub-D `compare_version` signature matched `compare_version(namespace, expected, actual)`:
   `src/cfm/data/sub_d/versions.py:37-41`.

3. Overture pin source of truth exists and is unambiguous:
   `configs/data/overture_release.yaml:3` pins `release: "2026-04-15.0"`.
   Policy says `configs/data/overture_release.yaml` is the single source of truth and `release` is the identifier at `docs/data/overture_pinning_policy.md:7-13`.

4. sub-E hash precedent exists:
   `src/cfm/data/sub_e/provenance.py:19-21` imports the shared helper,
   `src/cfm/data/sub_e/provenance.py:34-43` defines `SUB_E_EXCLUDED_FROM_SHA`,
   and `src/cfm/data/sub_e/provenance.py:110-118` uses it for provenance.
   Manifest hashing uses the same exclusion table at `src/cfm/data/sub_e/manifest.py:15-19`
   and `src/cfm/data/sub_e/manifest.py:114-123`.

5. ID namespace state is unchanged:
   BP1 `0..199` LOCKED at `configs/sub_f/sentinel_inventory.yaml:2-7`;
   BP4 `200..255` LOCKED at `configs/sub_f/sentinel_inventory.yaml:35-41`;
   dataloader `256..299` LOCKED and `on_disk: false` at `configs/sub_f/sentinel_inventory.yaml:50-70`;
   BP2 `300..1499` LOCKED at Halt 2 at `configs/sub_f/sentinel_inventory.yaml:9-34`;
   BP7 `1500..1599` remains PLACEHOLDER at `configs/sub_f/sentinel_inventory.yaml:42-49`.

6. sub-C output identity surface exists for SOURCE:
   cached manifest release is `2026-04-15.0` at `data/processed/sub_c/2026-04-15.0/singapore/manifest.yaml:34`;
   `initial_extraction.commit_sha` is present at `data/processed/sub_c/2026-04-15.0/singapore/manifest.yaml:26-30`;
   `sub_c_schema_version` is present at `data/processed/sub_c/2026-04-15.0/singapore/manifest.yaml:37`;
   per-tile `provenance_sha256` is present at `data/processed/sub_c/2026-04-15.0/singapore/manifest.yaml:38-41`.
   Source schema support is also visible in `RegionManifest` at `src/cfm/data/sub_c/manifest.py:33-61`,
   write serialization at `src/cfm/data/sub_c/manifest.py:82-98`,
   and tile provenance aggregation at `src/cfm/data/sub_c/manifest.py:146-177`.

## Plan Revision Evidence

Pre-edit sub-D had five axes. Task 6 added `SOURCE = "source"` only, producing six axes total:
`src/cfm/data/sub_d/versions.py:18-26`.

The extension path is clean: `compare_version` still accepts one namespace and two `VersionRef` arguments,
checks namespaces, then compares opaque values at `src/cfm/data/sub_d/versions.py:37-57`.

## SOURCE Semantic

Reviewer override resolves SOURCE as reading (b):

`SOURCE = overture release pin + explicit sub-C output identity`.

Implemented manifest shape:

```yaml
sub_f_source_version:
  overture_release: <from configs/data/overture_release.yaml>
  sub_c_schema_version: <from cached sub-C region manifest>
  sub_c_commit_sha: <from cached sub-C region manifest>
```

Implementation evidence:

- Composite loader reads the Overture pin and cached sub-C region manifest at `src/cfm/data/sub_f/versions.py:41-63`.
- Manifest exposes the composite mapping at `src/cfm/data/sub_f/manifest.py:69-87`.
- Tests assert the concrete chain at `tests/data/sub_f/test_manifest.py`.

## VersionRef Composite Shape

sub-D `VersionRef` is scalar-only:

- `VersionRef.value` is documented as an opaque string at `src/cfm/data/sub_d/versions.py:29-34`.
- `compare_version` compares `expected.value != actual.value` directly at `src/cfm/data/sub_d/versions.py:53-57`.

Canonical-string fallback is therefore used only for `VersionRef.value`:

`overture=2026-04-15.0;subc_schema=1.1;subc_commit=12b1cdf8838d9f8b601ea4b2a859f905ee5ab368`

Implementation is at `src/cfm/data/sub_f/versions.py:66-73`.
This does not violate `compare_version` semantics because the helper treats values as opaque strings and only checks equality.

## Provisional Manifest Surface

Task 6 manifest is intentionally provisional until BP7 locks:

- `vocab_sources` is region-scope shared metadata, not per-tile provenance.
- `vocab_sources_status` is always `partial_pending_bp7`.
- `vocab_sources` covers BP1/BP2/BP4 only:
  `configs/sub_f/semantic_vocab.yaml`,
  `configs/sub_f/unknown_family.yaml`,
  `configs/sub_f/encoding_primitives.yaml`.
- BP7 boundary-reference vocab is intentionally absent.

Implementation evidence:

- Task 6 vocab source paths are defined at `src/cfm/data/sub_f/manifest.py:31-41`.
- Vocab source digests are computed at `src/cfm/data/sub_f/manifest.py:44-54`.
- Manifest fields and `partial_pending_bp7` are set at `src/cfm/data/sub_f/manifest.py:57-87`.
- Hash exclusions for manifest/provenance are defined at `src/cfm/data/sub_f/provenance.py:9-18`.

Deferred obligations:

- Add BP7 boundary-ref vocab source to region manifest after Task 7 BP7 lock.
- Recompute final complete `manifest_sha256` at sub-F close / Task 15 after BP7 source is present.
- Remove or update `vocab_sources_status` from `partial_pending_bp7` only when the manifest is content-complete.

## VALIDATOR Axis

Implemented shared v1 `SUB_F_VALIDATOR_VERSION = "1.0"` at `src/cfm/data/sub_f/versions.py:12-16`.

INLINE/CROSS_TILE split is deferred to sub-F v2 unless the reviewer chooses otherwise.

## ID Namespace Confirmation

- BP1: `0..199`, LOCKED.
- BP4: `200..255`, LOCKED.
- Dataloader sentinels: `256..299`, LOCKED, `on_disk: false`.
- BP2: `300..1499`, LOCKED at Halt 2.
- BP7: `1500..1599`, PLACEHOLDER pending Task 7.

No ID namespace change was made. BP7 remains PLACEHOLDER.

## Verification

Commands run:

```text
uv run pytest tests/data/sub_d/ -v
```

Result: `93 passed, 4 deselected`.

```text
uv run pytest tests/data/sub_f/test_provenance.py tests/data/sub_f/test_manifest.py -v
```

Result: `14 passed`.

```text
uv run pytest tests/data/sub_f/test_encoder.py tests/data/sub_f/test_vocab.py -v
```

Result: `63 passed`.

```text
git diff --check
```

Result: passed with no output.

Note: `uv` commands required approved access to the external uv cache because sandboxed runs failed with:
`failed to open file /Users/umaraslam/.cache/uv/sdists-v9/.git: Operation not permitted`.

## Reviewer Ratification Checklist

- SOURCE composite chain file:line evidence: included above for Overture pin and sub-C manifest fields.
- `VersionRef` shape: scalar-only; canonical-string fallback documented in code/tests/report.
- `compare_version` enum-add path: clean; sub-D tests green.
- Provisional manifest: BP1/BP2/BP4 only, `vocab_sources_status="partial_pending_bp7"`, deferred obligations present.
- Shared VALIDATOR axis for v1; INLINE/CROSS_TILE split deferred to v2.
- ID namespace unchanged; BP7 remains PLACEHOLDER.

## Section 10.5 Telemetry

- implementer-time-to-data-surface: same-session implementation and verification on 2026-05-28; no separate wall-clock timer was instrumented.
- reviewer-time-to-approval: pending.
- reviewer-time-to-rejection-or-cascade: pending.

## Halt Decision

Status: DONE_WITH_CONCERNS.

Concerns are limited to expected Halt 6 reviewer ratification items:

- Approve SOURCE composite semantic and canonical-string `VersionRef.value` fallback.
- Approve provisional manifest surface and deferred BP7 obligation.
- Approve shared v1 VALIDATOR axis with split deferred to v2.

Do not proceed past Halt 6 until reviewer approval.

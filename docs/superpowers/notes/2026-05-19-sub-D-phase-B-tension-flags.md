# Sub-D Phase B tension flags (drafted while Gate 2B blocked)

> Drafted 2026-05-19 during Gate 2B halt (sub-C Singapore output unavailable
> locally). This document collects spec/plan/implementation tensions noticed
> during Tasks 1-7 that the Task 8-16 implementer needs to resolve. None of
> these are session-4 anti-patterns — they are real ambiguities that will
> bite if not surfaced before code lands.

Each item names: what was noticed, where it appears in spec/plan, severity
(blocking vs heads-up), and a recommended resolution.

---

## A. Blocking decisions needed before Task 8/9 implementation

These three items change the shape of artifacts that Task 8-13 produces. They
need explicit reviewer answers before Task 8 commits.

### A1. Per-namespace `derivation_version` vs single global constant

**What I noticed.** My current `src/cfm/data/sub_d/evidence.py` uses a single
module constant `DERIVATION_VERSION = "1.0"` carried on every
`EvidenceMetric`. The spec treats derivation versions as **per-aspect**:

- Spec §11.5 (provenance.yaml) lists 4 separate fields: `zoning_derivation_version`,
  `cell_density_derivation_version`, `tile_population_density_derivation_version`,
  `road_skeleton_derivation_version`.
- Spec §11.7 (macro vocab artifact) lists the same 4 separately under
  `derivation_versions:`.
- Spec §12 lists "Derivation versions: zoning, cell-density,
  tile-population-density, road-skeleton" as separate namespace entries.

**Severity:** Blocking. A single global constant cannot express "zoning
algorithm bumped, density unchanged."

**Recommended resolution before Task 9.** Split into:

```python
# evidence.py
ZONING_DERIVATION_VERSION: str = "1.0"
CELL_DENSITY_DERIVATION_VERSION: str = "1.0"
TILE_POPULATION_DENSITY_DERIVATION_VERSION: str = "1.0"
ROAD_SKELETON_DERIVATION_VERSION: str = "1.0"
```

Each `derive_*_evidence` function stamps the appropriate version on its
emitted `EvidenceMetric` rows. The locked macro vocab artifact (Task 8) and
provenance.yaml (Task 11) record all four separately. Validator (Task 13)
compares per-namespace.

**Impact on existing tests.** `test_evidence.py` currently asserts
`sample.derivation_version` is a non-empty string. That test still passes
after the split — each metric still has a version, just from a different
constant.

### A2. Frequency-analysis file structure: one consolidated YAML vs four per-namespace YAMLs

**What I noticed.** My current implementation writes one consolidated YAML
(`frequency_analysis.yaml` or `macro_vocab_proposal.yaml`) with all four
namespace sections nested inside. The spec strongly implies **four separate
files**, one per analysis namespace:

- Spec §11.7 macro vocab artifact has
  `generated_from.frequency_analysis.{zoning_sha256, cell_density_sha256,
  tile_population_density_sha256, road_skeleton_sha256}` — separate digests
  for separate files.
- Plan Task 8 Step 5 says: "Copy approved golden frequency-analysis
  artifacts to `tests/golden/sub_d/frequency_analysis/<analysis_name>.yaml`"
  (plural "artifacts", placeholder `<analysis_name>`).
- Spec §11.3 says "machine-readable analysis artifacts under
  `reports/phase-1-sub-D/` with stable names" (also plural).

**Severity:** Blocking. The locked macro vocab embeds per-namespace digests;
those digests must be computable from the on-disk files. If one consolidated
file is the input, the digest must be over a section, not over a file.

**Resolution options:**

1. **Restructure to 4 files.** Task 7's CLI writes
   `zoning_analysis.yaml`, `cell_density_analysis.yaml`,
   `tile_population_density_analysis.yaml`, `road_skeleton_analysis.yaml`
   plus an index/proposal file that references them. Task 8's promote
   reads four files, computes four digests, embeds them in the locked
   artifact. Matches spec literally. Higher reviewer surface area (4
   files to scan).
2. **Keep consolidated; sha each section.** Task 7 writes one
   `macro_vocab_proposal.yaml`; Task 8's promote canonicalises each
   section subsection independently and embeds those subsection-shas. The
   reviewer reads one file. Departs from spec's "separate files" implication
   but matches the digest contract.

**My lean:** option (2). The reviewer-facing artifact stays one file (easier
to read and diff); per-section digests still pin each namespace's evidence
independently. Implementation cost: a `section_sha256(analysis, section_key)`
helper that canonicalises just the section subtree and hashes it.

**Open question for the reviewer:** confirm option (1) or option (2) before
Task 7 runs against real sub-C output. If option (1), Task 7's CLI needs
restructuring before Gate 2B closes; the README also needs an update.

### A3. `tile_population_density` evidence: where does it get derived?

**What I noticed.** Spec §11.3 carries `metric_namespace: 2=tile_population_density`
as a first-class derivation_evidence namespace. Spec §6 lists "Tile-level
population-density proxy analysis over tile aggregate built-form evidence"
as one of five empirical gates. Spec §8 says
`population_density_bucket` is a built-form proxy (mean / area-weighted /
median / percentile of building density).

**Tension:** Plan Task 5 (evidence primitives) lists only
`derive_cell_scope_metrics`, `derive_zoning_evidence`,
`derive_density_evidence`, `derive_road_skeleton_evidence` — **no**
`derive_tile_population_density_evidence`. My current `evidence.py` does not
compute it. Plan Task 6 (frequency analysis) doesn't surface it either.
But the locked vocab artifact (§11.7, plan Task 8) carries
`tile_population_density.tokens` and the conditioning overlay (plan Task 10)
fills `population_density_bucket` per tile.

So somewhere between Task 5 and Task 10, tile-level population-density
evidence needs to be computed. The plan does not name where.

**Severity:** Blocking. Task 8's locked vocab artifact references
`tile_population_density_vocab_version` and the frequency-analysis digest.
If the analysis doesn't produce a `tile_population_density_proposal`
section, Task 8 has nothing to lock and the field stays a placeholder.

**Resolution options:**

1. **Add to Task 5/Task 6.** Extend `evidence.py` with
   `derive_tile_population_density_evidence(meta, cells, features) ->
   list[EvidenceMetric]` (returns one `slot_kind=TILE, slot_index=0` row
   per tile carrying e.g. mean building footprint ratio or
   area-weighted building density). Extend frequency analysis with a
   `tile_population_density_proposal` section parallel to the other three.
2. **Defer to Task 10.** Compute `population_density_bucket` directly in
   the conditioning overlay using a fixed algorithm; treat the
   `tile_population_density` namespace as conditioning-only and skip the
   evidence parquet rows.

**My lean:** option (1). The spec explicitly lists tile_population_density
as a `derivation_evidence.parquet` namespace, and the locked vocab carries
its tokens. Skipping evidence would put sub-D out of contract with §11.3
and §11.7. Implementation cost: ~80 lines in evidence.py mirroring
density derivation but aggregating to tile.

**Open question for the reviewer:** confirm (1) before Task 7 produces a
real proposal — Task 7 needs to know whether to include a
`tile_population_density_proposal` section in the proposal artifact.

---

## B. Implementation heads-up (no spec change; future-me reminders)

### B1. SlotKind cardinality differs between artifacts

- `macro_core.parquet` (§11.2): `slot_kind enum: 0=cell, 1=internal_edge,
  2=external_edge` — 3 values.
- `derivation_evidence.parquet` (§11.3): `0=cell, 1=internal_edge,
  2=external_edge, 3=tile` — 4 values.

My `SlotKind` IntEnum already has 4 values (including `TILE`). The
macro_core writer (Task 9) must reject `SlotKind.TILE` rows. The validator
(Task 13) must check macro_core has no `slot_kind=3` rows.

### B2. `value_type` dispatch ordering — Python bool/int gotcha

`derivation_evidence.parquet` has `value_type` enum `{0=float64, 1=int64,
2=string, 3=bool}`. The writer (Task 9) dispatches on Python type.

**Trap:** `isinstance(True, int)` is `True` in Python. Order matters:

```python
if isinstance(value, bool):       # MUST come before int
    value_type = 3
elif isinstance(value, int):
    value_type = 1
elif isinstance(value, float):
    value_type = 0
elif isinstance(value, str):
    value_type = 2
```

Spec doesn't mention this. A Task 9 test should exercise both `True` and
`1` as separate values, asserting they get different `value_type` tags.

### B3. AST meta-test (Task 13) — every version comparison must use `compare_version`

Plan Task 13 Step 3 specifies an AST scanner over `validator.py` and any
`validator_*.py` files that fails on direct `==`/`!=` comparisons where
either side references a name/attribute/subscript containing `version` or
`_version`. Implementation discipline:

- Use `compare_version(VersionNamespace.X, expected, actual)` everywhere.
- Do NOT write `if x.version == y.version` even in helper functions if
  they live in a `validator*.py` file.
- If a non-version field happens to be named with the word "version"
  embedded (none currently planned), the meta-test will fail on it
  spuriously. Watch for this and rename if it happens.

### B4. provenance.yaml self-integrity uses excluding-timestamp; sub-C input digests use bytes-sha

Reviewer pre-flagged this for Task 11: sub-D uses two digest semantics
simultaneously. To preserve the distinction:

- `inputs.sub_c_*_sha256` = `sha256(file_bytes)` from sub-D's reader.
  These were snapshotted in `SubCTileInputs.digests` at read time. They go
  in sub-D's provenance.yaml directly.
- `provenance.yaml`'s OWN self-integrity sha (recorded in
  `manifest.tiles[*].provenance_sha256`) = `compute_sha256_excluding(data,
  "provenance.yaml", SUB_D_EXCLUDED_FROM_SHA)`. Sub-D's exclusion table
  excludes `extraction.extracted_utc` and any final-segment `*_sha256`
  fields (chain-of-custody).

**rerun_reason MUST be INCLUDED** in sub-D's exclusion-stripped sha (per
sub-C F2 fix and plan Task 11 test name
`test_provenance_sha_includes_rerun_reason`). Sub-D's
`SUB_D_EXCLUDED_FROM_SHA` must NOT contain `extraction.rerun_reason`.

### B5. Conditioning copy rule: skip `*_owner` AND sub-D-owned fields

Plan Task 10 Step 3 says: "Copy every field in `conditioning_per_tile`
except fields ending with `_owner` and fields explicitly owned by sub-D
before filling `population_density_bucket`."

Define:

```python
SUB_D_OWNED_FIELDS = {"population_density_bucket"}

def _is_sub_c_owned(field_name: str) -> bool:
    return not (field_name.endswith("_owner") or field_name in SUB_D_OWNED_FIELDS)
```

Apply this to `sub_c_meta["conditioning_per_tile"].items()`. Then add
sub-D's computed `population_density_bucket` value.

Spec §11.4 says the implementation must NOT use a static allowlist of
sub-C fields. The rule is schema-driven via the owner-suffix marker.

### B6. Manifest `config` block: copy entire sub-C config dict verbatim

Spec §11.6 example shows only a few keys (`cell_grid`, `cell_size_m`,
`tile_size_m`, `internal_edge_count`, `external_edge_count`). Sub-C's
actual `manifest.config` contains at least `sliver_drop_rule`. Recommend
copying the entire sub-C config dict; validator checks the full dict
matches. This auto-tracks any new keys sub-C adds.

### B7. `provenance_schema_version` etc. — NOT bare `schema_version`

Plan Task 11 test `test_provenance_schema_uses_provenance_schema_version_not_bare_schema_version`
pins this. Sub-C's `TileProvenance` uses bare `schema_version`; sub-D's
Provenance dataclass MUST use the namespaced field name
`provenance_schema_version`. Same pattern for manifest
(`manifest_schema_version`), effective_conditioning
(`effective_conditioning_schema_version`).

---

## C. Spec-vs-plan inconsistencies (plan wins; documenting for traceability)

### C1. Module names — spec mirrors sub-C split, plan consolidates

- Spec §14 lists: `zoning.py`, `density.py`, `road_skeleton.py`,
  `validator_inline.py`, `validator_cross_tile.py`.
- Plan File Map: `evidence.py` (one file for all three Layer-1 evidence
  derivations), `validator.py` (one file for inline + cross-tile checks).
- My implementation follows the plan.

**Resolution:** plan is more recent than spec; treat plan as source of
truth. No code change. Future-me, if you read the spec and think
"shouldn't there be a `zoning.py`?" — no, the consolidated evidence.py is
the agreed structure. The plan's consolidation was a deliberate choice
during Topic-10 brainstorming (not in this doc; cross-reference the
brainstorm if needed).

---

## D. Layer-3 cached Singapore tile IDs source-of-truth (Task 15)

Plan Task 15 has `test_cached_singapore_subset_tile_ids_have_rationales`.
Where do the committed tile IDs live?

**Options:**

1. Hardcoded list in the test file.
2. In `tests/golden/sub_d/frequency_analysis/macro_vocab_proposal.yaml`
   (committed at Task 8 after Gate 2 approval) — the test reads them from
   there.
3. In a new `tests/data/sub_d/layer3_tile_subset.yaml` config.

**My lean:** option (2). Task 8 already commits the locked vocab + golden
frequency-analysis artifacts that include `selected_layer3_tiles`. The
slow integration test (Task 15) reads `selected_layer3_tiles` from the
locked artifact. This keeps the tile subset and the locked vocab in one
artifact pair — they were derived together and locked together.

**Open question:** confirm before Task 15. Not blocking Task 8 directly,
but Task 8's locked artifact format should include `selected_layer3_tiles`
verbatim if option (2) is chosen.

---

## Summary: questions for the reviewer to answer before Task 8 starts

1. **A1 derivation_version split:** confirm split into per-namespace
   constants (`ZONING_DERIVATION_VERSION`, etc.) before Task 9 runs?
2. **A2 frequency-analysis file structure:** option (1) four files, or
   option (2) one consolidated file with per-section digests?
3. **A3 tile_population_density evidence:** option (1) extend Tasks 5/6/7
   to compute and propose it, or option (2) defer to Task 10
   conditioning-only?
4. **D1 Layer-3 tile-IDs source-of-truth:** option (2) read from the
   locked artifact at Task 15?

The other items (B, C) are implementation discipline reminders, not
blocking decisions. They're documented so the Task 8-16 implementer
(possibly me in a future session) doesn't have to re-derive them.

---

## Resolution status (post-review, 2026-05-19)

Reviewer answered all four blocking decisions:

- **A1:** approved — split into four per-namespace constants.
- **A2:** option (1) approved — four namespace files
  (`zoning_analysis.yaml`, `cell_density_analysis.yaml`,
  `tile_population_density_analysis.yaml`, `road_skeleton_analysis.yaml`)
  plus one index file `macro_vocab_proposal.yaml`. The
  byte-identity-modulo-status-marker test applies to the index file only;
  namespace files are content-pinned via digest references in the index.
  Locked-bucket choices live in the index (one reviewer-editable
  location, not four).
- **A3:** option (1) approved with a critical addition — emit *all*
  candidate proxies per tile as separate `metric_name`s (e.g.
  `mean_building_footprint_ratio`, `area_weighted_building_density`,
  `median_building_footprint_ratio`, `p75_building_footprint_ratio`).
  Layer-1 emits raw, Layer-3 reviewer picks. Don't pre-commit to one
  proxy formula at evidence time.
- **D1:** option (2) approved — Task 15 reads `selected_layer3_tiles`
  from the locked vocab artifact's index file verbatim. Task 8's locked
  artifact format must include the field.

A1/A2/A3 land in a separate implementation commit; the B/C items remain
implementer's-checklist material for Task 8-16.

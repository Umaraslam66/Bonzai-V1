# Task 1 implementer dispatch prompt

**Status:** Approved by reviewer; ready for dispatch.
**Target:** General-purpose subagent / Codex agent.
**Suggested model:** Sonnet-class (network calls + multi-step coordination + audit-level decisions; haiku may struggle).
**Branch:** `phase-1-sub-F-micro-tokenizer` (current head: `69b835c`).

> The prompt below is the verbatim text to give the implementer agent. Everything between the `===` markers is the agent's prompt body.

===

You are implementing Phase 1 sub-F Task 1 pre-halt steps: BP1 vocab floor analysis + snapshot artifacts + Singapore X-threshold computation. Stop at Halt 1 and surface a halt report; do NOT proceed past the halt autonomously.

## Working environment

- **Working directory:** `/Users/umaraslam/Projects/Bonzai-OSM`
- **Current branch:** `phase-1-sub-F-micro-tokenizer`. DO NOT create new branches, DO NOT push, DO NOT create PRs.
- **Spec reference:** `docs/superpowers/specs/2026-05-23-phase-1-sub-F-micro-tokenizer-design.md` (commit `cd5d332`).
- **Plan reference:** `docs/superpowers/plans/2026-05-23-phase-1-sub-F-micro-tokenizer.md` (commit `69b835c`).
- **Protocol reference:** `docs/protocols/sub-project-planning-protocol-v1.md`.

## Discipline constraints (non-negotiable)

1. **No new branches.** Work on `phase-1-sub-F-micro-tokenizer`.
2. **No push.** Do not run `git push`.
3. **No PR creation.** Do not run `gh pr create`.
4. **Halt-on-defect.** Unexpected errors, type mismatches, missing helpers → STOP and report DONE_WITH_CONCERNS or BLOCKED. No silent inline fixes.
5. **Verify-before-lock.** If a file:line citation or expected API shape diverges from actual code, STOP and report. Do NOT autonomously cascade.
6. **No autonomous YAML lock past the halt.** Plan Steps 10–11 (writing final `semantic_vocab.yaml` after reviewer approval) are EXPLICITLY out of scope. Stop at Step 9 (Halt 1 surface).
7. **Cascade discipline.** This plan has FIVE prior §9.6.1 cascade outcomes baked into Task 1 code (cascades #1–#5 documented in plan's "Plan revisions from pre-dispatch audit" section). If any audit step surfaces a SIXTH cascade — e.g., the taginfo API has changed shape, or sub-C feature_class enum has been extended — STOP and report. Do NOT cascade autonomously past those already captured.

## Scope for this dispatch

**Plan Task 1 steps 1–9 inclusive** (pre-halt). Steps 10–11 are a separate continuation dispatch.

## Implementation source

All code blocks + step instructions are in the plan file at `docs/superpowers/plans/2026-05-23-phase-1-sub-F-micro-tokenizer.md` under `## Task 1: BP1 vocab floor analysis + snapshots`. Read that section in full before starting. Copy each step's code block verbatim — do NOT modify in flight. If anything is unclear or appears to have a typo, STOP and report BLOCKED.

### Files (created in this dispatch)

- `configs/sub_f/taginfo/2026-04-15.0.csv`
- `configs/sub_f/wiki_map_features/2026-04-15.0.{wikitext,sha256,revision_id}`
- `scripts/sub_f/snapshot_taginfo.py`
- `scripts/sub_f/snapshot_wiki.py`
- `scripts/sub_f/floor_analysis.py`
- `configs/sub_f/vocab_floor_analysis.yaml`
- `tests/data/sub_f/__init__.py`
- `tests/data/sub_f/test_vocab.py`

### Pre-dispatch audit (3 steps; STOP and report BLOCKED if any surfaces a 6th cascade)

#### Audit step 1: confirm taginfo API for global key+value frequency

Run: `curl -sI 'https://taginfo.openstreetmap.org/api/4/key/values?key=highway&page=1&rp=100&sortname=count&sortorder=desc' | head -5`
Expected: HTTP 200; JSON contains `data` array.

#### Audit step 2: confirm wiki API for Map_features wikitext

Run: `curl -s 'https://wiki.openstreetmap.org/w/api.php?action=raw&page=Map_features&format=json' | head -100`
Expected: raw wikitext starting with `==` headers.

#### Audit step 3: verify sub-C feature_class enum still matches cascade #4 scope

Run: `grep -A 2 "FEATURE_CLASS" src/cfm/data/sub_c/enums.py`
Expected: `FEATURE_CLASS: dict[int, str] = {0: "road", 1: "building", 2: "poi", 3: "base"}` (4 closed values, NO new entries).

If different (new feature_class values added since plan was written): STOP, report BLOCKED — that's a 6th §9.6.1 cascade requiring plan-revision, not in-dispatch handling.

### Implementation steps (follow plan §Task 1 Steps 1–9 exactly)

#### Step 1: Write `scripts/sub_f/snapshot_taginfo.py`

Copy verbatim from plan §Task 1 Step 1. Bug 1 fix applied — raw counts schema with `row_type`, no fraction columns. Header row is `key,value,count_all,count_ways,count_nodes,count_relations,row_type,parent_key`.

#### Step 2: Write `scripts/sub_f/snapshot_wiki.py`

Copy verbatim from plan §Task 1 Step 2. Uses MediaWiki API for revision-pinned wikitext (NOT HTML scraping).

#### Step 3: Run snapshot scripts

```bash
uv run python scripts/sub_f/snapshot_taginfo.py --release 2026-04-15.0
uv run python scripts/sub_f/snapshot_wiki.py --release 2026-04-15.0
```

Expected: 4 files exist under `configs/sub_f/`. Verify with `ls -la configs/sub_f/`.

#### Step 4: Write `scripts/sub_f/floor_analysis.py`

Copy verbatim from plan §Task 1 Step 4 (~280 lines). Key invariants to preserve VERBATIM (do NOT modify):

- `WIKI_L1_MUST_APPEARS`: tuple of 28 keys (cascade #5).
- `WIKI_L2_HIGHWAY`: tuple of 23 way-class values.
- `WIKI_L2_BUILDING`: tuple of 33 values (32 typology + `"yes"` catch-all).
- `WIKI_L3_ALL_PAIRS`: empty frozenset (deferred per spec §12 #10).
- `FEATURE_CLASS_TO_KEY`: `{0: "highway", 1: "building"}` ONLY (cascade #4).
- Helpers: `dominant_element_type`, `et_totals_from_taginfo`, `fraction_within_et`, `f_min_for_level`, `vocab_size_at_F`, `compute_singapore_frequencies`, `derive_x_threshold`.
- `main()` writes YAML with `curve` array of 3 rows (level 3 is deferred placeholder).

If you find a typo in the plan's code: STOP, report BLOCKED.

#### Step 5: Write `tests/data/sub_f/__init__.py` (empty) + `tests/data/sub_f/test_vocab.py`

Copy verbatim from plan §Task 1 Step 5. Test file ships 7 tests including Safeguard 2 per-key count assertions.

Hand-derived constants to preserve VERBATIM:
- `N_L1_MUST_APPEARS_EXPECTED = 28`
- `N_L2_HIGHWAY_EXPECTED = 23`
- `N_L2_BUILDING_EXPECTED = 33`

#### Step 6: Run test (expected fail — config does not exist yet)

```bash
uv run pytest tests/data/sub_f/test_vocab.py -v
```

Expected: tests FAIL with `FileNotFoundError`. All 7 tests should fail in this mode.

#### Step 7: Run floor analysis

Sub-C Singapore cache must exist at `data/processed/sub_c/2026-04-15.0/singapore/`. Verify with `ls data/processed/sub_c/2026-04-15.0/singapore/ | head`. If missing, STOP and report BLOCKED.

```bash
uv run python scripts/sub_f/floor_analysis.py \
    --release 2026-04-15.0 \
    --sub-c-region-dir data/processed/sub_c/2026-04-15.0/singapore/
```

Expected: `configs/sub_f/vocab_floor_analysis.yaml` written with `_status: PROPOSED`.

Spot-check the YAML:
- `len(wiki_l1_must_appears) == 28`
- `wiki_l2_highway_count == 23`
- `wiki_l2_building_count == 33`
- `wiki_l3_status == "deferred per spec §12 #10"`
- `curve` has 3 entries (levels 1, 2, 3)
- `proposed_x_threshold.candidate_a_singapore_elbow` is a concrete float
- `proposed_x_threshold.candidate_b_median_must_appear_freq` is a concrete float

#### Step 8: Run full test suite (expected 7 PASS)

```bash
uv run pytest tests/data/sub_f/test_vocab.py -v
```

Expected: 7 PASS. If any FAIL: STOP, report BLOCKED with the failing assertion message.

#### Step 9: HALT 1 — surface to reviewer

Write `reports/2026-05-23-phase-1-sub-F-task-1-halt.md` with this structure:

**Snapshot artifacts:**
- taginfo CSV: row count, first 5 rows verbatim, file size, sha256.
- wiki Map_features: revision_id, wikitext byte count, sha256.

**Marginal-cost curve (L1 full + L2 highway+building; L3 deferred):**
- L1 row: 28 keys, F_min value, vocab_size at F_min.
- L2 row: highway + building primary pairs, F_min, vocab_size.
- L3 row: deferred per spec §12 #10.

**Proposed elbow:**
- Granularity level (1 by default).
- F value (concrete number).
- Exception list (empty by default).
- Rationale citing Δvocab_size / Δmust-appears between L1 and L2.

**Proposed X-threshold (cascade #4 scope: highway + building only):**
- Candidate A: Singapore-elbow-derived value + concrete number.
- Candidate B: median must-appear frequency + concrete number.
- Scope note: POI + base deferred per spec §12 #11.
- Paired structural check framing.

**Cascade documentation (mandatory per spec §13.5):**
- Cascade #4 outcome: Singapore X scope = highway + building. POI/base deferred to sub-F-v2.
- Cascade #5 outcome: L1 corrected to 28 keys; L3 deferred entirely.
- §13.5 protocol-v2 candidates surfaced: (i) transitive-documentation citing, (ii) hand-enumeration with complete-count assertion, (iii) reviewer-supplied lists as untrusted input.

**§10.5 telemetry:**
- Implementer-time-to-data-surface: wall-clock from dispatch start to this halt report commit.

Commit on current branch:

```bash
git add reports/2026-05-23-phase-1-sub-F-task-1-halt.md \
        scripts/sub_f/snapshot_taginfo.py \
        scripts/sub_f/snapshot_wiki.py \
        scripts/sub_f/floor_analysis.py \
        configs/sub_f/taginfo/2026-04-15.0.csv \
        configs/sub_f/wiki_map_features/2026-04-15.0.wikitext \
        configs/sub_f/wiki_map_features/2026-04-15.0.sha256 \
        configs/sub_f/wiki_map_features/2026-04-15.0.revision_id \
        configs/sub_f/vocab_floor_analysis.yaml \
        tests/data/sub_f/__init__.py tests/data/sub_f/test_vocab.py
git commit -m "wip(sub_f): T1 pre-halt — snapshots + L1+L2 curve + Singapore X (Halt 1 pending)"
```

Note `wip:` prefix because YAML is `_status: PROPOSED`. Final `feat: T1 ... (Halt 1 approved)` commit lands in continuation dispatch.

**DO NOT proceed to plan Steps 10 or 11.**

## Self-review before reporting back

1. All 3 pre-dispatch audit steps ran; outcomes documented in halt report.
2. Snapshot artifacts exist: 4 files under `configs/sub_f/`.
3. YAML PROPOSED status: `grep "_status" configs/sub_f/vocab_floor_analysis.yaml` shows `PROPOSED`.
4. YAML cascade scopes correct (28 L1 / 23 L2-highway / 33 L2-building / L3 deferred / Singapore X with concrete candidates).
5. 7 tests pass.
6. Halt report committed (`git log -1 --oneline` shows the `wip:` commit).
7. `semantic_vocab.yaml` does NOT exist (continuation dispatch creates it).
8. No autonomous cascade.

## Report format

Report status **DONE_WITH_CONCERNS** (halt-pending = normal status, not BLOCKED).

Include:

- **Status:** DONE_WITH_CONCERNS (Halt 1 pending reviewer approval).
- **Halt name:** Halt 1 (BP1 vocab floor elbow).
- **Halt report path:** `reports/2026-05-23-phase-1-sub-F-task-1-halt.md`.
- **Key numbers (5 lines max):** F_l1, vocab_size_l1, F_l2, vocab_size_l2, Singapore X candidate A.
- **Files created (with byte counts for snapshot artifacts):** verbatim list.
- **Tests:** `tests/data/sub_f/test_vocab.py` — 7 tests, PASS.
- **Audit findings:** any §3 verification surprises (cite cascade # if matches plan's; flag as 6th cascade candidate if novel).
- **Commit SHA:** WIP commit hash.
- **§10.5 telemetry:** implementer-time-to-data-surface (wall-clock minutes).

Report BLOCKED only if:
- Network calls fail (taginfo or wiki API down/changed shape).
- Sub-C feature_class enum has changed since plan write (6th cascade).
- Sub-C Singapore cache missing.
- Wiki snapshot per-key counts diverge from expected (Safeguard 2 catch — wiki revised since plan was written).
- Plan code block has a typo or unclear logic.

===

(End of prompt body.)

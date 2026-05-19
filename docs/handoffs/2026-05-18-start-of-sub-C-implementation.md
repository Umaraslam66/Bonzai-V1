# Session handoff — start of Phase 1 sub-C implementation (2026-05-18)

> **For the Claude Code session reading this:** you are picking up Bonzai-OSM at the start of Phase 1 sub-project C (multi-cell tile extraction) **implementation**. The brainstorm-spec-plan cycle is COMPLETE; this handoff exists because the design-cycle session ran long and the user chose to spawn a fresh session for the implementation phase. Confirm you've read this doc + the §1 mandatory files before any tool calls. The next concrete piece of work is **sub-C plan Task 1** — see §6.

## 0. Project summary (one paragraph)

Bonzai-OSM is building a generative foundation model for *city geometry* — roads, buildings, points of interest, and land-use polygons — emitted as standards-compliant GeoJSON. The analogy: **GPT for cities**. Sub-C is the geometric extraction stage of the Phase 1 data pipeline: it turns sub-A's cached Overture themes into per-tile structured parquet + YAML artifacts (cells.parquet, features.parquet, crossings.parquet, meta.yaml, provenance.yaml) under a region-level manifest + `_SUCCESS` integrity chain. Sub-C's per-tile output IS the contract with sub-D (macro plan), sub-E (boundary contracts), sub-F (stitcher), sub-G (end-to-end validator). Compute lives on CINECA Leonardo (EuroHPC); allocation `AIFAC_P02_222`, 40k core-hours through 2026-06-11.

## 1. Required reading (in order)

| File | Mandatory or skim | What it gives you |
|---|---|---|
| `CLAUDE.md` | **MANDATORY in full** | Working agreement; user is new to deep learning, technical but learning; uv + ruff + pytest; design sketches in plain language before code. |
| `docs/superpowers/specs/2026-05-17-phase-1-sub-C-tile-extraction-design.md` | **MANDATORY in full** | The sub-C spec (~1000 lines, 22 sections). Canonical design source. Spec §12.1 is implementation-complete for the 10 inline validator invariants; spec §14 is the determinism contract with Categories A-K. |
| `docs/superpowers/plans/2026-05-17-phase-1-sub-C-tile-extraction.md` | **MANDATORY in full** | 19-task TDD-checkboxed plan. Branch + Test discipline sections at top. Per-task: spec sections implemented + named tests + determinism categories + dependencies. |
| `docs/superpowers/plans/2026-05-18-phase-1-sub-B2-followup-not-in-vocab.md` | skim | The B2 follow-up plan that landed before sub-C started. Useful only if Task 1 finds something unexpected in `_LOCKED_MISSING_POLICIES`. |
| `configs/data/missing_value_policy.yaml` | reference | The B2-follow-up-regenerated policy YAML with the four-case schema (both `missing_value` AND `not_in_vocab` axes per field). |
| `configs/tokenizer/vocab_phase1.yaml` | reference | Phase 1 vocab; sub-C reads it for the not-in-vocab handling in Task 6 (policy.py). |
| `docs/data/handoffs.md` | reference | The sub-A → sub-C contract (admin-polygon clipping obligation). |
| `docs/known_issues.md` | skim | Cold-fetch (#1) and subtype deferral (#2). Sub-C plan Task 19 will add #3 (Sweden densification revisit) and #4 (tokenizer emit_unknown_token enhancement training-path dependency). |
| `docs/handoffs/2026-05-16-end-of-sub-B2.md` | skim | The prior handoff (end of B2 → start of sub-C brainstorm). Most of its content is superseded by this doc + the sub-C spec. |

Auto-memory at `~/.claude/projects/-Users-umaraslam-Projects-Bonzai-OSM/memory/MEMORY.md` is loaded automatically — every linked entry there is pre-loaded into context. Key entries that codify the working agreements for this implementation phase are listed in §5.

## 2. Where we are in the project plan

```
Phase 0 (tokenizer round-trip)               DONE
Phase 1 sub-A (Overture loader)              DONE
Phase 1 sub-B1 (frequency analysis report)   DONE
Phase 1 sub-B2 (vocab YAML from B1 review)   DONE
  + B2 follow-up (not_in_vocab axis)         DONE (2026-05-18; merge commit 6ee4b14)
Phase 1 sub-C (multi-cell tile extraction)   IMPLEMENTATION IN PROGRESS (Task 0 done; Task 1 next)
Phase 1 sub-D (macro plan derivation)
Phase 1 sub-E (boundary contracts)
Phase 1 sub-F (deterministic stitcher)
Phase 1 sub-G (end-to-end pipeline + validator)
Phase 2  architecture bake-off (deferred — happens on Leonardo)
```

Sub-C plan Task 0 (B2 prereq gate + branch sync) was completed by the prior session. The B2 follow-up was merged to main and the sub-C branch was synced (merge commit `15e421b`).

## 3. State of the world (verify before starting)

**Repo:** `~/Projects/Bonzai-OSM`.

**Current branch:** `phase-1-sub-C-tile-extraction`.
```
git rev-parse --abbrev-ref HEAD
# expected: phase-1-sub-C-tile-extraction
```

**Test baseline:** 188 passed, 1 xfailed, 6 deselected.
```
uv run pytest -q
# expected: 188 passed, 6 deselected, 1 xfailed in ~8-10s
```

**Last commit on sub-C branch:** `15e421b Merge branch 'main' into phase-1-sub-C-tile-extraction` (picked up the B2 follow-up).

**Branches that exist (and why):**
- `main` — project trunk
- `phase-0-tokenizer`, `phase-1-sub-A-overture-loader`, `phase-1-sub-B1-frequency-analysis`, `phase-1-sub-B2-vocab-yaml` — historical merged feature branches (don't touch)
- `phase-1-sub-B2-followup-not-in-vocab` — merged to main on 2026-05-18 (don't touch)
- `phase-1-sub-C-tile-extraction` — **active branch; this is where sub-C implementation happens**

**Files on sub-C branch but not yet implemented:**
- `src/cfm/data/sub_c/` — does NOT EXIST yet; Task 1 creates it
- `scripts/extract_tiles.py`, `scripts/validate_extraction.py` — do NOT EXIST yet; Task 14 creates them
- `tests/data/sub_c/`, `tests/fixtures/sub_c/` — do NOT EXIST yet
- Sub-C spec + plan + B2 follow-up plan ARE committed on this branch.

**B2 follow-up gate (run if you doubt the prereq landed):**
```bash
uv run python -c "
from cfm.data.vocab_derivation import _LOCKED_MISSING_POLICIES
assert isinstance(_LOCKED_MISSING_POLICIES['buildings.class'], dict)
assert 'not_in_vocab' in _LOCKED_MISSING_POLICIES['buildings.class']
print('B2 follow-up OK: four-case schema present')
"
```

## 4. Plan task summary (19 tasks, T1–T19 remaining)

| # | Task | Spec sections | Complexity | Model rec. |
|---|---|---|---|---|
| T1 | Package skeleton (epsilon, enums, errors) | §4.3, §14.3, §14.4, §17 | small | cheap |
| T2 | Reprojection + tile-ID derivation | §7.1, §7.2 | small | cheap |
| T3 | Cell partitioning + densify + clip | §7.2, §7.3, §7.4 | small-medium | standard |
| T4 | Split-at-boundaries + crossing records (7 edge cases) | §8 | **large** | **most capable** |
| T5 | derive_sea_polygons + sea mask + overlap | §9 | medium-large | standard |
| T6 | apply_missing_value_policy (four-case) | §10.1, §10.2 | medium | standard |
| T7 | compute_conditioning_per_tile | §11.9 | small | cheap |
| T8 | io.py + determinism.py | §14.3, §14.6 | medium | standard |
| T9 | Per-tile write helpers (cells/features/crossings/meta/provenance) | §11.1–§11.6 | medium-large | standard |
| T10 | manifest.py + _SUCCESS protocol | §11.7, §11.8, §14.6 | medium | standard |
| T11 | validator_inline (10 invariants) | §12.1, §12.4, §14.3 | medium | standard |
| T12 | Pipeline orchestrator (sequential) | §6 | medium | **most capable** |
| T13 | Process-pool parallelization | §14.5 | medium | **most capable** |
| T14 | CLI scripts + cross-tile validator | §12.2, §15.2, §11.8 | medium | standard |
| T15 | Test fixtures (torture + cross-tile) | §13.2 | medium | standard |
| T16 | Layer 2 tests (~16 tests) | §13.2 | **large** | **most capable** |
| T17 | Layer 3 cached-Singapore integration | §13.3 | small-medium | standard |
| T18 | Pre-commit lint (no pandas in write-path) | §14.7 | small | cheap |
| T19 | known_issues + final verify + merge | §3, §7.4, §17 | small | standard |

**Heaviest tasks** (likely review-loop magnets): T4 (geometric edge cases), T11 (10 invariants — but spec §12.1 makes it mechanical), T16 (test count + diagnostic-payload determinism), T12+T13 (orchestrator + parallelism integration).

## 5. Critical constraints — must appear VERBATIM in every implementer dispatch

1. **Branch discipline.** All work on `phase-1-sub-C-tile-extraction`. NO new branches. NO push to remote. NO PR creation. Commit task-by-task to the existing branch. The sub-B2 Task 3 incident (subagent created its own branch + opened a PR) MUST NOT recur. Per saved memory `feedback_subagent_branch_pattern.md` + `project_branch_pattern.md`.

2. **Test discipline.** If a test fails because real data violates an assumed invariant, STOP and escalate. Do NOT modify the assertion to make broken code pass. Per saved memory `feedback_test_weakening_to_pass.md`. The specific sub-C trap: kept-cell rule (`NOT (sea_water_fraction >= 1.0 - EPS_RATIO AND zero non-sea features)`) — a real-tile violation means the 2.5a drop-rule wasn't applied; fix the code, not the test.

3. **Spec-driven implementation.** Every task in the plan cites the sub-C spec sections it implements. Implementer should re-read cited sections before writing code. The plan provides exact code skeletons + test patterns for unambiguous tasks; the spec is implementation-complete for invariant logic (e.g., Task 11's 10 inline validator invariants are exhaustively defined in spec §12.1).

4. **Determinism contract is a deliverable.** Spec §14 has Categories A-K of determinism rules with topic-of-origin citations on every rule. Each task lists which categories it satisfies; implementer must honor (e.g., parquet writer config pinned per §14.3; sort keys canonical per §14.2; EPS_RATIO for α structural-boundary comparisons per §14.4; pool-size independence per §14.5).

5. **B2 follow-up has landed.** Sub-C plan Task 0 Step 1 verifies `_LOCKED_MISSING_POLICIES` is the four-case schema; Step 2 verifies the YAML carries `not_in_vocab` axis. Both PASS as of merge commit 6ee4b14 on main and 15e421b on the sub-C branch.

## 6. First-message instructions for the new session

The new session should, in order:

1. **Confirm it has read this handoff** and the §1 mandatory files. Reply explicitly: "Read CLAUDE.md, sub-C spec, sub-C plan, handoffs.md, known_issues.md, this handoff doc."

2. **Sanity-check the workspace:**
   ```bash
   git rev-parse --abbrev-ref HEAD     # expect: phase-1-sub-C-tile-extraction
   uv run pytest -q                     # expect: 188 passed, 6 deselected, 1 xfailed
   ```
   If anything else — stop and report.

3. **Invoke `superpowers:subagent-driven-development`** with:
   - Plan path: `docs/superpowers/plans/2026-05-17-phase-1-sub-C-tile-extraction.md`
   - Spec path (for reviewers): `docs/superpowers/specs/2026-05-17-phase-1-sub-C-tile-extraction-design.md`
   - The 5 critical constraints from §5 above (VERBATIM in every implementer dispatch)
   - Start at **Task 1** (Task 0 is already complete)

4. **Per-task discipline (per the skill):**
   - Implementer subagent → spec compliance reviewer → code quality reviewer → mark complete
   - Two-stage review between every task; pause for human checkpoint after each
   - Use model selection per §4 task table (cheap/standard/most-capable)
   - Verify reports independently (don't trust subagent self-reports without inspection)

5. **After all 19 tasks complete**, invoke `superpowers:finishing-a-development-branch` to merge sub-C to main via `git merge --no-ff` (matches sub-A/B1/B2 pattern).

### Agent-to-agent notes (specific to sub-C implementation)

- **The plan's Task 0 says "create branch"; that's already done.** Task 0 was adapted by the prior session (branch already existed from when sub-C spec + plan were parked there; merge main into it instead of creating fresh). New session starts at Task 1.

- **Direct commits to `main` are blocked by a hook** (per CLAUDE.md "always go through a PR" rule). The project's actual pattern is `git merge --no-ff` from feature branch (matches sub-A/B1/B2). The hook allows feature-branch commits + `--no-ff` merge commits from `git merge`, just not bare `git commit` on main. Task 19's final merge to main via `git merge --no-ff phase-1-sub-C-tile-extraction` should work.

- **Two known wrinkles to watch for in T4 (split-at-boundaries):**
  - The plan provides a working approximation of `_derive_crossings_on_edge`; the corner-crossing case (intersection collapses to single Point on shared boundary corner) may need refinement to emit TWO records (one per axis). Test `test_corner_crossing_emits_two_records_one_per_axis` drives the refinement. If the test fails on the simple intersection approach, refine the geometric logic — but DO NOT weaken the test per §5 constraint #2.
  - The plan's `_flatten_intersection` uses a placeholder `ring_index=0` for all rings; real implementation needs to distinguish exterior (0) from interior rings (≥1) by source-polygon ring origin. The polygon-interior-ring test drives this.

- **Task 11 (inline validator) is mechanical** despite involving 10 invariants — spec §12.1 lists each invariant's exact assertion + structured-payload shape. The plan shows one invariant (bbox-matches-WKB) in full; the other 9 follow the same shape. Don't redesign; translate spec to Python.

- **Determinism test surface concentrates in T16** (~16 Layer 2 tests, including pool-size independence, sha stability, diagnostic-payload determinism, 8 per-invariant payload tests). This is the most test-density of any task — use the most capable model.

- **Sub-A's cold-fetch is ~8 hours** if the Singapore cache isn't present. Sub-C reads from the cache (cache-hit ~1s). Layer 3 tests in T17 use `load_region("singapore")` cache-hit; if the cache is somehow gone, sub-C is blocked by sub-A's known_issues #1, not sub-C itself. Verify `data/cache/overture/2026-04-15.0/singapore/` exists before starting Layer 3.

## 7. Things the prior session wishes the new session knew

- **The α/β EPSILON framework is project-wide, not just sub-C.** When implementer subagents touch any float comparison, the rule is: apply EPSILON at structural boundaries (0, 1, computed-value equality); use strict comparison for user thresholds (500m, 0.01m²). Saved as `feedback_epsilon_structural_vs_user_threshold.md`. Future sub-projects inherit this.

- **Denormalization decision rule:** "Denormalize iff (a) every consumer benefits AND (b) the access pattern is established." Both conditions together. Sub-C's plan/spec applies this 10 times (8 rejections, 2 acceptances: `geometry_type` and `bbox_*` with consistency tests). If T9's `features.parquet` schema looks like it grows new denormalized fields, push back.

- **Schema-polymorphism is cheap; re-deriving stored data is not.** Saved as `feedback_schema_vs_data_cost_asymmetry.md`. Specific recurring traps on this project: CRS choices (`tile.crs` polymorphic per region; stored coordinates ARE the choice), vocabulary token IDs (stored checkpoints ARE the choice), tile origin / grid alignment (tile IDs ARE the choice). T2/T3 implementer should not "improve" the EPSG:3414 choice or the half-open interval convention.

- **The handoff agenda is a floor, not a ceiling.** Saved as `feedback_handoff_agenda_is_floor.md`. If this handoff missed something the new session notices is needed, surface it before invoking the brainstorming/spec/plan cycle for that thing.

- **B2 follow-up landed during this design cycle, not before it.** The B2 follow-up (`docs/superpowers/plans/2026-05-18-phase-1-sub-B2-followup-not-in-vocab.md`) was a half-day mechanical migration to extend `_LOCKED_MISSING_POLICIES` from `(type, rationale, is_provisional)` 3-tuple to `{missing_value: PolicyAxis, not_in_vocab: PolicyAxis}` per sub-C spec §10.2. If the new session sees test_locked_missing_policies_four_case_table_matches_sub_c_spec_10_2 failing, that's a B2-follow-up regression — escalate, don't paper over.

- **The sub-C branch is currently the ONLY branch with the sub-C spec + plan committed.** Those files are NOT on main. When sub-C eventually merges to main (T19's `--no-ff` merge), they land along with the implementation. The B2 follow-up plan is on main (it landed with the B2 follow-up merge).

---

End of handoff. Total length: ~250 lines. Next session: §6, in order.

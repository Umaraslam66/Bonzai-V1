# Session handoff — end of Phase 1 sub-A (2026-05-16)

> **For the Claude Code session reading this:** you are picking up Bonzai-OSM right after Phase 1 sub-project A merged. The user is the project lead, new to deep learning, technically experienced. Confirm you've read this document and the files in §1 below before doing anything else. The next concrete piece of work is **B1 brainstorming** — see §6.

## 0. Project summary (one paragraph)

Bonzai-OSM is building a generative foundation model for *city geometry* — roads, buildings, points of interest, and land-use polygons — emitted as standards-compliant GeoJSON. The analogy: **GPT for cities**. The model trains on global Overture Maps data and learns to generate plausible urban form conditioned on a small set of inputs (region, density, era, style, etc.). The output is structured data for AV/robotics simulation, defense and security simulators, and procedural games — quality bar is geometric validity and plausibility, not photorealism. Compute lives on CINECA Leonardo (EuroHPC); allocation is `AIFAC_P02_222`, 40 k core-hours through 2026-06-11. We've explicitly rejected raster intermediates (a prior attempt failed for documented reasons in `PRD.md` §3). Everything works in discrete tokens end to end.

## 1. Required reading (in order)

| File | Mandatory or skim | What it gives you |
|---|---|---|
| `PRD.md` | **MANDATORY in full** | Locked plan: what we're building, why, the four-candidate architecture bake-off, eval suite, Leonardo budget. Section 2 (what we are NOT building) saves time. |
| `CLAUDE.md` | **MANDATORY in full** | How we collaborate. Read it before responding to any non-trivial user message. The "blast-radius before editing" rule and the "ask before doing" list are non-negotiable. |
| `docs/superpowers/specs/2026-05-16-phase-1-sub-A-overture-loader-design.md` | **MANDATORY** | The sub-A spec the just-shipped code implements. §3 (load-bearing decisions) and §7 (manifest format) are the most important. |
| `docs/superpowers/plans/2026-05-16-phase-1-sub-A-overture-loader.md` | skim | The 15-task implementation plan. Useful as a template for B1's plan; don't re-read end to end. |
| `docs/data/overture_pinning_policy.md` | skim | Re-pin procedure. Read in full only if you actually need to re-pin. |
| `docs/data/handoffs.md` | **MANDATORY** | The A→C contract: themes are bbox-filtered only; C must apply `admin_polygon` for precise clipping. This is the one easy-to-miss invariant in sub-A. |
| `docs/boundary_contracts_v0.md` | skim | Hypothesis doc parked for sub-project E. Not a spec; nothing here is locked yet. |
| `docs/known_issues.md` | **MANDATORY** | The 8.2-hour cold-fetch issue is here. Read before suggesting anything about Sweden or re-pinning. |
| `docs/superpowers/specs/2026-05-15-phase-0-tokenizer-roundtrip-design.md` | skim | Phase 0 spec. Read if you touch the tokenizer; otherwise skim. |
| `docs/LEONARDO_REFERENCE.md` | on-demand | Cluster paths, partitions, login. Read only when work crosses to Leonardo (not for B1). |

Auto-memory at `~/.claude/projects/-Users-umaraslam-Projects-Bonzai-OSM/memory/MEMORY.md` is loaded automatically — every linked entry there has already been pre-loaded into context when you started.

## 2. Where we are in the project plan

```
Phase 0 (tokenizer round-trip)             DONE  — 56 tests + 1 xfailed on main
Phase 0 patch round                        DONE  — 5 fixes; 56→56 + 1 xfailed
Phase 1 sub-A (Overture loader)            DONE  — 117 tests + 1 xfailed + 6 slow opt-in
Phase 1 sub-B1 (frequency analysis report) NEXT
Phase 1 sub-B2 (vocab YAML from B1 review)
Phase 1 sub-C  (multi-cell tile extraction)
Phase 1 sub-D  (macro plan derivation)
Phase 1 sub-E  (boundary contracts)
Phase 1 sub-F  (deterministic stitcher)
Phase 1 sub-G  (end-to-end pipeline + validator)
Phase 2  architecture bake-off (deferred — happens on Leonardo)
```

Dependency graph: `A → {B1 → B2, C} → {D, E} → F → G`. B1 and C can run in parallel; B2 follows B1 (review gate); D and E follow C; F follows E; G chains.

The next concrete step is brainstorming sub-project B1. See §6.

## 3. What's on disk that the next session can use

**Repo:** `~/Projects/Bonzai-OSM`. Main branch HEAD is `85ca344` (after this commit). Remote is `https://github.com/Umaraslam66/Bonzai-V1`.

**Cached Singapore data** at `data/cache/overture/2026-04-15.0/singapore/` (gitignored, ~147 MB total):

| Theme | Rows | Bytes |
|---:|---:|---:|
| `divisions.parquet` | 729 | 5.4 MB |
| `buildings.parquet` | 339,972 | 56.2 MB |
| `places.parquet` | 149,657 | 27.3 MB |
| `transportation.parquet` | 202,334 | 55.3 MB |
| `base.parquet` | 8,636 | 3.4 MB |
| `manifest.yaml` | — | 1.6 KB |

These are real Overture data at release `2026-04-15.0`. B1 reads them directly via `load_region("singapore")` — cache-hit path is ~1 s. Do **not** trigger a re-fetch.

**Python env:** `uv` managed, Python 3.11, `.venv` lives in repo (gitignored). Key pins: `duckdb==1.5.2`, `pyarrow>=15.0` (resolved 24.0), `shapely>=2.0`, `pyyaml>=6.0`, `pytest`, `ruff`. The repo is outside iCloud Drive (~/Projects/, not ~/Documents/) so `.venv` does not get its `.pth` files hidden — that was a Phase-0 quirk, see memory entry.

**Test suite quick reference:**
- `uv run pytest` — fast suite, ~118 tests, ~0.5 s. Slow tests deselected by default via `pyproject.toml` `addopts = "... -m 'not slow'"`.
- `uv run pytest -m slow` — opt-in real-S3 tests (5 cases, will hit network).
- `uv run python -c "from cfm.data.overture import load_region; r = load_region('singapore'); print(r.release, list(r.themes), len(r.themes['buildings']))"` — sanity check, ~1 s with cache.

## 4. Open issues with status

| Issue | Location | Status | Notes |
|---|---|---|---|
| Cold-fetch ~8 h | `docs/known_issues.md` #1 | deferred | **Hard blocker before Sweden.** B1 unaffected. |
| Boundary-entry marker | `tests/tokenizer/test_encode.py` xfail | parked | Documented as a Phase-1 gap to address with boundary contracts (sub-E). |
| `places.categories` typed as `string` in `schema.py` | `src/cfm/data/overture/schema.py` | accepted | Real Overture stores it as a struct; we only check column presence, not dtype. B1 inspects the struct shape. |
| Two-pass admin-polygon scoping | spec §3 decision 1 fallback | accepted | Phase 1 uses bbox-only fetch + handoff polygon. C does precise clipping. See `docs/data/handoffs.md`. |

No outstanding code review feedback, no failing tests, no uncommitted changes (after this handoff commits).

## 5. Working-pattern reminders

- **User is new to deep learning.** Explain concepts with concrete analogies before code. "Sketcher / Inker" (macro plan / micro generator) is one we've used. When jargon appears, define it inline. Treat basic-sounding questions as real questions.
- **"Blast radius before editing" is mandatory.** For non-trivial changes: state which files you'll touch, which tests will break, before opening any of them. The user uses this to spot wrong assumptions early.
- **Lock decisions before coding.** Before writing more than ~50 non-trivial lines, restate the load-bearing assumption in plain words and confirm. CLAUDE.md §"What to do before writing code" is authoritative.
- **B1/B2 split is the whole point.** B1 ships a *markdown report* with tables and plots. User reviews it. *Then* B2 derives `configs/tokenizer/vocab_phase1.yaml` from the reviewed numbers. Do **not** collapse the two — the review gate is the mechanism by which vocabulary decisions get user oversight rather than hiding in code.
- **Follow the data, not the PRD's hypotheses.** PRD §5 estimates "80–150 place categories, 8–15 building classes" etc. Those are predictions. If B1's frequency analysis on real Singapore data lands at 60 or 180, **the PRD updates to match.** See `feedback_follow_data_over_prd.md` in memory.
- **Architectural decisions live in `CLAUDE.md` and `PRD.md`.** Don't relitigate. If you find yourself disagreeing with a locked decision, flag it explicitly to the user before changing anything; quote the section.
- **Subagent-driven development for plan execution.** Phase 0 and sub-A both used `superpowers:subagent-driven-development`. Each task gets a fresh subagent with the full task text + scene-setting. Two-stage review (spec compliance, then code quality) is the standard discipline; in practice many config-only and small-test tasks were verified by the controller directly. Use judgment.
- **One commit per logical unit. Conventional prefixes: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `data`, `expt`.** Branch off main for non-trivial work; merge with `--no-ff` to keep the sub-branch visible in main's log.
- **Auto mode is active in this repo.** Execute autonomously on low-risk work; ask before destructive or shared-state actions (force push, delete data, modify PRD/CLAUDE without confirmation).
- **Don't push to `main` directly outside of merges.** The user explicitly authorizes pushes in the session.

## 6. First-message instructions for the next session

The new session should, in order:

1. **Confirm it has read this handoff** and the §1 mandatory files. Reply explicitly: "Read PRD.md, CLAUDE.md, sub-A spec, handoffs.md, known_issues.md".
2. **Sanity-check the Singapore cache** with the one-liner from §3:
   ```bash
   uv run python -c "from cfm.data.overture import load_region; r = load_region('singapore'); print(r.release, list(r.themes), len(r.themes['buildings']))"
   ```
   Expected output: `2026-04-15.0 ['divisions', 'buildings', 'places', 'transportation', 'base'] 339972`. Should complete in ~1 second. If anything other than that — stop and report; don't trigger a cold re-fetch.
3. **Propose the B1 brainstorming agenda** as a few short questions (one at a time per the brainstorming-skill discipline) covering at minimum:
   - **Scope.** Just Singapore frequency analysis? Or also include synthetic-fixture sanity comparison?
   - **B1's deliverable is a markdown report.** Confirm the format: tables for each categorical field (`buildings.class`, `places.categories`, `transportation.class`, `base.subtype`, `divisions.country`), counts and percentages, optional plots, *no* committed vocab. Output path proposal: `reports/2026-MM-DD-phase-1-sub-B1-singapore-frequency-analysis.md`.
   - **Frequency-floor calibration.** PRD §5's 10,000-instance global floor for vocabulary admission **does not apply at single-region scale.** Singapore has 339,972 buildings; 10k of those is ~3 % — a floor that high would over-prune. Propose a scaled floor (the natural starting point is ~100 instances or ~0.03 %, two orders of magnitude below the global floor, but the user should pick). Explicitly mark B1's output as **provisional pending Sweden** so the reviewed vocab in B2 can be revised when the second region lands.
   - **Plots.** matplotlib / plotly / none? The user has expressed a terminal-prose preference for architectural discussions; for B1 a few static rank-frequency plots are probably the right level. Confirm.
   - **Computation cost.** B1 runs locally on the cached parquets; tens-of-seconds for the frequency counts. No GPU. No Leonardo touch.
4. **Then invoke `superpowers:brainstorming` with B1 as the subject** once the user greenlights the agenda.

Do NOT start B1 implementation. Do NOT touch `src/cfm/` or `configs/` or `scripts/` until B1's spec is written and approved.

## 7. Things I (this session) wish I'd known at the start

- **The repo started life under iCloud Drive (`~/Documents/`).** Two harmless workarounds remain in code: `pyproject.toml` has `pythonpath = ["src"]` and `scripts/smoke.py` has a `sys.path` injection. Both are no-ops on the current non-iCloud path. Don't remove them unless the user asks; they're insurance.
- **Overture's actual S3 layout has an extra partition layer.** `s3://overturemaps-us-west-2/release/{release}/theme={theme}/type={type}/*` — the `type=` part was missing from the original spec and surfaced during the Task 13 slow test. `cfm.data.overture.backend.THEME_TO_TYPE` hard-codes the Phase-1 type per theme (e.g., `transportation → segment`, `base → water`, `divisions → division_area`). When B1 examines real columns, expect those types specifically.
- **DuckDB 1.5.2 has a real bug.** Calling `.arrow()` directly on a successful S3 httpfs result raises `InternalException ("TransactionContext::ActiveTransaction called without active transaction")`. The fix in `backend.py` uses `.to_arrow_reader()` + `pa.Table.from_batches`. Don't refactor it back. Also `SET http_retries=5` etc. is needed for multi-minute scans.
- **The Region dataclass's docstring is load-bearing.** A test (`test_region_docstring_states_handoff_contract`) asserts it mentions both `bbox` and `admin_polygon`. The split into `BboxScope` + `RegionGeometry` is *the* type-level expression of the A→C handoff. If you ever feel like collapsing those into one class, re-read `docs/data/handoffs.md` first.
- **The 1-second bump in `loader.py`'s `fetched_at`** is an intentional hack to keep manifest second-resolution timestamps unique across rapid refreshes. The corresponding test (`test_load_region_refresh_true_ignores_cache`) was rewritten during sub-A — the version in the plan as originally written had a read-after-overwrite bug. If you re-run that test and it flakes, the issue is timestamp resolution, not the loader.
- **The synthetic test fixtures' geometry is placeholder bytes** (WKB POINT(0,0)). Fast tests don't decode it. If a future test does, expect (0,0) for every row. `scripts/snapshot_overture_fixtures.py --mode s3` regenerates fixtures with real Singapore geometry.
- **The user reads end-of-turn summaries; long ones get skimmed.** Headers and tables help. CLAUDE.md "Tone and style" §"End-of-turn summary" is right — one or two sentences plus a tight table when there's tabular state.
- **The user's review questions are sharp.** Sub-A's user-review round caught a real bug (test_load_region_refresh_true read-after-overwrite that I'd already shipped). Plan for the user to surface issues you didn't; don't get defensive.
- **`/loop` and `ScheduleWakeup` are useful for long-running waits** (the 8-hour cold fetch used `ScheduleWakeup`). Pick 60s–270s when you expect change soon (stays in cache window), or 1200s+ when you're committed to a long wait.
- **GitHub remote was set up mid-project.** No issues are filed there yet — convention so far is `docs/known_issues.md` for tracked deferred work. The user said "whichever fits" when asked.

---

End of handoff. Total length: under 400 lines. Next session: §6, in order.

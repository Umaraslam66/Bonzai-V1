# Session handoff — mid-sub-F, Task 1 pending dispatch (2026-05-23)

> **For the incoming agent (Claude Code / Codex / any general-purpose agent):** this is a resumable handoff. You have NOT inherited the prior session's claude-mem memory or conversational context. This document + the referenced repo files contain everything you need.

## Where we are

Sub-F (Phase 1 micro-tokenizer) is mid-implementation. Brainstorm + spec + plan all approved and committed. Subagent prompt for Task 1 (first implementation task) is approved and committed. **Task 1 has NOT been dispatched yet.** Your job (or your dispatched subagent's job) is to execute it.

**Branch:** `phase-1-sub-F-micro-tokenizer` (5 commits ahead of `main`).

**Commit chain:**
```
69b835c docs(sub_f): plan revision — Task 1 cascades #4 + #5 + 5 code-bug fixes
cd5d332 docs(sub_f): spec sync — BP1 scope amendments from cascades #4 + #5
b52df6e docs(sub_f): spec sync — four-axis manifest → six-axis (audit-after-fixup)
07bd4bd docs(sub_f): phase 1 sub-F micro-tokenizer implementation plan
38ff72b docs(sub_f): phase 1 sub-F micro-tokenizer design spec
```

## What to do next

**Dispatch Task 1** using the approved prompt at:

```
reports/dispatch-prompts/2026-05-23-task-1-prompt.md
```

The prompt is self-contained — copy the text between the `===` markers into your subagent dispatch tool's `prompt` parameter. Suggested subagent model: Sonnet-class. Expected outcome: subagent reports `DONE_WITH_CONCERNS` at Halt 1 with a halt report committed via `wip:` prefix on the sub-F branch.

After the dispatched subagent halts and surfaces the report, **the human reviewer approves Halt 1** before any continuation. You do NOT auto-continue past the halt.

## Critical authoritative documents (read these; don't infer)

1. **Spec:** `docs/superpowers/specs/2026-05-23-phase-1-sub-F-micro-tokenizer-design.md`
   - §10.1 Halt 1: what data the subagent surfaces.
   - §10.3 Halt protocol: how halts work.
   - §12 Deferral ledger: what's intentionally NOT in sub-F-v1.
   - §13 Revision ledger: what changed during brainstorm + plan-write.
   - §13.5 Protocol-bump candidates: discipline lessons for future agents.

2. **Plan:** `docs/superpowers/plans/2026-05-23-phase-1-sub-F-micro-tokenizer.md`
   - "Plan revisions from pre-dispatch audit": 5 cascade outcomes baked into Task 1 code.
   - §Task 1: full TDD step list with code blocks (~750 lines, lines ~165–950).
   - §Task 2–15: subsequent tasks (you dispatch those after Halt 1 + reviewer continuation approval, one at a time per the halt-protocol pattern).

3. **Protocol:** `docs/protocols/sub-project-planning-protocol-v1.md`
   - Six gates + 4 principles + lint-cosmetic exemption + audit-after-fixup pattern.
   - All discipline lives here.

## Non-negotiable rules (these would normally be in agent memory; restating for Codex)

These rules govern ALL sub-F work, dispatched subagents and yourself alike:

### Branch + push discipline
- **Work only on branch `phase-1-sub-F-micro-tokenizer`.** Do NOT create new branches.
- **NEVER push** (`git push` forbidden). Sub-F merges to `main` only at sub-F-close via human-initiated merge.
- **NEVER create PRs** (`gh pr create` forbidden).

### Halt discipline
- Every task with a halt gate (Tasks 1, 2, 3c, 4, 5a, 6, 7 per spec §10.1) requires the implementer subagent to STOP at the halt and surface a halt report.
- The subagent's exit status at a halt is `DONE_WITH_CONCERNS` (NOT `BLOCKED`). Halt-and-wait is the protocol's normal flow, not a failure.
- The HUMAN reviewer approves each halt before continuation. Do not auto-continue.
- Use `wip:` git commit prefix for halt-pending commits; final `feat: ... (Halt N approved)` commits land post-approval.

### Verify-before-lock (per spec §13.5 protocol-v2 candidate)
- When the plan cites a file:line or expected API shape, the subagent **verifies against current source BEFORE relying on it.**
- If verification surfaces a mismatch with what plan/spec expects: STOP, report — do NOT autonomously cascade.
- The plan has already absorbed 5 §9.6.1 cascades (#1–#5); any 6th cascade is YOUR escalation trigger, not your in-flight fix.

### Halt-on-defect (per spec discipline)
- Implementer subagents do NOT silently fix unexpected errors. STOP and report.
- Halt-on-defect is the institutional capital from sub-E's Task 10 lesson (silent inline fixes contaminated the commit history).

### Audit-after-fixup (per protocol §8)
- Every plan-fixup commit triggers a fresh audit of surrounding code, not just the diff.
- Strengthening one gate can expose defects the prior weakness was masking.

### Cascade resolution (per spec §9.6.1)
- When verification surfaces upstream-contract mismatch:
  1. Upstream wins by default (sub-D / sub-C / sub-E / sub-A is canonical).
  2. Lock value updates in the originating spec section.
  3. §2 paired check re-validates against updated lock.
  4. §13 revision ledger entry committed (deferred to sub-F-close).

### Gate 6 hand-enumeration discipline (per spec §13.5, derived from sub-F cascade #5)
- Hand-enumerations of upstream lists (e.g., wiki must-appears) must be derived from the **canonical source directly** — NOT from downstream documentation referencing the source, NOT from reviewer-supplied lists.
- Per-section count assertions (hand-derived independently from the enumeration content) catch enumeration drift that flat set comparison misses.

## What's in this session's progress that's NOT obvious from the repo

Most context IS in the committed repo files (spec + plan + protocol). The few things only documented here:

1. **Task 1 prompt approval state:** the prompt at `reports/dispatch-prompts/2026-05-23-task-1-prompt.md` was reviewed and approved by the human in conversation. No code change has been made to anything outside `configs/sub_f/`, `scripts/sub_f/`, `tests/data/sub_f/`, `src/cfm/data/sub_f/` (none of which exist yet — Task 1 creates the first files there).

2. **Cascade backstory (in case the subagent encounters confusing-looking plan text):**
   - Cascade #1: `compare_version` is enum-add, not kwarg-add (sub-D `versions.py`).
   - Cascade #2: 6-axis manifest, not 4-axis (sub-D has 5 namespaces, sub-F adds SOURCE).
   - Cascade #3: sub-C feature sort key concretized to `(cell_i, cell_j, feature_class, source_feature_id)`.
   - Cascade #4: Singapore X-threshold scope narrowed to highway + building (POI/base deferred — sub-C `feature_class=2` has NULL `class_raw`).
   - Cascade #5: L1 must-appears expanded 15 → 28 keys (original 15-key list was reviewer-supplied; corrected from wikitext).
   - All five are documented in plan top section "Plan revisions from pre-dispatch audit" + spec §13.1.

3. **Memory entries that exist only in claude-mem (Claude Code's persistent memory; not in repo):**
   - Many `feedback_*` entries — most relevant ones for sub-F are CAPTURED in spec §13.5 + protocol file. You don't need direct access to claude-mem memory to execute Task 1 correctly.
   - If you (or your subagent) want the original feedback entries, they're at `~/.claude/projects/-Users-umaraslam-Projects-Bonzai-OSM/memory/` — but the spec + protocol distill what matters.

## After Task 1 halt approval

The human reviewer approves Halt 1. The continuation dispatch covers plan Steps 10–11 (write final `semantic_vocab.yaml` post-approval). Then Tasks 2–15 dispatch one at a time, following the same prompt-derivation + reviewer-approval + dispatch + halt-and-report pattern. Each task has its own halt (or no halt for pure-implementation tasks like 8–11).

The recommended dispatch sequence (per spec §11.3 parallelism notes — can start in parallel at T0):
- Task 1 (in progress / pending dispatch — THIS dispatch)
- Tasks 2, 3a, 5a, 6, 7 (independent; can dispatch concurrently after Task 1's first halt)
- Then dependent tasks (3b after T2; 4 after T1; 8 after T1+T2+T4+T5a+T6+T7; etc.)

## Pointers

- **Spec:** `docs/superpowers/specs/2026-05-23-phase-1-sub-F-micro-tokenizer-design.md`
- **Plan:** `docs/superpowers/plans/2026-05-23-phase-1-sub-F-micro-tokenizer.md`
- **Task 1 prompt:** `reports/dispatch-prompts/2026-05-23-task-1-prompt.md`
- **Protocol:** `docs/protocols/sub-project-planning-protocol-v1.md`
- **Sub-E handoff (inherited residuals):** `docs/handoffs/2026-05-20-end-of-sub-E.md`
- **Project working agreement:** `CLAUDE.md` (root) — applies to all agents on this project, not just Claude Code.

## Merge note

Do NOT merge `phase-1-sub-F-micro-tokenizer` to `main`. Sub-F merges only at sub-F-close (Task 15 handoff) via human-initiated merge.

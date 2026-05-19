# Session handoff - start of Phase 1 sub-D implementation (2026-05-19)

> **For the new session:** read this doc, then begin Task 1 from the implementation plan.

## 1. Current State

Sub-D brainstorm is closed (Topics 0-10). The sub-D design spec is committed at `ce62d77`; the first implementation plan commit is `7f01f0b`; the plan-tighten commit is `06a22b7`. Phase 1 sub-A/B1/B2/C are merged; sub-E/F/G are unstarted. The implementation branch is `phase-1-sub-D-macro-plan-derivation`.

## 2. Implementation Entry Point

Begin with **Task 1: Shared Determinism And I/O Helpers** in `docs/superpowers/plans/2026-05-19-phase-1-sub-D-macro-plan-derivation.md`.

Task 1 is the neutral helper extraction from sub-C into shared `cfm.data` modules. It is also the first hard halt at **Gate 1**. Do not start Task 2 until Gate 1 has explicit reviewer approval.

## 3. Gate 1 Requirements

Sub-C tests must pass with zero behavior change after re-pointing sub-C imports to the neutral helpers.

Gate 1 fallback triggers from the plan:

- Hidden coupling discovered mid-extraction that would require breaking helper signatures used by sub-C.
- Sub-C tests fail after repointing imports and the fix path is non-trivial.
- Sub-C tests pass but a byte-output or golden-artifact comparison shows behavior changed.

If any trigger fires, halt at Gate 1 and ask whether to use the spec fallback: duplicate determinism helpers locally in `cfm.data.sub_d` rather than sharing neutral helpers.

After Task 1 Step 8 commits, halt and report:

- Commit SHA.
- Full fast-suite output.
- Whether any fallback trigger fired.

Do not proceed to Task 2 without explicit reviewer approval.

## 4. Reusable Feedback Memory

- **Cost-asymmetry:** cheap-to-keep values that are expensive or impossible to recover later should be preserved unless there is a concrete cost. Citation: `docs/handoffs/2026-05-18-end-of-sub-C.md` section 4, `feedback_schema_vs_data_cost_asymmetry.md`.
- **Denormalization criterion:** denormalize only when every consumer benefits and the access pattern is established. Citation: `docs/handoffs/2026-05-18-end-of-sub-C.md` section 4, sub-C spec section 4.4.
- **Branch discipline:** every implementer dispatch repeats this verbatim: "Do NOT create new branches. Do NOT push to remote. Do NOT open pull requests. Commit task-by-task to the `phase-1-sub-D-macro-plan-derivation` branch via the user's git config." Citation: `docs/superpowers/plans/2026-05-19-phase-1-sub-D-macro-plan-derivation.md` branch discipline and task dispatch text.
- **Halt-on-validator-fail:** if real data violates an invariant, stop and escalate; never weaken invariants or tests just to pass. Citation: `docs/superpowers/plans/2026-05-19-phase-1-sub-D-macro-plan-derivation.md` implementation discipline; `docs/superpowers/specs/2026-05-19-phase-1-sub-D-macro-plan-derivation-design.md` validation strategy.
- **Read-codebase-before-designing:** verify the concrete sub-C artifact/schema before implementing assumptions. Topic 6 locked only after checking `crossings.parquet`; implementation should keep that discipline. Citation: `docs/superpowers/specs/2026-05-19-phase-1-sub-D-macro-plan-derivation-design.md` sections 3 and 9.
- **AST-over-grep for meta-tests:** use AST-based checks for validator meta-tests where comments and string literals could fool grep. Citation: `docs/superpowers/plans/2026-05-19-phase-1-sub-D-macro-plan-derivation.md` Task 13, `test_validator_files_do_not_compare_version_strings_directly`.

## 5. Cross-References

- Spec: `docs/superpowers/specs/2026-05-19-phase-1-sub-D-macro-plan-derivation-design.md`
- Plan: `docs/superpowers/plans/2026-05-19-phase-1-sub-D-macro-plan-derivation.md`
- Previous handoff: `docs/handoffs/2026-05-18-end-of-sub-C.md`

Open `docs/known_issues.md` items remain deferred per sub-D Topic 10 / spec section 15:

- #1 cold-fetch.
- #3 Sweden.
- #4 tokenizer.
- #7 rerun CLI.
- #8 validator coupling.

## 6. Reviewer Continuity

Umar is the reviewer. The reviewer's senior-reviewer agent persists across sessions. The paste-back loop continues from the new agent's first message after Task 1 / Gate 1 completion.

New session opens with: "Read docs/handoffs/2026-05-19-start-of-sub-D-implementation.md, then begin Task 1 from the implementation plan."

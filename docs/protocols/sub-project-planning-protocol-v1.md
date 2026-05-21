# Sub-project planning protocol v1

**Status:** active, derived from Phase 1 sub-E experience (2026-05-20 close).
**Scope:** all remaining Phase 1 sub-projects (sub-F, tokenizer fix, training scaffold) and future sub-projects within Bonzai-OSM.
**Source memory entry:** `feedback_external_source_of_truth_gate.md` records the sub-E defect class and the six-gate framework. This protocol operationalizes that entry — translating "thing we learned" into "thing the next sub-project will do."

## Why this protocol exists

Sub-E shipped with 41 commits, 18+ verify-before-asserting catches, and two structural defects that **survived every internal-consistency gate** until the first real-data integration test fired. The defects were not exotic — they were a swapped axis convention and a type mismatch. They survived because every gate sub-E had at the time verified **internal consistency** within sub-E's own contract surface. None of the gates asked "does sub-E's new abstraction agree with the upstream module it's translating?"

This protocol records the corrected baseline: five internal-consistency gates plus one external-correctness gate, plus four supporting principles (threshold pairing, proactive contract verification, reactive corollary on audit-bug propagation, per-stage prediction decomposition), plus a diagnostic heuristic (rough numbers). Each item below has a worked sub-E example and an operationalization pattern. The intent is that a reviewer or implementer can read this protocol cold and apply each principle without re-deriving it from sub-E's narrative.

## 1. The six gates

Each gate has: what it checks, what it cannot check, example defect caught, example defect missed. The "cannot check" column is load-bearing — it tells you which gaps the gate leaves and which subsequent gates need to fill them.

### Gate 1 — Plan review

- **What it checks:** text-level coverage against spec; scope alignment (what's in v1, what's deferred); cross-task coherence (does Task N's output match Task N+1's input); topic-by-topic gate discipline during brainstorm (one decision per assistant message).
- **What it cannot check:** logical reachability (would the fixture's row actually exercise this branch?); API correctness (does this helper exist? does it return this type?); real-data shape (will sub-D actually produce this row pattern?); per-row semantic interaction across tasks (will Task 7's invariant #5 fire before Task 7's invariant #3 because the row is non-null?).
- **Defect caught (sub-E):** Topic-by-topic brainstorm gates surfaced scope locks, lever-3 plumbing, the §10.1 #9 SlotKind carry-forward, the Topic 9 spec-amendment refinements, the 12-deferral ledger structure — all before plan-write. Plan review caught additions and gaps the spec missed.
- **Defect missed (sub-E):** All 18+ verify-before-asserting catches landed BELOW the plan-review layer. Plan review does not run code; it cannot detect the gap between what plan text says and what plan code does.

### Gate 2 — Pre-dispatch audit

- **What it checks:** helper API signatures by grep against shipped code (does `write_parquet_deterministic` exist? does `compare_version` have the kwargs the plan claims?); import existence; spec-vs-plan consistency (does plan honor spec §X.Y's mandate?); watch items the reviewer specifies (e.g., halt-on-validator-fail discipline, lever-3 propagation).
- **What it cannot check:** synthetic-fixture-vs-real-data agreement (audit reads code statically; cannot see what real upstream data shape looks like); shared-assumption errors where the audit's own code makes the same mistake as the code under audit; defects that manifest only at the intersection of multiple correct-looking layers.
- **Defect caught (sub-E):** `write_parquet_deterministic` doesn't exist (Task 6, fixup `fabce8b`); `write_yaml_canonical` doesn't exist (Task 8, fixup `4ffa101`); EXCLUDED_FROM_SHA mechanism missing per spec §9.2 (Task 8); rotation-aware invariant #5 weakening per spec §10.2 #5 (Task 9, fixup `3ed0554`); bare `!=` instead of `compare_version` for cross-tile version checks (Task 9); `_SUCCESS` write-then-unlink instead of validate-then-touch (Task 10, fixup `fd53fdd`); invariant #8 vacuous because orchestrator passed the same constant on both sides (Task 10).
- **Defect missed (sub-E):** Rotation axis convention swap — the audit's prediction step queried sub-C crossings directly, bypassing the rotation function, so the swap was structurally invisible to the audit. `feature_class` type mismatch — the reviewer's prediction script had the same bug (filtered `feature_class == "road"` against int8 column), caught it in audit code, fixed the audit script, never traced back to whether sub-E's production code had the same bug.

### Gate 3 — Implementer test-run

- **What it checks:** tests pass or fail; tests behave as expected on the test set the plan specified; lint findings under ruff `check` and `format`; full fast-suite regressions.
- **What it cannot check:** tests that pass for the wrong reason; tests that share the same buggy assumption as the code (self-consistent wrong); tests whose fixtures encode upstream contract incorrectly so that fixture + reader + pipeline agree internally but disagree with real data; verification of properties the test wasn't written to check.
- **Defect caught (sub-E):** Task 6 lint deviation (`zip(strict=True)` standardization); Task 7 invariant #5 fixture unreachability (loop ordering); Task 8 expected-count off-by-one (12 → 11); Task 10 modulo-arithmetic key collision (`edge_scope` dict collapses 144 rows to 128); Task 13 dead-code ternary `mean_gap if mean_tokens == 0 else mean_gap`.
- **Defect missed (sub-E):** Rotation axis swap survived for 11 tasks because every test ran against rotation's own wrong output and agreed self-consistently. `feature_class` type mismatch survived because every fixture encoded `"road"` as a string, sub-E read it as a string, pipeline filtered against string — all agreed with each other while disagreeing with sub-C's int8 schema.

### Gate 4 — Halt-and-report

- **What it checks:** discipline holds — implementer halts on first defect rather than improvising a fix or silently inlining; defects route to reviewer for plan-fixup decision; no contaminated commits land before the reviewer ratifies the fix shape.
- **What it cannot check:** defects beyond what the implementer can see (if the implementer halts on a test failure, the test failure's root cause might be upstream and not visible from the halt-point); defects the implementer rationalizes as "mechanical enough to fix inline" (the discipline erodes if not enforced).
- **Defect caught (sub-E):** Most Task 6–14 dispatches halted on the first plan defect surfaced. Task 9 implementer halted on YAML quote-style + parquet-mutation + corrupt_idx-collision defects in a single report — three defects in one halt. Task 14 implementer halted on first contact with real sub-D data showing the rotation axis mismatch.
- **Defect missed (sub-E):** Task 10 implementer applied three fixes inline before committing (3-tuple → 4-tuple key, `.parent` → `.parent.parent`, missing import). The fixes were correct but bypassed the reviewer gate. Retroactive plan-fixup at `aa0004a` closed the plan-vs-code drift; per-dispatch reminder for Tasks 11–15 explicitly forbade inline fixes. Pattern: any time "mechanically obvious" is the rationalization for inlining, the discipline is eroding.

### Gate 5 — Pre-code data-flow reasoning

- **What it checks:** predictive mental simulation of code paths against fixture data ("would this dict have collisions if I work through the modulo arithmetic?", "does this branch fire on this row?"); reasoning about cardinalities, key uniqueness, ordering before any code is written.
- **What it cannot check:** real-data shape (the reasoner uses synthetic fixture data); contract mismatches with upstream modules (the reasoner doesn't read upstream source unless explicitly told to); defects in the reasoner's own mental model.
- **Defect caught (sub-E):** Task 10 implementer caught the modulo-arithmetic `edge_scope` collision pre-code by walking through the dict semantics: "internal idx=0 → (0, 0, 0); internal idx=64 → (0, 0, 0); these collide." Caught before any code was written. Most epistemically valuable verify-before-asserting instance in sub-E.
- **Defect missed (sub-E):** Rotation axis (data-flow reasoning on rotation gives buggy answers because rotation is buggy — the reasoner has no way to know rotation disagrees with sub-D). `feature_class` (reasoning never queried sub-C's schema; the inferred-from-name shortcut won).

### Gate 6 — External-source-of-truth cross-reference

- **What it checks:** new abstraction's outputs match upstream module's documentation as ground truth; hand-enumerated expected values from the upstream module's docstring or schema; assertion logic does NOT use the new abstraction in expected-value computation.
- **What it cannot check:** defects internal to the new module that don't touch any upstream contract; defects in the upstream module itself (if upstream is wrong, the cross-reference test will faithfully verify the wrong behavior).
- **Defect caught (sub-E):** Would have caught rotation axis swap on day 1 of Task 3 implementation if it had existed. Hand-enumerate sub-D's lattice.py:130-149 external slot order, assert `cell_to_edge_ids` produces the same set. Would have caught `feature_class` type mismatch on day 1 of Task 5: hand-enumerate sub-C's parquet schema, assert `SubCFeatureRow` field types match.
- **Defect missed (sub-E):** This gate was missing for sub-E's first 11 tasks. After Task 14 surfaced the rotation defect, the gate was added as `tests/data/sub_e/test_rotation.py::test_external_set_matches_sub_d_hand_enumeration` + `test_internal_set_matches_sub_d_hand_enumeration` — and immediately caught the second instance (feature_class) within hours.

### Operationalization

Every sub-project's pre-dispatch audit for tasks that introduce a new abstraction over an existing module must include an explicit check: **"where in the test suite does this new abstraction get cross-referenced against the existing module's external documentation? If 'nowhere', halt the dispatch and require the test be written."** This is the audit-time addition that closes the sub-E gap.

The six gates are not redundant. Each catches a different defect class. Each leaves a different residual. The "cannot check" column tells subsequent gates what to look for. Skipping a gate leaves the gap unfilled; relying on one gate to cover another's residual fails the way sub-E failed twice in Task 14.

## 2. Threshold-pairing principle

**Every threshold-based verdict requires a paired structural-correctness check that asserts specific falsifiable properties the working system must satisfy.**

Thresholds-as-verdicts can be satisfied by broken systems if the broken system's output happens to land inside the threshold range. Empirical thresholds are **distribution-shape verdicts**, not **correctness verdicts**. They presume the underlying derivation is working as designed. If derivation is broken in a way that produces a threshold-passing distribution, the threshold cannot tell you. Only an independent correctness check can.

### Worked example (sub-E)

Sub-E's empirical gate had thresholds `max class fraction ≤ 0.90, min ≥ 0.02`. The `feature_class` type-mismatch defect produced the distribution `NONE 16.05% / MAJOR_ROAD 0% / MINOR_ROAD 84%`. Both `0.16` and `0.84` are inside the gate's bounds — **the gate would have green-lit a fundamentally broken derivation** where MAJOR_ROAD never appeared because the road-filter never matched any features.

The paired structural-correctness check (`test_layer3_writer_round_trips_major_and_minor`) asserts that **both `BoundaryClass.MAJOR_ROAD` and `BoundaryClass.MINOR_ROAD` must appear at least once** in the Layer-3 active-row distribution. This is a property a correct derivation must satisfy and a broken derivation cannot fake — the broken-derivation distribution had `MAJOR_ROAD = 0`, the test failed, the implementer halted, the defect was traced and fixed.

The threshold and the structural check together cover the full failure surface. The threshold alone does not.

### Pattern (operationalization)

For any new verdict mechanism in a future sub-project's plan:

1. **Identify the verdict threshold** — what aggregate statistic (mean, fraction, count, etc.) gates the decision.
2. **Identify the broken-but-in-range shape** — what plausible failure modes of the system under test would produce statistics that pass the threshold but with the system semantically broken. (Sub-E example: filter no-op → 0% one class but rest in range.)
3. **Write a structural-correctness check** — an independent assertion about a specific falsifiable property the working system must satisfy. Examples: "every class in the vocab must appear in the output"; "the row count must equal N"; "every external slot's tuple must match rotation's enumeration." The structural check must be falsifiable by exactly the broken-but-in-range shape from step 2.
4. **Both must pass for the verdict to be valid.** Threshold alone is not a verdict.

### When this applies

Any task that ships an aggregate-statistics verdict mechanism. Examples in Phase 1: perplexity-gap thresholds (sub-F + training scaffold), tokenizer round-trip accuracy thresholds (tokenizer fix), training loss thresholds (training scaffold).

## 3. Proactive contract verification

**Any time you write code that interacts with an upstream module's contract, verify the contract by reading the upstream module's source or schema, NOT by inferring from semantically-named field strings.**

Field names are documentation, not contracts. A field called `feature_class` could store a string-encoded category name OR an int8 enum code OR a path to a file OR a pointer to a vocabulary entry. The name does not constrain the type; the upstream module's schema does. Reading the schema costs ~30 seconds. Inferring from the name and being wrong costs the entire downstream defect chain.

### Worked example (sub-E)

Sub-E's `SubCFeatureRow.feature_class: str` was declared by inferring from the field name "feature_class" that the field stores a category name as a string. Real sub-C parquet schema (sub_c/io.py:44) has `pa.field("feature_class", pa.int8())` with sub_c/enums.py:22 defining `FEATURE_CLASS: {0: "road", 1: "building", 2: "poi", 3: "base"}`. The same mental shortcut produced both the audit-script bug (filtered `== "road"` against int8) and the production bug (same filter, same module). Neither author read sub-C's schema declaration.

### Pattern (operationalization)

Before declaring a type or writing a filter against any upstream module's data:

1. **Identify the upstream module's contract source** — typically a schema declaration (parquet/protobuf/typeddict), an enum definition, a dataclass docstring, or a spec section. Locate it by file path and line number.
2. **Read the actual declaration** — do not skim, do not assume from naming. Note the exact type, the enum values (if any), the nullability, the cardinality constraints.
3. **Encode your code against the actual declaration** — if the upstream is int8 enum, your code consumes int8 and compares against `encode_enum(UPSTREAM_ENUM, "category_name")`, not against the string `"category_name"` or the magic number `0`.
4. **Cite the upstream source in a comment** — `# feature_class is int8 per sub_c/io.py:44 + sub_c/enums.py:22 FEATURE_CLASS`. Cite by file:line so future readers can verify the contract without re-deriving.

### When this applies

Every task that consumes an upstream module's output. Phase 1 examples: sub-E reading sub-C/sub-D, sub-F reading sub-E, tokenizer reading sub-F, training scaffold reading all of the above. The proactive principle is the cheapest discipline in the protocol (~30 seconds per contract) and prevents the most opaque class of bug (silent type-comparison no-ops).

## 4. Reactive corollary on audit-bug propagation

**When your audit code has a bug against an upstream contract, audit whether the system under audit has the same bug.**

Same mental shortcut produces both bugs. The shortcut is usually "infer from the field name" or "assume the obvious convention." If you made the mistake once, the production-code author probably made it too — they may even be you. The reactive corollary triggers AFTER an audit-code bug is caught and fixed.

### Worked example (sub-E)

In the Task 14 pre-dispatch audit's prediction step, the reviewer's script filtered `feature_class_by_id.get(fid) != "road"` against int8 column. Got zero crossings, debugged, found the int8 enum, fixed the audit script, produced the prediction. **The reviewer never asked: "does sub-E's actual reader and pipeline have this same bug?"** Result: production bug shipped through 11 tasks, surfaced 5 days later via the writer-regression-guard test.

The reactive corollary would have caught this on the same day the audit-script bug was caught. The fix would have been "while I'm here, grep `feature_class` across sub-E's source and check the types match."

### Pattern (operationalization)

Every time you fix a bug in audit/verification/prediction code:

1. **Identify the upstream contract the bug was against** — what assumption did the audit code make that turned out to be wrong?
2. **Grep the system under audit for the same upstream contract** — does production code make the same assumption? Use the upstream contract name (e.g., `feature_class`) as the grep target.
3. **If yes, fix both in the same atomic plan-fixup** — production bug is found before it ships rather than after.
4. **If no, note the asymmetry in your report** — "audit code had X bug; checked production code, it correctly uses Y." This documents that the check was done.

### When this applies

Anywhere audit/verification/prediction code is written or debugged. The pre-dispatch audit pattern in particular benefits — audit scripts that exercise the same contracts as production code are exactly the place this trap surfaces.

## 4.5 Lint-cosmetic exemption rule

**Implementer-applied lint-driven changes are accepted as cosmetic** (not requiring halt-and-report) when ALL three conditions hold:

1. **Semantic equivalence** under the project's target Python version — verifiable trivially: same MRO, same `isinstance`, same comparison semantics.
2. **Precedent alignment** — either internal precedent in the sub-project matches, OR no contrary precedent exists.
3. **Automatic scope** — within `ruff --fix`'s automatic scope.

When all three hold, the implementer applies the change inline AND flags the deviation explicitly in their report. The reviewer ratifies as cosmetic in approval. When any one fails, halt-on-defect (Gate 4) still applies.

### Worked example (sub-E)

Task 12's `class ShuffleStrategy(str, Enum)` → `class ShuffleStrategy(StrEnum)` (ruff `UP042`) was ratified as cosmetic. `StrEnum` is semantically identical to `(str, Enum)` on py3.11+, `rotation.py` precedent already used `StrEnum`, and ruff `--fix` auto-applied. The implementer flagged the deviation explicitly; the reviewer ratified in approval. No plan-fixup was needed.

### What does NOT qualify

Structural changes do NOT qualify — different rule, halt-on-defect still holds. Worked counter-example: Task 10's 3-tuple → 4-tuple `edge_scope` key change was a semantic change to data layout, not a lint-cosmetic rewrite; the implementer's inline application of that fix was exactly the discipline erosion that Gate 4 is designed to prevent. The retroactive plan-fixup at `aa0004a` closed the drift, and per-dispatch reminders for Tasks 11–15 explicitly forbade inline fixes.

### Pattern (operationalization)

Before applying any lint-driven change inline:

1. **Verify semantic equivalence on target Python** — for `UP042` and similar rules, confirm the resulting construct has the same MRO, `isinstance`, and comparison semantics. Cite the equivalence in the report.
2. **Check internal precedent** — does another module in the sub-project already use the post-fix form? If yes, precedent aligns; if a module uses the pre-fix form deliberately, precedent contradicts and the deviation requires reviewer judgment.
3. **Confirm ruff `--fix` automatic scope** — manual rewrites or rule changes outside `--fix`'s automatic scope are not cosmetic by default.
4. **Flag explicitly in the implementer report** — "ruff `UP042` auto-applied at file:line; semantically equivalent on py3.11+; precedent at module:line; flagging per lint-cosmetic exemption."

### When this applies

Any task where the implementer encounters a ruff finding with `--fix` available. Operationally surfaced in sub-E Tasks 6, 9, 12, 13 (4 of 15 sub-E tasks). Without an explicit exemption rule, the implementer either halts unnecessarily (slowing the cycle) or applies inline without flagging (eroding the discipline). The exemption codifies the narrow case where inline-and-flag is correct, preserving halt-on-defect for everything else.

## 5. Per-stage prediction decomposition

**Predictions over multi-step pipelines decompose into per-stage assumptions, so divergence diagnostics can attribute deltas to specific stages.** Single-step predictions over multi-step processes conflate multiple effects and force diagnostic work on divergence.

### Worked example (sub-E)

Sub-E's prediction step for the Layer-3 empirical gate produced `NONE 29.66% / MAJOR_ROAD 18.55% / MINOR_ROAD 51.79%`. Actual: `NONE 16.05% / MAJOR_ROAD 19.36% / MINOR_ROAD 64.60%`. MAJOR_ROAD landed within 0.81 pp; NONE was off by −13.61 pp and MINOR_ROAD by +12.81 pp.

The 13.6 pp swap had two distinct compounded effects:

- **Activity ratio effect:** prediction assumed all 112 internal edges per tile were active (1008 worst case); reality 966 (95.8%) because some edges are scope_boundary or fully_masked from sub-D. Contributes ~5% of the redistribution.
- **Crossing prevalence effect:** prediction model assumed a specific zero-crossing-edge fraction; reality has more edges with at least one road crossing than the model estimated (dense urban Singapore). Drives the bulk of the 13.6 pp swap.

The single-step prediction conflated `(activity ratio × crossing prevalence × class-grouping map)` into one number. Divergence diagnostic work was needed to attribute. **If the prediction had been written as three per-stage assumptions multiplied through, the divergence would have attributed directly: "stage A (activity ratio) was 5% off, stage B (crossing prevalence) was the main driver, stage C (class-grouping map) was within 1%."**

### Pattern (operationalization)

For any prediction over a multi-step pipeline:

1. **Identify the stages** — what transformations does the input pass through before reaching the verdict? (Example: raw upstream data → activity filter → crossing filter → class grouping → distribution.)
2. **Write one assumption per stage** — "stage 1 expects N% of rows active"; "stage 2 expects M% of rows to have at least one match"; "stage 3 maps X categories to Y bins." Each assumption is a separate model.
3. **Compute the prediction by multiplying through** — show the intermediate values, not just the final number.
4. **On divergence, attribute** — compare actual stage-N output against stage-N's prediction. The first stage that diverges is the one where the prediction model needs refinement.

### When this applies

Any prediction-based verdict for a multi-step pipeline. Phase 1 examples: tokenizer round-trip accuracy (raw geometry → token → reconstructed geometry, multiple stages); perplexity gap (input tokens → conditioning → model output → NLL reduction → comparison, multiple stages).

## 6. Rough-numbers heuristic

**Suspiciously clean verification output is more likely a tool bug than improbably clean data.** Round numbers (zero, exactly half, all-equal, exactly-1000) are anti-signals. Rough numbers (29.66%, 18.55%, 51.79%) carry the texture of real measurement.

### Worked example (sub-E)

The first version of the Task 14 prediction script produced `0 crossings across all 9 Layer-3 tiles, NONE 100%`. Zero crossings is implausibly clean for dense urban Singapore. The reviewer noticed the round number, traced back to the verification code, found `feature_class != "road"` against int8 column → fixed → re-ran → got the realistic 29.66/18.55/51.79 distribution.

The same heuristic applies to test pass rates: 100% pass after a major refactor or new-abstraction introduction is suspicious if no test cross-references against an external source of truth. "All green" with no external-correctness gate is the same anti-signal as "0% in one class."

### Pattern (operationalization)

On every verification output:

1. **Scan for round numbers** — zero, exactly-N where N is a power of 10, all-equal fractions, exactly-half, exactly-100%. These are anti-signals.
2. **Compare against texture** — real measurements have noise. If the output looks like a fixture or a textbook example, suspect the verification.
3. **Trace the verification before accepting the result** — re-read the verification code, check assumptions against upstream, run the verification a different way.
4. **Real measurement texture is the green light** — 0.5179 is more trustworthy than 0.5000 even though they're nearly the same number, because 0.5179 carries the noise signature of real data.

### When this applies

Every prediction, audit, test run, smoke check. The heuristic is cheap (one glance at the output) and catches verification bugs that would otherwise be invisible.

## 7. Decision tree — when to invoke each principle

**About to write a verdict mechanism (threshold, gate, pass/fail criterion)?**
→ Apply **threshold pairing**. Identify what broken-but-in-range shape the threshold cannot detect. Write the paired structural-correctness check.

**About to write code interacting with an upstream module's contract (type declaration, filter, comparison)?**
→ Apply **proactive contract verification**. Read the upstream source/schema; do not infer from field names. Cite the upstream by file:line in a comment.

**About to fix a bug in audit/verification/prediction code?**
→ Apply **reactive corollary**. Grep the system under audit for the same upstream contract. Check whether production code has the same bug.

**About to interpret divergence between predicted and actual?**
→ Apply **per-stage prediction decomposition**. Identify the stages; attribute the divergence to a specific stage. Refine that stage's model.

**About to accept a verification output as evidence?**
→ Apply **rough-numbers heuristic**. Scan for round numbers or all-equal patterns; treat them as anti-signals. Trace the verification before accepting.

**About to introduce a new abstraction over an existing module?**
→ Apply **Gate 6 (external-source-of-truth cross-reference)**. Hand-enumerate expected values from the upstream module's docstring or schema; the assertion must not use the new abstraction in the expected-value computation.

**About to ship a fix to a defect surfaced by an implementer's halt?**
→ Apply the **plan-fix-then-resume** pattern. Reviewer applies plan-fixup commit (no inline fixes by implementer); implementer resumes via SendMessage with explicit per-file diffs. Discipline is non-negotiable; the cost of one plan-fixup commit is lower than the cost of one plan-vs-code drift.

**About to find a defect during pre-dispatch audit?**
→ Halt the dispatch. Apply plan-fixup. The pre-dispatch audit is the cheapest defect-fix point in the cycle; defects caught here don't contaminate any commits.

**About to apply a ruff lint finding during implementation?**
→ Apply the **lint-cosmetic exemption rule** (§4.5). Verify all three conditions: semantic equivalence on target Python, internal precedent alignment, ruff `--fix` automatic scope. If all three hold, apply inline AND flag in the report. If any one fails, halt-on-defect still applies.

**About to commit a plan-fixup?**
→ Apply the **audit-after-fixup pattern** (§8). Run a fresh pre-dispatch audit on the surrounding code, not just the diff. Strengthening one gate can expose defects that the prior gate's weakness was masking by coincidence.

## 8. Audit-after-fixup pattern

**Every plan-fixup commit triggers a fresh pre-dispatch audit on the surrounding code, not just the diff.** Strengthening one gate can expose defects that the prior gate's weakness was masking by coincidence. The act of correcting one assumption can change what other assumptions in the system are now exercised — and assumptions that were previously latent because a buggy fixture or weak gate was hiding them become live the moment the fix lands.

### Worked example (sub-E)

Task 10's plan-fixup `d95ace7` corrected the synthetic fixture so that its rotation reflected sub-D's external slot order. That correction was, in isolation, a clean win — the fixture now matched the upstream contract. But the now-correct rotation exposed the production pipeline's 3-tuple `edge_scope` key collision (modulo-arithmetic on internal idx) that the bad fixture had been masking by coincidence: the buggy fixture happened to produce row patterns where the colliding keys had identical values, so the collision was silent. Once the fixture rotated correctly, the row values diverged, the dict overwrote, the test failed. Without a post-fixup audit, the collision would have remained latent until a downstream test happened to surface it days later.

### Pattern (operationalization)

Every plan-fixup commit:

1. **Identify the prior gate's weakness** that the fixup is closing — what was the original gate's residual that this fix is now addressing? (Sub-E example: the synthetic fixture's rotation was inverted, masking real rotation behavior.)
2. **Audit the surrounding code, not just the diff** — grep for related contracts in modules the fix touches; run integration tests on pipelines that consume the fixed output; walk through stages that depend on the corrected invariant.
3. **Treat any newly-surfaced anomaly as a candidate latent defect** — if a test that previously passed now fails, or a stage that previously produced clean output now produces something noisy, the strengthening exposed something that was masked.
4. **Cost:** ~15–20 minutes per fixup. **Coverage:** latent defects exposed by the fixup's strengthening — defects that would otherwise wait for an unrelated test to surface them, often days later, with attribution harder.

### When this applies

Every plan-fixup commit in a sub-project. Particularly load-bearing when the fixup touches a fixture, schema, or upstream-contract assumption — those are the cases where the prior gate's residual was most likely to be masking adjacent defects in pipelines that consumed the wrong-but-internally-consistent output.

## Closing notes

These six gates + five principles are the consolidated discipline as of sub-E close. They are not exhaustive — sub-F may surface a seventh gate or a sixth principle that this protocol does not anticipate. If it does, add it here with a worked example and operationalization, the same way this protocol records sub-E's lessons.

The protocol's purpose is institutional compounding: each sub-project's lessons should reduce the next sub-project's defect surface, not relearn from zero. Sub-D's `feedback_test_weakening_to_pass`, sub-B2's `feedback_subagent_branch_pattern`, sub-C's `feedback_pyarrow_hive_partition_inference`, and now sub-E's six-gate framework + five principles are all part of the same compounding capital.

Read this protocol before sub-F's brainstorm. Reference its sections during sub-F's plan-write. Apply its decision tree during sub-F's dispatch. When sub-F closes, update this protocol with whatever it teaches that the current version doesn't anticipate.

The memory entry `feedback_external_source_of_truth_gate.md` is the narrative source for what happened in sub-E. This protocol is the operational source for what to do going forward. Both are required; neither is sufficient.

## Versioning

Each sub-project close is a potential bump point. The version increments when the protocol adds a new gate, principle, or operationalization pattern derived from that sub-project's experience. If the sub-project closes without surfacing new institutional capital, v1 holds.

Anticipated bump points: sub-F close (potential v2), tokenizer-fix close, training-scaffold close, end of Phase 1.

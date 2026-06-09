# Phase-2 Bake-off — Phases A–C Local Build (2026-06-09)

**Interim summary, written at the clean tree before any Leonardo/Phase-D work.** Captures the local-buildable portion of the bake-off delta-reconciliation so the milestone is durable and faithfully reconstructable once Phase-D churn starts.

## State (verified from git, not memory)

- **Branch `phase-2-bakeoff` @ `1d80a9f`** — local-only, **NOT pushed**.
- **`main` untouched @ `738a8fa`** (rebase target; the branch was rebased onto it cleanly, 0 manual conflicts).
- **Full test suite: `1317 passed`, 2 skipped (Leonardo-only berlin), 36 deselected, 2 xfailed (pre-existing).** ruff format + check clean.
- **Phases A–C (all local, no GPU) COMPLETE.** Phase D (Tasks 8–12, all Leonardo) is **PARKED on the allocation word**.

## What this is

The delta-reconciliation of the 2026-06-02 bake-off onto the EU/eval-set reality merged to `main` (112 commits: multiregion CRS → EU extract → corpus-completion → eval-set-gen). The bake-off decides architecture on **KS-realism over the 4-city EU held-out set** (coherence + its power gate are Phase-3), with a **data-feasible, diagnostic-determined ladder** (the Chinchilla-frontier `r→ladder` is a pre-committed function of a measured `r`, not a fixed 30M/100M/300M/1B).

- **Spec** (authoritative): `docs/superpowers/specs/2026-06-09-phase-2-bakeoff-delta-design.md` — content sha `cb4a906…` (amended in-flight with the §3 CORRECTION; committed at `e7ab4a1`).
- **Plan**: `docs/superpowers/plans/2026-06-09-phase-2-bakeoff-delta-reconciliation.md` — content sha `39d56c7…` (carries the re-sequenced Task 1, region-aware bullet, and the three carry-forward notes).
- **Execution mode**: subagent-driven (fresh implementer per task, implementer≠reviewer, Umar = chat reviewer), stop-before-commit on every halt-gate and design fork; batched review only for the pure-arithmetic rule functions.

## Commits per phase (12 delta commits atop the rebased branch)

**Phase A — rebase + obligation (a)**
- `3fd14fb` feat: region-aware holdout repoint + schema selection (obligation a)
- `e7ab4a1` docs: delta-reconciliation spec + plan land on the branch

**Phase B — the 4 pre-committed rule functions**
- `ec652f4` feat: Rule 1 `feasible_ladder` + Rule 2 `decision_basis` (`ladder.py`)
- `ecc7a7d` feat: worst-case city aggregate + #21 binding-city power gate (`city_aggregate.py`)
- `06b567e` docs: munich power-floor first-model expectation (Task-9 note)
- `7fda0dc` feat: §4 conditioning-discrimination gate (`conditioning_gate.py`)
- `e89b548` fix: Phase-B batched-review fixes (I1 ragged-input guard + M2 type-safety)

**Phase C — net-new T5/T6 builds**
- `d3e550b` docs: Rec-2 dict→list `decision_ks` seam (Task-6 obligation)
- `0f8c0de` feat: T5 per-city KS + worst-case decision axis, no pooling (`multiregion_realism.py`)
- `f38c315` docs: Task-12 input-completeness precondition
- `abc9505` feat: T6 multi-region train build (per-city manifests + datamodule union; whole-city exclusion)
- `1d80a9f` docs: Task-12 precondition active-filter linkage

## Gates / properties proven, with red-on-divergence evidence

Every gate was proven **non-vacuous** — it fires on the bad regime and passes on the good one — and each red-on-divergence was reproduced by hand at the controller level (not taken from the implementer's report).

| Gate / property | What it proves | Red-on-divergence (reproduced) |
|---|---|---|
| **Obligation-(a) red-before** (Task 1) | the holdout repoint fixes a real EU-vs-SG drift | stash the (a) edits → `_holdout_ids("munich")` → `KeyError: 'munich'` (genuine runtime, not a collection error); SG stays green |
| **Region-aware repoint** (Task 1) | SG survives as the local-test fixture; EU served | blanket repoint → 2 locked SG tests `KeyError: 'singapore'`; region-routing → both green (incl. un-masked slow `test_emergence`) |
| **#21 binding-city power gate** (Task 4) | a binding city under its own `C/√n` floor is demoted, not trusted | neuter `if gap > floor` → `if True` → under-power test fails (munich wrongly binds, `demoted_from=()`) |
| **Conditioning-discrimination gate** (Task 5) | residual city-style → HALT/REOPEN; macro-tracked → PASS; empty → fail-closed | neuter to always-pass → the residual-city-style FIRE test fails (0.22 wrongly passes) |
| **I1 ragged-input guard** (Task 4 review) | a city missing-from / only-in a non-first backbone raises a clear `ValueError`, not a bare `StopIteration` or silent worst-city drop | remove guard → (a) `RuntimeError/StopIteration`, (b) `DID NOT RAISE` (silent drop) |
| **No-pooling worst-case axis** (Task 6) | the decision is per-city-then-worst-case, never a concatenated reference | neuter `decision_ks` to pooled → worst city's 1.0 dilutes to 0.5, both worst-case tests fail |
| **Rec-2 dict→list bridge** (Task 6) | `decision_ks` hands a *list* to `worst_case_city` | a dict would `AttributeError` on str keys; the regression-lock returns 1.0 instead |
| **Task-7 `_holdout_ids` boundary** (Task 7) | a non-held-out train city is built with empty holdout, not a raise | naive path → `ValueError: unknown region 'prague'`; driver's all-validated bypass → no raise (`_holdout_ids` never called, monkeypatch-confirmed) |
| **CellExample.key collision-lock** (Task 7) | two cities sharing `(tile_i, tile_j)` get DISTINCT keys | revert key to 4-tuple → prague/barcelona keys collide `(3,4,1,2)==(3,4,1,2)` → test reds |

## Three carry-forwards (recorded in the committed plan)

1. **Rec-2 — Task 6 / Task 12 dict→list seam.** `per_city_ks` returns `{city: PerCityKS}`; `worst_case_city` takes a list. The `list(per_city.values())` bridge is localized to `decision_ks`; Task 12 must assemble `{backbone: list(per_city_ks(...).values())}` for `binding_city_verdict`.
2. **munich power-floor — Task 9 first-model expectation.** munich is the smallest held-out city (n=156) → its #21 floor is the highest (`1.358/√156 ≈ 0.1087`). It is the **most likely city to be demoted-for-under-power at first model**, so the munich→manchester swap reserve (parked Phase-3) may fire on **POWER** grounds — by design, not a defect. Don't pre-empt it.
3. **Task-12 input-completeness precondition.** The worst-case bar's safety rests on a complete `real_by_city`. The held-out exclusion is an **active filter** (the 4 are present as `validated: true` in the G4 roll-up and removed by `train_cities`, NOT structurally absent), so a held-out city CAN slip through an upstream gap → Task 12 must assert `set(real_by_city.keys()) == {eisenhuttenstadt, glasgow, krakow, munich}` and fail loud.

## Notable corrections caught during execution (the gate structure earning its cost)

- **Vacuous-green red-before premise (Task 1):** the plan's "rebase → expect RED" assumed the local suite drives the EU holdout path; it doesn't (SG-pinned / slow-deselected). Re-sequenced to a genuine red-before on (a)'s actual surface.
- **Blanket-repoint broke Singapore (Task 1):** the holdout readers are dual-region; resolved with region-conditional manifest+schema selection (`holdout_manifest_for_region`), reconciling spec §3 with §5/§6.
- **Train-city count is 38, not the spec's ~22 (Task 7):** the real G4 roll-up has 42 `validated: true` cities − 4 held-out = 38; corrected against source, the exclusion is an active filter.
- **`train_order` type-lie (Task 7):** the region-keyed 5-tuple key left the annotation saying 4 ints; made honest (`tuple[str, int, int, int, int]`).

## Phase D — PARKED (Tasks 8–12, all Leonardo)

Gated on the **allocation word**. Two calls ride in with it: **(1)** worst-case vs mean aggregation (recommendation: worst-case; gated on the Task-9 conditioning-discrimination diagnostic, which can reopen T5); **(2)** allocation/timing (`AIFAC_P02_222` soft-ends 2026-06-11; ~36k/40k core-h remain; top-up ~1–2 d post-expiry).

**Owed before any scored GPU-h (brought for review before Task 8/9, not after):** a real **core-h → GPU-h conversion** from the Leonardo boost-node spec (NOT core-h ≈ GPU-h), and a **diagnostic/cheapest-first ordering** inside the guaranteed pre-11 window. Task 9 carries two HALT-gates (conditioning-discrimination fail → T5 reopens; ∅-ladder → escalate to more-data) and Task 10.5 the mamba-ssm verify-before-lock gate. No push/merge to `main` without the explicit word, `--no-ff`, suite green.

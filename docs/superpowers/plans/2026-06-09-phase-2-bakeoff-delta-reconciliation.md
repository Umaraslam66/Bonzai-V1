# Phase-2 Bake-off Delta-Reconciliation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconcile the 2026-06-02 bake-off branch onto the EU/eval-set reality (delta-spec `docs/superpowers/specs/2026-06-09-phase-2-bakeoff-delta-design.md`), so the bake-off decides on KS-realism over the 4-city EU held-out set with a data-feasible, diagnostic-determined ladder.

**Architecture:** A **fork on the Task-1 diagnostic**. Phases A–C build, locally and test-first, (A) the rebase + obligation-(a) repoint, (B) the pre-committed *rule functions* that encode the fork (`feasible_ladder`, `decision_basis`, the worst-case aggregator + #21 artifact-check, the conditioning-discrimination gate), and (C) the net-new T5 eval aggregation and T6 multi-region train build. Phase D runs on Leonardo (gated by Umar's allocation word): the Task-1 diagnostic *measures*, the rule functions *decide*, and only then are the ladder, decision basis, and scored runs determined. **Nothing downstream of Task-1 is hardcoded** — scales and decision basis are the *outputs* of the rule functions applied to measured `r`.

**Tech Stack:** Python 3.11+, pytest, ruff, PyTorch Lightning (existing), `mamba-ssm` (later), Leonardo Slurm (4×A100). No new heavyweight deps (KS/curve are hand-rolled, matching existing precedent).

---

## Cross-cutting discipline (apply to EVERY task)

- **TDD:** failing test → run-it-fails → minimal impl → run-it-passes → commit. `from __future__ import annotations` + type hints on public functions. `ruff format` + `ruff check` before every commit (run **unpiped** — never pipe a check into a filter that swallows its exit code, per `feedback_tool_output_silence_is_not_pass`).
- **Halt-gates, not reports:** where a step says **HALT-GATE**, a failure stops the plan and routes to the reviewer — it is never "run and note." The implementer does not improvise a fix (`feedback_subagent_branch_pattern`, protocol-v3 Gate 4).
- **Branch discipline:** all work on `phase-2-bakeoff` (after Task 1's rebase). Subagents: **no new branches, no push, no PR, no merge.** No GPU runs. No `main` history rewrite.
- **Gate 2 (pre-dispatch audit):** before any step that edits an existing module, READ the current module and verify the signatures named here still hold; hand corrections forward. Plan snippets may have drifted.
- **Leonardo (Phase D):** verified-end-state, never exit codes — re-read the artifact / recompute the sha / count the real units (`feedback_no_marker_without_endstate_verify`). Gated on Umar's allocation word.

---

## File structure

**New (pure, complete code below):**
- `src/cfm/eval/ladder.py` — Rule 1 (`feasible_ladder`) + Rule 2 (`decision_basis`).
- `src/cfm/eval/city_aggregate.py` — per-city worst-case aggregator + #21 binding-city power gate.
- `src/cfm/eval/conditioning_gate.py` — the §4 conditioning-discrimination gate.

**Modified (Gate-2 read first; teeth complete below):**
- `src/cfm/eval/realism.py`, `src/cfm/eval/feature_resolution.py`, `src/cfm/eval/curve.py` — wire per-city → worst-case into the decision axis (T5).
- `src/cfm/eval/geometry.py`, `src/cfm/data/training/build_shards.py`, `src/cfm/data/training/datamodule.py` — multi-region build + holdout re-point (T6 + obligation a).
- `scripts/train_scaffold.py`, the DDP scripts — obligation-(a) schema flip + path re-point.

**Tests:** `tests/eval/test_ladder.py`, `tests/eval/test_city_aggregate.py`, `tests/eval/test_conditioning_gate.py`, `tests/eval/test_realism_multiregion.py`, `tests/data/training/test_build_shards_multiregion.py`.

---

## PHASE A — Rebase & obligation-(a) (the opening sequence; halt-gates)

### Task 1: Rebase onto main, red-before/green-after, atomic obligation-(a)

**Pre-req:** plan reviewed + Umar's go for the branch op. This is the only task that touches branch state.

> **Task-1 correction (2026-06-09, after the rebase landed).** The original red-before premise (old Step 2) was WRONG: it assumed the existing local suite drives the EU holdout path. It does not. Verified from disk after the rebase — the non-slow suite exercises the holdout consumers only via `region="singapore"` against the still-valid SG manifest (the EU path is `-m 'not slow'`-deselected / Leonardo-skipped), so the rebased suite came back **vacuous-GREEN (`1281 passed`), not RED** — exactly the §4 GAP-not-DRIFT / "tests don't exercise the live holdout path" alarm. The HALT-gate caught it (implementer stopped, did not apply (a)); Umar adjudicated → re-sequence Task 1 to manufacture a GENUINE red-before on obligation **(a)'s actual surface**, now feasible because the EU manifest is local. Three corrections fold in:
> - **(a)/(c) scope split (LOAD-BEARING).** Obligation **(a)** = repoint the holdout-**MANIFEST** consumers + flip the schema sites — **locally testable** now the EU manifest is on disk (`data/processed/eval_set/2026-04-15.0/multiregion/holdout_manifest.yaml`, schema 2.0). Obligation **(c)** = the EU-train-split KS resolved-gap recompute that feeds `assert_resolution_sufficient` (`resolution.py:46`, which reads `eval_set_locked_marker` = the **SG** marker, NOT the manifest) — needs the held-out **tile token data, which is Leonardo-only → Task 9.** `assert_resolution_sufficient` is **NOT in (a)'s scope** and stays on the SG marker until Task 9. **The red-before MUST fire on (a)'s manifest-consumer surface ONLY, never the (c)/SG-marker path** — a red there would clear after (a) without proving (a) did anything (a vacuous green wearing a red coat). (This corrects the spec §3 Step-2 parenthetical, which conflated (a)'s manifest drift with (c)'s `assert_resolution_sufficient`/SG-marker surface.)
> - **"3 schema sites" is phantom → there are 2.** `scripts/ddp_audit_halt_check.py` carries no `expected_holdout_schema` param (verified on both refs). The flip sites are exactly `scripts/train_scaffold.py:66` and `scripts/ddp_resume_check.py:72`.
> - **`uv sync` needs `--extra dev --extra training`.** Dev-only omits torch/lightning/pydantic → 9 vacuous `ModuleNotFoundError` collection errors (itself a vacuous-failure source to eliminate before trusting any red or green).
> - **Region-aware repoint, NOT blanket (LOAD-BEARING; 2nd correction, Umar-ruled 2026-06-09).** A *blanket* repoint breaks Singapore: `_holdout_ids`/`compute_training_tile_ids` and `geometry.holdout_polygons_per_active_cell` are **dual-region** (called with `region="singapore"` in local tests AND EU cities at runtime); a blanket swap to the EU manifest fail-closes the SG path (`KeyError: 'singapore'`), which spec §5/§6 say SURVIVES ("NEVER touches the Singapore set"; "local = Singapore + the frozen holdout manifest"). **Resolution: §3's "repoint the consumers" is REGION-CONDITIONAL** — the `region` arg selects manifest **and** schema: `singapore` → SG single-region manifest / schema-1.0; the 4 EU cities → multiregion manifest / schema-2.0. Mechanism: a small `holdout_manifest_for_region(release, region)` helper (returns path + expected schema), so the region-conditionality is EXPLICIT and the blanket swap can't be re-introduced. This honors BOTH §3 (EU consumers repoint) AND §5/§6 (SG never touched). The "flip 2 schema sites to 2.0" is likewise region-conditional. **Why keep SG:** at bake-off runtime SG is unused (train=~22 EU cities T6, eval=4 EU held-out T5); SG is now the **local-test fixture** — EU tile data is Leonardo-only, so SG is the ONLY local coverage of `_holdout_ids`/emergence-floor/datamodule. Retiring it would blind local TDD on exactly the data-path functions this phase rewrites.

**Files:** Add a `holdout_manifest_for_region(release, region)` helper (returns path + expected schema, region-conditional: `singapore`→SG manifest/1.0, the 4 EU cities→multiregion/2.0) — natural home `src/cfm/eval/holdout/paths.py` (beside `multiregion_holdout_manifest_path`). Route the dual-region holdout-manifest readers through it: `src/cfm/eval/geometry.py::holdout_polygons_per_active_cell`, `src/cfm/data/training/build_shards.py::_holdout_ids`. For the region-parameterized scripts (`scripts/train_scaffold.py` `cfg.region`, `scripts/ddp_resume_check.py` `_REGION`), select manifest + `expected_holdout_schema` by region too (an SG smoke run stays 1.0/SG; an EU run is 2.0/EU) — NOT a blanket flip. Do NOT touch `src/cfm/eval/holdout/pipeline.py:306` (the eval-set-gen SG builder — "NEVER touches the Singapore set") or `assert_resolution_sufficient` (obligation (c)/Task 9). New test: the TWO-SIDED red-before (EU red→green; SG stays green).

- [x] **Step 1: Rebase — DONE (verified clean).** `git checkout phase-2-bakeoff && git rebase main`. Git auto-resolved the predicted `train_scaffold.py` conflict (additive sides non-overlapping) with **0 manual conflicts**; both sides survive (main's holdout-audit wiring + the branch's bake-off CLI). Branch `605d4b8 → 740fd21`; merge-base = main tip `738a8fa`; `main` untouched.

- [ ] **Step 2: Establish the vacuous-green baseline.** `uv sync --extra dev --extra training`, then `uv run pytest -q`. Expected: **GREEN** (`1281 passed, 2 skipped, 36 deselected, 2 xfailed`), documenting that the existing suite is blind to (a)'s surface — this is the premise that JUSTIFIES the new red-before test (Step 3), not a pass to trust.

- [ ] **Step 3: Author the TWO-SIDED red-before on (a)'s ACTUAL surface (new local test). EXPECT GENUINE RED on the EU side.** Drive a region-routed consumer (natural choice: `build_shards._holdout_ids`, manifest-read only — `geometry`'s tile-data round-trip needs Leonardo). Assert BOTH sides: **(EU)** `_holdout_ids(release, <an EU city, e.g. "munich">)` resolves holdout tile-ids from the multiregion manifest — RED before the region-aware routing exists (the consumer reads the SG manifest → `KeyError: 'munich'`), GREEN after; **(SG)** `_holdout_ids(release, "singapore")` STILL resolves from the SG manifest — must STAY GREEN across the change (proving region-aware routing didn't break the path §5/§6 protect). The EU red MUST be a real assertion/KeyError on the manifest surface — **confirm it is NOT a collection/import error** — and must **NOT** touch the `assert_resolution_sufficient`/SG-marker path. (Principled deviation from the earlier "no new tests in Task 1": restoring the evidence chain the T4 design rests on.)

- [ ] **Step 4: Apply obligation (a) — REGION-AWARE, atomically.** In one edit: add `holdout_manifest_for_region(release, region)` (SG manifest + schema-1.0 for `singapore`; multiregion manifest + schema-2.0 for the 4 EU cities); route `geometry.holdout_polygons_per_active_cell` and `build_shards._holdout_ids` through it; make the scripts' `expected_holdout_schema` region-conditional (NOT a blanket 1.0→2.0 flip). Repoint + schema-selection travel together (flip-ahead-of-repoint fail-closes; repoint-ahead-of-flip audits EU against the SG schema). If a consumer must parse the 2.0 structure (`regions:{city:{tiles:[{tile_i,tile_j}]}}`) vs 1.0, handle whatever the two-sided test demands. Do NOT touch `resolution.py`/`assert_resolution_sufficient`.

- [ ] **Step 5: Re-run, EXPECT GREEN — including the un-masked SG regression test.** `uv run pytest -q` (full suite, incl. the new test) → PASS, no regressions; the 2 locked SG `test_build_shards.py` tests STAY GREEN. **Un-mask the deselected pocket:** `tests/eval/test_emergence.py:61` calls the consumer with `region="singapore"` but is slow/deselected — the vacuous-green pocket that HID this regression. Run it explicitly (`uv run pytest tests/eval/test_emergence.py -v` or include the slow marker) and confirm GREEN — the SG-survives claim must NOT rest on a test that doesn't run. The EU-side Step-3-RED → Step-5-GREEN **and** SG-stays-GREEN (incl. the un-masked emergence test) is the two-sided evidence.

- [ ] **Step 6: Commit obligation (a) + the new test.**
```bash
git add -A
git commit -m "fix(bakeoff): rebase onto EU main + atomic holdout re-point + schema 2.0 flip (obligation a)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 7: Add the delta-spec + plan to the branch.** `git add docs/superpowers/specs/2026-06-09-phase-2-bakeoff-delta-design.md docs/superpowers/plans/2026-06-09-phase-2-bakeoff-delta-reconciliation.md && git commit -m "docs(bakeoff): delta-reconciliation spec + plan"`. (The spec/plan were authored uncommitted on the working tree; they land here.)

---

## PHASE B — Pre-committed rule functions (local TDD; these ENCODE the fork)

### Task 2: Rule 1 — `feasible_ladder` (r → on-frontier scales)

**Files:** Create `src/cfm/eval/ladder.py`; Test `tests/eval/test_ladder.py`.

- [ ] **Step 1: Write the failing test.**
```python
from __future__ import annotations
import pytest
from cfm.eval.ladder import feasible_ladder, LadderDecision

M30, M100, M300, B1 = 30_000_000, 100_000_000, 300_000_000, 1_000_000_000

def test_r20_unique_only_30M():
    d = feasible_ladder(r=20.0)
    assert d.feasible == (M30,)

def test_r10_unique_only_30M():
    assert feasible_ladder(r=10.0).feasible == (M30,)

def test_r5_unique_adds_100M():
    assert feasible_ladder(r=5.0).feasible == (M30, M100)

def test_r5_epoch4_adds_300M():
    assert feasible_ladder(r=5.0, epoch_factor=4.0).feasible == (M30, M100, M300)

def test_1B_dropped_even_at_low_r_and_E4():
    # 1B needs r <= 624M*E/1e9; at E=4 that is r<=2.496. r=2.5 must still drop it.
    assert B1 not in feasible_ladder(r=2.5, epoch_factor=4.0).feasible

def test_empty_ladder_sets_escalate():
    d = feasible_ladder(r=1000.0)  # nothing clears
    assert d.feasible == () and d.escalate_more_data is True

def test_conservative_rounding_uses_upper_r_ci():
    # CI [5,7] straddles the 100M boundary (6.24). Conservative => use 7 => drop 100M.
    from cfm.eval.ladder import feasible_ladder_conservative
    assert feasible_ladder_conservative(r_ci_high=7.0).feasible == (M30,)
```

- [ ] **Step 2: Run, expect FAIL.** Run: `uv run pytest tests/eval/test_ladder.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement.**
```python
# src/cfm/eval/ladder.py
"""Pre-committed bake-off ladder + decision-basis rules (delta-spec §2; T3).

Rule 1 (`feasible_ladder`): scale N is on-frontier-feasible iff r*N <= train_tokens*E.
Failing scales are DROPPED, never run data-limited. Conservative boundary rounding uses
the UPPER r-CI bound (higher effective r -> fewer rungs).
"""
from __future__ import annotations
from dataclasses import dataclass

#: Authoritative frozen EU train-token count (_EVAL_SET_LOCKED, release 2026-04-15.0).
TRAIN_TOKENS: int = 623_900_790
#: The PRD/baseline candidate scales, in params.
LADDER_SCALES_PARAMS: tuple[int, ...] = (30_000_000, 100_000_000, 300_000_000, 1_000_000_000)

@dataclass(frozen=True)
class LadderDecision:
    feasible: tuple[int, ...]
    dropped: tuple[int, ...]
    escalate_more_data: bool  # True iff feasible is empty (the ∅-case)

def feasible_ladder(
    r: float, *, epoch_factor: float = 1.0,
    train_tokens: int = TRAIN_TOKENS, scales: tuple[int, ...] = LADDER_SCALES_PARAMS,
) -> LadderDecision:
    if r <= 0:
        raise ValueError("r must be positive")
    budget = train_tokens * epoch_factor
    feasible = tuple(n for n in scales if r * n <= budget)
    dropped = tuple(n for n in scales if n not in feasible)
    return LadderDecision(feasible=feasible, dropped=dropped, escalate_more_data=not feasible)

def feasible_ladder_conservative(r_ci_high: float, **kw) -> LadderDecision:
    """Boundary-straddle rule: size by the UPPER r-CI bound so we never add a rung
    the data can't clearly support."""
    return feasible_ladder(r_ci_high, **kw)
```

- [ ] **Step 4: Run, expect PASS.** Run: `uv run pytest tests/eval/test_ladder.py -v` → PASS.
- [ ] **Step 5: Commit.** `git add src/cfm/eval/ladder.py tests/eval/test_ladder.py && git commit -m "feat(bakeoff): Rule 1 feasible_ladder (r->on-frontier scales)"`

### Task 3: Rule 2 — `decision_basis` (#feasible points → curve vs fixed-scale+§13)

**Files:** Modify `src/cfm/eval/ladder.py`; Modify `tests/eval/test_ladder.py`.

- [ ] **Step 1: Write the failing test.**
```python
def test_decision_basis_step_function():
    from cfm.eval.ladder import decision_basis, DecisionBasis
    assert decision_basis(0) is DecisionBasis.ESCALATE_MORE_DATA
    assert decision_basis(1) is DecisionBasis.FIXED_SCALE_PLUS_S13
    assert decision_basis(2) is DecisionBasis.FIXED_SCALE_PLUS_S13
    assert decision_basis(3) is DecisionBasis.SCALING_CURVE
    assert decision_basis(4) is DecisionBasis.SCALING_CURVE
```

- [ ] **Step 2: Run, expect FAIL.** Run: `uv run pytest tests/eval/test_ladder.py::test_decision_basis_step_function -v` → FAIL.

- [ ] **Step 3: Implement (append to `ladder.py`).**
```python
from enum import Enum

class DecisionBasis(Enum):
    ESCALATE_MORE_DATA = "escalate_more_data"      # 0 feasible points
    FIXED_SCALE_PLUS_S13 = "fixed_scale_plus_s13"  # 1-2 points: curve REPORTED, never decision-bearing
    SCALING_CURVE = "scaling_curve"                # 3+ points: falsifiable curve (>=1 DoF)

def decision_basis(n_feasible: int) -> DecisionBasis:
    """Rule 2 (delta-spec §2). <3 points -> curve reported, never decision-bearing:
    decide at the top feasible scale + §13. =3 -> falsifiable curve (lever-arm sanity-checked)."""
    if n_feasible <= 0:
        return DecisionBasis.ESCALATE_MORE_DATA
    if n_feasible < 3:
        return DecisionBasis.FIXED_SCALE_PLUS_S13
    return DecisionBasis.SCALING_CURVE
```

- [ ] **Step 4: Run, expect PASS.** → PASS.
- [ ] **Step 5: Commit.** `git commit -am "feat(bakeoff): Rule 2 decision_basis (curve vs fixed-scale+§13)"`

### Task 4: Worst-case city aggregator + #21 binding-city power gate

**Files:** Create `src/cfm/eval/city_aggregate.py`; Test `tests/eval/test_city_aggregate.py`.
**Gate 2:** read `src/cfm/eval/feature_resolution.py` and confirm `single_region_floor_gap(*, n_reference_features)` returns `C/√n` (the per-city floor); use it, do not re-derive.

- [ ] **Step 1: Write the failing test (the T5 teeth).**
```python
from __future__ import annotations
import math
import pytest
from cfm.eval.city_aggregate import PerCityKS, worst_case_city, binding_city_verdict

def test_worst_case_binds_on_worst_city():
    cities = [PerCityKS("glasgow", 0.10, 5000), PerCityKS("munich", 0.42, 5000)]
    assert worst_case_city(cities).city == "munich"  # decision binds on the WORST, not the average

def test_pooling_is_not_silently_reintroduced():
    # The aggregate of [0.1, 0.5] must be the worst (0.5), NEVER a cell-count-weighted blend.
    cities = [PerCityKS("a", 0.1, 100_000), PerCityKS("b", 0.5, 100)]
    assert worst_case_city(cities).ks == 0.5  # b's tiny n must not let a's mass dominate
    # structural guard: no pooled-concatenation helper exists in the module
    import cfm.eval.city_aggregate as agg
    assert not any("pool" in name.lower() for name in dir(agg))

def test_underpowered_binding_city_is_demoted():
    # munich is worst but its winner-vs-runner-up gap < its own C/sqrt(n) floor -> demote to glasgow.
    per_backbone = {
        "AR":   [PerCityKS("munich", 0.42, 156), PerCityKS("glasgow", 0.30, 5000)],
        "diff": [PerCityKS("munich", 0.425, 156), PerCityKS("glasgow", 0.20, 5000)],
    }
    v = binding_city_verdict(per_backbone)
    assert "munich" in v.demoted_from        # munich gap 0.005 << its floor -> under-powered
    assert v.binding_city == "glasgow"       # decision falls to the next-worst RESOLVED city
    assert v.winner == "diff"                # glasgow: 0.20 < 0.30 -> diff wins where it's weakest
```

- [ ] **Step 2: Run, expect FAIL.** → FAIL.

- [ ] **Step 3: Implement.**
```python
# src/cfm/eval/city_aggregate.py
"""Per-city worst-case aggregation + the #21 binding-city power gate (delta-spec §4).

Generalization is a worst-case property: the decision binds on the WORST held-out city,
NEVER a mean and NEVER a pooled (cell-count-weighted) reference. munich is INCLUDED (KS has
no null to saturate); the #21 gate only DEMOTES a binding city whose winner-vs-runner-up gap
is below that city's OWN C/√n resolution floor (under-powered, cannot decide).
"""
from __future__ import annotations
from dataclasses import dataclass
from cfm.eval.feature_resolution import single_region_floor_gap

@dataclass(frozen=True)
class PerCityKS:
    city: str
    ks: float
    n_features: int  # reference feature count for this city's binding metric

@dataclass(frozen=True)
class BindingVerdict:
    binding_city: str
    winner: str
    runner_up: str
    gap: float
    city_floor: float
    demoted_from: tuple[str, ...]

def worst_case_city(per_city: list[PerCityKS]) -> PerCityKS:
    if not per_city:
        raise ValueError("no per-city KS supplied")
    return max(per_city, key=lambda c: c.ks)

def binding_city_verdict(per_backbone_per_city: dict[str, list[PerCityKS]]) -> BindingVerdict:
    """Worst-case decision with the #21 power gate. Cities are considered worst-first; a city
    whose winner-vs-runner-up KS gap < its own C/√n floor is DEMOTED (under-powered); the
    decision uses the first city that is both binding and resolved."""
    backbones = list(per_backbone_per_city)
    cities = [c.city for c in per_backbone_per_city[backbones[0]]]
    # mean KS per city across backbones, to order cities worst-first
    def city_mean(city: str) -> float:
        return sum(_ks(per_backbone_per_city[b], city) for b in backbones) / len(backbones)
    demoted: list[str] = []
    for city in sorted(cities, key=city_mean, reverse=True):
        ranked = sorted(backbones, key=lambda b: _ks(per_backbone_per_city[b], city))
        winner, runner_up = ranked[0], ranked[1]
        gap = _ks(per_backbone_per_city[runner_up], city) - _ks(per_backbone_per_city[winner], city)
        floor = single_region_floor_gap(n_reference_features=_n(per_backbone_per_city[winner], city))
        if gap > floor:
            return BindingVerdict(city, winner, runner_up, gap, floor, tuple(demoted))
        demoted.append(city)
    raise ValueError(f"no resolved binding city; all under-powered: {demoted}")

def _ks(per_city: list[PerCityKS], city: str) -> float:
    return next(c.ks for c in per_city if c.city == city)

def _n(per_city: list[PerCityKS], city: str) -> int:
    return next(c.n_features for c in per_city if c.city == city)
```

- [ ] **Step 4: Run, expect PASS.** → PASS.
- [ ] **Step 5: Commit.** `git add ... && git commit -m "feat(bakeoff): worst-case city aggregate + #21 binding-city power gate"`

### Task 5: Conditioning-discrimination gate (the §4 gate that validates the worst-case rule)

**Files:** Create `src/cfm/eval/conditioning_gate.py`; Test `tests/eval/test_conditioning_gate.py`.

- [ ] **Step 1: Write the failing test.**
```python
from __future__ import annotations
from cfm.eval.conditioning_gate import conditioning_discrimination_gate

def test_gate_passes_when_same_stratum_tiles_share_distributions():
    # same-macro-stratum tiles across cities have near-identical feature distributions
    # (KS within tolerance) -> conditioning explains per-city variation -> worst-case is valid.
    r = conditioning_discrimination_gate({"glasgow_v_munich": 0.04, "krakow_v_munich": 0.05}, tolerance=0.08)
    assert r.passes is True

def test_gate_fails_on_residual_city_style_and_signals_reopen():
    # same-stratum tiles DIFFER across cities -> residual un-conditioned city-style ->
    # per-city miss is ambiguous -> T5 must reopen.
    r = conditioning_discrimination_gate({"glasgow_v_munich": 0.22}, tolerance=0.08)
    assert r.passes is False
    assert "REOPEN" in r.reason.upper()
```

- [ ] **Step 2: Run, expect FAIL.** → FAIL.

- [ ] **Step 3: Implement.**
```python
# src/cfm/eval/conditioning_gate.py
"""The §4 conditioning-discrimination gate (delta-spec T5).

Worst-case (or mean) per-city scoring is coherent IFF per-city KS tracks macro-plan
differences, not residual un-conditioned city-style. Operationalized on the Task-1
diagnostic: for tiles at the SAME macro-stratum, the cross-city feature-distribution KS
must sit within tolerance (the per-city KS noise floor). If it exceeds tolerance, a per-city
miss is ambiguous ("wasn't told the city") and T5 REOPENS before any scored run.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class GateResult:
    passes: bool
    reason: str

def conditioning_discrimination_gate(
    same_stratum_cross_city_ks: dict[str, float], *, tolerance: float,
) -> GateResult:
    if not same_stratum_cross_city_ks:
        return GateResult(False, "no same-stratum cross-city KS supplied; cannot discharge gate (fail-closed)")
    worst = max(same_stratum_cross_city_ks.values())
    if worst <= tolerance:
        return GateResult(True, f"same-stratum cross-city KS {worst:.3f} <= {tolerance}: conditioning explains per-city variation")
    return GateResult(False, f"same-stratum cross-city KS {worst:.3f} > {tolerance}: residual city-style — T5 REOPENS before any scored run")
```

- [ ] **Step 4: Run, expect PASS.** → PASS.
- [ ] **Step 5: Commit.** `git add ... && git commit -m "feat(bakeoff): §4 conditioning-discrimination gate (validates worst-case bar)"`

---

## PHASE C — Net-new builds (local code + synthetic fixtures)

### Task 6: T5 — per-city KS wired into the decision axis (worst-case aggregate)

**Gate 2 (read first):** `src/cfm/eval/realism.py` (`ks_distance`, `feature_samples`, `FeatureMetric`), `src/cfm/eval/feature_resolution.py` (winner-vs-runner-up), `src/cfm/eval/curve.py` (the y-value it consumes), `src/cfm/eval/geometry.py` (per-region holdout read). Confirm current signatures before editing.

**Files:** Create `src/cfm/eval/multiregion_realism.py` (the per-city driver — keep `realism.py` a leaf); Modify `curve.py` / `feature_resolution.py` call sites to consume the worst-case aggregate; Test `tests/eval/test_realism_multiregion.py`.

> **[REC-2 from Phase-B batched review — the dict→list composition seam, with a regression-lock test].** `per_city_ks` returns `{city: PerCityKS}` (a **dict**) but `worst_case_city` takes a **`list[PerCityKS]`**. `decision_ks` MUST bridge it explicitly — `worst_case_city(list(per_city.values())).ks` — and that adaptation lives ONCE inside `multiregion_realism.py`, never leaked to another caller. **Why a test and not a comment:** passing the dict directly does NOT fail loudly — `max(dict, key=lambda c: c.ks)` iterates the dict's KEYS (strings) → `AttributeError` (`'str' has no 'ks'`) at the wrong layer. **Add a regression-lock test** that `decision_ks` returns the WORST city's KS from a multi-city dict (so a dict-passed-as-list bug would surface here, not in Phase D). Forward note for Task 12: when it assembles `{backbone: list(per_city_ks(...).values())}` for `binding_city_verdict`, every backbone must cover the same cities or the Phase-B I1 guard raises a clear `ValueError`.

- [ ] **Step 1: Write the failing test (teeth).**
```python
from __future__ import annotations
from cfm.eval.multiregion_realism import per_city_ks, decision_ks
from cfm.eval.realism import FeatureMetric

def test_per_city_ks_is_computed_against_each_city_separately_not_pooled():
    # generated[city] vs real[city], per city — never a concatenated reference.
    generated = {"glasgow": [10.0, 11.0, 12.0], "munich": [50.0, 51.0]}
    real =      {"glasgow": [10.0, 11.0, 12.0], "munich": [10.0, 11.0]}
    pc = per_city_ks(generated, real, metric=FeatureMetric.BUILDING_AREA)
    assert pc["glasgow"].ks == 0.0          # identical -> 0
    assert pc["munich"].ks > 0.5            # far -> large, NOT diluted by glasgow's match

def test_decision_ks_is_worst_case_not_mean():
    generated = {"a": [1.0], "b": [1.0]}
    real = {"a": [1.0], "b": [100.0]}
    # worst city dominates; a mean would halve it
    assert decision_ks(generated, real, metric=FeatureMetric.BUILDING_AREA) == 1.0
```

- [ ] **Step 2: Run, expect FAIL.** → FAIL.
- [ ] **Step 3: Implement** `multiregion_realism.py`: `per_city_ks(generated_by_city, real_by_city, *, metric)` calling `realism.ks_distance` **once per city** and returning `{city: PerCityKS(...)}` (n_features = len(real[city])); `decision_ks(...)` = `worst_case_city(...).ks` (delegates to `city_aggregate`). Wire `curve.py`'s per-scale y-value to `decision_ks` (the worst-case aggregate per the §2 composition note). **Do NOT add any pooled-concatenation path.**
- [ ] **Step 4: Run, expect PASS.** → PASS. Also run the full eval suite (`uv run pytest tests/eval -q`) — no regressions.
- [ ] **Step 5: Commit.** `git commit -am "feat(bakeoff): T5 per-city KS + worst-case decision axis (no pooling)"`

### Task 7: T6 — multi-region train build (per-city manifests + datamodule union)

> **[PRECONDITION from Task-1 review — I1, do NOT miss].** Task 7's train-city path MUST treat a **non-held-out city as empty-holdout (not an error)**. The vestigial tile-level `build_shards._holdout_ids` routes through `holdout_manifest_for_region`, which **raises `ValueError` on an EU train city** (a train city is neither `singapore` nor one of the 4 EU held-out cities). This is **correct until the train loop reaches it** — spec §5 makes whole-city exclusion primary and `_holdout_ids` a vestigial backstop. **Task 7 handles this AT THE LOOP** (a train city has no tile-level holdout → empty set), **NOT by changing `_holdout_ids`** or the region-aware helper (changing them would break the Task-1 fail-closed guarantee + the SG fixture). When the multi-region driver iterates `train_cities(...)`, it must NOT call `_holdout_ids` for a train city expecting a manifest entry. (Surfaced + verified clean in the Task-1 code-quality review; the raise is reached, unswallowed, and right for the boundary.)

**Gate 2 (read first):** `src/cfm/data/training/build_shards.py` (all `(release, region)` fns), `datamodule.py:236` (the single-region load), the G4 roll-up `reports/2026-06-05-phase-2-g4-corpus-dod.yaml` (the authoritative shipped-corpus city list).

**Files:** Modify `build_shards.py` (add a multi-region driver + a `train_cities()` reader); Modify `datamodule.py` (union over train cities); Test `tests/data/training/test_build_shards_multiregion.py` (synthetic multi-CRS fixtures — the EU corpus is Leonardo-only).

- [ ] **Step 1: Write the failing test (teeth).**
```python
def test_train_cities_excludes_the_4_held_out_by_construction(tmp_path):
    # train-city source = G4 roll-up MINUS held-out-4; the 4 must be absent from the build list.
    from cfm.data.training.build_shards import train_cities
    cities = train_cities(release="2026-04-15.0", g4_rollup=_fake_rollup(), holdout_manifest=_fake_holdout())
    for held in ("glasgow", "eisenhuttenstadt", "munich", "krakow"):
        assert held not in cities          # PRIMARY exclusion, in the build (not just leak-audited)

def test_datamodule_union_loads_multiple_cities(tmp_path):
    # union over 2 synthetic cities yields cells from BOTH
    ...  # build 2 per-city manifests, union, assert example provenance spans both regions

def test_non_z32_city_tokens_load_without_crs_leakage(tmp_path):
    # synthetic city tagged EPSG:25833 loads through the union; tokens are int ids, no CRS field.
    ...  # assert loaded examples are pure token ints; no CRS attribute anywhere on the example
```

- [ ] **Step 2: Run, expect FAIL.** → FAIL.
- [ ] **Step 3: Implement:** `train_cities(release, *, g4_rollup, holdout_manifest)` = roll-up cities − held-out cities (region-level whole-city exclusion, PRIMARY); a multi-region driver that calls the existing single-region `build_training_shards` per train city (per-city manifests, schema 1.0 unchanged); datamodule unions the per-city manifests over `train_cities(...)`. Keep `_holdout_ids` as the vestigial backstop. **Do not bump the training-manifest schema.**
- [ ] **Step 4: Run, expect PASS.** → PASS + full suite no-regress.
- [ ] **Step 5: Commit.** `git commit -am "feat(bakeoff): T6 multi-region train build (per-city manifests + datamodule union; whole-city exclusion)"`

---

## PHASE D — Leonardo execution (GATED on Umar's allocation word; verified-end-state; the fork resolves here)

> These tasks RUN on Leonardo and are gated. Their downstream parameters (scales, decision basis) are the **outputs** of Phase-B rule functions applied to Task-9's measurements — never hardcoded. Bite-sized run-steps are finalized at dispatch (they depend on measured `r`).

### Task 8: Build EU multi-region train shards on Leonardo (CPU)
- [ ] Run the Task-7 driver over `train_cities(...)` on Leonardo. **Verified-end-state:** count built city dirs == `len(train_cities)`; sum `n_training_tiles` across per-city manifests; assert no held-out city dir was built. Never trust the job exit code alone.

### Task 9: Task-1 diagnostic (GPU) — measures; decides nothing
- [ ] Run the diagnostic (existing `bakeoff_diagnostic.sbatch`, re-pointed to EU): rule out truncation → measure **`r` with a CI** → emergence floor → per-scale eval-cost.
- [ ] **Measure the two gate inputs.** **(i) is a DATA property, model-INDEPENDENT:** same-macro-stratum cross-city feature-KS computed from **REAL held-out tiles** (real building-area/road-length distributions of same-stratum tiles, city-vs-city) — **never from any pilot model's generations.** It is knowable before any model exists (that is *why* the gate fires pre-scored-run); computing it from model output would conflate "conditioning can't distinguish cities" (what the gate tests) with "the pilot model is bad" (the exact ambiguity the gate exists to kill). It may be measured as soon as the EU held-out tiles are accessible (after Task 8), independent of any generation. **(ii)** per-city winner-vs-runner-up resolvability at the pilot scale (KS-power adequacy, munich included) — this one IS model-facing (pilot generations), and is separate from (i).
- **[FIRST-MODEL EXPECTATION — munich power floor; recorded from Task-4 review, do NOT pre-empt].** munich is the SMALLEST held-out city (n=156 features) → its #21 floor is the HIGHEST: `single_region_floor_gap(156) = 1.358/√156 ≈ 0.1087` (vs glasgow ≈ 0.0192 at n=5000). So at first-model munich is the **most likely city to be demoted-for-under-power** by `binding_city_verdict` (Task 4), and the **munich→manchester swap reserve** (parked Phase-3) has a real chance of firing here on **POWER grounds** — independent of, and earlier than, the #21 coherence-saturation reason. This is the gate **working as designed, not a defect**; expect it when (ii) is measured. Don't pre-empt the swap — just don't be surprised when munich demotes.
- [ ] **HALT-GATE — conditioning-discrimination:** feed (i) to `conditioning_discrimination_gate`. If it FAILS → **HALT; T5 reopens** (the worst-case bar is invalid) — do not proceed to any scored run; report to reviewer.
- [ ] **HALT-GATE — ∅-ladder:** compute `feasible_ladder_conservative(r_ci_high)`. If `escalate_more_data` → **HALT; escalate to more-data (C)** — no scored runs are feasible on-frontier.
- [ ] **Verified-end-state:** persist the measured `r` (+CI), the gate verdicts, and the ladder to a `reports/` YAML; re-read it before proceeding.

### Task 10: Resolve the fork (deterministic, off the measurement)
- [ ] `ladder = feasible_ladder_conservative(r_ci_high, epoch_factor=E)`; `basis = decision_basis(len(ladder.feasible))`. **Persist BOTH** `ladder.feasible` and `basis` to a `reports/` YAML.
- [ ] Generate the per-scale run configs **from `ladder.feasible`** — assert the configured scale set **== `ladder.feasible`** (anti-hardcoding teeth on the *scales* output).
- [ ] **Close the loop on the OTHER fork output too:** the persisted `basis` is the *only* decision path Task 12 may take — Task 12 asserts its path **== persisted `basis`**. (Right scales + wrong rule — e.g., fitting a curve on 2 feasible points — must be impossible. Both fork outputs are enforced, not just the scale set.)

### Task 10.5: `mamba-ssm` verify-before-lock — HALT-GATE, BEFORE any scored GPU-h
- [ ] Verify `mamba-ssm` (+ `causal-conv1d`, `triton`) **import + run a fwd/bwd** on Leonardo under the **exact locked torch stack** (torch 2.5.1+cu121). **HALT-GATE:** if it forces *any* torch/CUDA change → **re-lock-all** (every backbone, incl. the already-run transformer-AR, re-runs under the new lock — a mid-bake-off version change breaks curve comparability). Do **not** commit any scored-run GPU-h until this passes. (Baseline §10; elevated here from a Task-11 clause to its own gate so a kernel/stack incompatibility surfaces *before* GPU-h is spent on the transformer arm, not mid-bake-off.) **Verified-end-state:** the fwd/bwd output + the resolved lock manifest, re-read.

### Task 11: Scored runs at the determined scales (GPU; gated)
- [ ] For each scale in `ladder.feasible` × {transformer-AR, mamba-hybrid, discrete-diffusion}: run under the locked recipe, identical `E`, across-job `$WORK` resume. (mamba-ssm already lock-verified at Task 10.5.) Verified-end-state per run (checkpoint sha + eval artifact).

### Task 12: Decision + report

> **[PRECONDITION from Task-6 review — input-completeness guard for the worst-case bar].** `per_city_ks`/`decision_ks` are safe-by-construction GIVEN a complete `real_by_city` (they iterate the real/held-out keys, so the silent-drop that hit `binding_city_verdict` cannot happen INSIDE them). But the safety rests on the CALLER passing a complete `real_by_city`: if Task 12 builds it short a held-out city (its tiles failed to load, an upstream filter), the worst-case bar silently binds on worst-of-three and misses the fourth. **When the worst-case decision is computed, assert `set(real_by_city.keys()) == {"eisenhuttenstadt", "glasgow", "krakow", "munich"}` (the frozen 4-city held-out set) and FAIL LOUD if any held-out city is missing** — the analog of the Phase-A manifest-drift guard. The silent-drop can't happen in the function; the input's completeness is the caller's responsibility, guarded here.

- [ ] **Assert the path == the persisted `basis` from Task 10** (the loop-closing teeth — the decision rule is not re-chosen here, only executed). Then: if `basis == SCALING_CURVE`: fit `curve.py`, **report the extrapolation distance alongside the CI verdict**, lever-arm sanity-checked before it's decision-bearing. If `FIXED_SCALE_PLUS_S13`: decide at the top feasible scale via `binding_city_verdict`; on no resolved separation → §13 `transformer-AR`. (`ESCALATE_MORE_DATA` cannot reach here — it HALTed at Task 9's ∅-ladder gate.)
- [ ] Write the `reports/` summary (config + commit + data-snapshot) and the PRD §6/§10 update (experiments win). **Verified-end-state.** Merge to `main` only on Umar's word, `--no-ff`, suite green.

---

## Self-Review

**Spec coverage:** T1/T2 → Task 1 (re-point) + the KS-only axis is enforced by Task 6 (no coherence in the decision path). T3 → Tasks 2,3 (rules), 9,10 (fork). T4 → Task 1 (rebase + 4-step). T5 → Tasks 4,5,6 (+ the §4 gate as a Task-9 HALT). T6 → Task 7,8. T7 (Phase-3 parking, comparability bookkeeping, budget) → carried as baseline-unchanged + Task 10.5 mamba-lock (own gate) + Task 12 report; **note:** coherence/(b)/(c)-swap are explicitly NOT in this plan (Phase-3) — correct per the ledger. The three Umar-calls live at Task 12 / Phase-D gating, not pre-resolved.

**Placeholder scan:** Phase A–C steps carry complete test code + complete pure-function impl; existing-module edits carry complete teeth + a Gate-2 read step (deliberate, per the verify-at-dispatch rule). Phase D run-steps are intentionally parameterized (they depend on measured `r`) — flagged as such, not silent TODOs.

**Type consistency:** `PerCityKS`/`BindingVerdict`/`LadderDecision`/`DecisionBasis`/`GateResult` are used identically across Tasks 4–6 and 9–12; `feasible_ladder`/`decision_basis`/`worst_case_city`/`binding_city_verdict`/`conditioning_discrimination_gate` signatures match between definition and call sites.

**Gap to watch (recorded, not a silent cap):** the conditioning-discrimination gate's `tolerance` is set to the per-city KS noise floor at dispatch (Task 9) — its derivation is a Task-9 sub-step, not pre-baked here, because it depends on the measured per-city `n`.

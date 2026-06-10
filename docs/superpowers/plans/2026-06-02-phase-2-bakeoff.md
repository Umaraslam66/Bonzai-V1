# Phase-2 Architecture Bake-off Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Branch discipline (every implementer dispatch):** work on `phase-2-bakeoff`; commit task-by-task; **NO new branches, NO push, NO PR** ([[feedback_subagent_branch_pattern]]). All commit messages end with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` footer (omitted from the templates below for brevity). Run `uv run ruff format && uv run ruff check && uv run pytest -q` before each commit; if `pytest` falls through to system Python 3.9, run `uv sync --extra dev` first ([[feedback_uv_sync_dev_extras]]).

**Goal:** Decide which sequence-model backbone (transformer-AR / mamba-hybrid / discrete-diffusion) to scale up for production, by fitting per-backbone scaling curves of geometry-fidelity vs measured compute over {30M, 100M, 300M, 1B} and extrapolating to the production compute budget.

**Architecture:** A shared, identity-locked scaffold (embedding + value-bearing conditioning prefix + sub-F vocab head 1508 + training harness + eval) with a swappable backbone. Three backbones differ only in their sequence-mixing layers (+ diffusion's quarantined loss/generation/mask). Decision axis = a per-feature geometry-realism KS distance on decoded output (architecture-agnostic; NLL is an AR-family-only diagnostic). Eval (autoregressive generation) is the binding cost; training is ~7.6× under the PRD envelope.

**Tech Stack:** PyTorch 2.5.1+cu121, Lightning 2.6.5, pydantic 2.13.4 (the locked comparability stack); `mamba-ssm`/`causal-conv1d` (new, verify-before-lock); `shapely` (already a dep) for geometry; the per-feature KS statistic is hand-rolled (no scipy — see Task 2 Step 8); Slurm `boost_usr_prod` / `AIFAC_P02_222` on 4×A100 nodes.

**Spec:** `docs/superpowers/specs/2026-06-02-phase-2-bakeoff-design.md` (this plan implements its §14 task matrix; task→spec mapping noted per task).

---

## The hard gate (read before executing)

Execution has one **non-negotiable ordering gate**, spanning Tasks 4→5→(6+):

1. The eval-measurement machinery (Tasks 1–3) must exist before the diagnostic can measure anything.
2. **The diagnostic (Task 4) must COMPLETE** — emergence floor, geometry-r, recipe locked — **before** the mamba-ssm lock decision.
3. **The mamba-ssm verify-before-lock (Task 5) must be SETTLED** (clean OR re-lock-all executed) **before ANY backbone build (Tasks 6–8) or scored run (Task 12)**. A mid-bake-off version change breaks comparability → all runs re-run under the new lock.

Tasks 1–3 (eval machinery) and Tasks 6 (conditioning) may be built in parallel with each other, but a SCORED run never starts before Task 5 settles.

---

## File structure (decomposition map)

**New files:**
- `src/cfm/models/backbone.py` — the `Backbone` protocol/ABC + a `build_backbone(name, cfg)` factory; the abstraction all three backbones implement.
- `src/cfm/models/mamba_hybrid.py` — `MambaHybrid` backbone (Jamba-style ~7:1 interleave via `mamba-ssm`).
- `src/cfm/models/discrete_diffusion.py` — `DiscreteDiffusion` backbone (absorbing-state/MDLM-family); + `src/cfm/models/diffusion/{loss.py,generate.py,mask.py}` for the three quarantined divergences.
- `src/cfm/eval/realism.py` — per-feature geometry-realism KS distance (building-area, road-length), the new scipy-based machinery.
- `src/cfm/eval/feature_resolution.py` — per-feature resolution re-derivation + worst-resolved-feature selection + winner-vs-runner-up pairwise seam feeding.
- `src/cfm/eval/emergence.py` — building-class-token instrumentation + the holdout-density-tied emergence floor (one source, shared with the §2 guard).
- `src/cfm/eval/curve.py` — bootstrap-CI scaling-curve fit + extrapolation + the §13 structural check + the pre-committed tie-break.
- `src/cfm/training/deviation_log.py` — the comparability-deviation log + the fails-to-train detector.
- `src/cfm/training/resume.py` — across-job `$WORK` checkpoint discovery + auto-resume.
- `scripts/bakeoff_diagnostic.sbatch`, `scripts/bakeoff_run.sbatch`, `scripts/verify_mamba_lock.sbatch` — Slurm entrypoints.
- `configs/experiments/bakeoff-*.yaml` — pydantic-validated per-run configs.

**Modified files:**
- `src/cfm/training/config.py` — add `backbone: str`, `grad_accum: int`, diffusion/Mamba sub-configs to `ScaffoldConfig`.
- `src/cfm/training/lit_module.py` — `ScaffoldLit` dispatches to `build_backbone(...)`; loss/generation route per backbone.
- `src/cfm/training/env_lock.py` — extend `_EXPECTED` with `mamba-ssm`/`causal-conv1d` (after Task 5 verification).
- `src/cfm/data/training/conditioning.py` — `conditioning_field_to_id`/`build_conditioning_prefix` → value-bearing (Task 6).
- `src/cfm/inference/generate.py` — route AR vs diffusion generation; building-token instrumentation hook.
- `src/cfm/eval/slice_metrics.py` — `slice_eval` gains the §2 emergence floor-score guard + the lexicographic verdict.
- `scripts/train_scaffold.py` — CLI gains `--backbone`, `--grad-accum`; `run_short` routes per backbone.

---

## Task 1: Emergence instrumentation + holdout building-density floor
*(spec §14 Task 1 prep; §5; §2 guard one-source)*

**Why first:** the diagnostic (Task 4) must (a) rule out eval-truncation by checking whether generated sequences contain building-class tokens *at all*, and (b) judge "buildings emerged" against a non-vacuous, holdout-density-tied threshold — NOT `n_polygons>0`. Both are small, testable, and are the single source the §2 guard (Task 2) reuses.

**Files:**
- Create: `src/cfm/eval/emergence.py`
- Test: `tests/eval/test_emergence.py`

- [ ] **Step 1: Write the failing test for building-token detection**

**CONTRACT CORRECTION (verified 2026-06-02 against the real sub-F vocab — §15 regime-transfer catch).** The exploration agent reported a `B_` tag prefix; that is the *raw Phase-0* `vocab_phase1.yaml` scheme, NOT the sealed sub-F vocab. `vocab_tag_to_id()` has **zero `B_` tags** — its building tags are `<key=value>` form with **L1 key `building`** (77 tags incl. the unknown-family `<unknown_building>`), exactly parallel to `ROAD_L1_KEY="highway"`. The authority is `semantic_tag_to_l1_key(tag) == "building"` (verified: 77 ids, raises on none of the 686 tags, zero road overlap). Detect by the L1 key, never a string prefix.

```python
# tests/eval/test_emergence.py
from cfm.data.sub_f.vocab import vocab_tag_to_id, semantic_tag_to_l1_key
from cfm.eval.emergence import sequence_has_building_tokens, building_token_ids, BUILDING_L1_KEY

def test_building_token_ids_are_the_building_l1_key_tags():
    vocab = vocab_tag_to_id()
    expected = {i for tag, i in vocab.items() if semantic_tag_to_l1_key(tag) == BUILDING_L1_KEY}
    assert building_token_ids() == expected
    assert len(expected) == 77  # verified count against the sealed sub-F vocab

def test_sequence_with_a_building_token_is_detected():
    a_building_id = min(building_token_ids())
    assert sequence_has_building_tokens([1, 2, a_building_id, 3]) is True

def test_sequence_without_building_tokens_is_not_detected():
    non_building = sorted(set(range(686)) - building_token_ids())[:5]
    assert sequence_has_building_tokens(non_building) is False
```

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/eval/test_emergence.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement detection**

```python
# src/cfm/eval/emergence.py
from __future__ import annotations

from functools import cache

from cfm.data.sub_f.vocab import semantic_tag_to_l1_key, vocab_tag_to_id

# Authority for "is this a building feature token": the BP1 L1 key, parallel to
# ROAD_L1_KEY="highway". Covers building=* / building=<value> AND the unknown-family
# <unknown_building>. Verified against the sealed sub-F vocab (77 ids). NOT a string prefix.
BUILDING_L1_KEY = "building"


@cache
def building_token_ids() -> frozenset[int]:
    """Token ids whose sub-F tag resolves to the ``building`` BP1 L1 key."""
    return frozenset(
        i for tag, i in vocab_tag_to_id().items() if semantic_tag_to_l1_key(tag) == BUILDING_L1_KEY
    )


def sequence_has_building_tokens(tokens: list[int]) -> bool:
    """True iff the generated token sequence contains ANY building-class token.

    The truncation discriminator (§5 stage 1): no building tokens at all means the
    model never tried to emit buildings; building tokens present but n_polygons==0
    means they did not CLOSE into polygons (a different cause than truncation).
    """
    ids = building_token_ids()
    return any(t in ids for t in tokens)
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/eval/test_emergence.py -v` → PASS.

- [ ] **Step 5: Write the failing test for the holdout-density-tied emergence floor**

The floor is "the model reliably produces buildings where real data has them," expressed as a per-cell polygon-count rate tied to the holdout's real building density (NOT an absolute count). It is a single function reused by the §2 guard.

```python
# append to tests/eval/test_emergence.py
from cfm.eval.emergence import emergence_floor_polygons_per_cell, buildings_emerged

def test_floor_is_a_fraction_of_holdout_density_not_an_absolute():
    # holdout has ~4.0 polygons/active-cell -> floor = frac * that, frac in (0,1]
    floor = emergence_floor_polygons_per_cell(holdout_polys_per_cell=4.0, frac=0.25)
    assert floor == 1.0

def test_buildings_emerged_requires_meeting_the_density_floor_not_one_stray():
    # 1 polygon across 100 cells = 0.01/cell -> below a 1.0 floor -> NOT emerged
    assert buildings_emerged(n_polygons=1, n_cells=100, floor_per_cell=1.0) is False
    # 120 polygons across 100 cells = 1.2/cell -> at/above floor -> emerged
    assert buildings_emerged(n_polygons=120, n_cells=100, floor_per_cell=1.0) is True
```

- [ ] **Step 6: Run to verify fail** — FAIL (functions missing).

- [ ] **Step 7: Implement the floor**

```python
# append to src/cfm/eval/emergence.py

# DECISION: emergence floor = frac of the holdout's real polygons-per-active-cell density.
# Tied to real data (one stray polygon != emergence), relative not absolute, so the same
# threshold meaning transfers across scales. frac is a recorded PI choice; default 0.25.
# Revisit if the Task-4 diagnostic shows it admits roads-only runs. Same source as the §2 guard.
EMERGENCE_FRAC_OF_HOLDOUT_DENSITY: float = 0.25


def emergence_floor_polygons_per_cell(*, holdout_polys_per_cell: float, frac: float = EMERGENCE_FRAC_OF_HOLDOUT_DENSITY) -> float:
    return frac * holdout_polys_per_cell


def buildings_emerged(*, n_polygons: int, n_cells: int, floor_per_cell: float) -> bool:
    if n_cells <= 0:
        return False
    return (n_polygons / n_cells) >= floor_per_cell
```

- [ ] **Step 8: Write the test that measures the holdout's real polygons-per-active-cell** (the one place the floor's `holdout_polys_per_cell` comes from), using the existing `decode_region_blocks` round-trip on holdout cells.

```python
# append to tests/eval/test_emergence.py
import pytest
from cfm.eval.emergence import holdout_polygons_per_active_cell

@pytest.mark.slow
def test_holdout_density_is_measured_from_real_roundtripped_geoms():
    # Real frozen holdout: round-trip real cells -> count polygons / active cells.
    density = holdout_polygons_per_active_cell(release="2026-04-15.0", region="singapore")
    assert density > 0.0  # dense urban Singapore has buildings
```

- [ ] **Step 9: Implement `holdout_polygons_per_active_cell`** reusing `cfm.eval.holdout.roundtrip.decode_region_blocks` and the holdout manifest loader; count decoded `Polygon`/`MultiPolygon` geoms over active cells. Mark the heavy path `@pytest.mark.slow`.

- [ ] **Step 10: Run + commit**

```bash
uv run pytest tests/eval/test_emergence.py -v
git add src/cfm/eval/emergence.py tests/eval/test_emergence.py
git commit -m "feat(bakeoff): emergence instrumentation + holdout-density-tied floor (one source)"
```

---

## Task 1.5: Building-ring promotion contract-fix (UNPLANNED — surfaced during execution)
*(6th catch: contract-not-read; §3 + §9)*

**Discovered at the Task-4 prerequisite (holdout density read 0.0 — a §6 anti-signal).** The sealed sub-F decoder returns building closed rings as `type:"LineString"` BY CONTRACT (`decoder.py:145-157`: closed-ring↔roundabout is ambiguous at decode time; consumer promotes). The scaffold's `slice_metrics` (and inherited Tasks 1/2) filtered `type=="Polygon"` and never promoted → `n_polygons` / right-angle / building-area / emergence-floor all read 0 on real data. This — not under-training — was the dominant cause of the probe's `n_polygons=0`.

**Fix (done):** `src/cfm/eval/geometry.py::promote_building_rings(blocks, geoms)` — the ONE promotion authority; promotes a feature to Polygon iff its block is a building feature-class (reuses Task-1 `building_token_ids`, §9 construction-identity) AND its ring is closed; roads incl. closed roundabouts stay LineString. Paired §9 guard (`test_geometry_promote.py`): building closed ring → Polygon (must-promote) **and** closed road ring → LineString (must-NOT-promote). Applied in `slice_metrics.slice_eval` + `geometry.holdout_polygons_per_active_cell` (moved from `emergence` to avoid a cycle); identity-locked (`slice_metrics.promote_building_rings is geometry.promote_building_rings`). Then re-measure holdout density → confirm **non-vacuous AND plausible** (a sane buildings-per-active-cell, not just >0) before the floor it feeds is trusted and before the diagnostic runs against it.

## Task 2: §2 emergence guard in slice_eval + lexicographic ranking + per-feature realism KS
*(spec §14 Task 4; §2, §7)*

**Files:**
- Create: `src/cfm/eval/realism.py`
- Modify: `src/cfm/eval/slice_metrics.py` (the `slice_eval` return dict + a verdict)
- Test: `tests/eval/test_realism.py`, `tests/eval/test_slice_metrics_emergence_guard.py`

- [ ] **Step 1: Write the failing test for the §2 floor-score guard**

The guard must turn `n_polygons < floor` into a FLOOR verdict, never a vacuous pass. The discrimination test: a run with `ogc_valid_rate=1.0` but `n_polygons` below the floor must be verdict `ROADS_ONLY` (floor), and the guard must still FIRE on a real defect in the *kept* set (a run that clears the floor but has genuinely invalid polygons scores low on validity, not vacuously high).

```python
# tests/eval/test_slice_metrics_emergence_guard.py
from cfm.eval.slice_metrics import emergence_verdict, EmergenceVerdict

def test_roads_only_run_is_floored_not_a_vacuous_pass():
    # ogc_valid_rate=1.0 but zero polygons across many cells -> ROADS_ONLY (floor), NOT a pass
    v = emergence_verdict(n_polygons=0, n_cells=110, floor_per_cell=1.0)
    assert v is EmergenceVerdict.ROADS_ONLY

def test_run_clearing_the_floor_is_scoreable():
    v = emergence_verdict(n_polygons=200, n_cells=110, floor_per_cell=1.0)
    assert v is EmergenceVerdict.SCOREABLE

def test_guard_distinguishes_regimes_floor_keys_on_density_not_validity():
    # Two runs with identical ogc_valid_rate=1.0 diverge only on emergence density.
    assert emergence_verdict(n_polygons=0, n_cells=110, floor_per_cell=1.0) is EmergenceVerdict.ROADS_ONLY
    assert emergence_verdict(n_polygons=300, n_cells=110, floor_per_cell=1.0) is EmergenceVerdict.SCOREABLE
```

- [ ] **Step 2: Run to verify fail** — FAIL (symbol missing).

- [ ] **Step 3: Implement `emergence_verdict` in `slice_metrics.py`** using `cfm.eval.emergence.buildings_emerged` (one source — import the function, do not reinline the threshold):

```python
# add to src/cfm/eval/slice_metrics.py
from enum import Enum
from cfm.eval.emergence import buildings_emerged

class EmergenceVerdict(Enum):
    SCOREABLE = "scoreable"
    ROADS_ONLY = "roads_only"  # below the density floor -> building metrics are FLOORED, not vacuous

def emergence_verdict(*, n_polygons: int, n_cells: int, floor_per_cell: float) -> EmergenceVerdict:
    return EmergenceVerdict.SCOREABLE if buildings_emerged(
        n_polygons=n_polygons, n_cells=n_cells, floor_per_cell=floor_per_cell
    ) else EmergenceVerdict.ROADS_ONLY
```

- [ ] **Step 4: Extend `slice_eval`'s return dict** with `emergence_verdict` + a `building_metrics_floored: bool` flag (when `ROADS_ONLY`, the building-geometry metrics are reported but explicitly marked floored so the curve never reads `ogc_valid=1.0` as a good score). Keep the existing keys; add the new ones. Run the existing `tests/eval/test_slice_metrics*.py` to confirm no regression.

- [ ] **Step 5: Run + commit the guard**

```bash
uv run pytest tests/eval/test_slice_metrics_emergence_guard.py tests/eval/test_slice_metrics.py -v
git add src/cfm/eval/slice_metrics.py tests/eval/test_slice_metrics_emergence_guard.py
git commit -m "feat(bakeoff): §2 emergence floor-score guard in slice_eval (one source with Task 1)"
```

- [ ] **Step 6: Write the failing test for the per-feature realism KS distance**

This is NEW machinery (no scipy in the tree today). The KS distance compares a generated feature distribution (e.g. building areas, road lengths) against the holdout `ReferenceDistribution.samples`, per `cell_density_bucket` stratum. Lower = more realistic.

> **ERRATUM 2026-06-10 (F4-C1d):** `feature_samples` consumers MUST apply `promote_building_rings` first — the decoder returns building rings as LineString by contract. This snippet predates that finding; the readiness plan Task 26 makes `feature_samples` promote internally.

```python
# tests/eval/test_realism.py
from cfm.eval.realism import feature_samples, ks_distance, FeatureMetric

def test_building_area_samples_extracted_from_polygon_geoms():
    geoms = [
        {"type": "Polygon", "coordinates": [[[0, 0], [0, 2], [2, 2], [2, 0], [0, 0]]]},  # area 4
        {"type": "LineString", "coordinates": [[0, 0], [3, 0]]},  # not a building
    ]
    areas = feature_samples(geoms, metric=FeatureMetric.BUILDING_AREA)
    assert areas == [4.0]

def test_road_length_samples_extracted_from_linestring_geoms():
    geoms = [{"type": "LineString", "coordinates": [[0, 0], [3, 0], [3, 4]]}]  # length 3+4=7
    assert feature_samples(geoms, metric=FeatureMetric.ROAD_LENGTH) == [7.0]

def test_ks_distance_is_zero_for_identical_distributions_and_grows_with_divergence():
    a = [1.0, 2.0, 3.0, 4.0]
    assert ks_distance(a, a) == 0.0
    assert ks_distance(a, [10.0, 20.0, 30.0, 40.0]) > ks_distance(a, [1.1, 2.1, 3.1, 4.1])
```

- [ ] **Step 7: Run to verify fail** — FAIL (module missing).

- [ ] **Step 8: Implement `realism.py`** with shapely area/length extraction and a **hand-rolled two-sample KS statistic** (`D = max_x |F_gen(x) - F_ref(x)|` via `bisect` over the empirical CDFs). **DECISION (refinement during execution): NO scipy dependency** — the existing codebase computes KS quantities without scipy (`holdout/sizing.py`'s `1.358·√(2/n)`), the statistic is ~10 lines, and adding scipy would be a new heavyweight dependency to install on Leonardo. Default-to-simplicity + codebase precedent. (Empty either sample → 1.0, maximally far.)

- [ ] **Step 9: (no dependency change)** — KS is hand-rolled, so `pyproject.toml`/`uv.lock` are untouched and there is no Leonardo install to manage. The tensorboard-out-of-lock precedent is moot here (nothing added to `env_lock._EXPECTED`).

- [ ] **Step 10: Run + commit**

```bash
uv run pytest tests/eval/test_realism.py -v
git add src/cfm/eval/realism.py tests/eval/test_realism.py
git commit -m "feat(bakeoff): per-feature geometry-realism KS distance (building-area, road-length)"
```

---

## Task 3: Per-feature resolution re-derivation + winner-vs-runner-up seam feeding
*(spec §14 Task 5; §8; §10.3)*

**Why:** the frozen `0.076`/`0.049` were derived for per-CELL density representativeness; the bake-off ranks on a per-FEATURE KS. The resolution must be re-derived in the per-feature unit on the REAL generated+holdout feature populations, checked against the worst-resolved feature, and the seam fired only on the winner-vs-runner-up pair.

**Files:**
- Create: `src/cfm/eval/feature_resolution.py`
- Test: `tests/eval/test_feature_resolution.py`

- [ ] **Step 1: Failing test — resolution re-derived in the per-feature unit, NOT inherited**

The existing `pipeline.py` uses `gap = 1.358 * sqrt(2/n)`. The per-feature resolution must use the *feature* count (n building areas, n road lengths), not the cell count, and must be computed on the actual populations — not the 0.076 constant.

```python
# tests/eval/test_feature_resolution.py
import math
from cfm.eval.feature_resolution import per_feature_resolved_gap, binding_resolution

def test_resolution_uses_feature_count_not_inherited_0076():
    # 400 features -> gap = 1.358*sqrt(2/400) ~= 0.096 ; NOT the cell-derived 0.076
    g = per_feature_resolved_gap(n_features=400)
    assert math.isclose(g, 1.358 * math.sqrt(2 / 400), rel_tol=1e-9)
    assert g != 0.076

def test_binding_resolution_is_the_worst_resolved_feature():
    # building-area has fewer samples (coarser/larger gap) than road-length -> it binds
    res = binding_resolution({"building_area_m2": 100, "road_length_m": 2000})
    assert res.binding_metric == "building_area_m2"
    assert res.binding_gap == per_feature_resolved_gap(n_features=100)
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement `feature_resolution.py`**

```python
# src/cfm/eval/feature_resolution.py
from __future__ import annotations

import math
from dataclasses import dataclass

_KS_C_ALPHA_05 = 1.358  # KS two-sample critical coefficient at alpha=0.05 (matches holdout/sizing.py)


def per_feature_resolved_gap(*, n_features: int) -> float:
    """The finest KS gap resolvable with n features (alpha=0.05). Same formula family as
    the eval-set sizing, but in the PER-FEATURE unit (building areas / road lengths), NOT
    per-cell. Re-derived on the real population, never the inherited per-cell 0.076 (§10.3)."""
    if n_features <= 0:
        return float("inf")
    return _KS_C_ALPHA_05 * math.sqrt(2.0 / n_features)


@dataclass(frozen=True)
class BindingResolution:
    binding_metric: str
    binding_gap: float
    per_metric_gap: dict[str, float]


def binding_resolution(n_features_by_metric: dict[str, int]) -> BindingResolution:
    """The WORST-resolved (coarsest-gap) feature distribution binds the seam — a pair
    resolvable on roads but not buildings must not be green-lit (§8)."""
    per_metric = {m: per_feature_resolved_gap(n_features=n) for m, n in n_features_by_metric.items()}
    binding_metric = max(per_metric, key=lambda m: per_metric[m])
    return BindingResolution(binding_metric, per_metric[binding_metric], per_metric)
```

- [ ] **Step 4: Failing test — the seam fires on winner-vs-runner-up only, in the per-feature unit**

```python
# append to tests/eval/test_feature_resolution.py
import pytest
from cfm.eval.feature_resolution import check_decision_resolvable, DecisionUnresolvable

def test_seam_fires_when_winner_runnerup_gap_below_binding_resolution():
    # ranked KS scores (lower=better): winner 0.20, runner-up 0.205 -> gap 0.005, tiny
    with pytest.raises(DecisionUnresolvable):
        check_decision_resolvable(ranked_scores=[0.20, 0.205, 0.40], binding_gap=0.05)

def test_seam_silent_when_winner_runnerup_gap_clears():
    check_decision_resolvable(ranked_scores=[0.20, 0.30, 0.40], binding_gap=0.05)  # no raise

def test_last_place_tie_does_not_fire_the_seam():
    # winner clearly separated; 2nd and 3rd are tied -> irrelevant to the decision, no raise
    check_decision_resolvable(ranked_scores=[0.20, 0.39, 0.395], binding_gap=0.05)
```

- [ ] **Step 5: Implement `check_decision_resolvable`** — compute the gap between the best (`ranked_scores[0]`) and second-best (`ranked_scores[1]`) only; raise `DecisionUnresolvable` if `< binding_gap`. Ignore all other pairs.

- [ ] **Step 6: 3-tier escalation, native to the per-feature unit (DECISION, locked with PI).** The bake-off's generated eval is NOT write-once (unlike the eval-set), so a too-fine gap has a cheap escalation before second-region: **generate more eval cells** (tightens `1.358·√(2/n)`). Escalation tiers, with the **fixed holdout-reference feature count** as the tier-2/3 floor (KS is two-sample; the reference can't be enlarged, so as generated→∞ the gap asymptotes to `1.358/√(n_ref)`):
  - `gap ≥ current binding gap` → **RESOLVED** (decide).
  - `floor ≤ gap < binding` → **GENERATE_MORE_CELLS**: compute the n_cells that resolves THIS gap **once** (`n_features = ⌈2·(1.358/gap)²⌉`, `n_cells = ⌈n_features/features_per_cell⌉`) — **not a loop**; and tier-2 generation **reuses the §4 locked eval content** (same conditioning/seeds/holdout — "more of the same," never a faster variant), or the tightened gap measures a different distribution than the one being ranked.
  - `gap < floor` → **SECOND_REGION** (fundamental — no number of generated cells beats the fixed reference). **Termination guard:** if the n_cells a tier-2 gap would need implies a gap already below the floor, skip generation and go straight to tier 3 (don't burn eval budget chasing an unreachable gap).
  - `check_decision_resolvable` owns the per-feature gate **natively** — it does NOT call the per-cell `assert_resolution_sufficient` (that compares against the per-cell 0.076 marker — the §8 unit-inheritance trap). `assert_resolution_sufficient` stays the frozen-SET representativeness seam, a different question.

- [ ] **Step 7: Run + commit**

```bash
uv run pytest tests/eval/test_feature_resolution.py -v
git add src/cfm/eval/feature_resolution.py tests/eval/test_feature_resolution.py
git commit -m "feat(bakeoff): per-feature resolution re-derivation + winner-vs-runner-up seam (§10.3)"
```

---

## Task 4: The diagnostic run — emergence floor, geometry-r, eval-cost, recipe lock  ⛔ HARD GATE
*(spec §14 Task 1; §3, §5)*

**This is an EMPIRICAL run, not a TDD task.** It uses the EXISTING transformer-AR (`MicroAR`) + Tasks 1–3 machinery. Staged in cost-to-rule-out order. Its outputs (emergence floor, geometry-r, per-scale eval node-h, the locked recipe) feed every later task. **Task 5 must not start until this completes.**

**Files:**
- Create: `scripts/bakeoff_diagnostic.sbatch`
- Create: `reports/phase-2-bakeoff/2026-XX-XX-diagnostic.md` (written by the run)

- [ ] **Step 1: Author `scripts/bakeoff_diagnostic.sbatch`** from the `scaffold_scaleup_probe.sbatch` template. Key differences: `--qos=boost_qos_lprod` (longer than the 30-min dbg cap, since this trains to r≈20–40); a **generous `--eval-max-new`** (so generation provably CAN reach building tokens — stage-1 truncation control); a 100M config (`--d-model 768 --n-layers 12 --n-heads 12`, ≈100M). Pre-build the manifest in the preamble (single process) before `srun --no-build`.

```bash
# scripts/bakeoff_diagnostic.sbatch (header mirrors scaffold_scaleup_probe.sbatch)
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=AIFAC_P02_222
#SBATCH --job-name=bakeoff-diagnostic-100m
#SBATCH --time=08:00:00
#SBATCH --nodes=1 --ntasks-per-node=4 --cpus-per-task=8 --gres=gpu:4 --mem=480G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7 cuda/12.2
source .venv/bin/activate
echo "git_sha=$(git rev-parse HEAD)"
python -c "from cfm.data.training.build_shards import build_training_shards as b; b('2026-04-15.0','singapore')"
# r-target: 100M * 20 = 2B tokens; tokens/step = batch*devices*max_len. Tune --max-steps to hit r~=20-40.
srun --kill-on-bad-exit=1 python -u scripts/train_scaffold.py \
  --devices 4 --no-build --no-compile --backbone transformer-ar \
  --d-model 768 --n-layers 12 --n-heads 12 --max-len 2048 --batch-size 4 \
  --max-steps 122000 --eval-cells 64 --eval-max-new 2048
```

- [ ] **Step 2: STAGE 1 — rule out truncation.** Open the SSH master socket; sync via git bundle; submit. After it completes, in the eval read **`sequence_has_building_tokens`** (Task 1) over generated cells. Decision branch:
  - **Building tokens present but `n_polygons` low** → not truncation; the model emits building tokens that don't close into polygons → proceed to stage-2 interpretation (training).
  - **No building tokens at all even with generous `--eval-max-new`** → either still under-trained or conditioning-blind; record and proceed; do NOT conclude "conditioning required" yet.

- [ ] **Step 3: STAGE 2 — read the loss trajectory + emergence vs step.** From the CSVLogger loss curve and periodic eval, report **where loss flattens** and **where `buildings_emerged` (Task 1, holdout-density floor) first goes true**. Apply the stopping rule: *train until loss flattens OR buildings emerge* — if neither by r≈40, that is the finding "geometry-r is high," NOT a reason to stop. Record:
  - `geometry_r_regime`: at the chosen r, is loss descending (r too low) or flat (r about right)?
  - `emergence_step` / `emergence_r`: where buildings cleared the density floor (or "not by r≈40").
  - `eval_node_h_per_cell_100M`: measured per-cell generation cost at 100M (for the §6 budget).

- [ ] **Step 4: STAGE 3 — conditioning (ONLY if buildings absent after stages 1–2 cleared).** If and only if buildings did not emerge despite cleared truncation and adequate training, this is the signal to build value-bearing conditioning (Task 6) and re-run. Otherwise, record the decision: **"value-bearing conditioning demoted to a §7 quality lever, not an emergence gate."**

- [ ] **Step 5: Lock the recipe + r.** From the diagnostic, set: the geometry-verified `r` (= `max(Chinchilla, emergence)`), the optimizer recipe (lr/schedule/warmup that trained cleanly), and the per-scale eval budget. Write these into `configs/experiments/bakeoff-base.yaml`.

- [ ] **Step 5b: Record the 100M→1B scale-transfer of the emergence floor as a STATED assumption (§15 regime-transfer caution).** The diagnostic measures emergence-r and the density floor at **100M**, but they are applied to the **300M and 1B** runs too. Write into the diagnostic report explicitly: *"the emergence floor and geometry-r are assumed scale-stable; measured at 100M, applied to all scales."* If there is any reason to expect emergence-r to differ by scale (e.g. the loss-flatten point shifts materially between the 100M and 300M points once those run), flag it as a known extrapolation and re-check the floor against the 300M run's `n_polygons` before the 1B run — never let the 100M→1B application be silent (it is exactly the "number measured in one regime, applied in another" meta-pattern).

- [ ] **Step 6: Write `reports/phase-2-bakeoff/2026-XX-XX-diagnostic.md`** (config + commit + data snapshot + the measured emergence floor, geometry-r regime, per-scale eval node-h, recipe). Commit:

```bash
git add scripts/bakeoff_diagnostic.sbatch configs/experiments/bakeoff-base.yaml reports/phase-2-bakeoff/
git commit -m "expt(bakeoff): task-1 diagnostic — emergence floor + geometry-r + eval-cost measured; recipe locked"
```

---

## Task 5: mamba-ssm verify-before-lock + env-lock extension  ⛔ HARD GATE
*(spec §14 Task 2; §10)*

**Settle this BEFORE any backbone build or scored run.** If `mamba-ssm` forces a torch/CUDA change, it is a re-lock-all event.

**Files:**
- Create: `scripts/verify_mamba_lock.sbatch`
- Modify: `src/cfm/training/env_lock.py`
- Test: `tests/training/test_env_lock_mamba.py`

- [ ] **Step 1: Author `scripts/verify_mamba_lock.sbatch`** — a single-GPU job (`boost_qos_dbg`, 30 min) that, under the EXACT locked stack, installs/imports `mamba-ssm` + `causal-conv1d` and runs a forward+backward through one Mamba layer:

```bash
# scripts/verify_mamba_lock.sbatch
#SBATCH --partition=boost_usr_prod --qos=boost_qos_dbg --account=AIFAC_P02_222
#SBATCH --job-name=verify-mamba-lock --time=00:30:00
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G
#SBATCH --output=logs/%x-%j.out --error=logs/%x-%j.err
set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7 cuda/12.2
source .venv/bin/activate
python - <<'PY'
import torch; print("torch", torch.__version__)               # MUST stay 2.5.1+cu121
import mamba_ssm, causal_conv1d; print("mamba_ssm", mamba_ssm.__version__)
from mamba_ssm import Mamba
m = Mamba(d_model=256).cuda()
x = torch.randn(2, 64, 256, device="cuda", requires_grad=True)
m(x).sum().backward()                                          # fwd+bwd under the locked stack
print("MAMBA_FWD_BWD_OK", torch.__version__)
PY
```

- [ ] **Step 2: Submit and inspect.** Two outcomes:
  - **Clean:** `MAMBA_FWD_BWD_OK 2.5.1+cu121` printed, torch unchanged → proceed to Step 3, pin the verified `mamba-ssm`/`causal-conv1d` versions.
  - **Forces a torch/CUDA change:** STOP. This is a **re-lock-all event** — record the new pinned stack, update `LOCKED_TORCH` etc., and note that the Task-4 diagnostic (run under the old stack) must be **re-run under the new lock** before it counts. Do not proceed to backbones until the re-lock is executed and the diagnostic re-validated. (Per the spec, comparability requires identical stack across ALL runs.)

- [ ] **Step 3: Failing test — the lock includes mamba-ssm**

```python
# tests/training/test_env_lock_mamba.py
from cfm.training.env_lock import _EXPECTED, check_versions, TrainingEnvMismatch
import pytest

def test_lock_pins_mamba_ssm_and_causal_conv1d():
    assert "mamba-ssm" in _EXPECTED and "causal-conv1d" in _EXPECTED

def test_drift_in_mamba_ssm_is_caught():
    actual = dict.fromkeys(_EXPECTED, "x")  # all wrong
    actual.update({k: v for k, v in _EXPECTED.items() if k != "mamba-ssm"})
    with pytest.raises(TrainingEnvMismatch):
        check_versions(actual)
```

- [ ] **Step 4: Run to verify fail** — FAIL (`mamba-ssm` not in `_EXPECTED`).

- [ ] **Step 5: Extend `_EXPECTED`** in `env_lock.py` with the versions verified in Step 2, and add the imports to `assert_training_env_locked()`:

```python
# src/cfm/training/env_lock.py — add to the constants and _EXPECTED dict
LOCKED_MAMBA_SSM: str = "..."     # the version verified in Step 2 under torch 2.5.1+cu121
LOCKED_CAUSAL_CONV1D: str = "..."
_EXPECTED["mamba-ssm"] = LOCKED_MAMBA_SSM
_EXPECTED["causal-conv1d"] = LOCKED_CAUSAL_CONV1D
# in assert_training_env_locked(): import mamba_ssm, causal_conv1d and pass their __version__
```

- [ ] **Step 6: Run + commit**

```bash
uv run pytest tests/training/test_env_lock_mamba.py -v
git add scripts/verify_mamba_lock.sbatch src/cfm/training/env_lock.py tests/training/test_env_lock_mamba.py
git commit -m "feat(bakeoff): mamba-ssm verify-before-lock + env-lock extension (re-lock-all contingency settled)"
```

---

## Task 6: Value-bearing conditioning wiring + identity-lock
*(spec §14 Task 3; §7; §3/§Gate-6)*

**Files:**
- Modify: `src/cfm/data/training/conditioning.py`
- Test: `tests/data/training/test_conditioning_value_bearing.py`

- [ ] **Step 1: Failing test — the prefix encodes real values, identity-locked to the one source**

The conditioning derivation must remain the SAME object as `cfm.eval.holdout.labels.derive_tile_conditioning` (the trigger-2 one source) — an `is` assertion, not "produces equal output today."

```python
# tests/data/training/test_conditioning_value_bearing.py
from cfm.data.training import conditioning
from cfm.eval.holdout.labels import derive_tile_conditioning

def test_conditioning_derivation_is_the_one_source_by_identity():
    assert conditioning.derive_tile_conditioning is derive_tile_conditioning

def test_value_bearing_prefix_differs_across_distinct_conditioning():
    # two tiles with different density buckets must produce DIFFERENT prefix ids
    p_low = conditioning.build_value_bearing_prefix(population_density_bucket=0, cell_density_bucket=0, zoning_class=1, road_skeleton_class=1, region="singapore", coastal_inland_river=0, sub_c_morphology_class="Asian-megacity", seed=7)
    p_high = conditioning.build_value_bearing_prefix(population_density_bucket=5, cell_density_bucket=5, zoning_class=1, road_skeleton_class=1, region="singapore", coastal_inland_river=0, sub_c_morphology_class="Asian-megacity", seed=7)
    assert p_low != p_high  # value-bearing, not value-agnostic

def test_prefix_ids_stay_above_subf_vocab_and_are_append_only():
    p = conditioning.build_value_bearing_prefix(population_density_bucket=0, cell_density_bucket=0, zoning_class=1, road_skeleton_class=1, region="singapore", coastal_inland_river=0, sub_c_morphology_class="Asian-megacity", seed=7)
    assert all(i >= conditioning.CONDITIONING_ID_BASE for i in p)  # never collides with sub-F vocab
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement value-bearing conditioning.** Extend `conditioning.py`: keep `CONDITIONING_ID_BASE` (686) and the append-only `_CONDITIONING_FIELDS`; add per-field value sub-ranges *above* the base (each field reserves a block of ids for its bucketed values; the layout is recorded and append-only). Add `derive_tile_conditioning` re-exported from labels (the `is` source) and `build_value_bearing_prefix(...)` that maps each field's value to its id. Update `flatten_shards_to_cells` (datamodule) to call the value-bearing prefix using the shard's `tile_conditioning` + per-cell `cell_density_bucket`.

```python
# src/cfm/data/training/conditioning.py — re-export the one source by identity
from cfm.eval.holdout.labels import derive_tile_conditioning  # the trigger-2 single source (is-lock)
```

- [ ] **Step 4: Update the model's `n_cond`** path. `ScaffoldLit.__init__` reads `n_cond = len(conditioning_field_to_id())`; the value-bearing layout changes the prefix *content* but the prefix LENGTH stays 8 (one id per field, now value-encoded). Confirm `MicroAR`'s `max_len = cfg.max_len + n_cond` still holds and the embedding table spans `n_subf_vocab + (max value-id offset)`. **Discrimination test:** assert the embedding table is large enough for the maximum value-bearing id (no out-of-range index).

- [ ] **Step 5: Run + commit**

```bash
uv run pytest tests/data/training/test_conditioning_value_bearing.py -v
git add src/cfm/data/training/conditioning.py src/cfm/data/training/datamodule.py tests/data/training/test_conditioning_value_bearing.py
git commit -m "feat(bakeoff): value-bearing conditioning prefix (identity-locked to read_tile_labels)"
```

---

## Task 7: Swappable-backbone abstraction + mamba-hybrid + identity-lock test
*(spec §14 Task 6; §9)*

**Files:**
- Create: `src/cfm/models/backbone.py`, `src/cfm/models/mamba_hybrid.py`
- Modify: `src/cfm/training/lit_module.py`, `src/cfm/training/config.py`, `scripts/train_scaffold.py`
- Test: `tests/models/test_backbone_identity_lock.py`, `tests/models/test_mamba_hybrid.py`

- [ ] **Step 1: Failing test — the identity-lock (the comparability proof)**

All backbones must share the SAME conditioning-prefix builder, the SAME vocab/head sizing, and the SAME eval-content path — by identity. The test asserts the shared objects are shared, not that outputs match.

```python
# tests/models/test_backbone_identity_lock.py
from cfm.models.backbone import build_backbone, shared_head_dim, shared_conditioning_builder
from cfm.data.training.conditioning import build_value_bearing_prefix

def test_all_backbones_use_the_same_conditioning_builder_by_identity():
    assert shared_conditioning_builder() is build_value_bearing_prefix

def test_all_backbones_share_the_same_head_dim():
    # the sub-F head size (686) is one source for every backbone
    dims = {name: build_backbone(name, _tiny_cfg()).head_out_features for name in ("transformer-ar", "mamba-hybrid")}
    assert len(set(dims.values())) == 1  # identical head dim across backbones
```

- [ ] **Step 2: Run to verify fail** — FAIL (module missing).

- [ ] **Step 3: Implement `backbone.py`** — a `Backbone` protocol (forward signature identical to `MicroAR.forward`, `head_out_features` property, `training_loss` for AR backbones) + `build_backbone(name: str, cfg)` returning `MicroAR` for `"transformer-ar"`. Factor the embedding/pos/head construction so both AR backbones share it; only the mixing stack differs. Expose `shared_conditioning_builder()` and `shared_head_dim()` as the identity anchors.

- [ ] **Step 4: Failing test for mamba-hybrid forward shape + interleave ratio**

```python
# tests/models/test_mamba_hybrid.py
import torch, pytest
from cfm.models.mamba_hybrid import MambaHybrid, MambaHybridConfig

@pytest.mark.slow  # mamba-ssm needs CUDA
def test_mamba_hybrid_forward_matches_micro_ar_output_contract():
    cfg = MambaHybridConfig(d_model=256, n_layers=8, n_heads=8, n_subf_vocab=686, n_cond=8, max_len=128)
    m = MambaHybrid(cfg).cuda()
    out = m(torch.zeros(2, 16, dtype=torch.long, device="cuda"))
    assert out.shape == (2, 16, 686)  # same head contract as MicroAR

def test_interleave_is_7_mamba_to_1_transformer():
    cfg = MambaHybridConfig(d_model=64, n_layers=8, n_heads=8, n_subf_vocab=686, n_cond=8, max_len=64)
    m = MambaHybrid(cfg)
    assert m.n_transformer_layers == 1 and m.n_mamba_layers == 7  # 8 layers, ~7:1 Jamba ratio
```

- [ ] **Step 5: Implement `mamba_hybrid.py`** — Jamba-style interleave (1 transformer layer per 7 Mamba layers via `mamba_ssm.Mamba`), sharing the embedding/pos/head from `backbone.py`, same causal AR `training_loss` and `forward` contract as `MicroAR`. Mark CUDA tests `@pytest.mark.slow`.

- [ ] **Step 6: Wire `backbone` into config + lit_module + CLI.** Add `backbone: str = "transformer-ar"` to `ScaffoldConfig`; `ScaffoldLit.__init__` calls `build_backbone(cfg.backbone, ...)` instead of constructing `MicroAR` directly; add `--backbone` to `train_scaffold.py`'s argparse + override loop. Run the full fast suite to confirm transformer-ar still trains (regression guard).

- [ ] **Step 7: Run + commit**

```bash
uv run pytest tests/models/test_backbone_identity_lock.py tests/models/test_mamba_hybrid.py -v -m "not slow"
git add src/cfm/models/backbone.py src/cfm/models/mamba_hybrid.py src/cfm/training/lit_module.py src/cfm/training/config.py scripts/train_scaffold.py tests/models/
git commit -m "feat(bakeoff): swappable-backbone abstraction + mamba-hybrid + identity-lock test"
```

---

## Task 8: Discrete-diffusion backbone (absorbing-state) + identity-lock
*(spec §14 Task 7; §9; the diffusion accommodations)*

**Files:**
- Create: `src/cfm/models/discrete_diffusion.py`, `src/cfm/models/diffusion/{mask.py,loss.py,generate.py}`
- Modify: `src/cfm/training/lit_module.py` (route loss per backbone), `src/cfm/inference/generate.py` (route generation per backbone)
- Test: `tests/models/test_discrete_diffusion.py`

- [ ] **Step 1: Failing test — absorbing-state masking + the three quarantined divergences exist and the shared parts are identity-locked**

```python
# tests/models/test_discrete_diffusion.py
import torch, pytest
from cfm.models.discrete_diffusion import DiscreteDiffusion, DiffusionConfig
from cfm.models.diffusion.mask import apply_absorbing_mask, MASK_ID
from cfm.models.backbone import shared_conditioning_builder
from cfm.data.training.conditioning import build_value_bearing_prefix

def test_diffusion_uses_the_same_shared_conditioning_builder_by_identity():
    assert shared_conditioning_builder() is build_value_bearing_prefix  # not forked

def test_absorbing_mask_replaces_a_fraction_with_MASK_ID_seeded():
    seq = torch.arange(100).reshape(1, 100)
    masked, target_mask = apply_absorbing_mask(seq, frac=0.5, seed=7)
    assert (masked == MASK_ID).sum() > 0
    # reproducible under the same seed (DDP determinism — diffusion mask sampling seeded)
    masked2, _ = apply_absorbing_mask(seq, frac=0.5, seed=7)
    assert torch.equal(masked, masked2)

def test_diffusion_head_contract_matches_other_backbones():
    cfg = DiffusionConfig(d_model=64, n_layers=4, n_heads=4, n_subf_vocab=686, n_cond=8, max_len=64)
    out = DiscreteDiffusion(cfg)(torch.zeros(2, 16, dtype=torch.long))
    assert out.shape == (2, 16, 686)  # same head dim; bidirectional mask (no causal)
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement the three quarantined modules + the backbone.**
  - `diffusion/mask.py`: `MASK_ID` (a reserved id ABOVE the conditioning block — append-only, never collides with sub-F vocab or conditioning ids), `apply_absorbing_mask(seq, frac, seed)` (seeded — DDP determinism), and the noise schedule.
  - `diffusion/loss.py`: the denoising cross-entropy over masked positions (NOT next-token; the MDLM-family objective).
  - `diffusion/generate.py`: `denoise(model, prefix, max_new, T, seed)` — start all-MASK, run T un-masking passes (bidirectional forward each pass), seeded.
  - `discrete_diffusion.py`: `DiscreteDiffusion` — shares embedding/pos/head from `backbone.py`; forward uses a **bidirectional** (non-causal) mask; exposes `diffusion_loss(...)` and `generate(...)` from the quarantined modules.

- [ ] **Step 4: Route per backbone in `lit_module.py` and `generate.py`.** `ScaffoldLit._loss` calls `diffusion_loss` when `cfg.backbone == "discrete-diffusion"`, else `model.training_loss`. `generate_cell_tokens` routes to `denoise(...)` for diffusion (T from config), else the AR multinomial loop. **No NLL cross-check for diffusion** (geometry-only ranking, §3).

- [ ] **Step 5: Failing test — T is a config knob set by quality-convergence, not budget-capped.** Add a config field `diffusion_T` and a test asserting `generate` honors it; document (in a comment + the deviation log) that T is chosen by quality-convergence (denoise until the geometry score plateaus), never capped to a budget. The quality-convergence sweep itself is an empirical sub-step of Task 12 for the diffusion runs.

- [ ] **Step 6: Run + commit**

```bash
uv run pytest tests/models/test_discrete_diffusion.py -v -m "not slow"
git add src/cfm/models/discrete_diffusion.py src/cfm/models/diffusion/ src/cfm/training/lit_module.py src/cfm/inference/generate.py tests/models/test_discrete_diffusion.py
git commit -m "feat(bakeoff): absorbing-state diffusion backbone, divergences quarantined, shared parts identity-locked"
```

---

## Task 9: Comparability lock + deviation valve + grad-accum effective batch
*(spec §14 Task 8; §10)*

**Files:**
- Create: `src/cfm/training/deviation_log.py`
- Modify: `src/cfm/training/config.py` (add `grad_accum: int = 1`), `src/cfm/training/train.py` (`accumulate_grad_batches`)
- Test: `tests/training/test_deviation_valve.py`, `tests/training/test_grad_accum.py`

- [ ] **Step 1: Failing test — the deviation valve trigger is fails-to-train, NOT scores-lower, and is testable**

```python
# tests/training/test_deviation_valve.py
from cfm.training.deviation_log import is_train_failure, DeviationLog, DeviationError

def test_train_failure_is_diverged_nan_or_flatline_not_low_score():
    assert is_train_failure(loss_history=[5.0, float("nan")]) is True          # NaN
    assert is_train_failure(loss_history=[5.0, 6.0, 8.0, 12.0]) is True          # diverging
    assert is_train_failure(loss_history=[5.0, 5.0, 5.0, 5.0]) is True           # flatline from step 0
    assert is_train_failure(loss_history=[5.0, 4.0, 3.5, 3.2]) is False          # trains fine (just maybe loses later)

def test_deviation_must_be_logged_with_a_uniform_rule_not_a_bespoke_number():
    log = DeviationLog()
    log.record(backbone="discrete-diffusion", scale="100M", rule="loss-scale-normalized-lr", trigger="flatline")
    # a bespoke per-backbone number with no named rule is rejected
    import pytest
    with pytest.raises(DeviationError):
        log.record(backbone="discrete-diffusion", scale="100M", rule=None, trigger="scores-lower")
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement `deviation_log.py`** — `is_train_failure(loss_history)` (NaN / monotone-increasing / flat-from-start detectors); `DeviationLog.record(...)` that REQUIRES a named principled `rule` and rejects `trigger="scores-lower"` (raises `DeviationError`). The log is written to the run's `reports/` entry.

- [ ] **Step 4: Failing test — effective batch is held constant via grad-accum across scales.**

```python
# tests/training/test_grad_accum.py
from cfm.training.config import ScaffoldConfig
from cfm.training.train import effective_batch_size

def test_effective_batch_constant_across_per_gpu_batch_via_grad_accum():
    # 30M fits batch 8; 1B fits batch 2 -> grad_accum 4 keeps effective batch identical
    small = ScaffoldConfig(batch_size=8, grad_accum=1, devices=4)
    large = ScaffoldConfig(batch_size=2, grad_accum=4, devices=4)
    assert effective_batch_size(small) == effective_batch_size(large)  # 8*4*1 == 2*4*4
```

- [ ] **Step 5: Implement `grad_accum` in config + `effective_batch_size(cfg) = cfg.batch_size * cfg.devices * cfg.grad_accum`; pass `accumulate_grad_batches=cfg.grad_accum` to `build_trainer`'s `L.Trainer(...)`.** Add `--grad-accum` to the CLI.

- [ ] **Step 6: Run + commit**

```bash
uv run pytest tests/training/test_deviation_valve.py tests/training/test_grad_accum.py -v
git add src/cfm/training/deviation_log.py src/cfm/training/config.py src/cfm/training/train.py scripts/train_scaffold.py tests/training/test_deviation_valve.py tests/training/test_grad_accum.py
git commit -m "feat(bakeoff): deviation valve (fails-to-train-only, logged) + grad-accum effective-batch lock"
```

---

## Task 10: Across-job `$WORK` checkpoint resume (the 1B requirement)
*(spec §14 Task 10 new requirement; §10)*

**Files:**
- Create: `src/cfm/training/resume.py`
- Modify: `src/cfm/training/train.py` (checkpoint `dirpath` on `$WORK`; auto-resume `ckpt_path`), `scripts/bakeoff_run.sbatch` (relaunch-on-timeout)
- Test: `tests/training/test_resume.py`

- [ ] **Step 1: Failing test — find the latest checkpoint on $WORK and resume from it, not step 0**

```python
# tests/training/test_resume.py
from pathlib import Path
from cfm.training.resume import latest_checkpoint, resume_ckpt_path

def test_latest_checkpoint_picks_highest_step(tmp_path: Path):
    (tmp_path / "epoch=0-step=100.ckpt").write_text("x")
    (tmp_path / "epoch=0-step=500.ckpt").write_text("x")
    (tmp_path / "last.ckpt").write_text("x")
    assert latest_checkpoint(tmp_path).name == "last.ckpt"  # Lightning's last.ckpt preferred

def test_resume_returns_none_on_fresh_run(tmp_path: Path):
    assert resume_ckpt_path(tmp_path) is None  # empty dir -> fresh, no false resume

def test_resume_returns_checkpoint_when_present(tmp_path: Path):
    (tmp_path / "last.ckpt").write_text("x")
    assert resume_ckpt_path(tmp_path) == tmp_path / "last.ckpt"
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement `resume.py`** — `latest_checkpoint(dir)` (prefer `last.ckpt`, else max-step), `resume_ckpt_path(dir)` (None if empty). Set the `ModelCheckpoint(dirpath=...)` to a `$WORK`-rooted per-run path (`$WORK/Bonzai-OSM/checkpoints/bakeoff/<backbone>-<scale>/`), and pass `trainer.fit(..., ckpt_path=resume_ckpt_path(dir))` so a relaunched job continues.

- [ ] **Step 4: Author `scripts/bakeoff_run.sbatch` with relaunch-on-timeout** — `#SBATCH --signal=B:USR1@120` + a trap that resubmits the same job (`sbatch $0`) when the wall-clock signal fires, so a 1B run spanning multiple jobs auto-continues from `$WORK`.

```bash
# scripts/bakeoff_run.sbatch (relaunch-on-timeout skeleton)
#SBATCH --signal=B:USR1@120
#SBATCH --time=24:00:00
# ... (standard 4-GPU header) ...
trap 'echo "wall-clock approaching; resubmitting"; sbatch "$0"; exit 0' USR1
srun --kill-on-bad-exit=1 python -u scripts/train_scaffold.py --backbone "$BACKBONE" ... &
wait
```

- [ ] **Step 5: The across-job resume integration test (the user-required test).** As a `@pytest.mark.slow` test on a tiny config: start a short run, kill it mid-way, relaunch pointing at the same `$WORK` dir, assert `trainer.global_step` resumes from the last checkpoint's step (not 0) and the loss continues rather than restarting. (Local CPU surrogate acceptable for the unit-level resume logic; the real Slurm relaunch is verified manually on Leonardo.)

```python
# append to tests/training/test_resume.py
import pytest
@pytest.mark.slow
def test_across_job_resume_continues_from_last_checkpoint(tmp_path):
    # run a few steps -> checkpoint -> simulate new job -> resume -> global_step > 0 at start
    ...  # uses run_short with a tiny config and ckpt_path=resume_ckpt_path(dir)
```

- [ ] **Step 6: Run + commit**

```bash
uv run pytest tests/training/test_resume.py -v -m "not slow"
git add src/cfm/training/resume.py src/cfm/training/train.py scripts/bakeoff_run.sbatch tests/training/test_resume.py
git commit -m "feat(bakeoff): across-job \$WORK checkpoint resume (1B runs span jobs; kill-mid-run continues)"
```

---

## Task 11: Pilot winner-vs-runner-up gap → conditional parallel second-region extraction
*(spec §14 Task 9; §8 early-warning)*

**EMPIRICAL/ops, gated after backbones 1–2 (transformer-AR + mamba-hybrid).**

- [ ] **Step 1: After the 30M+100M points for transformer-AR and mamba-hybrid exist** (a subset of Task 12), compute the per-feature KS scores (Task 2) and the binding per-feature resolution (Task 3) on those pilot runs.
- [ ] **Step 2: Estimate the pilot winner-vs-runner-up gap** via `check_decision_resolvable` (Task 3). Decision branch:
  - **Gap clears the binding resolution** → no extraction; proceed.
  - **Gap within noise** → the action contract fires: **kick off second-region extraction NOW, in parallel** (it is slow — ~8h+ cold fetch; ≈ its own data sub-project), so it is ready by the time the full ladder finishes. Record the trigger in the report.
- [ ] **Step 3: If extraction fires,** open it as the documented second-region escalation (Sweden or Sri Lanka, the locked de-risking set), serving the triple duty (resolution + generalization + compliance). Do NOT block the remaining bake-off runs on it.

```bash
git add reports/phase-2-bakeoff/
git commit -m "expt(bakeoff): pilot winner-vs-runner-up gap; second-region extraction trigger evaluated"
```

---

## Task 12: Run the 12-run ladder (3 backbones × {30M,100M,300M,1B})
*(spec §14 Task 10; §12)*

**EMPIRICAL/ops. Each run = one (measured node-h, KS-realism) point. Final-checkpoint eval only.**

- [ ] **Step 1: Per-run configs.** Generate `configs/experiments/bakeoff-<backbone>-<scale>.yaml` from `bakeoff-base.yaml` (Task 4), each with the geometry-verified `r` (→ `max_steps`), the locked recipe, `grad_accum` to hold effective batch constant (Task 9), and the `$WORK` checkpoint dir (Task 10).
- [ ] **Step 2: Sequencing — decision-relevant scales first.** Submit in an order that lands the most decision-relevant points first if the June-11 window tightens: the **300M and 1B** points of all three backbones before the 30M/100M curve-shape filler. (The winner-vs-runner-up gap is dominated by the points nearest C_prod.)

- [ ] **Step 2b: HARD ORDERING — prove across-job resume (Task 10) on a REAL Leonardo relaunch before submitting ANY 1B job.** The 1B runs are simultaneously the ones most likely to span job boundaries AND the ones sequenced first (Step 2) — so the across-job `$WORK` resume path must be verified working on a real Slurm timeout-relaunch (not just the unit test) **before** the first 1B submission. Run the relaunch verification on a 300M job first (let it hit a short `--time` cap, confirm the resubmitted job continues from `$WORK`'s `last.ckpt`, not step 0). Only after that passes does any 1B job submit. Discovering a broken resume mid-1B-run costs ~159 node-h per backbone.
- [ ] **Step 3: Diffusion T quality-convergence sweep.** For the diffusion runs, before scoring the final point, sweep `diffusion_T` upward until the per-feature KS score plateaus; that plateau-T is the run's T (cost follows). Log the sweep; never cap T to a budget (§4/§6).
- [ ] **Step 4: For each run** — verify `assert_training_env_locked()` passes (it runs at the GPU entrypoint), train to convergence (**"converged" = loss-flat AND past the Task-4 emergence floor**), final-checkpoint eval on the frozen holdout (n_cells/seed-repeats per the §6 measured minimum), record the (measured node-h, KS-realism score, emergence_verdict, right_angle_rate, density-compliance) point + the NLL cross-check for the two AR backbones. Each run writes a `reports/phase-2-bakeoff/<backbone>-<scale>.md`.
- [ ] **Step 5: Commit each run's report as it completes** (task-by-task, local-first):

```bash
git add reports/phase-2-bakeoff/<backbone>-<scale>.md configs/experiments/bakeoff-<backbone>-<scale>.yaml
git commit -m "expt(bakeoff): <backbone> <scale> run — (node-h, KS-realism) point recorded"
```

---

## Task 13: Bootstrap-CI curve fit + extrapolation + §13 structural check + tie-break
*(spec §14 Task 11; §11)*

**Files:**
- Create: `src/cfm/eval/curve.py`
- Test: `tests/eval/test_curve.py`

- [ ] **Step 1: Failing test — fit returns a CI, and a non-monotonic fit fails the §2 structural check**

```python
# tests/eval/test_curve.py
import pytest
from cfm.eval.curve import fit_scaling_curve, extrapolate, structural_check_ok, pick_winner, TIEBREAK_BACKBONE

def test_fit_returns_point_estimate_and_confidence_interval():
    pts = [(0.14, 0.40), (1.6, 0.30), (14.6, 0.22), (160.0, 0.18)]  # (node-h, KS) lower=better
    fit = fit_scaling_curve(pts, n_bootstrap=200)
    lo, hi = extrapolate(fit, target_node_h=500.0)
    assert lo < hi  # a confidence interval, not a bare point

def test_structural_check_rejects_non_monotonic_fit():
    bad = [(0.14, 0.30), (1.6, 0.35), (14.6, 0.20), (160.0, 0.50)]  # non-improving / noisy
    assert structural_check_ok(fit_scaling_curve(bad, n_bootstrap=200)) is False

def test_tiebreak_is_transformer_ar_when_extrapolated_cis_overlap():
    # winner & runner-up extrapolated CIs overlap -> doesn't separate -> simplest backbone
    overlapping = {"transformer-ar": (0.17, 0.21), "mamba-hybrid": (0.18, 0.22), "discrete-diffusion": (0.30, 0.34)}
    assert pick_winner(overlapping) == TIEBREAK_BACKBONE == "transformer-ar"

def test_clear_winner_is_picked_when_cis_separate():
    separated = {"transformer-ar": (0.30, 0.33), "mamba-hybrid": (0.17, 0.20), "discrete-diffusion": (0.40, 0.44)}
    assert pick_winner(separated) == "mamba-hybrid"
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement `curve.py`** — `fit_scaling_curve(points, n_bootstrap)` (power-law fit in log-log of KS-vs-node-h, bootstrap over points/seeds → CI); `extrapolate(fit, target_node_h)` → `(lo, hi)`; `structural_check_ok(fit)` (monotonic-improving + residual/CI bound — the §2 paired check); `pick_winner(extrapolated_cis)` — best point estimate IF its CI does not overlap the runner-up's, ELSE `TIEBREAK_BACKBONE = "transformer-ar"` (the pre-committed §13 simplest-backbone tie-break). Wire `check_decision_resolvable` (Task 3) at the extrapolated point so "doesn't separate" reuses the resolution seam.

- [ ] **Step 4: Run + commit**

```bash
uv run pytest tests/eval/test_curve.py -v
git add src/cfm/eval/curve.py tests/eval/test_curve.py
git commit -m "feat(bakeoff): bootstrap-CI scaling-curve fit + extrapolation + §13 structural check + tie-break"
```

---

## Task 14: Final report + PRD update + Phase-3-opening gate
*(spec §14 Task 12)*

**Files:**
- Create: `reports/phase-2-bakeoff/2026-XX-XX-bakeoff-decision.md`
- Modify: `PRD.md` (§6 roster reshape, §6.4/§10 cost correction), `docs/protocols/sub-project-planning-protocol-v3.md` (bump if a new principle surfaced)

- [ ] **Step 1: Write the decision report** — every run's config+commit+snapshot+point; the fitted curves + extrapolated CIs; the winner (or the §13 tie-break invocation with evidence); the resolution-seam outcome (fired → second region, or cleared); the five regime-transfer catches as realized; the deviation log.
- [ ] **Step 2: Update `PRD.md`** (flag "experiments win"): §6 roster reshape (4×3 → 3 backbones × 4 scales, cell-level-collapse rationale); §6.4/§10 cost correction (~7.6× under, not 80×); the eval-as-binding-cost reframe.
- [ ] **Step 3: Write the named Phase-3-opening winner-preview gate** — a committed checkpoint in the report + PRD §11: "Phase 3 OPENS by building the macro-planner + boundary-contract stitching for the winning backbone and running a tile-coherence preview BEFORE committing the full hierarchical build."
- [ ] **Step 4: Protocol bump check** — if the bake-off surfaced a new gate/principle (e.g., the regime-transfer meta-pattern as a formal §11), add it to the protocol with a worked example; else note "no new institutional capital, v3 holds."
- [ ] **Step 5: Final suite + merge prep**

```bash
uv sync --extra dev && uv run ruff format && uv run ruff check && uv run pytest -q -m "not slow"
git add reports/phase-2-bakeoff/ PRD.md docs/protocols/
git commit -m "report(bakeoff): architecture decision + PRD update + Phase-3-opening winner-preview gate"
# Merge to main + push happens ONLY after the user confirms suite-green + report-written (CLAUDE.md).
```

---

## Self-review (run against the spec)

**Spec coverage:** §3 axis → Tasks 2 (KS metric), 13 (curve); §4 compute axis → Tasks 4 (r), 9 (effective batch); §5 emergence → Tasks 1, 4; §6 eval-cost → Tasks 4 (per-scale measure), 12 (final-checkpoint, T-convergence); §7 metric set → Tasks 2, 6; §8 resolution → Tasks 3, 11; §9 builds → Tasks 5, 7, 8; §10 comparability → Tasks 5, 9, 10; §11 robustness → Tasks 12 (ladder), 13 (fit/tie-break). All spec sections map to a task.

**User's review criteria, each satisfied:** every tier-1-flavored lock has a discrimination test — comparability version lock (Task 5 `test_drift_in_mamba_ssm_is_caught`), identity-lock (Tasks 7/8 `is`-assertions), resolution-unit (Task 3 `test_resolution_uses_feature_count_not_inherited_0076`); the Task 4→5→6+ hard gate is stated at the top and in each task's dependency; the across-job-resume test exists (Task 10 Step 5); the deviation-valve fails-to-train-only trigger is testable (Task 9 `test_train_failure_is_diverged_nan_or_flatline_not_low_score`).

**Placeholders:** the `LOCKED_MAMBA_SSM = "..."` in Task 5 is filled in Step 2 (the verified version) — it is a measured value, not a plan gap. All other steps carry real code or real commands.

**Type consistency:** `EmergenceVerdict`, `FeatureMetric`, `BindingResolution`, `DeviationLog`, `build_backbone`, `build_value_bearing_prefix`, `fit_scaling_curve` are referenced consistently across tasks; the `MicroAR(cfg)` / `head_out_features` / `n_subf_vocab=686` / `n_cond=8` contracts match the explored surface.

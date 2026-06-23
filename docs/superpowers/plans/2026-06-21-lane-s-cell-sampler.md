> NOTE (2026-06-23): current design, but the scored run / architecture crown / realism eval is DEFERRED per
> `docs/PROJECT_FOCUS.md` (current focus = methodology validation). Do not run the scored matrix, decide(),
> or build the realism eval without Umar's word.

# Lane-S Cell Sampler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a budget-bounded, stratified, sha-locked DOWN-sampler that picks which held-out cells the bake-off backbones generate, so generated-side feature distributions clear the conditioning floor's `min_n=50` per scored stratum and `decide()` produces a real verdict, not an under-sampling artifact.

**Architecture:** A pure-logic core (`cfm.eval.lane_s_sampler`) sizes `n_cells` per floored `(city, 4-tuple)` stratum against the *scarce* floored metric, hash-rank-selects cells, and seals a write-once manifest using the existing locked-YAML grammar. A regen script runs the only data-dependent step — a per-stratum cell-key census over the held-out pool (read-only, no GPU, on Leonardo) — then builds the manifest. A consumer-side coverage check applies the §9 ceiling-bound split after generation. Real feature counts come from the locked floor artifact's `n_a`/`n_b` (local), not the 29 MB parquet.

**Tech Stack:** Python 3.11+, `pyarrow` (parquet census), stdlib `hashlib.blake2b` (selection), `cfm.data.locked_yaml` (seal/verify), `cfm.eval.conditioning_floor.load_verified_floor` (lineage), `pytest`.

**Spec:** `docs/superpowers/specs/2026-06-21-lane-s-cell-sampler.md` (committed `a6eff0b`). Gates 1–5 + §6/§7 + R3-RESOLVED.

**Standing gates (do NOT cross during execution):** no scored run, no generation, no merge. Code commits per task are fine; the BUILD (running the regen script on real held-out data) is gated on PI word and the GROUND_TRUTH §5 Leonardo redeploy.

**LOCKED INVARIANT (floor→sampler boundary, PI 2026-06-21):** `floor_n` is READ from the locked floor artifact (sha `95abb88…`), NEVER recomputed. The build CLI fails loud if the loaded floor's sha ≠ the pinned `EXPECTED_FLOOR_SHA256`; Task 4's external-source-of-truth test asserts the real floor's sha equals that constant, so a future floor re-derivation that changes `n_a`/`n_b` turns the test RED and forces the constant + guard to update in the SAME commit (lock-and-guards-travel-together). The parquet-drop deviation (`real_fpc = floor_n / census_cells`, parquet not read) is **RATIFIED + LOCKED**: `available_cells` cancels in the ceiling-bound flag, so the R3-proven feasibility logic is fully local-testable and Leonardo shrinks to census-only.

---

## File structure

| File | Responsibility |
|---|---|
| `src/cfm/eval/lane_s_sampler.py` (create) | Pure logic + manifest IO: constants/errors, floor adapter, sizing, selection, census read/write, manifest build + seal + verified load, consumer coverage check. Mirrors `conditioning_floor.py`'s one-module structure. |
| `scripts/_heldout_cell_count.py` (modify) | Add an `--emit <path>` mode that writes the per-stratum cell-key census parquet (keeps the existing print summary). |
| `scripts/build_lane_s_sampler.py` (create) | CLI orchestrator: verified-load floor → read census → build → seal manifest → print cost re-derivation (§6). |
| `scripts/build_lane_s_sampler.sbatch` (create) | Leonardo no-GPU serial job: run census emit, then build. |
| `tests/eval/test_lane_s_sampler.py` (create) | Fast unit tests (fixtures): sizing, selection determinism, §9 split, lock round-trip/tamper. |
| `tests/eval/test_lane_s_sampler_real_floor.py` (create, `@pytest.mark.slow`) | External-source-of-truth: assert against the real locked floor (146/119 strata, min building `n`=59, R3 ceiling-bound counts). |

Artifacts (NOT committed — data): `data/processed/lane_s_sampler/2026-04-15.0/{heldout-cell-census.parquet, sampler-manifest.yaml, _LANE_S_SAMPLER_LOCKED}`.

---

## Task 1: Module skeleton — constants, errors, manifest seal + verified load

**Files:**
- Create: `src/cfm/eval/lane_s_sampler.py`
- Test: `tests/eval/test_lane_s_sampler.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_lane_s_sampler.py
from __future__ import annotations

import pytest

from cfm.eval import lane_s_sampler as ls


def _minimal_payload() -> dict:
    return {
        "sampler_schema_version": ls.SAMPLER_SCHEMA_VERSION,
        "release": "test.0",
        "floor_sha256": "deadbeef",
        "methodology": {"target_features": 50, "headroom": 2.0, "seed": 7,
                        "selection": "blake2b_hash_rank"},
        "held_out_cities": ["glasgow"],
        "strata": [],
        "cells": [],
    }


def test_seal_then_verified_load_round_trips(tmp_path):
    path = tmp_path / "sampler-manifest.yaml"
    ls.seal_manifest(_minimal_payload(), path)
    assert (tmp_path / ls.SAMPLER_LOCK_NAME).exists()
    loaded = ls.load_verified_manifest(path)
    assert loaded["release"] == "test.0"
    assert loaded["methodology"]["target_features"] == 50


def test_verified_load_refuses_tampered_content(tmp_path):
    path = tmp_path / "sampler-manifest.yaml"
    ls.seal_manifest(_minimal_payload(), path)
    text = path.read_text().replace("release: test.0", "release: tampered.9")
    path.write_text(text)
    with pytest.raises(ls.SamplerArtifactError, match="sha mismatch"):
        ls.load_verified_manifest(path)


def test_seal_is_write_once(tmp_path):
    path = tmp_path / "sampler-manifest.yaml"
    ls.seal_manifest(_minimal_payload(), path)
    with pytest.raises(FileExistsError):
        ls.seal_manifest(_minimal_payload(), path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_lane_s_sampler.py -k "round_trips or tampered or write_once" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cfm.eval.lane_s_sampler'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cfm/eval/lane_s_sampler.py
"""Lane-S held-out CELL SAMPLER (spec 2026-06-21).

A budget-bounded stratified DOWN-sampler over the held-out cell pool. Picks
which held-out cells the bake-off backbones generate so generated-side feature
distributions clear the conditioning floor's min_n per scored stratum.

UNIT DISCIPLINE (spec, protocol §10.3): the obligation is FEATURES (>= min_n per
floored (city, metric, stratum)); the lever is CELLS (per (city, 4-tuple)). The
scarce floored metric (building_area where owed) binds n_cells.

The artifact is sha-locked write-once via cfm.data.locked_yaml, mirroring the
conditioning floor's grammar (sha excludes itself; a _LANE_S_SAMPLER_LOCKED
marker beside the file; reader refuses absent/unsealed/sha-mismatch/skew).
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass

from cfm.data.locked_yaml import stamp_and_seal, verify_sealed_yaml

logger = logging.getLogger(__name__)

SAMPLER_SCHEMA_VERSION = "1.0"
SAMPLER_LOCK_NAME = "_LANE_S_SAMPLER_LOCKED"
SAMPLER_SHA_FIELD = "sampler_sha256"

#: Metric token strings as the floor freezes them (conditioning_discrimination._tile_features).
BUILDING_METRIC = "building_area_m2"
ROAD_METRIC = "road_length_m"

#: LOCK-AND-GUARDS-TRAVEL-TOGETHER (spec invariant, PI 2026-06-21): floor_n is READ from THIS
#: locked floor (sha 95abb88), NEVER recomputed. The build CLI fails loud if the loaded floor's
#: sha differs (a re-derived floor could change n_a/n_b silently); Task 4's external-SoT test
#: RED-flags a change so the guard + this constant update in the SAME commit as the floor.
EXPECTED_FLOOR_SHA256 = "95abb88bfaf0a79d4254883478aa5e5b558ed63c27a3c0a5845e8bb65f3a6be6"

#: DECISION: default target = the floor's locked min_n (the obligation unit). Revisit only if
#: the floor's min_n changes (then cells re-derive automatically). Spec §6.
DEFAULT_TARGET_FEATURES = 50
#: DECISION: headroom=2.0 default (spec Gate 5 + R3: 6/119 ceiling-bound at 2.0, glasgow-
#: concentrated, #21 risk negligible). Config knob; refined after first generation. Spec §6.
DEFAULT_HEADROOM = 2.0


class SamplerArtifactError(RuntimeError):
    """The sampler manifest failed verification (absent / unsealed / tampered / skewed)."""


def seal_manifest(payload: dict, path) -> None:
    """Stamp the sha, write canonical YAML ONCE, touch the lock marker."""
    stamp_and_seal(payload, path, sha_field=SAMPLER_SHA_FIELD, lock_name=SAMPLER_LOCK_NAME)


def load_verified_manifest(path) -> dict:
    """Verified read; refuses absent/unsealed/sha-mismatch/version-skew (fail-closed)."""
    return verify_sealed_yaml(
        path,
        sha_field=SAMPLER_SHA_FIELD,
        lock_name=SAMPLER_LOCK_NAME,
        schema_field="sampler_schema_version",
        schema_version=SAMPLER_SCHEMA_VERSION,
        required_key="strata",
        error=SamplerArtifactError,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_lane_s_sampler.py -k "round_trips or tampered or write_once" -v`
Expected: PASS (3 passed). The `sha mismatch` message comes from `verify_sealed_yaml`.

- [ ] **Step 5: Commit**

```bash
git add src/cfm/eval/lane_s_sampler.py tests/eval/test_lane_s_sampler.py
git commit -m "feat(cell-eos): lane-s sampler module skeleton + sha-locked manifest IO"
```

---

## Task 2: Sizing — scarce-metric binding + ceiling-bound

**Files:**
- Modify: `src/cfm/eval/lane_s_sampler.py`
- Test: `tests/eval/test_lane_s_sampler.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/eval/test_lane_s_sampler.py
def test_binding_metric_is_building_when_owed():
    assert ls.binding_metric(frozenset({ls.BUILDING_METRIC, ls.ROAD_METRIC})) == ls.BUILDING_METRIC
    assert ls.binding_metric(frozenset({ls.ROAD_METRIC})) == ls.ROAD_METRIC


def test_size_stratum_ceiling_bound_depends_only_on_floor_n():
    # floor_n=50, target=50, headroom=1.0 -> raw=ceil(50*1*A/50)=A -> NOT ceiling-bound
    r = ls.size_stratum(target_features=50, headroom=1.0, floor_n_binding=50, available_cells=200)
    assert not r.ceiling_bound and r.n_cells_selected == 200 and r.n_cells_target == 200
    # floor_n=59 (the real min), headroom=2.0 -> target*headroom=100 > 59 -> ceiling-bound
    r2 = ls.size_stratum(target_features=50, headroom=2.0, floor_n_binding=59, available_cells=40)
    assert r2.ceiling_bound and r2.n_cells_selected == 40  # take-all
    # plentiful stratum: floor_n=3000, headroom=2.0, available big -> small draw, not ceiling
    r3 = ls.size_stratum(target_features=50, headroom=2.0, floor_n_binding=3000, available_cells=2000)
    assert not r3.ceiling_bound and r3.n_cells_selected < 2000


def test_size_stratum_rejects_unfloored_n():
    with pytest.raises(ValueError, match="floor_n_binding"):
        ls.size_stratum(target_features=50, headroom=2.0, floor_n_binding=0, available_cells=10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_lane_s_sampler.py -k "binding_metric or size_stratum" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'binding_metric'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/cfm/eval/lane_s_sampler.py (after load_verified_manifest)

@dataclass(frozen=True)
class SizingResult:
    n_cells_target: int   # raw demand = ceil(target/real_fpc * headroom)
    n_cells_selected: int # min(raw, available_cells)
    ceiling_bound: bool   # raw > available_cells (pool exhausted at the target)


def binding_metric(owed_metrics: frozenset[str]) -> str:
    """The SCARCE floored metric that binds n_cells: building_area where owed (it emits
    ~0-1/cell vs roads ~5-15/cell), else road_length. building_area is never owed alone
    (building_area subset road_length in the floor), so this is total over the floored set."""
    if BUILDING_METRIC in owed_metrics:
        return BUILDING_METRIC
    if ROAD_METRIC in owed_metrics:
        return ROAD_METRIC
    raise ValueError(f"no known floored metric in owed set {sorted(owed_metrics)}")


def size_stratum(
    *, target_features: int, headroom: float, floor_n_binding: int, available_cells: int
) -> SizingResult:
    """n_cells = ceil(target / real_fpc[binding] * headroom), real_fpc = floor_n/available.

    Algebraically raw = ceil(target * headroom * available / floor_n); `available` cancels in
    the ceiling test, so ceiling_bound <=> floor_n < target*headroom (independent of available —
    why R3 was computable from floor_n alone). floor_n must be >= 1 (a floored stratum has
    n >= the floor's min_n by construction)."""
    if floor_n_binding < 1:
        raise ValueError(f"floor_n_binding must be >= 1 (got {floor_n_binding}); a floored "
                         "stratum has n >= min_n by construction")
    if available_cells < 1:
        raise ValueError(f"available_cells must be >= 1 (got {available_cells})")
    raw = math.ceil(target_features * headroom * available_cells / floor_n_binding)
    selected = min(raw, available_cells)
    return SizingResult(n_cells_target=raw, n_cells_selected=selected,
                        ceiling_bound=raw > available_cells)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_lane_s_sampler.py -k "binding_metric or size_stratum" -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/cfm/eval/lane_s_sampler.py tests/eval/test_lane_s_sampler.py
git commit -m "feat(cell-eos): lane-s sampler sizing (scarce-metric binding + ceiling-bound)"
```

---

## Task 3: Selection — blake2b hash-rank (PYTHONHASHSEED-proof)

**Files:**
- Modify: `src/cfm/eval/lane_s_sampler.py`
- Test: `tests/eval/test_lane_s_sampler.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/eval/test_lane_s_sampler.py
import random


def _cells(n):
    return [ls.SampledCell(city="glasgow", tile_i=i, tile_j=0, cell_i=i % 7, cell_j=i // 7,
                           density_bucket=1) for i in range(n)]


def test_select_cells_take_all_when_capped():
    cells = _cells(10)
    out = ls.select_cells(cells, 25, seed=7)
    assert len(out) == 10  # take-all, never over-draw


def test_select_cells_is_input_order_independent():
    cells = _cells(100)
    a = ls.select_cells(cells, 30, seed=7)
    shuffled = cells[:]
    random.Random(123).shuffle(shuffled)
    b = ls.select_cells(shuffled, 30, seed=7)
    assert a == b  # hash-rank keys on identity, not input order -> PYTHONHASHSEED-proof


def test_select_cells_seed_changes_subset():
    cells = _cells(100)
    assert ls.select_cells(cells, 30, seed=7) != ls.select_cells(cells, 30, seed=8)


def test_select_cells_output_canonically_sorted():
    out = ls.select_cells(_cells(100), 30, seed=7)
    assert out == sorted(out, key=ls._cell_sort_key)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_lane_s_sampler.py -k "select_cells" -v`
Expected: FAIL — `AttributeError: ... 'SampledCell'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/cfm/eval/lane_s_sampler.py (after the imports block, near top with other dataclasses)

@dataclass(frozen=True, order=True)
class SampledCell:
    """One held-out cell to condition generation on. Identity = grid coordinate (cell_i,
    cell_j) within (city, tile_i, tile_j); density_bucket is the conditioned stratum dim."""
    city: str
    tile_i: int
    tile_j: int
    cell_i: int
    cell_j: int
    density_bucket: int


def _cell_sort_key(c: SampledCell) -> tuple:
    return (c.city, c.tile_i, c.tile_j, c.cell_i, c.cell_j)


def _rank_digest(seed: int, c: SampledCell) -> str:
    raw = f"{seed}:{c.city}:{c.tile_i}:{c.tile_j}:{c.cell_i}:{c.cell_j}"
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()


def select_cells(cells: list[SampledCell], n: int, *, seed: int) -> list[SampledCell]:
    """Deterministically select <= n cells by blake2b hash-rank of the cell identity.

    stdlib hashlib is byte-stable across Python/numpy versions (a seeded numpy shuffle is
    not) and order-independent (the digest is a total order), so the result is reproducible
    for a sha-locked write-once manifest. Take-all when n >= len(cells). Output is sorted
    canonically (not by digest) so the manifest bytes are stable."""
    if n >= len(cells):
        chosen = list(cells)
    else:
        chosen = sorted(cells, key=lambda c: _rank_digest(seed, c))[:n]
    return sorted(chosen, key=_cell_sort_key)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_lane_s_sampler.py -k "select_cells" -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Add the cold-process PYTHONHASHSEED determinism test**

```python
# add to tests/eval/test_lane_s_sampler.py
import os
import subprocess
import sys


def test_select_cells_stable_across_pythonhashseed(tmp_path):
    snippet = (
        "from cfm.eval import lane_s_sampler as ls\n"
        "cells=[ls.SampledCell('glasgow',i,0,i%7,i//7,1) for i in range(200)]\n"
        "out=ls.select_cells(cells,40,seed=7)\n"
        "print(';'.join(f'{c.tile_i},{c.cell_i},{c.cell_j}' for c in out))\n"
    )
    outs = []
    for hs in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hs}
        r = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1] == outs[2], "selection drifted across PYTHONHASHSEED"
```

- [ ] **Step 6: Run it**

Run: `uv run pytest tests/eval/test_lane_s_sampler.py::test_select_cells_stable_across_pythonhashseed -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/cfm/eval/lane_s_sampler.py tests/eval/test_lane_s_sampler.py
git commit -m "feat(cell-eos): lane-s sampler blake2b hash-rank selection (determinism-tested)"
```

---

## Task 4: Floor adapter — floored targets + held-out feature counts

**Files:**
- Modify: `src/cfm/eval/lane_s_sampler.py`
- Test: `tests/eval/test_lane_s_sampler.py` (fixture) + `tests/eval/test_lane_s_sampler_real_floor.py` (real)

- [ ] **Step 1: Write the failing fixture test**

```python
# add to tests/eval/test_lane_s_sampler.py
def _floor_payload():
    # Two held-out cities so a family-1 D-D pair exists per stratum. stratum lists per the floor.
    S = ["R", "S1", 1, "inland"]
    return {
        "held_out_cities": ["glasgow", "krakow"],
        "floors": [
            {"city": "glasgow", "metric": ls.BUILDING_METRIC, "stratum": S},
            {"city": "glasgow", "metric": ls.ROAD_METRIC, "stratum": S},
            {"city": "krakow", "metric": ls.ROAD_METRIC, "stratum": S},
        ],
        "pairs": [
            {"city_a": "glasgow", "city_b": "krakow", "metric": ls.BUILDING_METRIC,
             "stratum": S, "n_a": 59, "n_b": 120},
            {"city_a": "glasgow", "city_b": "krakow", "metric": ls.ROAD_METRIC,
             "stratum": S, "n_a": 800, "n_b": 950},
        ],
        "cross_pairs": [],
    }


def test_floored_targets_groups_owed_metrics_and_binding():
    targets = ls.floored_targets(_floor_payload())
    assert ("glasgow", ("R", "S1", 1, "inland")) in targets
    g = targets[("glasgow", ("R", "S1", 1, "inland"))]
    assert g.owed_metrics == frozenset({ls.BUILDING_METRIC, ls.ROAD_METRIC})
    assert g.binding_metric == ls.BUILDING_METRIC
    k = targets[("krakow", ("R", "S1", 1, "inland"))]
    assert k.owed_metrics == frozenset({ls.ROAD_METRIC}) and k.binding_metric == ls.ROAD_METRIC


def test_heldout_feature_counts_reads_n_from_pairs():
    counts = ls.heldout_feature_counts(_floor_payload())
    assert counts[("glasgow", ls.BUILDING_METRIC, ("R", "S1", 1, "inland"))] == 59
    assert counts[("krakow", ls.ROAD_METRIC, ("R", "S1", 1, "inland"))] == 950
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_lane_s_sampler.py -k "floored_targets or heldout_feature_counts" -v`
Expected: FAIL — `AttributeError: ... 'floored_targets'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/cfm/eval/lane_s_sampler.py

@dataclass(frozen=True)
class FlooredTarget:
    city: str
    stratum: tuple
    owed_metrics: frozenset[str]
    binding_metric: str


def floored_targets(floor_payload: dict) -> dict[tuple[str, tuple], FlooredTarget]:
    """Per (city, 4-tuple) targeted stratum: the owed floored metrics + the binding metric.
    Target set = the distinct (city, 4-tuple) carrying >= 1 floored metric row (spec Gate 1)."""
    owed: dict[tuple[str, tuple], set[str]] = {}
    for rec in floor_payload["floors"]:
        key = (rec["city"], tuple(rec["stratum"]))
        owed.setdefault(key, set()).add(rec["metric"])
    return {
        (city, stratum): FlooredTarget(
            city=city, stratum=stratum, owed_metrics=frozenset(ms),
            binding_metric=binding_metric(frozenset(ms)),
        )
        for (city, stratum), ms in owed.items()
    }


def heldout_feature_counts(floor_payload: dict) -> dict[tuple[str, str, tuple], int]:
    """Per (held-out city, metric, stratum): the real feature count n, read from the floor's
    pair records (n_a/n_b) across BOTH families. These ARE the floor's qualify counts (same
    extraction), so n >= min_n for any floored stratum. Equals the optimistic gen ceiling at
    gen_ratio=1, full draw (= available_cells * real_fpc). Source: conditioning_floor n_a/n_b."""
    held = set(floor_payload["held_out_cities"])
    out: dict[tuple[str, str, tuple], int] = {}
    for table in ("pairs", "cross_pairs"):
        for p in floor_payload.get(table, []):
            stratum = tuple(p["stratum"])
            for city, n in ((p["city_a"], p["n_a"]), (p["city_b"], p["n_b"])):
                if city in held:
                    out[(city, p["metric"], stratum)] = int(n)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_lane_s_sampler.py -k "floored_targets or heldout_feature_counts" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Write the external-source-of-truth test against the REAL locked floor (protocol Gate 6)**

```python
# tests/eval/test_lane_s_sampler_real_floor.py
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cfm.eval import lane_s_sampler as ls

FLOOR = Path("reports/conditioning_floor/2026-04-15.0/conditioning-floor.yaml")


@pytest.mark.slow
def test_real_floor_target_shape_and_min_building_n():
    payload = yaml.safe_load(FLOOR.read_text())
    # LOCK-AND-GUARDS-TRAVEL-TOGETHER: a floor re-derivation that changes n_a/n_b changes this
    # sha -> this assertion goes RED, forcing EXPECTED_FLOOR_SHA256 + the guard to update with it.
    assert payload["floor_sha256"] == ls.EXPECTED_FLOOR_SHA256
    targets = ls.floored_targets(payload)
    assert len(targets) == 146  # distinct (city, 4-tuple) floored 4-tuples (spec Gate 1)
    both = [t for t in targets.values() if t.binding_metric == ls.BUILDING_METRIC]
    assert len(both) == 119  # building_area owed (binds) in 119 of 146
    counts = ls.heldout_feature_counts(payload)
    building_ns = [counts[(t.city, ls.BUILDING_METRIC, t.stratum)] for t in both]
    assert min(building_ns) == 59  # R3: min real building feature count


@pytest.mark.slow
def test_real_floor_reproduces_R3_ceiling_bound_counts():
    payload = yaml.safe_load(FLOOR.read_text())
    counts = ls.heldout_feature_counts(payload)
    both = [t for t in ls.floored_targets(payload).values() if t.binding_metric == ls.BUILDING_METRIC]
    ns = [counts[(t.city, ls.BUILDING_METRIC, t.stratum)] for t in both]
    # ceiling_bound <=> floor_n < target*headroom (available cancels). R3 census numbers:
    assert sum(1 for n in ns if n < 50 * 1.0) == 0
    assert sum(1 for n in ns if n < 50 * 2.0) == 6
```

- [ ] **Step 6: Run it**

Run: `uv run pytest tests/eval/test_lane_s_sampler_real_floor.py -v -m slow`
Expected: PASS (2 passed) — these confirm the adapter against the real floor and re-derive R3.

- [ ] **Step 7: Commit**

```bash
git add src/cfm/eval/lane_s_sampler.py tests/eval/test_lane_s_sampler.py tests/eval/test_lane_s_sampler_real_floor.py
git commit -m "feat(cell-eos): lane-s floor adapter (floored targets + n_a/n_b feature counts) + real-floor SoT test"
```

---

## Task 5: Per-stratum cell-key census — emit + read

**Files:**
- Modify: `src/cfm/eval/lane_s_sampler.py` (census write/read helpers — testable)
- Modify: `scripts/_heldout_cell_count.py` (add `--emit <path>` calling the helper)
- Test: `tests/eval/test_lane_s_sampler.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/eval/test_lane_s_sampler.py
def test_cell_census_round_trips_grouped_by_city_4tuple(tmp_path):
    rows = [
        ls.SampledCell("glasgow", 1, 2, 0, 0, 1),
        ls.SampledCell("glasgow", 1, 2, 0, 1, 1),
        ls.SampledCell("krakow", 3, 4, 5, 6, 2),
    ]
    strata = {  # (city, tile_i, tile_j) -> (zoning, skeleton, coastal)
        ("glasgow", 1, 2): ("R", "S1", "inland"),
        ("krakow", 3, 4): ("C", "S2", "coastal"),
    }
    path = tmp_path / "census.parquet"
    ls.write_cell_census(rows, strata, path)
    pool = ls.read_cell_census(path)
    # grouped by (city, 4-tuple); density from the cell, (zoning,skeleton,coastal) from the tile
    assert len(pool[("glasgow", ("R", "S1", 1, "inland"))]) == 2
    assert pool[("krakow", ("C", "S2", 2, "coastal"))][0].cell_i == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_lane_s_sampler.py -k "cell_census_round_trips" -v`
Expected: FAIL — `AttributeError: ... 'write_cell_census'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/cfm/eval/lane_s_sampler.py
# NOTE pyarrow import is local to these two fns so the pure-logic core stays import-light.

#: census parquet column order (sorted-row determinism relies on this fixed schema).
_CENSUS_COLS = ("city", "tile_i", "tile_j", "cell_i", "cell_j", "zoning", "skeleton",
                "density", "coastal")


def write_cell_census(
    cells: list[SampledCell],
    tile_strata: dict[tuple[str, int, int], tuple],
    path,
) -> None:
    """Write the per-cell census parquet: one row per conditionable held-out cell, carrying
    the cell's density and its tile's (zoning, skeleton, coastal). Rows sorted canonically for
    byte-determinism. ``tile_strata[(city, ti, tj)] = (zoning, skeleton, coastal)``."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for c in cells:
        z, sk, co = tile_strata[(c.city, c.tile_i, c.tile_j)]
        rows.append((c.city, c.tile_i, c.tile_j, c.cell_i, c.cell_j, z, sk, c.density_bucket, co))
    rows.sort()  # canonical order => deterministic bytes
    cols = {name: [r[i] for r in rows] for i, name in enumerate(_CENSUS_COLS)}
    table = pa.table({k: cols[k] for k in _CENSUS_COLS})
    pq.write_table(table, str(path))


def read_cell_census(path) -> dict[tuple[str, tuple], list[SampledCell]]:
    """Read the census back, grouped by (city, 4-tuple stratum). The 4-tuple is
    (zoning, skeleton, density, coastal) — byte-for-byte the floor's grammar."""
    import pyarrow.parquet as pq

    tbl = pq.ParquetFile(str(path)).read()
    col = {n: tbl.column(n).to_pylist() for n in tbl.column_names}
    pool: dict[tuple[str, tuple], list[SampledCell]] = {}
    for i in range(tbl.num_rows):
        stratum = (col["zoning"][i], col["skeleton"][i], int(col["density"][i]), col["coastal"][i])
        cell = SampledCell(col["city"][i], int(col["tile_i"][i]), int(col["tile_j"][i]),
                           int(col["cell_i"][i]), int(col["cell_j"][i]), int(col["density"][i]))
        pool.setdefault((col["city"][i], stratum), []).append(cell)
    return pool
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_lane_s_sampler.py -k "cell_census_round_trips" -v`
Expected: PASS.

- [ ] **Step 5: Wire the emit mode into `_heldout_cell_count.py`**

Modify `scripts/_heldout_cell_count.py`. Inside `main()`, accumulate cells while the existing loop runs, then emit if requested. Add after the imports:

```python
import argparse

from cfm.eval.lane_s_sampler import SampledCell, write_cell_census
```

Replace the `by_stratum`/`by_city_stratum` accumulation block (lines 42-78) so it also collects rows and tile strata, and add the emit at the end of `main()`:

```python
    cell_rows: list[SampledCell] = []
    tile_strata: dict[tuple[str, int, int], tuple] = {}
    # ... inside the per-tile loop, after computing zoning/skeleton/coastal/cdbc/tokens:
            tile_strata[(city, ti, tj)] = (zoning, skeleton, coastal)
            for (ci, cj), toks in tokens.items():
                if not toks:
                    empty += 1
                    continue
                nonempty += 1
                density = cdbc.get((ci, cj), -1)
                if density is None:
                    density = -1
                stratum = (zoning, skeleton, density, coastal)
                by_stratum[stratum] += 1
                by_city_stratum[(city, stratum)] += 1
                cell_rows.append(SampledCell(city, ti, tj, ci, cj, int(density)))
    # ... after the existing prints, before `return 0`:
    args = _parse_args()
    if args.emit:
        write_cell_census(cell_rows, tile_strata, args.emit)
        print(f"census parquet emitted: {args.emit}  rows={len(cell_rows)}")
```

Add an arg parser:

```python
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="held-out per-stratum cell census")
    ap.add_argument("--emit", default=None, help="write the per-cell census parquet to this path")
    return ap.parse_args()
```

- [ ] **Step 6: Lint + verify the script still imports**

Run: `uv run ruff check scripts/_heldout_cell_count.py src/cfm/eval/lane_s_sampler.py && uv run python -c "import ast; ast.parse(open('scripts/_heldout_cell_count.py').read())"`
Expected: no lint errors; parse OK. (Full run is a Leonardo job — Task 8; do NOT run it against real data here.)

- [ ] **Step 7: Commit**

```bash
git add src/cfm/eval/lane_s_sampler.py scripts/_heldout_cell_count.py tests/eval/test_lane_s_sampler.py
git commit -m "feat(cell-eos): per-stratum cell-key census emit/read (extend _heldout_cell_count)"
```

---

## Task 6: Build orchestrator — manifest assembly

**Files:**
- Modify: `src/cfm/eval/lane_s_sampler.py` (`build_manifest`)
- Create: `scripts/build_lane_s_sampler.py`
- Test: `tests/eval/test_lane_s_sampler.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/eval/test_lane_s_sampler.py
def test_build_manifest_sizes_and_selects_per_stratum(tmp_path):
    floor = _floor_payload()  # glasgow: building n=59 + road n=800; krakow: road n=950
    S = ("R", "S1", 1, "inland")
    pool = {
        ("glasgow", S): [ls.SampledCell("glasgow", 0, 0, i % 9, i // 9, 1) for i in range(40)],
        ("krakow", S): [ls.SampledCell("krakow", 0, 0, i % 9, i // 9, 1) for i in range(500)],
    }
    payload = ls.build_manifest(
        floor_payload=floor, floor_sha256="abc123", cell_pool=pool, release="test.0",
        seed=7, target_features=50, headroom=2.0,
    )
    by_key = {(s["city"], tuple(s["stratum"])): s for s in payload["strata"]}
    g = by_key[("glasgow", S)]
    # glasgow building: target*headroom=100 > 59 -> ceiling-bound -> take all 40
    assert g["binding_metric"] == ls.BUILDING_METRIC and g["ceiling_bound"] is True
    assert g["n_cells_selected"] == 40
    k = by_key[("krakow", S)]
    # krakow road n=950: raw=ceil(50*2*500/950)=ceil(52.6)=53 -> not ceiling-bound
    assert k["binding_metric"] == ls.ROAD_METRIC and k["ceiling_bound"] is False
    assert k["n_cells_selected"] == 53
    # cells[] holds exactly the selected union
    assert len(payload["cells"]) == 40 + 53
    assert payload["floor_sha256"] == "abc123"


def test_build_manifest_skips_strata_absent_from_pool(tmp_path, caplog):
    floor = _floor_payload()
    payload = ls.build_manifest(floor_payload=floor, floor_sha256="abc", cell_pool={},
                               release="test.0", seed=7, target_features=50, headroom=2.0)
    assert payload["strata"] == [] and payload["cells"] == []
    assert "no census cells" in caplog.text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_lane_s_sampler.py -k "build_manifest" -v`
Expected: FAIL — `AttributeError: ... 'build_manifest'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/cfm/eval/lane_s_sampler.py

def build_manifest(
    *,
    floor_payload: dict,
    floor_sha256: str,
    cell_pool: dict[tuple[str, tuple], list[SampledCell]],
    release: str,
    seed: int,
    target_features: int = DEFAULT_TARGET_FEATURES,
    headroom: float = DEFAULT_HEADROOM,
) -> dict:
    """Assemble the (unsealed) manifest payload: size + select per floored (city, 4-tuple)."""
    targets = floored_targets(floor_payload)
    counts = heldout_feature_counts(floor_payload)
    strata_records: list[dict] = []
    all_cells: list[SampledCell] = []
    for (city, stratum) in sorted(targets, key=lambda k: (k[0], tuple(map(str, k[1])))):
        t = targets[(city, stratum)]
        available = cell_pool.get((city, stratum), [])
        if not available:
            logger.warning("lane-s sampler: no census cells for floored stratum %s %s; skipping "
                           "(census/floor lineage mismatch)", city, stratum)
            continue
        floor_n = counts[(city, t.binding_metric, stratum)]
        sizing = size_stratum(target_features=target_features, headroom=headroom,
                              floor_n_binding=floor_n, available_cells=len(available))
        chosen = select_cells(available, sizing.n_cells_selected, seed=seed)
        all_cells.extend(chosen)
        strata_records.append({
            "city": city,
            "stratum": list(stratum),
            "owed_metrics": sorted(t.owed_metrics),
            "binding_metric": t.binding_metric,
            "floor_n_binding": floor_n,
            "available_cells": len(available),
            "real_fpc_binding": floor_n / len(available),
            "n_cells_target": sizing.n_cells_target,
            "n_cells_selected": sizing.n_cells_selected,
            "ceiling_bound": sizing.ceiling_bound,
        })
    all_cells.sort(key=_cell_sort_key)
    return {
        "sampler_schema_version": SAMPLER_SCHEMA_VERSION,
        "release": release,
        "floor_sha256": floor_sha256,
        "methodology": {
            "target_features": target_features,
            "headroom": headroom,
            "seed": seed,
            "selection": "blake2b_hash_rank",
            "sizing": "ceil(target_features * headroom * available / floor_n_binding)",
            "binding_rule": "scarce_floored_metric_building_area_else_road_length",
            "real_fpc_source": "heldout_floor_n_a_n_b_div_census_cells",
            "gen_ratio": "training_city_informed_proxy_validated_at_first_generation",
        },
        "held_out_cities": sorted(floor_payload["held_out_cities"]),
        "strata": strata_records,
        "cells": [
            {"city": c.city, "tile_i": c.tile_i, "tile_j": c.tile_j,
             "cell_i": c.cell_i, "cell_j": c.cell_j, "density_bucket": c.density_bucket}
            for c in all_cells
        ],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_lane_s_sampler.py -k "build_manifest" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Write the build CLI**

```python
# scripts/build_lane_s_sampler.py
"""Build + seal the Lane-S cell sampler manifest (CPU-only; read-only inputs).

Inputs: the locked conditioning floor (verified-load for lineage) + the per-stratum cell
census parquet (Task 5 emit). Output: a sha-locked sampler-manifest.yaml + marker. Prints the
§6 cost re-derivation. NO generation, NO GPU. Gated on PI word + Leonardo redeploy.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from cfm.eval.conditioning_floor import load_verified_floor
from cfm.eval.lane_s_sampler import (
    EXPECTED_FLOOR_SHA256,
    build_manifest,
    load_verified_manifest,
    read_cell_census,
    seal_manifest,
)

PER_CELL_GPU_H_TRANSFORMER = 0.0045  # GROUND_TRUTH §3 (~600-tok self-terminated, 4-GPU-sharded)
MATRIX_RUNS = 6  # 2 backbones x 3 seeds (one manifest consumed by all)


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", required=True, type=Path)
    ap.add_argument("--census", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--release", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--target-features", type=int, default=50)
    ap.add_argument("--headroom", type=float, default=2.0)
    args = ap.parse_args()

    verified = load_verified_floor(args.floor)  # raises if lineage broken (internal sha)
    floor_payload = verified.payload
    floor_sha = str(floor_payload["floor_sha256"])
    if floor_sha != EXPECTED_FLOOR_SHA256:
        raise SystemExit(
            f"floor sha {floor_sha} != pinned {EXPECTED_FLOOR_SHA256} — the sampler is locked to "
            "the 95abb88 floor (floor_n READ, never recomputed). A re-derived floor must update "
            "EXPECTED_FLOOR_SHA256 + the Task-4 SoT test in the SAME commit (lock-and-guards-"
            "travel-together); refusing to build against an unverified floor lineage."
        )
    pool = read_cell_census(args.census)

    payload = build_manifest(
        floor_payload=floor_payload, floor_sha256=floor_sha, cell_pool=pool,
        release=args.release, seed=args.seed,
        target_features=args.target_features, headroom=args.headroom,
    )
    seal_manifest(payload, args.out)
    loaded = load_verified_manifest(args.out)  # prove it reads back verified

    total = sum(s["n_cells_selected"] for s in loaded["strata"])
    ceil_bound = sum(1 for s in loaded["strata"] if s["ceiling_bound"])
    gpu_h = total * MATRIX_RUNS * PER_CELL_GPU_H_TRANSFORMER
    print(f"=== Lane-S sampler built: {args.out} ===")
    print(f"strata={len(loaded['strata'])}  cells_selected={total}  ceiling_bound={ceil_bound}")
    print(f"generations={total * MATRIX_RUNS} (x{MATRIX_RUNS} runs)")
    print(f"est transformer GPU-h={gpu_h:.1f}  (~{100*gpu_h/5000:.1f}% of 5,000 GPU-h grant); "
          f"MAMBA RATE UNVERIFIED — measure at next GPU smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Lint + import check (do NOT run against real data)**

Run: `uv run ruff check scripts/build_lane_s_sampler.py && uv run python -c "import ast; ast.parse(open('scripts/build_lane_s_sampler.py').read())"`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/cfm/eval/lane_s_sampler.py scripts/build_lane_s_sampler.py tests/eval/test_lane_s_sampler.py
git commit -m "feat(cell-eos): lane-s manifest build orchestrator + CLI with cost re-derivation"
```

---

## Task 7: Consumer-side coverage check — the §9 ceiling-bound split

**Files:**
- Modify: `src/cfm/eval/lane_s_sampler.py`
- Test: `tests/eval/test_lane_s_sampler.py`

- [ ] **Step 1: Write the failing test (incl. the regime-distinguishing guard)**

```python
# add to tests/eval/test_lane_s_sampler.py
def _manifest_with(stratum, ceiling_bound, owed, binding):
    return {
        "methodology": {"target_features": 50},
        "strata": [{"city": "glasgow", "stratum": list(stratum), "owed_metrics": sorted(owed),
                    "binding_metric": binding, "ceiling_bound": ceiling_bound}],
    }


def test_coverage_ok_when_all_metrics_meet_min_n():
    S = ("R", "S1", 1, "inland")
    man = _manifest_with(S, False, {ls.BUILDING_METRIC, ls.ROAD_METRIC}, ls.BUILDING_METRIC)
    gen = {"glasgow": {(ls.BUILDING_METRIC, S): [1.0] * 60, (ls.ROAD_METRIC, S): [1.0] * 400}}
    report = ls.verify_gen_coverage(gen, man)
    assert report.unexpected_short == [] and report.ceiling_bound_excluded == []
    assert report.ok == [("glasgow", ls.BUILDING_METRIC, S), ("glasgow", ls.ROAD_METRIC, S)]


def test_coverage_ceiling_bound_short_is_excluded_and_reported():
    S = ("R", "S1", 1, "inland")
    man = _manifest_with(S, True, {ls.BUILDING_METRIC, ls.ROAD_METRIC}, ls.BUILDING_METRIC)
    gen = {"glasgow": {(ls.BUILDING_METRIC, S): [1.0] * 30, (ls.ROAD_METRIC, S): [1.0] * 400}}
    report = ls.verify_gen_coverage(gen, man)  # building short but ceiling-bound -> report, no raise
    assert report.ceiling_bound_excluded == [("glasgow", ls.BUILDING_METRIC, S)]
    assert report.unexpected_short == []


def test_coverage_NOT_ceiling_bound_short_FAILS_LOUD():
    # The regime-distinguishing guard: same SYMPTOM (building short) but NOT ceiling-bound ->
    # sampler under-sized -> must raise. A symptom-keyed "skip if thin" would wrongly pass this.
    S = ("R", "S1", 1, "inland")
    man = _manifest_with(S, False, {ls.BUILDING_METRIC, ls.ROAD_METRIC}, ls.BUILDING_METRIC)
    gen = {"glasgow": {(ls.BUILDING_METRIC, S): [1.0] * 30, (ls.ROAD_METRIC, S): [1.0] * 400}}
    with pytest.raises(ls.SamplerCoverageError, match="not ceiling-bound"):
        ls.verify_gen_coverage(gen, man)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_lane_s_sampler.py -k "coverage" -v`
Expected: FAIL — `AttributeError: ... 'verify_gen_coverage'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/cfm/eval/lane_s_sampler.py

class SamplerCoverageError(RuntimeError):
    """A floored (metric, stratum) is below min_n on the GENERATED side WITHOUT being
    ceiling-bound — a sampler sizing / headroom bug, never hidden behind the ceiling
    exclusion (spec Gate 5 / protocol §9 regime-distinguishing guard)."""


@dataclass(frozen=True)
class CoverageReport:
    ok: list[tuple[str, str, tuple]]
    ceiling_bound_excluded: list[tuple[str, str, tuple]]  # data limit: report, drop from Lane-S
    unexpected_short: list[tuple[str, str, tuple]]        # always empty on return (we raise first)


def verify_gen_coverage(
    gen_by_city: dict[str, dict[tuple[str, tuple], list]],
    manifest: dict,
    *,
    min_n: int | None = None,
) -> CoverageReport:
    """Per floored (city, metric, stratum) in the manifest: assert achieved gen features >=
    min_n on the ACTUAL generated set (spec Gate 5, protocol §10.3 correct unit).

    §9 split: a short metric that is the binding metric of a CEILING-BOUND stratum is a data
    limit -> exclude-and-report (mirrors the floor's 'report, do NOT coarsen'; #21 demotion +
    SECOND_REGION downstream). Any other short -> FAIL LOUD (sampler under-sized)."""
    min_n = manifest["methodology"]["target_features"] if min_n is None else min_n
    ok: list = []
    excluded: list = []
    for s in manifest["strata"]:
        city, stratum = s["city"], tuple(s["stratum"])
        binding, ceiling = s["binding_metric"], bool(s["ceiling_bound"])
        for metric in s["owed_metrics"]:
            key = (city, metric, stratum)
            achieved = len(gen_by_city.get(city, {}).get((metric, stratum), []))
            if achieved >= min_n:
                ok.append(key)
            elif metric == binding and ceiling:
                excluded.append(key)
            else:
                raise SamplerCoverageError(
                    f"lane-s coverage: {key} has {achieved} gen features < min_n={min_n} but the "
                    f"stratum is not ceiling-bound for this metric (binding={binding}, "
                    f"ceiling_bound={ceiling}) — the sampler under-sized it; re-derive headroom, "
                    "do not exclude. (spec Gate 5 / protocol §9)"
                )
    return CoverageReport(ok=ok, ceiling_bound_excluded=excluded, unexpected_short=[])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_lane_s_sampler.py -k "coverage" -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the whole fast suite + lint**

Run: `uv run pytest tests/eval/test_lane_s_sampler.py -v && uv run ruff check src/cfm/eval/lane_s_sampler.py && uv run ruff format --check src/cfm/eval/lane_s_sampler.py`
Expected: all green; lint + format clean.

- [ ] **Step 6: Commit**

```bash
git add src/cfm/eval/lane_s_sampler.py tests/eval/test_lane_s_sampler.py
git commit -m "feat(cell-eos): lane-s consumer coverage check (§9 ceiling-bound split, regime-guarded)"
```

---

## Task 8: Leonardo build job (no GPU) — ops wiring (NOT run here)

**Files:**
- Create: `scripts/build_lane_s_sampler.sbatch`

- [ ] **Step 1: Write the sbatch**

Mirror the existing serial ops sbatch pattern (`scripts/heldout_cell_count.sbatch`): `--account=AIFAC_P02_548`, `--partition=lrd_all_serial`, no GPU, the gcc-12 `libstdc++` `LD_PRELOAD` only if a torch/mamba import is pulled (it is not — keep it out unless an import error appears).

```bash
#!/bin/bash
#SBATCH --account=AIFAC_P02_548
#SBATCH --partition=lrd_all_serial
#SBATCH --job-name=lane_s_sampler
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/lane_s_sampler_%j.out

set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
RELEASE=2026-04-15.0
OUTDIR=data/processed/lane_s_sampler/$RELEASE
mkdir -p "$OUTDIR" logs

# 1. per-stratum cell-key census (read-only over held-out tiles)
uv run python scripts/_heldout_cell_count.py --emit "$OUTDIR/heldout-cell-census.parquet"

# 2. build + seal the manifest (verified-load floor for lineage)
uv run python scripts/build_lane_s_sampler.py \
  --floor reports/conditioning_floor/$RELEASE/conditioning-floor.yaml \
  --census "$OUTDIR/heldout-cell-census.parquet" \
  --out "$OUTDIR/sampler-manifest.yaml" \
  --release "$RELEASE" --seed 7 --target-features 50 --headroom 2.0
```

- [ ] **Step 2: Shellcheck / sanity (no submit)**

Run: `bash -n scripts/build_lane_s_sampler.sbatch`
Expected: no syntax error. **Do NOT submit** — submission is gated on PI word + the GROUND_TRUTH §5 Leonardo redeploy (Leonardo must be at the Mac's committed HEAD, incl. this branch).

- [ ] **Step 3: Commit**

```bash
git add scripts/build_lane_s_sampler.sbatch
git commit -m "chore(cell-eos): Leonardo no-GPU sbatch for lane-s sampler census+build"
```

---

## Pre-build addendum (PI ratifications 2026-06-22 — docs-only now; code at PI word + Leonardo redeploy)

Two spec ratifications landed (spec §7 + §10 R5). They imply ONE small pre-build code step and one logged gap — neither implemented yet (Tasks 1–7 are frozen green; this is gated build prep):

1. **Wire `census_sha256` into the manifest (small code task, land BEFORE the first manifest seals).** The manifest must pin the EXACT cell pool via `census_sha256` (chosen over `source_holdout_manifest` because the 1,952-tile pool = 4 per-city manifests → ambiguous ref; spec §7). Change:
   - `scripts/build_lane_s_sampler.py`: after reading the census, compute `census_sha = compute_sha256(args.census.read_bytes())` (`from cfm.data.determinism import compute_sha256`) and pass it to `build_manifest`.
   - `build_manifest(...)`: add a `census_sha256: str` param; record `"census_sha256": census_sha256` as a top-level payload key (beside `floor_sha256`).
   - Test: extend `test_build_manifest_*` to assert the passed `census_sha256` round-trips into the payload. (TDD, same discipline as Tasks 1–7.)
   No artifact exists yet, so adding this field invalidates nothing.

2. **R5 (logged, NO code change):** the §9 `else` fail-loud branch also catches a ceiling-bound stratum's NON-binding metric short (a data limit, not a sampler bug). UNREACHABLE under R3 at `headroom ≤ 2.0`. The `verify_gen_coverage` comment (commit `5e80f1d`) carries the note. **Activation condition: revisit before raising `headroom` > 2.0** — PI then decides widen-exclusion vs keep-loud. No speculative handling.

3. **Naming-seam note (generation-loop wiring, no change now):** manifest `cells[].density_bucket` → `DecodedCell.cell_density_bucket` (`gen_realism.py:51`).

## Post-build (gated — execute only on PI word, after Leonardo redeploy)

These are NOT plan tasks (no code). Recorded so the executor does not improvise:
1. ✅ **DONE 2026-06-22** — `build_lane_s_sampler.sbatch` submitted (Leonardo job `47600021`, HEAD `69bca37`, serial no-GPU, COMPLETED 2:37). Manifest sealed; `load_verified_manifest` round-trips; `floor_sha256`=`95abb88…` (==EXPECTED), `census_sha256`=`236cea99…` recomputes byte-for-byte from the on-disk census parquet. 146/146 floored strata built, 5,705 cells, ~154 transformer GPU-h ≈ 3.1% of grant (mamba rate still unverified).
2. **`ceiling_bound = 10` is EXPECTED** (NOT "≤6"). The R3 census only ever counted **building_area-floored** ceiling-bound (6 @ headroom 2.0); the manifest counts ceiling-bound across **all** binding metrics, so the correct expectation is **6 building + 4 road-only = 10**:
   - **6 building-floored** reproduce R3 *exactly* by city — glasgow 4 / munich 1 / eisenhüttenstadt 1 / krakow 0 (all building `floor_n` ∈ [59,93] < 100).
   - **4 road-only** (one per city) — eisenh `floor_n`=59/avail=1, glasgow 57/8, krakow 69/2, munich 73/29 — each verified a **legit data limit** (`n_cells_target` > `available_cells`, took all available; `floor_n` < target×headroom=100), routing to §9 **exclude-and-report** (confirmed read-only via `verify_gen_coverage` + the regime-distinguishing guard, 2026-06-22).
   - Action contract: at headroom 2.0 this is the expected count — do **not** revisit `headroom`/`target_features`. Halt only if a *building-floored* ceiling-bound count exceeds 6, or a *not-ceiling-bound* stratum shorts at generation (`verify_gen_coverage` fails loud).
3. The matrix generation loop consumes `manifest.cells[]` → `gen_realism.gen_features_by_city` → `verify_gen_coverage` → `decide()`. Still gated: no scored run without PI word.

---

## Self-review

**Spec coverage:**
- Gate 1 (target set / per-metric obligation) → Task 4 (`floored_targets`, binding) + Task 6.
- Gate 2 (1,952 pool) → Task 5 census iterates the holdout manifest tiles (the 1,952 set, via `_heldout_cell_count.py`'s existing `holdout_manifest_for_region` loop).
- Gate 3 (bridge, scarce-metric, real_fpc from held-out, gen_ratio proxy) → Task 2 (sizing) + Task 4 (`heldout_feature_counts`) + Task 6 (`real_fpc_binding` recorded; `gen_ratio` proxy noted in methodology).
- Gate 4 (blake2b hash-rank, determinism, one manifest) → Task 3.
- Gate 5 / §9 (feasibility split, fail-loud guard) → Task 7.
- §6 (N/headroom config + §10.1 + cost) → Tasks 2/6 (config args; methodology records achieved property; build prints cost).
- §7 (sha-locked manifest + regen script + schema) → Tasks 1/6/8.
- §8 (work-owed: census emit; real_fpc) → Task 5 (census), Task 4 (real_fpc from floor `n`, parquet avoided), Task 8 (Leonardo).
- §9 test plan → determinism (T3), scarce-binding (T2), §9 guard (T7), lock grammar (T1), external-SoT (T4 real-floor).

**Placeholder scan:** none — every code/test step has runnable content; deferred items (mamba rate, the actual Leonardo run) are explicitly gated, not stubbed.

**Type consistency:** `SampledCell(city, tile_i, tile_j, cell_i, cell_j, density_bucket)` used identically in Tasks 3/5/6; `FlooredTarget.binding_metric` / `owed_metrics` consistent T4↔T6↔T7; `size_stratum(... floor_n_binding, available_cells)` signature matches its T6 call; `verify_gen_coverage` reads `methodology.target_features` written in T6.

**Deviation from spec §8 — RATIFIED + LOCKED (PI 2026-06-21):** `real_fpc` is sourced from the floor's `n_a`/`n_b` (local) ÷ census cells, NOT the 29 MB parquet — the parquet's per-stratum feature counts are identical to the floor's qualify counts (same extraction), so the parquet dependency is dropped. `available_cells` cancels in the ceiling-bound flag, so the R3-proven feasibility logic is fully local-testable; Leonardo shrinks to census-only. Guarded by the floor-sha invariant above (Task 1 constant + Task 6 CLI fail-loud + Task 4 SoT test).

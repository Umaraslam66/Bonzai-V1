# Eval-set-generation (multi-region) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the multi-region held-out evaluation set + the net-new macro-plan-coherence metric + the de-Singapore generalizations, implementing exactly `docs/superpowers/specs/2026-06-08-eval-set-gen-design.md` — nothing added, nothing reinterpreted.

**Architecture:** Generalize the Singapore-shaped eval machinery (`src/cfm/eval/holdout/*`, `src/cfm/eval/resolution.py`) to a 4-city whole-city holdout. Each step maps to a spec section (cited per task). Teeth-proofs are **gating halt-steps** (red-before/green-after, block on failure), not reporting. The locked dependency order is honored: **T2 manifest + §2.2 correct-by-construction assertions land and verify BEFORE the leak-guard teeth** (the guard reads the manifest's whole-city declaration; a faithful guard over a wrong declaration passes its own teeth and still leaks). Unattended steps verify **actual end-state**, never an exit code (the false-completion soft spot).

**Tech Stack:** Python 3.11, pyarrow, pydantic/yaml, pytest. Corpus on Leonardo (`/leonardo_work/AIFAC_P02_222/Bonzai-OSM/data/processed`, release `2026-04-15.0`, DERIVATION 1.2).

**Execution environment:** code + synthetic-fixture unit tests run **local**; the manifest build, the usable-n measurement, and the real-data teeth (real-vs-permuted; assertion (b) tile-set match) read the frozen corpus and run **on Leonardo** (`.venv/bin/python`). Each Leonardo task notes it.

**Standing rules:** branch `phase-2-eval-set-gen` off `main`; no merge/push without PI word; `ruff format` + `ruff check` + `pytest` green before each commit (run unpiped — tool-output silence is not a pass).

---

## File structure

| file | responsibility | task |
|---|---|---|
| `src/cfm/eval/holdout/paths.py` | per-region CRS-label resolution; **distinct multi-region eval-set path** (the SG set already holds `eval_set/<rel>/`) | T1 |
| `src/cfm/eval/resolution.py` | KS-resolution **produces the resolved-gap NUMBER only** (generic escalation; does NOT own the coherence verdict/swap) | T2 |
| `src/cfm/eval/holdout/labels.py` | regression-lock: no SG constant scored (no logic change) | T3 |
| `src/cfm/eval/holdout/manifest.py` | multi-region builder + `holdout_kind` + achieved-props + §2.2 assertions + freeze (to the distinct path) | T4, T6 |
| `src/cfm/eval/holdout/macro_graph.py` (new) | **shared** interior-road-graph builder (interior 1..6, road {1,2,3}, endpoints) — ONE definition | T5 |
| `src/cfm/eval/usable_tiles.py` (new) | `tile_is_usable` = `len(interior_road_graph(rows)) >= 3` (consumes the shared builder) | T5 |
| `scripts/eval/measure_usable_tiles.py` | read-only usable-n per held-out city (gate b) | T5 |
| `src/cfm/eval/holdout/lineage_audit.py` | city-identity leak guard (scoped to whole_city) | T7 |
| `src/cfm/data/training/{holdout_guard,datamodule}.py` (re-point) | **wire** the loader audit to the multi-region manifest + real-run fail-closed test | **T8.5** (§6 trigger-1) |
| `src/cfm/eval/holdout/coherence.py` (new) | S1 metric — **imports `interior_road_graph` from `macro_graph`** (no re-definition) | T9, T11 |
| `src/cfm/eval/resolution.py` (sibling fn) | `assert_coherence_power_sufficient` — **sole** architecture-discrimination verdict; owns munich→manchester | T12 |
| test files mirror each under `tests/eval/...` | teeth + unit tests | per task |

---

## Phase A — de-Singapore plumbing (spec §5)

### Task 1: `paths.py` — per-region CRS label (spec §5)

**Files:**
- Modify: `src/cfm/eval/holdout/paths.py:19,31-33`
- Test: `tests/eval/holdout/test_paths.py`

- [ ] **Step 0: Re-verify current state** (memory claims this was done; source said otherwise — confirm before editing)

Run: `rg -n "_EPSG_LABEL|DEFAULT_REGION|epsg_label" src/cfm/eval/holdout/paths.py`
Expected: `_EPSG_LABEL = "EPSG3414"` still present. If already generalized, skip to Step 4 and adapt the test.

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/holdout/test_paths.py
from cfm.eval.holdout.paths import tile_dirname, epsg_label_for_region

def test_tile_dirname_uses_passed_label():
    assert tile_dirname(337, 2662, epsg_label="EPSG25832") == "tile=EPSG25832_i337_j2662"

def test_epsg_label_for_region_reads_region_config():
    # munich's region config has projected_crs: EPSG:25832 -> label EPSG25832
    assert epsg_label_for_region("munich") == "EPSG25832"
    assert epsg_label_for_region("krakow") == "EPSG25834"
```

- [ ] **Step 2: Run, verify it fails**

Run: `uv run pytest tests/eval/holdout/test_paths.py -v`
Expected: FAIL — `epsg_label_for_region` not defined.

- [ ] **Step 3: Implement**

```python
# paths.py — add (keep _EPSG_LABEL as the SG back-compat default for the frozen SG set)
import yaml

def _region_config_path(region: str) -> Path:
    return _repo_root() / "configs" / "data" / "regions" / f"{region}.yaml"

def epsg_label_for_region(region: str) -> str:
    """CRS label embedded in a region's sub-D tile dir names, from its config's
    projected_crs (e.g. 'EPSG:25832' -> 'EPSG25832')."""
    cfg = yaml.safe_load(_region_config_path(region).read_text(encoding="utf-8"))
    return cfg["projected_crs"].replace(":", "")

# Distinct multi-region eval-set dir — the SG eval-set already occupies
# eval_set/<release>/ (frozen 2026-06-01, write-once), so the EU set CANNOT reuse it.
def multiregion_eval_set_dir(release: str) -> Path:
    return eval_set_dir(release) / "multiregion"

def multiregion_holdout_manifest_path(release: str) -> Path:
    return multiregion_eval_set_dir(release) / "holdout_manifest.yaml"

def multiregion_eval_set_locked_marker(release: str) -> Path:
    return multiregion_eval_set_dir(release) / "_EVAL_SET_LOCKED"
```
(`tile_dirname` already takes `epsg_label` — no signature change; callers pass `epsg_label_for_region(region)`.)

> **SPEC-PATH ERRATUM (surfaced, not silently changed):** spec §2 wrote the manifest path as `eval_set/2026-04-15.0/holdout_manifest.yaml` — but the **frozen SG eval-set already occupies it** (same release; write-once). The EU set lives in the **`multiregion/` subdir** instead. Note this back to the spec on approval (a one-line path correction); the SG set is untouched (write-once preserved).

- [ ] **Step 4: Run, verify pass**

Run: `uv run pytest tests/eval/holdout/test_paths.py -v` → PASS.
(`projected_crs` field confirmed present in region configs, e.g. `configs/data/regions/munich.yaml`.)

- [ ] **Step 5: Commit**

```bash
git add src/cfm/eval/holdout/paths.py tests/eval/holdout/test_paths.py
git commit -m "feat(eval): per-region CRS label in holdout paths (de-Singapore, spec §5)"
```

### Task 2: `resolution.py` — multi-region escalation + train-split note (spec §5, §7)

**Files:**
- Modify: `src/cfm/eval/resolution.py:54-67` (escalation messages)
- Test: `tests/eval/test_resolution_seam.py`

> **REVIEW FLAG (do not reinterpret):** spec §7 unifies the resolution check with the coherence power gate, and §5 says "recompute the gap-floor on the TRAIN split ONLY." The spec does **not** pin whether the existing KS-gap `assert_resolution_sufficient` is *repurposed* as the coherence power gate or runs *alongside* a new coherence-power gate. This task implements only the **unambiguous** parts (escalation wording; the train-split source of the numbers). The KS↔coherence unification + the first-model coherence-effect-size side are deferred to **T12** with an explicit PI-confirm. Flagged, not invented.

- [ ] **Step 1: Write the failing test** (escalation must NOT say "extract a second region")

```python
# tests/eval/test_resolution_seam.py — add
import pytest
from cfm.eval.resolution import assert_resolution_sufficient, ResolutionInsufficientError

def test_escalation_is_multiregion_not_second_region(tmp_path, monkeypatch):
    # marker stubbed with resolved=0.10, floor=0.05; needed_gap in [floor, resolved)
    _write_marker(tmp_path, ks_resolved_gap_binding=0.10, ks_single_region_floor=0.05)
    monkeypatch.setattr("cfm.eval.resolution.eval_set_locked_marker", lambda r: tmp_path / "_EVAL_SET_LOCKED")
    with pytest.raises(ResolutionInsufficientError) as e:
        assert_resolution_sufficient(0.07, release="2026-04-15.0")
    msg = str(e.value)
    assert "second region" not in msg.lower()
    assert "munich" not in msg.lower()          # the swap is the COHERENCE gate's (T12), NOT KS-resolution's
    assert "more/larger held-out" in msg.lower()  # generic escalation only
```

- [ ] **Step 2: Run, verify it fails** (current message says "extract a second region")

Run: `uv run pytest tests/eval/test_resolution_seam.py::test_escalation_is_multiregion_not_second_region -v` → FAIL.

- [ ] **Step 3: Implement** — replace the two escalation messages (`resolution.py:54-67`)

```python
    if needed_gap >= floor:
        raise ResolutionInsufficientError(
            f"needed gap {needed_gap} < this set's resolved gap {resolved}: this held-out set "
            f"CANNOT resolve it; more/larger held-out data could ('extract a second region' is "
            f"moot at 42 cities). NOTE: this is the KS-resolution concern only — it PRODUCES "
            f"the resolved-gap NUMBER; it does NOT own the architecture-discrimination verdict "
            f"or the munich->manchester swap (that is assert_coherence_power_sufficient, T12)."
        )
    raise ResolutionInsufficientError(
        f"needed gap {needed_gap} < single-region floor {floor}: the resolvable-gap CEILING; "
        f"finer needs more/larger held-out data."
    )
```

- [ ] **Step 4: Run, verify pass.** `uv run pytest tests/eval/test_resolution_seam.py -v` → PASS.
- [ ] **Step 5: Commit**

```bash
git add src/cfm/eval/resolution.py tests/eval/test_resolution_seam.py
git commit -m "feat(eval): multi-region resolution escalation (no 'second region', spec §7)"
```

### Task 3: `labels.py` — regression-lock "no SG constant is scored" (spec §4.1, §5)

**Files:**
- Modify: `src/cfm/eval/holdout/labels.py` (NO logic change — confirm only)
- Test: `tests/eval/holdout/test_labels_no_sg_constant.py`

- [ ] **Step 1: Write the failing test** (locks the §5 finding: SG constants never reach the scored surface)

```python
# tests/eval/holdout/test_labels_no_sg_constant.py
from cfm.eval.holdout.labels import UNSCORED_V1_DIMENSIONS, TileLabels
import dataclasses

def test_sg_constants_are_unscored_or_absent():
    # sub_c_morphology_class + admin_region (the SG-leaked fields that REACH TileLabels) are UNSCORED
    assert "morphology_class" in UNSCORED_V1_DIMENSIONS
    assert "region" in UNSCORED_V1_DIMENSIONS
    # country / climate_zone / era_class are NOT TileLabels fields at all (structurally unscorable)
    fields = {f.name for f in dataclasses.fields(TileLabels)}
    assert "country" not in fields and "climate_zone" not in fields and "era_class" not in fields
    # the scored morphology signal is the sub-D stratum, not the sub-C constant
    assert "morphology_stratum" in fields
```

- [ ] **Step 2: Run, verify pass** (this is a lock on existing behavior — should pass as-is; if it FAILS, the §5 claim is wrong and you HALT and report).

Run: `uv run pytest tests/eval/holdout/test_labels_no_sg_constant.py -v` → PASS expected.

- [ ] **Step 3: Commit**

```bash
git add tests/eval/holdout/test_labels_no_sg_constant.py
git commit -m "test(eval): lock no-SG-constant-scored invariant (spec §4.1/§5)"
```

---

## Phase B — write-once manifest (spec §2) — MUST PRECEDE Phase C

### Task 4: multi-region manifest builder + §2.2 correct-by-construction assertions

**Files:**
- Modify: `src/cfm/eval/holdout/manifest.py` (extend; bump `MANIFEST_SCHEMA_VERSION = "2.0"`)
- Test: `tests/eval/holdout/test_manifest_multiregion.py`

- [ ] **Step 1: Write the failing tests** (schema 2.0 + the two §2.2 assertions)

```python
# tests/eval/holdout/test_manifest_multiregion.py
import pytest
from cfm.eval.holdout.manifest import build_holdout_manifest_multiregion, HoldoutDeclarationError

REG = {  # minimal per-region payload (real build sources these from G4 + sub-D provenance)
  "krakow": dict(morphology="medieval-organic", density="moderate", geography="PL",
                 crs="EPSG:25834", n_tiles=2, tokens=100,
                 tiles=[dict(tile_i=0, tile_j=0, provenance_sha256="a", macro_vocab_sha256="v"),
                        dict(tile_i=0, tile_j=1, provenance_sha256="b", macro_vocab_sha256="v")]),
}

def test_schema_and_whole_city_declaration():
    m = build_holdout_manifest_multiregion(REG, corpus_release="2026-04-15.0",
            derivation_version="1.2", train_cities={"hamburg"}, corpus_tile_counts={"krakow": 2})
    assert m["manifest_schema_version"] == "2.0"
    assert m["regions"]["krakow"]["holdout_kind"] == "whole_city"
    assert m["held_out_cities"] == ["krakow"]

def test_assertion_a_holdout_train_disjoint():
    with pytest.raises(HoldoutDeclarationError, match="on both sides"):
        build_holdout_manifest_multiregion(REG, corpus_release="2026-04-15.0",
            derivation_version="1.2", train_cities={"krakow"}, corpus_tile_counts={"krakow": 2})

def test_assertion_b_tiles_match_frozen_corpus_set():
    # enumerated 2 tiles but corpus has 3 -> NOT a match (must not drop tiles)
    with pytest.raises(HoldoutDeclarationError, match="matches frozen-corpus tile set"):
        build_holdout_manifest_multiregion(REG, corpus_release="2026-04-15.0",
            derivation_version="1.2", train_cities={"hamburg"}, corpus_tile_counts={"krakow": 3})
```

- [ ] **Step 2: Run, verify it fails.** `uv run pytest tests/eval/holdout/test_manifest_multiregion.py -v` → FAIL (function missing).

- [ ] **Step 3: Implement** (extend `manifest.py`)

```python
MANIFEST_SCHEMA_VERSION = "2.0"

class HoldoutDeclarationError(Exception):
    """A whole-city holdout declaration that cannot be proven correct-by-construction."""

def build_holdout_manifest_multiregion(
    regions_payload: dict[str, dict], *, corpus_release: str, derivation_version: str,
    train_cities: set[str], corpus_tile_counts: dict[str, int],
) -> dict:
    held_out = sorted(regions_payload)
    # §2.2 assertion (a): held-out and train cities disjoint.
    both = set(held_out) & set(train_cities)
    if both:
        raise HoldoutDeclarationError(f"cities on both sides (holdout AND train): {sorted(both)}")
    regions = {}
    for city in held_out:
        p = regions_payload[city]
        tiles = [dict(tile_i=int(t["tile_i"]), tile_j=int(t["tile_j"]),
                      provenance_sha256=t["provenance_sha256"],
                      macro_vocab_sha256=t.get("macro_vocab_sha256"))
                 for t in sorted(p["tiles"], key=lambda t: (t["tile_i"], t["tile_j"]))]
        # §2.2 assertion (b): enumerated tiles MATCH FROZEN-CORPUS TILE SET (not "fully enumerated").
        if len(tiles) != corpus_tile_counts[city]:
            raise HoldoutDeclarationError(
                f"{city}: enumerated {len(tiles)} tiles but frozen corpus has "
                f"{corpus_tile_counts[city]} — manifest must match frozen-corpus tile set "
                f"(no invented/dropped tiles). NB this is corpus-tile-set match, NOT geographic "
                f"completeness (e.g. munich is inner-core by extent, #21)."
            )
        regions[city] = dict(
            partition_path=f"holdout/region={city}", holdout_kind="whole_city",
            morphology=p["morphology"], density=p["density"], geography=p["geography"],
            crs=p["crs"], n_tiles=len(tiles), n_usable_tiles=p.get("n_usable_tiles"),
            tokens=int(p["tokens"]), tiles=tiles,
        )
    held_tok = sum(r["tokens"] for r in regions.values())
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "corpus_release": corpus_release, "derivation_version": derivation_version,
        "held_out_cities": held_out, "regions": regions,
        "totals": {"held_out_tokens": held_tok},
    }
```

- [ ] **Step 4: Run, verify pass.** → PASS.
- [ ] **Step 5: Commit**

```bash
git add src/cfm/eval/holdout/manifest.py tests/eval/holdout/test_manifest_multiregion.py
git commit -m "feat(eval): multi-region holdout manifest + correct-by-construction assertions (spec §2.1/§2.2)"
```

### Task 5: usable-n measurement (gate b) — read-only, verified end-state (spec §2.1, §3.5(b))

**Files:**
- Create: `scripts/eval/measure_usable_tiles.py`
- Test: `tests/eval/test_measure_usable_tiles.py` (synthetic macro_core fixtures)

- [ ] **Step 1: Write the failing test** (a tile with ≥3 interior road edges is usable; a water tile is not)

```python
# tests/eval/test_measure_usable_tiles.py
from cfm.eval.usable_tiles import tile_is_usable  # MIN_ROAD_EDGES=3

def test_usable_requires_3_interior_road_edges(make_macro_core):
    assert tile_is_usable(make_macro_core(interior_road_edges=3)) is True
    assert tile_is_usable(make_macro_core(interior_road_edges=2)) is False
    assert tile_is_usable(make_macro_core(active_cells=0)) is False  # water
```

- [ ] **Step 2: Run, verify it fails.** → FAIL.

- [ ] **Step 3: Implement the SHARED graph builder + `tile_is_usable` on top of it.** "usable" and "scored" must be ONE definition — so `usable_tiles` does NOT reimplement interior/road/endpoints; it consumes `macro_graph.interior_road_graph`, which T9's coherence metric *also* consumes. Then "usable" == "≥3 edges in the scored coherence graph" by construction, and the two cannot drift. (Interior = `1..6`; road = `{1,2,3}`, bucket 0 = no crossing per `macro_plan_vocab.yaml:3489-3505`; endpoints per `sub_d/lattice.py`.)

```python
# src/cfm/eval/holdout/macro_graph.py  — THE single definition (T9 coherence imports the SAME)
from cfm.data.sub_d.enums import SlotKind
ROAD = {1, 2, 3}                       # road_skeleton bucket 0 = no crossing
def interior(i, j) -> bool: return 1 <= i <= 6 and 1 <= j <= 6
def endpoints(li, lj, axis): return ((li, lj), (li+1, lj)) if axis == 0 else ((li, lj), (li, lj+1))
def interior_road_graph(rows):
    """Road-carrying interior-interior internal edges as [(cellA, cellB), ...].
    Single source for BOTH the 'usable' power unit (T5) and the S1 coherence
    metric (T9), so they cannot diverge."""
    out = []
    for r in rows:
        if r.slot_kind != SlotKind.INTERNAL_EDGE or r.road_skeleton_class is None: continue
        if int(r.road_skeleton_class) not in ROAD: continue
        a, b = endpoints(r.lower_cell_i, r.lower_cell_j, r.axis)
        if interior(*a) and interior(*b):
            out.append((a, b))
    return out

# src/cfm/eval/usable_tiles.py — consumes the shared builder (no reimplementation)
from cfm.eval.holdout.macro_graph import interior_road_graph
MIN_ROAD_EDGES = 3
def tile_is_usable(rows) -> bool:
    return len(interior_road_graph(rows)) >= MIN_ROAD_EDGES
```

- [ ] **Step 4: Run, verify pass.** → PASS.

- [ ] **Step 5: Run on Leonardo (verified end-state, not exit code)**

Run (Leonardo): `ssh leonardo '<repo>/.venv/bin/python -m scripts.eval.measure_usable_tiles --release 2026-04-15.0 --cities glasgow,eisenhuttenstadt,munich,krakow --out reports/2026-06-08-usable-n.yaml'`
**End-state verification (mandatory):** re-open `reports/2026-06-08-usable-n.yaml` and confirm it contains a non-null `n_usable_tiles` for ALL FOUR cities and the per-city `n_tiles` matches the G4 yaml — do NOT trust the exit code. Expected (matches this session's measurement): glasgow 523, eisenhuttenstadt 579, munich 156, krakow 601. If munich's usable-n is unexpectedly low (≪156), HALT and escalate to the §7 swap question (reaches back to T1).

- [ ] **Step 6: Commit**

```bash
git add src/cfm/eval/usable_tiles.py scripts/eval/measure_usable_tiles.py tests/eval/test_measure_usable_tiles.py reports/2026-06-08-usable-n.yaml
git commit -m "feat(eval): usable-tile measurement (gate b, spec §2.1/§3.5b)"
```

### Task 6: freeze the manifest — verified-end-state write-once (spec §2.3)

**Files:**
- Modify: `src/cfm/eval/holdout/manifest.py` (reuse `freeze_holdout_manifest`); add a build driver
- Test: `tests/eval/holdout/test_manifest_freeze.py`

- [ ] **Step 1: Write the failing test** (freeze refuses overwrite; sha self-excludes; regeneration byte-identical)

```python
def test_freeze_is_write_once_and_byte_deterministic(tmp_path):
    m = build_holdout_manifest_multiregion(REG, corpus_release="2026-04-15.0",
            derivation_version="1.2", train_cities={"hamburg"}, corpus_tile_counts={"krakow": 2})
    p = tmp_path / "holdout_manifest.yaml"
    freeze_holdout_manifest(m, p)
    first = p.read_bytes()
    with pytest.raises(FileExistsError):              # write-once
        freeze_holdout_manifest(m, p)
    # regeneration to a fresh path is byte-identical (determinism)
    p2 = tmp_path / "again.yaml"; freeze_holdout_manifest(m, p2)
    assert p2.read_bytes() == first
```

- [ ] **Step 2: Run, verify it fails / passes-after.** (`freeze_holdout_manifest` exists; sorted-regions determinism may need adding — if the test fails on ordering, add `sort_keys`/sorted regions to the canonical dump.) → iterate to PASS.

- [ ] **Step 3: Build + freeze on Leonardo (verified end-state)**

Build the real 4-city manifest (sourcing tiles + provenance_sha256 from sub-D, n_usable_tiles from Task 5, n_tiles from G4), `freeze_holdout_manifest` to **`multiregion_holdout_manifest_path("2026-04-15.0")`** = `data/processed/eval_set/2026-04-15.0/multiregion/holdout_manifest.yaml` (the **distinct** EU path — the SG set still owns `eval_set/2026-04-15.0/holdout_manifest.yaml`, untouched), write `multiregion_eval_set_locked_marker(...)`.
**End-state verification (mandatory, not exit code):**
1. re-read the frozen file from disk; recompute `manifest_sha256(loaded)` and assert it equals the stored `manifest_sha256`;
2. assert `held_out_cities == [eisenhuttenstadt, glasgow, krakow, munich]` and each region `holdout_kind == whole_city`;
3. assert `n_tiles` per city matches G4; `n_usable_tiles` present;
4. `git add -f` the manifest + `_EVAL_SET_LOCKED` (gitignored dir), confirm `git status` shows them staged.
Only after all four pass is the freeze "done."

- [ ] **Step 4: Commit (local; push needs PI word)**

```bash
git add -f data/processed/eval_set/2026-04-15.0/multiregion/holdout_manifest.yaml data/processed/eval_set/2026-04-15.0/multiregion/_EVAL_SET_LOCKED
git add src/cfm/eval/holdout/manifest.py tests/eval/holdout/test_manifest_freeze.py
git commit -m "feat(eval): freeze multi-region held-out manifest (multiregion/ subdir), write-once verified vs disk (spec §2.3)"
```

---

## Phase C — leak guard (spec §6) — AFTER Phase B (the guard reads the §2.2-verified declaration)

### Task 7: `lineage_audit.py` — city-identity guard scoped to `whole_city`

**Files:**
- Modify: `src/cfm/eval/holdout/lineage_audit.py:51-73`
- Test: `tests/eval/holdout/test_lineage_audit_cityguard.py` (Task 8 is the gating teeth)

- [ ] **Step 1: Write the failing test** (city-guard trips on an un-enumerated held-out-city tile)

```python
from cfm.eval.holdout.lineage_audit import audit_no_holdout_leak, HoldoutLeakError, Artifact

MANIFEST = {"regions": {"krakow": {"holdout_kind": "whole_city",
            "tiles": [{"tile_i": 0, "tile_j": 0}]}}}  # only (0,0) enumerated

def test_cityguard_trips_on_unenumerated_holdout_city_tile():
    art = Artifact(path="train/x", lineage=frozenset({("krakow", 9, 9)}))  # NOT enumerated
    with pytest.raises(HoldoutLeakError, match="krakow"):
        audit_no_holdout_leak(MANIFEST, [art])
```

- [ ] **Step 2: Run, verify it fails** (current audit only intersects enumerated `(region,tile)` → misses (9,9)). → FAIL.

- [ ] **Step 3: Implement** — add the city-guard alongside the existing enumerated-intersection + fail-closed:

```python
def _whole_city_regions(m: dict) -> set[str]:
    return {r for r, p in m["regions"].items() if p.get("holdout_kind") == "whole_city"}

def audit_no_holdout_leak(holdout_manifest, training_reachable) -> None:
    holdout = _holdout_tile_refs(holdout_manifest)           # existing (region,tile_i,tile_j)
    whole_city = _whole_city_regions(holdout_manifest)       # NEW
    failures = []
    for art in training_reachable:
        if art.lineage is None:
            failures.append(LineageFailure(art.path, "absent lineage (fail-closed)")); continue
        leaked = art.lineage & holdout
        if leaked:
            failures.append(LineageFailure(art.path, f"lineage includes held-out tiles {sorted(leaked)}"))
        # NEW city-guard: any lineage tile whose region is a wholly-held-out city (enumerated or not).
        # Report city-leaks NOT already covered by the enumerated `leaked` set, ALWAYS (even when
        # `leaked` is non-empty) — a complete message names both classes (minor-fix: no `and not leaked`).
        city_only = sorted({(r, i, j) for (r, i, j) in art.lineage if r in whole_city} - set(leaked))
        if city_only:
            failures.append(LineageFailure(art.path,
                f"lineage touches wholly-held-out city/cities {sorted({r for r, _, _ in city_only})} "
                f"(city-guard; tiles not enumerated: {city_only})"))
    if failures:
        raise HoldoutLeakError(failures)
```

- [ ] **Step 4: Run, verify pass.** → PASS.
- [ ] **Step 5: Commit**

```bash
git add src/cfm/eval/holdout/lineage_audit.py tests/eval/holdout/test_lineage_audit_cityguard.py
git commit -m "feat(eval): city-identity leak guard for whole-city holdout (spec §6)"
```

### Task 8: leak-guard 4-case teeth — GATING halt-gate (spec §6)

**Files:**
- Test: `tests/eval/holdout/test_lineage_audit_teeth.py`

> **This is a gate, not a report:** each case is red-before/green-after; the suite BLOCKS Phase D until all four pass. Construct the failing case, prove it trips; prove the clean case passes.

- [ ] **Step 1: Write all four teeth cases**

```python
WC = {"regions": {"krakow": {"holdout_kind": "whole_city",
       "tiles": [{"tile_i": 0, "tile_j": 0}, {"tile_i": 0, "tile_j": 1}]}}}

def test_A_enumerated_leak_trips():               # tile-key
    art = Artifact("train/a", frozenset({("krakow", 0, 0)}))
    with pytest.raises(HoldoutLeakError): audit_no_holdout_leak(WC, [art])

def test_B_unenumerated_holdout_city_tile_trips():  # city-guard non-redundant
    art = Artifact("train/b", frozenset({("krakow", 7, 7)}))  # not in WC tiles
    with pytest.raises(HoldoutLeakError, match="city-guard"): audit_no_holdout_leak(WC, [art])

def test_C_clean_passes():                         # no held-out lineage
    art = Artifact("train/c", frozenset({("hamburg", 1, 1)}))
    audit_no_holdout_leak(WC, [art])               # returns None, no raise

def test_D_partial_holdout_does_NOT_overtrip():    # synthetic tile_sample (forward-protection)
    ts = {"regions": {"singapore": {"holdout_kind": "tile_sample",
           "tiles": [{"tile_i": 5, "tile_j": 5}]}}}
    train_sg = Artifact("train/sg", frozenset({("singapore", 1, 1)}))  # singapore train-side tile
    audit_no_holdout_leak(ts, [train_sg])          # MUST NOT raise (city-guard scoped to whole_city only)
```

- [ ] **Step 2: Run as a gate** — `uv run pytest tests/eval/holdout/test_lineage_audit_teeth.py -v`. **All four must pass.** If B fails → the city-guard is vacuous (HALT, fix Task 7). If D fails → the guard over-trips partial holdouts (HALT, fix the `whole_city` scoping). Do not proceed to Phase D until green.

- [ ] **Step 3: Commit**

```bash
git add tests/eval/holdout/test_lineage_audit_teeth.py
git commit -m "test(eval): leak-guard 4-case gating teeth (A/B/C/D, spec §6)"
```

### Task 8.5: WIRE the guard to the real frozen manifest (spec §6 trigger-1) — closes the tested-but-unwired gap

**Files:**
- Test: `tests/eval/holdout/test_lineage_audit_realrun.py`
- Modify: `src/cfm/data/training/holdout_guard.py` (fail-closed schema check) + the bake-off datamodule call-site/config (re-point to the multi-region manifest)

> **Why this task exists:** T7/T8 prove the guard's LOGIC on synthetic artifacts. That is the **tested-but-unwired class that already bit us (#16 `assert_lossless_clip`)** — a green teeth suite does NOT prove the guard runs over the real frozen manifest + real train enumeration. The loader audit IS wired (`holdout_guard.py:44-46`, `datamodule.py:setup()`), but it currently points at the **SG 132-tile** manifest; for the EU bake-off it must point at the frozen **multi-region** manifest (schema 2.0 / city-guard). Closing it here (option a), not deferring.

- [ ] **Step 1: Write the failing real-run test** (the audit FIRES fail-closed over the FROZEN multi-region manifest)

```python
# tests/eval/holdout/test_lineage_audit_realrun.py  (LEONARDO — reads the frozen EU manifest)
import yaml
from cfm.data.training.holdout_guard import run_holdout_audit
from cfm.eval.holdout.lineage_audit import Artifact, HoldoutLeakError
from cfm.eval.holdout.paths import multiregion_holdout_manifest_path

def _load(): return yaml.safe_load(multiregion_holdout_manifest_path("2026-04-15.0").read_text())

def test_realrun_fires_on_leaked_held_out_city_tile():
    m = _load()                                   # frozen EU manifest, schema 2.0
    krakow_tile = (m["regions"]["krakow"]["tiles"][0]["tile_i"], m["regions"]["krakow"]["tiles"][0]["tile_j"])
    leaked = Artifact("train/leak", frozenset({("krakow", krakow_tile[0], krakow_tile[1])}))
    import pytest
    with pytest.raises(HoldoutLeakError):
        run_holdout_audit(m, [leaked])

def test_realrun_passes_clean():
    m = _load()
    clean = Artifact("train/ok", frozenset({("hamburg", 1, 1)}))
    run_holdout_audit(m, [clean])                 # no raise
```

- [ ] **Step 2: Run (Leonardo, gating).** Requires the frozen manifest (T6). `.venv/bin/python -m pytest tests/eval/holdout/test_lineage_audit_realrun.py -v`. **Both must pass** — this proves the guard guards the *real* manifest, not just synthetic artifacts.

- [ ] **Step 3: Re-point + fail-closed default**

```python
# holdout_guard.py — add a fail-closed schema assertion so the guard cannot silently
# run over a stale (SG schema-1.0) manifest when auditing the EU corpus.
def run_holdout_audit(holdout_manifest: dict, reachable: list[Artifact]) -> None:
    if holdout_manifest.get("manifest_schema_version") != "2.0":
        raise HoldoutLeakError([LineageFailure("<manifest>",
            "expected multi-region manifest schema 2.0 (whole-city city-guard); got "
            f"{holdout_manifest.get('manifest_schema_version')!r} — refusing to audit the EU "
            "corpus against a non-multi-region manifest (fail-closed).")])
    audit_no_holdout_leak(holdout_manifest, reachable)
```
Then point the **bake-off datamodule construction** at `multiregion_holdout_manifest_path("2026-04-15.0")` (the datamodule takes `holdout_manifest: Path` as a constructor arg — `datamodule.py:201,210` — so this is the call-site/config, not datamodule internals).

> **Named residual (NOT silently absent):** the bake-off *training entrypoint/config* that constructs the datamodule does not exist until the bake-off phase. So the **call-site re-point lands at bake-off-setup** — recorded here AND it MUST go in the handoff as a §6 trigger-1 obligation: *"the EU bake-off datamodule MUST be constructed with `multiregion_holdout_manifest_path`; the fail-closed schema-2.0 assertion (above) backstops a forgotten re-point."* The schema-2.0 fail-closed default means a forgotten re-point **fails loud**, not silently leaks.

- [ ] **Step 4: Commit**

```bash
git add src/cfm/data/training/holdout_guard.py tests/eval/holdout/test_lineage_audit_realrun.py
git commit -m "feat(eval): wire holdout guard to multi-region manifest + fail-closed schema-2.0 (spec §6 trigger-1)"
```

---

## Phase D — coherence metric (spec §3)

### Task 9: `coherence.py` — S1 (continuity + giant-component) + zoning, shuffle-gap

**Files:**
- Create: `src/cfm/eval/holdout/coherence.py`
- Test: `tests/eval/holdout/test_coherence.py`

- [ ] **Step 1: Write failing unit tests** (the term definitions, on hand-built tiles)

```python
from cfm.eval.holdout.macro_graph import interior_road_graph   # the SHARED builder (T5)
from cfm.eval.holdout.coherence import continuity, giant_component_fraction, zoning_agreement

def test_continuity_counts_through_cells():
    # a 3-edge path through interior cells: 2 ends (deg1) + interior through-cells
    edges = [((1,1),(2,1)), ((2,1),(3,1)), ((3,1),(4,1))]
    assert continuity(edges) == pytest.approx(2/4)   # 4 touched cells, 2 with deg>=2

def test_giant_component_fraction_drops_on_fragmentation():
    one_net = [((1,1),(2,1)), ((2,1),(3,1)), ((3,1),(3,2))]
    assert giant_component_fraction(one_net) == 1.0
    two_islands = [((1,1),(2,1)), ((5,5),(6,5)), ((6,5),(6,6))]
    assert giant_component_fraction(two_islands) < 1.0

def test_zoning_agreement_excludes_inactive_edges():
    # built(class 0) <-> empty(None) edge is EXCLUDED, not disagreement (spec §3.1 active-active)
    cells = {(1,1): 0, (2,1): None, (1,2): 0}
    assert zoning_agreement(edges=[((1,1),(2,1)), ((1,1),(1,2))], zoning=cells) == 1.0  # only the 0-0 edge counts
```

- [ ] **Step 2: Run, verify it fails.** → FAIL.

- [ ] **Step 3: Implement** `coherence.py` (reads `MacroCoreRow`s; interior = `1..6`; road-carrying = `{1,2,3}`; null = inactive). Union-find for components. The shuffle-gap permutes the attribute among interior-interior slots only (fixing the interior count).

```python
import numpy as np
# ONE definition of interior / road / endpoints / road-graph — imported from the shared
# builder (T5), never re-defined here, so the "usable" power unit and the "scored" metric
# cannot drift (your item 3). The zoning-adjacency edge set (all interior-interior pairs,
# road-agnostic) is built from the SAME `interior`/`endpoints` helpers.
from cfm.eval.holdout.macro_graph import interior, endpoints, ROAD, interior_road_graph

def continuity(edges):  # fraction of road-touched interior cells with degree>=2
    if not edges: return None
    deg = {}
    for a, b in edges: deg[a] = deg.get(a,0)+1; deg[b] = deg.get(b,0)+1
    return sum(1 for c in deg if deg[c] >= 2) / len(deg)

def giant_component_fraction(edges):
    if not edges: return None
    parent = {}
    def find(x):
        parent.setdefault(x, x); r = x
        while parent[r] != r: r = parent[r]
        while parent[x] != r: parent[x], x = r, parent[x]
        return r
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    comp = {}
    for a, b in edges: comp[find(a)] = comp.get(find(a),0)+1
    return max(comp.values()) / len(edges)

def zoning_agreement(edges, zoning):  # active-active same-class fraction (spec §3.1)
    active = [(a,b) for (a,b) in edges if zoning.get(a) is not None and zoning.get(b) is not None]
    if not active: return None
    return sum(1 for a,b in active if zoning[a] == zoning[b]) / len(active)
```
(Plus `interior_road_graph(rows)` → list of road-carrying interior-interior edge endpoint-pairs, and a `coherence_gap(rows, *, rng, n_shuffle)` that computes `score(real) − mean(score(interior-permuted))` per term, with the interior-only permutation null.)

- [ ] **Step 4: Run, verify pass.** → PASS.
- [ ] **Step 5: Commit**

```bash
git add src/cfm/eval/holdout/coherence.py tests/eval/holdout/test_coherence.py
git commit -m "feat(eval): macro-plan-coherence metric S1 (continuity+giant-component) + active-active zoning (spec §3.1)"
```

### Task 10: coherence 3-way teeth — GATING halt-gate (spec §3.3)

**Files:**
- Test: `tests/eval/holdout/test_coherence_teeth.py`

> Gate, not report. Synthetic uniform/noise/disconnected-loops + real-vs-permuted on held-out real tiles. BLOCKS Phase E until green.

- [ ] **Step 1: Write the three-way teeth + shuffle teeth + real-vs-permuted**

```python
def test_uniform_plan_fails(make_rows):   # all edges road -> gap≈0 (shuffles to self)
    g = coherence_gap(make_rows(all_edges_road=True), rng=np.random.default_rng(0), n_shuffle=15)
    assert abs(g["continuity_gap"]) < 0.05 and abs(g["fragmentation_gap"]) < 0.05

def test_noise_plan_fails(make_rows):     # random scatter -> low continuity, low gap
    g = coherence_gap(make_rows(random_edges=True), rng=np.random.default_rng(0), n_shuffle=15)
    assert g["continuity_gap"] < 0.1

def test_disconnected_loops_fail_fragmentation_pass_continuity(make_rows):  # the non-redundant case
    g = coherence_gap(make_rows(ten_disconnected_loops=True), rng=np.random.default_rng(0), n_shuffle=15)
    assert g["continuity_real"] > 0.8 and g["giant_real"] < 0.5   # high continuity, low single-network

def test_real_vs_permuted_separates_on_held_out_sample():   # LEONARDO (read-only real tiles)
    gaps = [coherence_gap(load_real(t), rng=np.random.default_rng(i), n_shuffle=15)
            for i, t in enumerate(sample_held_out_tiles(n=50))]
    assert np.mean([g["fragmentation_gap"] > 0 for g in gaps]) >= 0.7   # real beats permuted
```

- [ ] **Step 2: Run as a gate.** Synthetic three-way + shuffle: local. `test_real_vs_permuted...`: Leonardo (`.venv/bin/python -m pytest ... -k real_vs_permuted`). **All must pass.** If the disconnected-loops case does NOT fail on fragmentation while passing continuity → the fragmentation term is redundant (HALT, fix Task 9). Do not proceed until green.

- [ ] **Step 3: Commit**

```bash
git add tests/eval/holdout/test_coherence_teeth.py
git commit -m "test(eval): coherence 3-way gating teeth (uniform/noise/disconnected-loops + real-vs-permuted, spec §3.3)"
```

### Task 11: threshold-after-measuring — per-stratum held-out reference gap (spec §3.1, §10.2)

**Files:**
- Create: `scripts/eval/measure_coherence_reference.py`
- Test: `tests/eval/test_coherence_reference.py`

- [ ] **Step 1: Write the failing test** (the reference is the held-out-real per-stratum gap, NOT a guessed absolute)

```python
def test_reference_is_per_stratum_measured_not_constant():
    ref = load_coherence_reference("reports/2026-06-08-coherence-reference.yaml")
    assert set(ref["per_stratum"]) == {"glasgow","eisenhuttenstadt","munich","krakow"}
    for s in ref["per_stratum"].values():
        assert "continuity_gap" in s and "fragmentation_gap" in s  # measured reference, not a literal threshold
```

- [ ] **Step 2: Run, verify it fails.** → FAIL (file missing).

- [ ] **Step 3: Measure on Leonardo (verified end-state)** — compute the per-held-out-stratum reference gap on the **held-out real** tiles (usable only), write `reports/2026-06-08-coherence-reference.yaml`. **End-state check:** re-read the file; assert all four strata present with non-null gaps. The model-vs-real threshold (the gate) is set relative to THIS reference at first model (§3.1 / §10.2), never a guessed absolute.

- [ ] **Step 4: Commit**

```bash
git add scripts/eval/measure_coherence_reference.py tests/eval/test_coherence_reference.py reports/2026-06-08-coherence-reference.yaml
git commit -m "feat(eval): per-stratum held-out coherence reference (threshold-after-measuring, spec §3.1/§10.2)"
```

---

## Phase E — first-model power gate (spec §7) — wired-but-dormant-until-model

### Task 12: `assert_coherence_power_sufficient` — the SOLE architecture-discrimination verdict (spec §7)

**Files:**
- Modify: `src/cfm/eval/resolution.py` (add the sibling function)
- Test: `tests/eval/test_power_gate.py`

> **§7 wiring — PI-CONFIRMED (sibling, NOT extend-in-place):** `assert_resolution_sufficient` (T2) stays scoped to the KS-resolution concern and **PRODUCES A NUMBER** (the train-split resolved-gap); it does NOT carry the coherence/bake-off verdict or the swap. The new `assert_coherence_power_sufficient` is the **SOLE verdict-producer** for architecture-discrimination adequacy: it **CONSUMES** (resolved-gap, held-out usable-n, first-model model-vs-real effect-size) and renders pass/fail **and owns the munich→manchester swap escalation**. One renders the verdict, the other only feeds it a number → they **cannot disagree** (§7's "one rule"). Dormant until the first trained model supplies the effect-size (T3/T4c/trigger-2 all collapse here).

- [ ] **Step 1: Write the failing test** (the verdict gate fail-louds + owns the swap)

```python
import pytest
from cfm.eval.resolution import assert_coherence_power_sufficient, CoherencePowerInsufficientError

def test_coherence_power_failloud_owns_the_swap():
    with pytest.raises(CoherencePowerInsufficientError) as e:
        assert_coherence_power_sufficient(stratum="munich", usable_n=156,
                                          resolved_gap=0.10, model_vs_real_effect=0.04)
    assert "munich->manchester" in str(e.value)   # THIS gate owns the swap (KS-resolution does not)

def test_coherence_power_passes_when_resolvable():
    assert_coherence_power_sufficient(stratum="munich", usable_n=156,
                                      resolved_gap=0.10, model_vs_real_effect=0.20)  # no raise
```

- [ ] **Step 2: Run, verify it fails.** `uv run pytest tests/eval/test_power_gate.py -v` → FAIL (function missing).

- [ ] **Step 3: Implement the sibling verdict-producer** (consumes resolution's NUMBER + usable-n + effect-size)

```python
# src/cfm/eval/resolution.py — sibling to assert_resolution_sufficient
class CoherencePowerInsufficientError(Exception):
    """Held-out usable-n cannot resolve the model-vs-real coherence effect for architecture
    discrimination on this stratum (spec §7). SOLE verdict for that question."""

def assert_coherence_power_sufficient(*, stratum: str, usable_n: int,
                                      resolved_gap: float, model_vs_real_effect: float) -> None:
    """The ONE architecture-discrimination verdict. resolved_gap is the NUMBER produced by
    assert_resolution_sufficient (train-split); usable_n is the held-out power side;
    model_vs_real_effect arrives at the first trained model. Owns the munich->manchester swap."""
    if model_vs_real_effect < resolved_gap:        # finer than the train split can resolve
        raise CoherencePowerInsufficientError(
            f"stratum {stratum!r}: model-vs-real coherence effect {model_vs_real_effect} is finer "
            f"than the train-resolved gap {resolved_gap}; held-out usable_n={usable_n} cannot "
            f"discriminate architectures here. Escalate (owned by THIS gate): munich->manchester "
            f"(swap the floor stratum to a larger held-out city) or add-a-train-city, then re-lock "
            f"the multi-region eval set (write-once-per-version)."
        )
```

- [ ] **Step 4: Run, verify pass; commit.**

```bash
git add src/cfm/eval/resolution.py tests/eval/test_power_gate.py
git commit -m "feat(eval): assert_coherence_power_sufficient — sole arch-discrimination verdict, owns the swap (spec §7)"
```

> **Dormant-until-model:** `model_vs_real_effect` exists only once a model is trained, so this gate is wired now and FIRES at the bake-off first-model checkpoint — record in the handoff alongside the T8.5 datamodule re-point (both are bake-off-first-model obligations).

---

## Self-review (plan vs spec)

**Spec coverage:** §1 held-out set → T4/T5/T6 (frozen on the locked set, distinct `multiregion/` path). §2 manifest contract → T4 (builder+§2.2), T5 (usable-n + shared graph builder), T6 (freeze). §3 coherence → T9 (S1+active-active zoning, importing the shared `interior_road_graph`), T10 (3-way teeth), T11 (threshold-after-measuring). §3.4 density-dropped → not implemented (correctly); T3 locks no-SG-constant-scored. §4.1 admin_region excluded → T3. §4.2/§4.3 seam/§9.113 deferred → no task (correctly deferred). §5 de-Singapore → T1 (paths + distinct EU eval-set path), T2 (KS-resolution number + generic escalation), T3 (labels lock); §5.1 #22 → already recorded (known_issues). §6 leak guard → T7 (city-key), T8 (4-case gating teeth), **T8.5 (trigger-1 real-run wiring to the frozen manifest + fail-closed schema-2.0)**. §7 power gate → **T12 (`assert_coherence_power_sufficient`, sole verdict, owns the swap; PI-confirmed sibling)**. §10.1/§10.2/§9 discipline → honored in T4/T6/T11.

**Placeholder scan:** none. T12 is now fully specified (PI-confirmed sibling); T8.5's bake-off-caller re-point is an explicitly-named residual (not a hidden TODO) with a fail-closed schema-2.0 backstop so a forgotten re-point fails loud.

**Type consistency:** `interior`/`endpoints`/`ROAD`/`interior_road_graph` are defined **once** in `macro_graph.py` (T5) and imported by both `usable_tiles` (T5) and `coherence` (T9) — `tile_is_usable` == `len(interior_road_graph(rows)) >= 3` by construction, so "usable" and "scored" cannot drift. `continuity`/`giant_component_fraction`/`zoning_agreement`/`coherence_gap` consistent across T9/T10/T11. `Artifact`/`HoldoutLeakError`/`audit_no_holdout_leak`/`run_holdout_audit` match `lineage_audit.py`/`holdout_guard.py`. `build_holdout_manifest_multiregion`/`HoldoutDeclarationError`/`freeze_holdout_manifest`/`multiregion_holdout_manifest_path` consistent across T1/T4/T6/T8.5. `assert_resolution_sufficient` (number) vs `assert_coherence_power_sufficient`/`CoherencePowerInsufficientError` (verdict) cleanly separated across T2/T12.

**Dependency order:** Phase B (manifest+§2.2) precedes Phase C (guard teeth + T8.5 real-run wiring) — the guard reads the §2.2-verified `holdout_kind` declaration. T8/T10 are gating halt-steps. Orchestration: T5/T6/T11/T8.5 verify actual end-state (re-read disk + recompute sha + match G4 / load the frozen manifest), never an exit code.

**Resolved since v1:** (1) §7 unification — PI-confirmed sibling (T12 verdict-producer consumes resolution's number; cannot disagree). (2) Leak-guard tested-but-unwired gap — closed by T8.5 (real-run test + fail-closed schema-2.0 default; bake-off-caller re-point named for the handoff). (3) T5/T9 duplication — one shared `macro_graph` builder. (4) **NEW spec-path erratum** — the EU eval-set uses `eval_set/<release>/multiregion/` because the SG set already owns `eval_set/<release>/` (write-once); note back to spec §2 on approval.

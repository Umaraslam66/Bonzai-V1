# Bounded Multi-Region Extract Orchestrator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a testable Python state-machine orchestrator that fetches, processes (sub_c→sub_g), and validates many European cities into one diversity-spanning corpus sized for a 30M-ceiling compute-optimal ladder at r=20.

**Architecture:** A thin Python driver under `src/cfm/data/multiregion/` reuses the five existing stage scripts as subprocess calls (mirroring `sub_g/pipeline.py`), tracks per-city/per-stage `_SUCCESS` stamped with the commit-sha it was produced under, re-runs a stage (and everything downstream) when that stage's source changed since its sha, isolates per-city failures (continue-but-loud), and writes a roll-up manifest with an axis-coverage matrix. Fetch runs on a Leonardo login node (S3 egress); all processing runs on a CPU Slurm partition (never `boost_usr_prod`).

**Tech Stack:** Python 3.11, `subprocess`, `git` CLI for sha-diff, `pydantic`/dataclasses, `pyarrow`, `pytest`. Spec: `docs/superpowers/specs/2026-06-03-phase-2-multiregion-extract-orchestrator-design.md`.

**Conventions (from CLAUDE.md):** `from __future__ import annotations` atop every module; type hints on public functions; `ruff format` + `ruff check` before every commit; `logging`, never `print`; run `uv run pytest` (env: `uv sync --extra dev` if collection falls back to system Python).

---

## File Structure (decomposition lock)

**New package `src/cfm/data/multiregion/`:**
- `__init__.py` — public exports.
- `stages.py` — the `Stage` table: per-stage name, output-dir resolver, marker filename, subprocess-arg builder, and **source-path globs** (for invalidate-on-fix). `STAGE_ORDER`.
- `state.py` — `StageCompletion` (sha-stamped), `CityState`, `stage_source_changed()` (git-diff), `stages_to_run()` (invalidate-on-fix cascade), load/save sidecar state.
- `partition.py` — `assert_cpu_partition()` (the hard non-boost assertion) + Slurm submission helper.
- `driver.py` — `run_city()` (per-city chain with sha-aware gating, continue-but-loud) + `run_batch()`.
- `selection.py` — single-UTM-zone filter (`utm_epsg_for_centroid`), candidate-city generation spanning the diversity axes, region-config writer.
- `proxy.py` — redundancy measurement + the pre-committed decision rule.
- `rollup.py` — roll-up manifest model, axis-coverage matrix, fail-loud "ready for next batch" gate.

**New script:** `scripts/extract_region_batch.py` — CLI over `driver.run_batch`.
**New Slurm template:** `scripts/multiregion_process.sbatch` — per-city CPU process job.
**Modified (in-scope sub_f fix):** `src/cfm/data/sub_f/manifest.py`, `src/cfm/data/sub_f/pipeline.py`.
**Tests:** `tests/data/multiregion/test_*.py`, `tests/data/sub_f/test_manifest_region_crs.py`.

**Task order rationale:** Phase A (sub_f fix) is independent and small. Phase B builds the state-machine skeleton (`stages.py`+`state.py`) — the load-bearing core. **Phase C is Gate-6 contract verification, run immediately after the skeleton and before any city run** (cheapest place to catch a wrong stage dir/marker/arg in `stages.py`). Phases D–F finish the orchestrator and prove it on Berlin. Phase G is the operational extract (canary → gate → batch 2).

---

## Phase A — In-scope sub_f `region_crs` fix

### Task A1: sub_f manifest carries `region_crs` (non-Singapore-CRS test)

**Why:** Spec §8. sub_f's manifest omits `region_crs` while sub_c/sub_d/sub_e carry it. "Non-blocking because Berlin passed" is the regime-blindness trap — close it with a non-Singapore test.

**Files:**
- Modify: `src/cfm/data/sub_f/manifest.py` (add field), `src/cfm/data/sub_f/pipeline.py` (read sub_e manifest's `region_crs`, thread it in)
- Test: `tests/data/sub_f/test_manifest_region_crs.py`

- [ ] **Step 1: Read the precedent before writing (proactive contract verification).**
  Read these to mirror exactly — do not infer field names:
  - `src/cfm/data/sub_e/manifest.py` lines ~65,87 — how `SubEManifest` declares and serializes `region_crs`.
  - `src/cfm/data/sub_e/pipeline.py` lines ~122-123 — how sub_e reads `region_crs = str(sub_d_manifest["region_crs"])` from the upstream manifest.
  - `src/cfm/data/sub_f/manifest.py` — the `SubFManifest` dataclass/serializer (full file; it's small).
  - `src/cfm/data/sub_f/pipeline.py` lines ~184-265 — where the sub_f manifest is built and where the sub_e dir is read.

- [ ] **Step 2: Write the failing test.**

```python
# tests/data/sub_f/test_manifest_region_crs.py
"""sub_f manifest must carry region_crs, verified on a NON-Singapore CRS.

Regime-blindness guard (spec §8): a Singapore-only test passes even if the
field is hardcoded or dropped; a non-Singapore CRS (EPSG:25833) fails loud if
region_crs is not actually threaded from the sub_e manifest.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cfm.data.sub_f import manifest as subf_manifest


@pytest.mark.slow
def test_subf_manifest_region_crs_threads_non_singapore(
    derived_region_25833: Path,  # fixture: a tiny region derived end-to-end at EPSG:25833
) -> None:
    m = subf_manifest.SubFManifest.from_yaml(derived_region_25833 / "manifest.yaml")
    assert m.region_crs == "EPSG:25833", (
        f"sub_f manifest region_crs={m.region_crs!r}; expected EPSG:25833 threaded "
        f"from the sub_e manifest, not Singapore's EPSG:3414 or a missing/hardcoded value"
    )
```

Note: if no `derived_region_25833` fixture exists, build it from the smallest existing sub_e fixture by overriding its manifest `region_crs` to `EPSG:25833` and renaming tile dirs to `tile=EPSG25833_*`. Reuse `tests/data/sub_e/_fixtures.py` helpers; keep it `@pytest.mark.slow` if it runs the real derive.

- [ ] **Step 3: Run the test to verify it fails.**

Run: `uv run pytest tests/data/sub_f/test_manifest_region_crs.py -o addopts="" -v`
Expected: FAIL — `SubFManifest` has no `region_crs` attribute (or it's `None`).

- [ ] **Step 4: Implement — add `region_crs` to `SubFManifest`** mirroring `SubEManifest` (same field declaration, same placement in `to_yaml`/`from_yaml`). In `sub_f/pipeline.py`, read `region_crs` from the sub_e manifest exactly as `sub_e/pipeline.py:122` reads it from sub_d (`region_crs = str(sub_e_manifest["region_crs"])`), and pass it into the `SubFManifest` constructor. Cite the source in a comment: `# region_crs threaded from sub_e manifest per spec §8; see sub_e/pipeline.py:122`.

- [ ] **Step 5: Run the new test + the sub_f suite to verify pass + no regression.**

Run: `uv run pytest tests/data/sub_f/ -o addopts="" -v`
Expected: PASS (new test green; existing sub_f integration/determinism still green).

- [ ] **Step 6: Commit.**

```bash
git add src/cfm/data/sub_f/manifest.py src/cfm/data/sub_f/pipeline.py tests/data/sub_f/test_manifest_region_crs.py
git commit -m "fix(sub_f): thread region_crs into manifest (+ non-Singapore-CRS guard)"
```

---

## Phase B — State-machine skeleton (the load-bearing core)

### Task B1: `stages.py` — the stage table with source globs

**Files:**
- Create: `src/cfm/data/multiregion/__init__.py` (empty for now)
- Create: `src/cfm/data/multiregion/stages.py`
- Test: `tests/data/multiregion/test_stages.py`

- [ ] **Step 1: Trace each stage's source dependencies (carry-in #1 — trace, don't guess).**
  The §3.3 invalidation test proves invalidation *fires*; it does NOT prove the glob set is *complete*. An under-inclusive glob silently leaves stale artifacts. So derive each stage's globs by tracing its script's transitive local imports, not from memory. Run:

```bash
# For each stage script, list the cfm.* modules it (transitively) imports under src/.
uv run python - <<'PY'
import ast, pathlib
ROOT = pathlib.Path("src")
def local_imports(path):
    tree = ast.parse(pathlib.Path(path).read_text())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("cfm"):
            mods.add(n.module)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("cfm"):
                    mods.add(a.name)
    return mods
for s in ["scripts/extract_tiles.py","scripts/derive_macro_plan.py",
          "scripts/derive_boundary_contracts.py","scripts/sub_f/derive.py",
          "src/cfm/data/sub_g/cli.py"]:
    print(s, "->", sorted(local_imports(s)))
PY
```
  Record the result. Each stage's `source_globs` = its own script + the `src/cfm/...` dirs of its imported modules **plus shared modules they depend on** (at minimum `src/cfm/data/sub_c/coords.py`, `src/cfm/data/overture/`). **Bias to over-inclusion** (extra re-run = wasted time; missing dep = stale-artifact bug). Repeat the trace one level deep on the shared modules (e.g. does `coords` import anything else shared?).

- [ ] **Step 2: Write the failing test.**

```python
# tests/data/multiregion/test_stages.py
from __future__ import annotations

from cfm.data.multiregion import stages


def test_stage_order_is_the_full_chain():
    assert [s.name for s in stages.STAGE_ORDER] == [
        "fetch", "sub_c", "sub_d", "sub_e", "sub_f", "validate"
    ]


def test_each_stage_has_nonempty_source_globs():
    for s in stages.STAGE_ORDER:
        assert s.source_globs, f"{s.name} has empty source_globs (invalidate-on-fix blind)"


def test_shared_coords_module_is_in_every_processing_stage_globs():
    # coords is a shared dependency; a coords change must invalidate all CRS-dependent stages.
    for s in stages.STAGE_ORDER:
        if s.name in {"sub_c", "sub_d", "sub_e", "sub_f", "validate"}:
            assert any("coords" in g for g in s.source_globs), (
                f"{s.name} source_globs miss the shared coords module (under-inclusion bug)"
            )
```

- [ ] **Step 3: Run to verify it fails.**
  Run: `uv run pytest tests/data/multiregion/test_stages.py -v`
  Expected: FAIL — module `cfm.data.multiregion.stages` does not exist.

- [ ] **Step 4: Implement `stages.py`.**

```python
# src/cfm/data/multiregion/stages.py
"""The per-city stage table: how to invoke each stage, where it writes, and
which source paths gate its invalidate-on-fix (spec §2, §3.3).

source_globs are the load-bearing field: a stage re-runs when any tracked file
matching its globs changed since the sha its _SUCCESS was stamped with. Globs
are intentionally OVER-inclusive (extra re-run = wasted CPU; missing dep = a
stale post-fix artifact, which is the bug). Derived by tracing imports (Task B1
step 1), not from memory.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StageContext:
    """Everything a stage needs to build its invocation, for one city."""
    region: str
    release: str
    repo_root: Path
    commit_sha: str
    sub_c_dir: Path
    sub_d_dir: Path
    sub_e_dir: Path
    sub_f_dir: Path
    sub_g_dir: Path


@dataclass(frozen=True)
class Stage:
    name: str
    #: marker filename that signals this stage completed ("" => fetch, see driver)
    marker: str
    #: dir whose marker we check (None for fetch, handled specially in driver)
    output_dir: Callable[[StageContext], Path] | None
    #: argv builder for subprocess (empty for fetch; fetch is in-process load_region)
    argv: Callable[[StageContext], list[str]]
    #: tracked paths (pathspecs) whose change invalidates this stage + downstream
    source_globs: tuple[str, ...]


# NOTE: source_globs below are SEED values. Task B1 step 1 MUST replace them with
# the traced set (this is the carry-in: trace, don't guess). The values here cover
# the obvious own-script + shared coords/overture; extend per the trace output.
_PY = sys.executable

STAGE_ORDER: tuple[Stage, ...] = (
    Stage(
        name="fetch",
        marker="manifest.yaml",
        output_dir=None,  # cache dir; driver resolves + checks specially
        argv=lambda c: [],  # in-process load_region; driver handles
        source_globs=("src/cfm/data/overture/",),
    ),
    Stage(
        name="sub_c",
        marker="_SUCCESS",
        output_dir=lambda c: c.sub_c_dir,
        argv=lambda c: [_PY, "scripts/extract_tiles.py", "--region", c.region,
                        "--release", c.release],
        source_globs=("scripts/extract_tiles.py", "src/cfm/data/sub_c/",
                      "src/cfm/data/overture/"),
    ),
    Stage(
        name="sub_d",
        marker="_SUCCESS",
        output_dir=lambda c: c.sub_d_dir,
        argv=lambda c: [_PY, "scripts/derive_macro_plan.py", "--region", c.region,
                        "--release", c.release, "--sub-c-dir", str(c.sub_c_dir),
                        "--output-dir", str(c.sub_d_dir),
                        "--macro-vocab", "configs/macro_plan/v1/macro_plan_vocab.yaml",
                        "--commit-sha", c.commit_sha],
        source_globs=("scripts/derive_macro_plan.py", "src/cfm/data/sub_d/",
                      "src/cfm/data/sub_c/coords.py"),
    ),
    Stage(
        name="sub_e",
        marker="_SUCCESS",
        output_dir=lambda c: c.sub_e_dir,
        argv=lambda c: [_PY, "scripts/derive_boundary_contracts.py", "--release",
                        c.release, "--region", c.region, "--sub-c-region-dir",
                        str(c.sub_c_dir), "--sub-d-region-dir", str(c.sub_d_dir),
                        "--output-region-dir", str(c.sub_e_dir)],
        source_globs=("scripts/derive_boundary_contracts.py", "src/cfm/data/sub_e/",
                      "src/cfm/data/sub_c/coords.py"),
    ),
    Stage(
        name="sub_f",
        marker="_SUCCESS",
        output_dir=lambda c: c.sub_f_dir,
        argv=lambda c: [_PY, "scripts/sub_f/derive.py", "--release", c.release,
                        "--region", c.region, "--sub-c-region-dir", str(c.sub_c_dir),
                        "--sub-d-region-dir", str(c.sub_d_dir), "--sub-e-region-dir",
                        str(c.sub_e_dir), "--output-region-dir", str(c.sub_f_dir)],
        source_globs=("scripts/sub_f/derive.py", "src/cfm/data/sub_f/",
                      "src/cfm/data/sub_c/coords.py"),
    ),
    Stage(
        name="validate",
        marker="_PHASE1_VALIDATED",
        output_dir=lambda c: c.sub_g_dir,
        argv=lambda c: [_PY, "-m", "cfm.data.sub_g.cli", "validate", "--region",
                        c.region, "--release", c.release, "--sub-c-region-dir",
                        str(c.sub_c_dir), "--sub-d-region-dir", str(c.sub_d_dir),
                        "--sub-e-region-dir", str(c.sub_e_dir), "--sub-f-region-dir",
                        str(c.sub_f_dir), "--output-dir", str(c.sub_g_dir)],
        source_globs=("src/cfm/data/sub_g/", "src/cfm/data/sub_c/coords.py"),
    ),
)
```

- [ ] **Step 5: Replace seed `source_globs` with the traced set from step 1**, then run the test.
  Run: `uv run pytest tests/data/multiregion/test_stages.py -v`
  Expected: PASS.

- [ ] **Step 6: Verify the `validate` argv against the real CLI (do not trust this draft).**
  Read `src/cfm/data/sub_g/cli.py` and confirm the `validate` subcommand name and arg flags match the `argv` above. If the CLI uses `validate_main` without a subcommand (per the Berlin chain's `python -c "...validate_main(sys.argv[1:])"`), adjust `argv` to match the actual entrypoint. (Phase C re-verifies this against real writes; fix here if the read shows a mismatch.)

- [ ] **Step 7: Commit.**

```bash
git add src/cfm/data/multiregion/__init__.py src/cfm/data/multiregion/stages.py tests/data/multiregion/test_stages.py
git commit -m "feat(multiregion): stage table with traced source globs"
```

### Task B2: `state.py` — sha-stamped completions + invalidate-on-fix cascade (the mandatory test)

**Files:**
- Create: `src/cfm/data/multiregion/state.py`
- Test: `tests/data/multiregion/test_state.py`

- [ ] **Step 1: Write the failing test — a REAL git repo, exercising the real git-diff (carry-in #1).**

```python
# tests/data/multiregion/test_state.py
"""invalidate-on-fix: a source change to stage N re-runs N + downstream, NOT upstream.

Uses a REAL temp git repo so the git-diff mechanism itself is tested, not just
the cascade logic — an under-inclusive glob must be catchable here.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from cfm.data.multiregion import state, stages


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True,
                          capture_output=True, text=True).stdout.strip()


def _make_repo_with_stage_sources(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    # one tracked file per stage source dir referenced by STAGE_ORDER globs
    for rel in ["scripts/extract_tiles.py", "scripts/derive_macro_plan.py",
                "scripts/derive_boundary_contracts.py", "scripts/sub_f/derive.py",
                "src/cfm/data/sub_c/coords.py", "src/cfm/data/sub_c/x.py",
                "src/cfm/data/sub_d/x.py", "src/cfm/data/sub_e/x.py",
                "src/cfm/data/sub_f/x.py", "src/cfm/data/sub_g/x.py",
                "src/cfm/data/overture/x.py"]:
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("v1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    return repo


def test_changing_sub_e_source_reruns_sub_e_and_downstream_not_upstream(tmp_path):
    repo = _make_repo_with_stage_sources(tmp_path)
    sha_a = _git(repo, "rev-parse", "HEAD")
    # All stages completed at sha_a.
    cs = state.CityState(region="x", completions={
        s.name: state.StageCompletion(stage=s.name, sha=sha_a) for s in stages.STAGE_ORDER
    })
    # Fix sub_e's source -> new sha B.
    (repo / "src/cfm/data/sub_e/x.py").write_text("v2 (regime fix)\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "fix sub_e")
    sha_b = _git(repo, "rev-parse", "HEAD")

    to_run = state.stages_to_run(cs, sha_b, stages.STAGE_ORDER, repo)
    assert to_run == ["sub_e", "sub_f", "validate"], (
        f"expected sub_e + downstream to re-run, got {to_run}"
    )


def test_unchanged_head_reruns_nothing(tmp_path):
    repo = _make_repo_with_stage_sources(tmp_path)
    sha_a = _git(repo, "rev-parse", "HEAD")
    cs = state.CityState(region="x", completions={
        s.name: state.StageCompletion(stage=s.name, sha=sha_a) for s in stages.STAGE_ORDER
    })
    assert state.stages_to_run(cs, sha_a, stages.STAGE_ORDER, repo) == []


def test_missing_completion_runs_that_stage_and_downstream(tmp_path):
    repo = _make_repo_with_stage_sources(tmp_path)
    sha_a = _git(repo, "rev-parse", "HEAD")
    cs = state.CityState(region="x", completions={
        s.name: state.StageCompletion(stage=s.name, sha=sha_a)
        for s in stages.STAGE_ORDER if s.name not in {"sub_f", "validate"}
    })
    assert state.stages_to_run(cs, sha_a, stages.STAGE_ORDER, repo) == ["sub_f", "validate"]


def test_changing_shared_coords_reruns_all_processing_stages(tmp_path):
    repo = _make_repo_with_stage_sources(tmp_path)
    sha_a = _git(repo, "rev-parse", "HEAD")
    cs = state.CityState(region="x", completions={
        s.name: state.StageCompletion(stage=s.name, sha=sha_a) for s in stages.STAGE_ORDER
    })
    (repo / "src/cfm/data/sub_c/coords.py").write_text("v2\n")
    _git(repo, "add", "-A"); _git(repo, "commit", "-qm", "coords")
    sha_b = _git(repo, "rev-parse", "HEAD")
    to_run = state.stages_to_run(cs, sha_b, stages.STAGE_ORDER, repo)
    # coords is in sub_c..validate globs -> first changed stage is sub_c -> all downstream
    assert to_run == ["sub_c", "sub_d", "sub_e", "sub_f", "validate"]
```

- [ ] **Step 2: Run to verify it fails.**
  Run: `uv run pytest tests/data/multiregion/test_state.py -v`
  Expected: FAIL — `cfm.data.multiregion.state` does not exist.

- [ ] **Step 3: Implement `state.py`.**

```python
# src/cfm/data/multiregion/state.py
"""Per-city orchestration state + invalidate-on-fix (spec §3.3).

A stage's completion is stamped with the commit-sha it was produced under. On a
re-run, a stage is invalidated when any tracked file in its source_globs changed
between its recorded sha and HEAD; an invalidated stage AND all downstream stages
re-run; unchanged upstream stages do not.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from cfm.data.multiregion.stages import Stage


@dataclass(frozen=True)
class StageCompletion:
    stage: str
    sha: str


@dataclass
class CityState:
    region: str
    completions: dict[str, StageCompletion] = field(default_factory=dict)


def stage_source_changed(
    recorded_sha: str, head_sha: str, source_globs: tuple[str, ...], repo_root: Path
) -> bool:
    """True iff any tracked path in source_globs differs between the two shas."""
    if recorded_sha == head_sha:
        return False
    proc = subprocess.run(
        ["git", "diff", "--quiet", recorded_sha, head_sha, "--", *source_globs],
        cwd=repo_root,
    )
    if proc.returncode == 0:
        return False  # no diff
    if proc.returncode == 1:
        return True  # diff present
    raise RuntimeError(
        f"git diff failed (rc={proc.returncode}) comparing {recorded_sha}..{head_sha} "
        f"for {source_globs}"
    )


def stages_to_run(
    city: CityState, head_sha: str, stage_order: tuple[Stage, ...], repo_root: Path
) -> list[str]:
    """Stages needing a (re)run, in dependency order.

    A stage runs if: it has no recorded completion, OR its source changed since
    its recorded sha, OR any upstream stage is (re)running (downstream cascade).
    """
    to_run: list[str] = []
    cascade = False
    for stage in stage_order:
        rec = city.completions.get(stage.name)
        if rec is None:
            cascade = True
        elif stage_source_changed(rec.sha, head_sha, stage.source_globs, repo_root):
            cascade = True
        if cascade:
            to_run.append(stage.name)
    return to_run
```

- [ ] **Step 4: Run the tests to verify they pass.**
  Run: `uv run pytest tests/data/multiregion/test_state.py -v`
  Expected: PASS (all four). The `sub_e`-change test is the spec's **mandatory invalidate-on-fix test**.

- [ ] **Step 5: Add state persistence (sidecar).** Append to `state.py`:

```python
import json


def load_city_state(path: Path, region: str) -> CityState:
    if not path.exists():
        return CityState(region=region)
    raw = json.loads(path.read_text())
    return CityState(
        region=raw["region"],
        completions={k: StageCompletion(**v) for k, v in raw["completions"].items()},
    )


def save_city_state(path: Path, city: CityState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "region": city.region,
        "completions": {k: {"stage": v.stage, "sha": v.sha}
                        for k, v in city.completions.items()},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
```

  Add a round-trip test in `test_state.py`:

```python
def test_city_state_roundtrips(tmp_path):
    p = tmp_path / "s.json"
    cs = state.CityState(region="berlin", completions={
        "sub_c": state.StageCompletion(stage="sub_c", sha="abc")})
    state.save_city_state(p, cs)
    assert state.load_city_state(p, "berlin") == cs
```

- [ ] **Step 6: Run + commit.**

```bash
uv run pytest tests/data/multiregion/test_state.py -v
git add src/cfm/data/multiregion/state.py tests/data/multiregion/test_state.py
git commit -m "feat(multiregion): sha-stamped state + invalidate-on-fix cascade"
```

---

## Phase C — Gate-6 contract cross-reference (FIRST after skeleton, before any run)

### Task C1: assert `stages.py` matches each stage module's real writes

**Why (carry-in #2 + protocol Gate 6):** `stages.py` is the orchestrator's *belief* about each stage's output dir, marker, and args. Drift between belief and reality is the silent-skip / wrong-path bug class. This task cross-references the belief against the stage modules' **actual writes**, not field names — and it runs before any city fetch so a wrong dir/marker/arg costs nothing.

**Files:**
- Test: `tests/data/multiregion/test_stage_contract_gate6.py`

- [ ] **Step 1: For each stage, read where it actually writes its marker.** Record file:line for each:
  - sub_c marker: `cfm.data.sub_c.manifest.write_success_marker` writes `_SUCCESS` (called from `extract_tiles.py`).
  - sub_d: `sub_d/pipeline.py` `write_success_marker` → `_SUCCESS`.
  - sub_e: `sub_e/pipeline.py:~269` `.touch()` → `_SUCCESS`.
  - sub_f: `sub_f/pipeline.py:~250` `.touch()` → `_SUCCESS`.
  - validate: `sub_g/validator.py:190` writes `_PHASE1_VALIDATED`.

- [ ] **Step 2: Write the Gate-6 test (assertion logic does NOT use `stages.py` to compute the expected value — it hand-enumerates from the modules).**

```python
# tests/data/multiregion/test_stage_contract_gate6.py
"""Gate 6 (protocol v3): cross-reference the orchestrator's stage table against
each stage module's ACTUAL marker writes. Expected values are hand-enumerated
from the upstream modules, NOT read from stages.py — so a wrong marker/dir in
stages.py fails here, before any city run."""
from __future__ import annotations

from cfm.data.multiregion import stages

# Hand-enumerated from the stage modules (Task C1 step 1), NOT from stages.py.
EXPECTED_MARKERS = {
    "fetch": "manifest.yaml",
    "sub_c": "_SUCCESS",
    "sub_d": "_SUCCESS",
    "sub_e": "_SUCCESS",
    "sub_f": "_SUCCESS",
    "validate": "_PHASE1_VALIDATED",
}


def test_stage_markers_match_module_writes():
    actual = {s.name: s.marker for s in stages.STAGE_ORDER}
    assert actual == EXPECTED_MARKERS, (
        "stages.py marker beliefs drifted from the stage modules' actual writes; "
        "re-read the module that writes each marker and fix stages.py"
    )


def test_required_commit_sha_only_on_sub_d():
    # Foundation map: --commit-sha is required only on sub_d. Guard the belief.
    from cfm.data.multiregion.stages import StageContext
    from pathlib import Path
    ctx = StageContext(region="r", release="2026-04-15.0", repo_root=Path("."),
                       commit_sha="SHA", sub_c_dir=Path("c"), sub_d_dir=Path("d"),
                       sub_e_dir=Path("e"), sub_f_dir=Path("f"), sub_g_dir=Path("g"))
    sub_d = next(s for s in stages.STAGE_ORDER if s.name == "sub_d")
    assert "--commit-sha" in sub_d.argv(ctx)
    for other in ("sub_c", "sub_e", "sub_f"):
        s = next(x for x in stages.STAGE_ORDER if x.name == other)
        assert "--commit-sha" not in s.argv(ctx), f"{other} must not pass --commit-sha"
```

- [ ] **Step 3: Run it; fix `stages.py` (not the test) on any mismatch.**
  Run: `uv run pytest tests/data/multiregion/test_stage_contract_gate6.py -v`
  Expected: PASS once `stages.py` matches reality. If it fails, the belief in `stages.py` is wrong — fix `stages.py` per the module read (do NOT weaken the hand-enumerated expectation; that is the external source of truth).

- [ ] **Step 4: Commit.**

```bash
git add tests/data/multiregion/test_stage_contract_gate6.py src/cfm/data/multiregion/stages.py
git commit -m "test(multiregion): Gate-6 stage-contract cross-reference vs module writes"
```

---

## Phase D — Topology guard + per-city driver

### Task D1: `partition.py` — hard non-boost assertion

**Files:**
- Create: `src/cfm/data/multiregion/partition.py`
- Test: `tests/data/multiregion/test_partition.py`

- [ ] **Step 1: Write the failing test.**

```python
# tests/data/multiregion/test_partition.py
from __future__ import annotations

import pytest

from cfm.data.multiregion import partition


def test_boost_partition_is_rejected_loud():
    with pytest.raises(ValueError, match="boost_usr_prod"):
        partition.assert_cpu_partition("boost_usr_prod")


@pytest.mark.parametrize("p", ["dcgp_usr_prod", "lrd_all_serial"])
def test_cpu_partitions_accepted(p):
    partition.assert_cpu_partition(p)  # no raise
```

- [ ] **Step 2: Run to verify fail.** `uv run pytest tests/data/multiregion/test_partition.py -v` → FAIL (no module).

- [ ] **Step 3: Implement.**

```python
# src/cfm/data/multiregion/partition.py
"""Topology guard (spec §3.2): extraction is CPU-only and must NEVER run on a
GPU-billed partition. boost_usr_prod bills a 4xA100 node for CPU work and would
silently consume the training budget — so this is a hard runtime assertion, not
a convention."""
from __future__ import annotations

FORBIDDEN_PARTITIONS = frozenset({"boost_usr_prod"})


def assert_cpu_partition(partition: str) -> None:
    if partition in FORBIDDEN_PARTITIONS:
        raise ValueError(
            f"refusing to submit CPU extraction to GPU-billed partition "
            f"{partition!r}; use a CPU partition (dcgp_usr_prod / lrd_all_serial). "
            f"boost_usr_prod bills a 4xA100 node and would eat the training budget."
        )
```

- [ ] **Step 4: Run to verify pass.** Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/cfm/data/multiregion/partition.py tests/data/multiregion/test_partition.py
git commit -m "feat(multiregion): hard non-boost partition assertion"
```

### Task D2: `driver.py` — per-city chain with sha-aware gating + continue-but-loud

**Files:**
- Create: `src/cfm/data/multiregion/driver.py`
- Test: `tests/data/multiregion/test_driver.py`

- [ ] **Step 1: Write the failing test (subprocess + fetch stubbed).**

```python
# tests/data/multiregion/test_driver.py
from __future__ import annotations

from pathlib import Path

from cfm.data.multiregion import driver, stages, state


def test_run_city_skips_completed_stages_when_code_unchanged(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(driver, "_run_stage_subprocess",
                        lambda stage, ctx: calls.append(stage.name))
    monkeypatch.setattr(driver, "_run_fetch", lambda ctx: calls.append("fetch"))
    # all stages believed complete at HEAD; no source change -> nothing runs
    head = driver._head_sha(tmp_path) if (tmp_path / ".git").exists() else "HEAD0"
    monkeypatch.setattr(driver, "_head_sha", lambda repo: "HEAD0")
    monkeypatch.setattr(state, "stages_to_run", lambda *a, **k: [])
    cs = state.CityState(region="x")
    result = driver.run_city(_ctx(tmp_path), cs, repo_root=tmp_path)
    assert calls == []
    assert result.status == "validated"


def test_run_city_marks_failed_loud_and_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "_head_sha", lambda repo: "HEAD0")
    monkeypatch.setattr(state, "stages_to_run", lambda *a, **k: ["sub_c"])
    monkeypatch.setattr(driver, "_run_fetch", lambda ctx: None)

    def boom(stage, ctx):
        raise RuntimeError("regime: multipolygon explosion")
    monkeypatch.setattr(driver, "_run_stage_subprocess", boom)
    cs = state.CityState(region="x")
    result = driver.run_city(_ctx(tmp_path), cs, repo_root=tmp_path)
    assert result.status == "failed"
    assert "regime" in result.detail  # loud: the failure reason is captured


def _ctx(root: Path) -> stages.StageContext:
    return stages.StageContext(
        region="x", release="2026-04-15.0", repo_root=root, commit_sha="HEAD0",
        sub_c_dir=root / "c", sub_d_dir=root / "d", sub_e_dir=root / "e",
        sub_f_dir=root / "f", sub_g_dir=root / "g")
```

- [ ] **Step 2: Run → FAIL (no module).**

- [ ] **Step 3: Implement `driver.py`** (mirrors `sub_g/pipeline.py:_run_derive` — subprocess + `_SUCCESS` recheck — adds sha-aware gating and continue-but-loud).

```python
# src/cfm/data/multiregion/driver.py
"""Per-city chain driver (spec §3). Mirrors sub_g/pipeline.py's subprocess pattern
and adds: (1) sha-aware gating (skip a stage whose code is unchanged since its
recorded sha; re-run it + downstream otherwise — state.stages_to_run); (2)
continue-but-loud per-city failure (a city's regime marks it failed and is
captured in the result; the BATCH continues, the city is never silently dropped).
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from cfm.data.multiregion import stages, state
from cfm.data.multiregion.stages import Stage, StageContext

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CityResult:
    region: str
    status: str  # "validated" | "failed"
    detail: str = ""


def _head_sha(repo_root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
                          capture_output=True, text=True).stdout.strip()


def _run_fetch(ctx: StageContext) -> None:
    """Stage 1a: in-process fetch on the (egress-capable) host. Cache-hit is cheap.
    Run this on a login node; the rest of the chain runs on a CPU Slurm node."""
    from cfm.data.overture import load_region
    load_region(ctx.region, confirm=True, repo_root=ctx.repo_root)


def _run_stage_subprocess(stage: Stage, ctx: StageContext) -> None:
    _log.info("city=%s stage=%s: running", ctx.region, stage.name)
    subprocess.run(stage.argv(ctx), cwd=ctx.repo_root, check=True)
    if stage.output_dir is not None:
        marker = stage.output_dir(ctx) / stage.marker
        if not marker.exists():
            raise RuntimeError(
                f"city={ctx.region} stage={stage.name} returned 0 but wrote no "
                f"{stage.marker} at {marker}"
            )


def run_city(ctx: StageContext, city: state.CityState, repo_root: Path) -> CityResult:
    head = _head_sha(repo_root)
    to_run = state.stages_to_run(city, head, stages.STAGE_ORDER, repo_root)
    if not to_run:
        return CityResult(region=ctx.region, status="validated")
    try:
        for stage in stages.STAGE_ORDER:
            if stage.name not in to_run:
                continue
            if stage.name == "fetch":
                _run_fetch(ctx)
            else:
                _run_stage_subprocess(stage, ctx)
            city.completions[stage.name] = state.StageCompletion(stage=stage.name, sha=head)
    except (subprocess.CalledProcessError, RuntimeError, ValueError) as exc:
        _log.error("city=%s FAILED-NEEDS-ATTENTION: %s", ctx.region, exc)
        return CityResult(region=ctx.region, status="failed", detail=str(exc))
    return CityResult(region=ctx.region, status="validated")
```

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Commit.**

```bash
git add src/cfm/data/multiregion/driver.py tests/data/multiregion/test_driver.py
git commit -m "feat(multiregion): per-city driver (sha-gated, continue-but-loud)"
```

---

## Phase E — Selection, roll-up, proxy

### Task E1: `selection.py` — single-UTM-zone filter + axis-spanning candidates

**Files:**
- Create: `src/cfm/data/multiregion/selection.py`
- Test: `tests/data/multiregion/test_selection.py`

- [ ] **Step 1: Write the failing test.**

```python
# tests/data/multiregion/test_selection.py
from __future__ import annotations

import pytest

from cfm.data.multiregion import selection


def test_single_utm_zone_filter_accepts_within_zone_city():
    # Berlin centroid ~13.40E,52.52N -> zone 33 -> EPSG:25833, no straddle for a tight bbox
    ok, crs = selection.single_utm_zone_ok(
        bbox=(13.0883, 52.3383, 13.7612, 52.6755))
    assert ok and crs == "EPSG:25833"


def test_single_utm_zone_filter_rejects_zone_straddle():
    # bbox spanning a 6-degree zone boundary (e.g. 11.9E..12.1E crosses zone 32/33 edge at 12E)
    ok, _ = selection.single_utm_zone_ok(bbox=(11.9, 52.0, 12.1, 52.5))
    assert not ok


def test_southern_hemisphere_rejected():
    ok, _ = selection.single_utm_zone_ok(bbox=(13.0, -1.0, 13.5, -0.5))
    assert not ok
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement.** (Uses `utm_epsg_for_centroid`; the straddle check compares the UTM zone of the bbox W and E edges.)

```python
# src/cfm/data/multiregion/selection.py
"""City selection (spec §4): hard filters (single-UTM-zone, defer cross-border-admin)
and axis-spanning candidate generation. The NAMED list is PI-ratified (Task G1)."""
from __future__ import annotations

from cfm.data.sub_c.coords import utm_epsg_for_centroid


def _utm_zone(lon: float) -> int:
    return int((lon + 180.0) / 6.0) + 1


def single_utm_zone_ok(bbox: tuple[float, float, float, float]) -> tuple[bool, str | None]:
    """(ok, projected_crs). ok iff the bbox lies in one UTM zone, northern hemisphere,
    inside the ETRS89 European range. Returns the centroid CRS when ok."""
    min_lon, min_lat, max_lon, max_lat = bbox
    if min_lat < 0 or max_lat < 0:
        return (False, None)
    if _utm_zone(min_lon) != _utm_zone(max_lon):
        return (False, None)  # straddles a zone boundary
    centroid_lon = (min_lon + max_lon) / 2.0
    centroid_lat = (min_lat + max_lat) / 2.0
    try:
        crs = utm_epsg_for_centroid(centroid_lon, centroid_lat)
    except ValueError:
        return (False, None)
    return (True, crs)
```

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Add the candidate-generation scaffold + region-config writer** (the axis-spanning selection consumes a curated candidate pool; the actual axis labels per city are assigned by the operator/PI in Task G1). Append:

```python
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CityCandidate:
    name: str
    country_code: str
    admin_level: str
    bbox: tuple[float, float, float, float]
    morphology: str   # medieval-organic | planned-grid | modernist-sprawl | mixed
    density: str      # dense-core | moderate | sparse
    projected_crs: str


def write_region_config(candidate: CityCandidate, configs_dir: Path) -> Path:
    """Emit configs/data/regions/<name>.yaml mirroring berlin.yaml."""
    out = configs_dir / f"{candidate.name}.yaml"
    payload = {
        "name": candidate.name,
        "admin": {"source": "overture://divisions",
                  "country_code": candidate.country_code,
                  "level": candidate.admin_level},
        "fallback_bbox": list(candidate.bbox),
        "crs": "EPSG:4326",
        "projected_crs": candidate.projected_crs,
    }
    out.write_text(yaml.safe_dump(payload, sort_keys=False))
    return out
```

  Add a test that `write_region_config` round-trips and that the written `projected_crs` equals `single_utm_zone_ok(bbox)`'s CRS.

- [ ] **Step 6: Run + commit.**

```bash
uv run pytest tests/data/multiregion/test_selection.py -v
git add src/cfm/data/multiregion/selection.py tests/data/multiregion/test_selection.py
git commit -m "feat(multiregion): single-UTM filter + region-config writer"
```

### Task E2: `rollup.py` — manifest, axis-coverage matrix, fail-loud gate

**Files:**
- Create: `src/cfm/data/multiregion/rollup.py`
- Test: `tests/data/multiregion/test_rollup.py`

- [ ] **Step 1: Write the failing test.**

```python
# tests/data/multiregion/test_rollup.py
from __future__ import annotations

import pytest

from cfm.data.multiregion import rollup


def _city(name, morph, dens, status, tiles=400, tokens=11_000_000):
    return rollup.CityRecord(
        name=name, morphology=morph, density=dens, geography="DE",
        region_crs="EPSG:25833", tile_count=tiles, fetch_seconds=800.0,
        stage_shas={"sub_c": "abc"}, release="2026-04-15.0",
        validation_status=status, token_count=tokens)


def test_ready_gate_raises_on_unaddressed_failure():
    r = rollup.RollUp(cities=[
        _city("a", "planned-grid", "dense-core", "validated"),
        _city("b", "medieval-organic", "moderate", "failed")])
    with pytest.raises(RuntimeError, match="failed-needs-attention"):
        rollup.assert_ready_for_next_batch(r)


def test_ready_gate_passes_when_all_validated():
    r = rollup.RollUp(cities=[_city("a", "planned-grid", "dense-core", "validated")])
    rollup.assert_ready_for_next_batch(r)  # no raise


def test_axis_coverage_matrix_counts_validated_only():
    r = rollup.RollUp(cities=[
        _city("a", "planned-grid", "dense-core", "validated"),
        _city("b", "planned-grid", "dense-core", "validated"),
        _city("c", "medieval-organic", "sparse", "failed")])
    cov = rollup.axis_coverage(r)
    assert cov["morphology"]["planned-grid"] == 2
    assert cov["morphology"].get("medieval-organic", 0) == 0  # failed excluded


def test_total_tokens_sums_validated_only():
    r = rollup.RollUp(cities=[
        _city("a", "planned-grid", "dense-core", "validated", tokens=10),
        _city("b", "mixed", "moderate", "failed", tokens=999)])
    assert rollup.total_validated_tokens(r) == 10
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement `rollup.py`.**

```python
# src/cfm/data/multiregion/rollup.py
"""Roll-up manifest (spec §6): the reproducibility record + the diversity-by-
construction proof (axis-coverage matrix). A failed city is present-and-LOUD,
never dropped; the next-batch gate refuses to proceed with unaddressed failures."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

VALIDATED = "validated"
FAILED = "failed"
PENDING = "pending"


@dataclass
class CityRecord:
    name: str
    morphology: str
    density: str
    geography: str
    region_crs: str
    tile_count: int
    fetch_seconds: float
    stage_shas: dict[str, str]
    release: str
    validation_status: str  # validated | failed | pending
    token_count: int
    regime: str = ""  # set when failed: the surfaced regime


@dataclass
class RollUp:
    cities: list[CityRecord] = field(default_factory=list)


def unaddressed_failures(r: RollUp) -> list[str]:
    return [c.name for c in r.cities if c.validation_status == FAILED]


def assert_ready_for_next_batch(r: RollUp) -> None:
    failed = unaddressed_failures(r)
    if failed:
        raise RuntimeError(
            f"{len(failed)} cities failed-needs-attention; address them before the "
            f"next batch (do not silently drop): {failed}"
        )


def axis_coverage(r: RollUp) -> dict[str, dict[str, int]]:
    """Per-axis label -> count of VALIDATED cities. Proves diversity by construction."""
    cov: dict[str, dict[str, int]] = {"morphology": defaultdict(int),
                                      "density": defaultdict(int),
                                      "geography": defaultdict(int)}
    for c in r.cities:
        if c.validation_status != VALIDATED:
            continue
        cov["morphology"][c.morphology] += 1
        cov["density"][c.density] += 1
        cov["geography"][c.geography] += 1
    return {k: dict(v) for k, v in cov.items()}


def total_validated_tokens(r: RollUp) -> int:
    return sum(c.token_count for c in r.cities if c.validation_status == VALIDATED)


def total_validated_tiles(r: RollUp) -> int:
    return sum(c.tile_count for c in r.cities if c.validation_status == VALIDATED)


def write_rollup(r: RollUp, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cities": [asdict(c) for c in r.cities],
        "totals": {
            "validated_cities": sum(1 for c in r.cities if c.validation_status == VALIDATED),
            "validated_tiles": total_validated_tiles(r),
            "validated_tokens": total_validated_tokens(r),
            "axis_coverage": axis_coverage(r),
        },
    }
    if extra:
        payload["totals"].update(extra)  # tile-budget vs achieved, proxy verdict
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
```

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Commit.**

```bash
git add src/cfm/data/multiregion/rollup.py tests/data/multiregion/test_rollup.py
git commit -m "feat(multiregion): roll-up manifest + axis-coverage + fail-loud gate"
```

### Task E3: `proxy.py` — redundancy proxy + pre-committed decision rule

**Files:**
- Create: `src/cfm/data/multiregion/proxy.py`
- Test: `tests/data/multiregion/test_proxy.py`

**Pinned constants (spec §7, §11 — pre-committed BEFORE measuring, per refinement #7):**
`X_AMBIGUOUS_BAND = 0.10` (geometry redundancy within ±10% of the language baseline ⇒ ambiguous), `Y_SIZE_UP = 0.5` (size the tile budget up by 50% when geometry is materially less redundant ⇒ r>20). These are committed now, before any canary measurement, so the result cannot be read to fit a pre-chosen city count. Rationale recorded in the module docstring; revisit only by an explicit pre-registered amendment, never post-hoc.

- [ ] **Step 1: Write the failing test (the rule, in all three regimes + the ambiguous=conservative+flag behavior).**

```python
# tests/data/multiregion/test_proxy.py
from __future__ import annotations

from cfm.data.multiregion import proxy


def test_more_redundant_than_language_confirms_r20():
    v = proxy.proxy_decision(geometry_redundancy=0.80, language_baseline=0.60,
                            singapore_redundancy=0.78, base_tile_budget=20_600)
    assert v.verdict == "r20_confirmed"
    assert v.recommended_tile_budget == 20_600


def test_materially_less_redundant_sizes_up():
    v = proxy.proxy_decision(geometry_redundancy=0.40, language_baseline=0.60,
                            singapore_redundancy=0.42, base_tile_budget=20_600)
    assert v.verdict == "size_up"
    assert v.recommended_tile_budget == 30_900  # 20_600 * 1.5


def test_within_band_is_ambiguous_conservative_and_flagged():
    v = proxy.proxy_decision(geometry_redundancy=0.62, language_baseline=0.60,
                            singapore_redundancy=0.61, base_tile_budget=20_600)
    assert v.verdict == "ambiguous_unresolved"
    assert v.recommended_tile_budget == 20_600  # conservative r=20 sizing held
    assert v.r_unresolved_flag is True  # NOT read as confirmation
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement `proxy.py`.**

```python
# src/cfm/data/multiregion/proxy.py
"""Data-only redundancy proxy (spec §7). NOT the compute-optimal r (that is
TRAINING-measured, in the bake-off). A RELATIVE comparison of the geometry token
stream's redundancy against a language baseline (the r=20 anchor) and Singapore.

Pre-committed decision rule (stated before measuring, refinement #7):
- geometry materially MORE redundant than language (rel >= +X) -> r<=20 ->
  r=20 sizing (already over-provisioned) stands: 'r20_confirmed'.
- geometry materially LESS redundant (rel <= -X) -> r>20 -> 'size_up' the tile
  budget by Y (add cities while cheap).
- within +/-X of the baseline -> 'ambiguous_unresolved': hold the CONSERVATIVE
  r=20 sizing AND raise r_unresolved_flag (resolved by the bake-off's training
  measurement) -- never read an ambiguous proxy as confirmation.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass

X_AMBIGUOUS_BAND = 0.10
Y_SIZE_UP = 0.5


def compression_redundancy(token_bytes: bytes) -> float:
    """1 - compressed/raw. Higher => more redundant. Cheap robust proxy."""
    if not token_bytes:
        return 0.0
    return 1.0 - (len(gzip.compress(token_bytes, compresslevel=6)) / len(token_bytes))


@dataclass(frozen=True)
class ProxyVerdict:
    geometry_redundancy: float
    language_baseline: float
    singapore_redundancy: float
    relative_delta: float
    verdict: str  # r20_confirmed | size_up | ambiguous_unresolved
    recommended_tile_budget: int
    r_unresolved_flag: bool


def proxy_decision(*, geometry_redundancy: float, language_baseline: float,
                  singapore_redundancy: float, base_tile_budget: int) -> ProxyVerdict:
    rel = (geometry_redundancy - language_baseline) / language_baseline
    if rel >= X_AMBIGUOUS_BAND:
        verdict, budget, flag = "r20_confirmed", base_tile_budget, False
    elif rel <= -X_AMBIGUOUS_BAND:
        verdict = "size_up"
        budget = int(round(base_tile_budget * (1.0 + Y_SIZE_UP)))
        flag = False
    else:
        verdict, budget, flag = "ambiguous_unresolved", base_tile_budget, True
    return ProxyVerdict(geometry_redundancy, language_baseline, singapore_redundancy,
                        rel, verdict, budget, flag)
```

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Commit.**

```bash
git add src/cfm/data/multiregion/proxy.py tests/data/multiregion/test_proxy.py
git commit -m "feat(multiregion): redundancy proxy + pre-committed decision rule"
```

---

## Phase F — CLI, Slurm template, Berlin integration

### Task F1: `scripts/extract_region_batch.py` — batch CLI

**Files:**
- Create: `scripts/extract_region_batch.py`
- Test: `tests/data/multiregion/test_cli.py`

- [ ] **Step 1: Write the failing test (argparse surface + dry-run plan).**

```python
# tests/data/multiregion/test_cli.py
from __future__ import annotations

import subprocess
import sys


def test_cli_dry_run_lists_planned_stages_per_city(tmp_path):
    # --dry-run prints, per city, the stages that WOULD run, and submits nothing.
    out = subprocess.run(
        [sys.executable, "scripts/extract_region_batch.py",
         "--cities", "berlin", "--partition", "lrd_all_serial", "--dry-run"],
        capture_output=True, text=True)
    assert out.returncode == 0
    assert "berlin" in out.stdout
    assert "fetch" in out.stdout  # plan is printed


def test_cli_rejects_boost_partition():
    out = subprocess.run(
        [sys.executable, "scripts/extract_region_batch.py",
         "--cities", "berlin", "--partition", "boost_usr_prod", "--dry-run"],
        capture_output=True, text=True)
    assert out.returncode != 0
    assert "boost_usr_prod" in (out.stderr + out.stdout)
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement `scripts/extract_region_batch.py`.** It: parses `--cities` (names) / `--cities-file`, `--partition` (asserted via `partition.assert_cpu_partition`), `--release`, `--dry-run`, `--state-dir`; for each city builds a `StageContext`, loads `CityState`, computes `stages_to_run` and prints the plan (dry-run) or runs `driver.run_city`; assembles a `RollUp` and calls `rollup.write_rollup`; at the end calls `rollup.assert_ready_for_next_batch` only when `--gate` is passed. Mirror the arg-parsing style of `scripts/derive_macro_plan.py`. Configure logging once at startup (`logging.basicConfig`). Inject `sys.path` for `src` if needed per `scripts/smoke.py` precedent ([[project_repo_location_icloud]]).

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Commit.**

```bash
git add scripts/extract_region_batch.py tests/data/multiregion/test_cli.py
git commit -m "feat(multiregion): extract_region_batch CLI (dry-run + non-boost guard)"
```

### Task F2: `scripts/multiregion_process.sbatch` — CPU process job template

**Files:**
- Create: `scripts/multiregion_process.sbatch`

- [ ] **Step 1: Write the template** (mirror `berlin_extract.sbatch` resources but **CPU partition**, no `--gres=gpu`). Header asserts the partition is CPU; body activates the venv and runs `extract_region_batch.py` for the `$CITY` passed via `--export`. Fetch is NOT here (fetch runs on the login node first; this job cache-hits).

```bash
#!/bin/bash
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_222
#SBATCH --job-name=mr-process
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Per-city PROCESS job (sub_c..validate), CPU-only, cache-hit (no egress).
# Fetch (1a) runs on the login node BEFORE submitting this. NEVER boost_usr_prod
# (CPU work must not bill a GPU node; extract_region_batch asserts this too).
set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7
source .venv/bin/activate
echo "host=$(hostname) sha=$(git rev-parse --short HEAD) city=${CITY} $(date -u +%FT%TZ)"
python -u scripts/extract_region_batch.py --cities "${CITY}" --partition lrd_all_serial
```

- [ ] **Step 2: Lint check the script is referenced correctly** (no test; verified in Task F3 / Phase G). Commit.

```bash
git add scripts/multiregion_process.sbatch
git commit -m "feat(multiregion): CPU per-city process sbatch template"
```

### Task F3: Berlin end-to-end through the orchestrator (real-data integration)

**Why:** Spec §9 — the pilot proved the *chain*; this proves the *orchestrator drives it*. Berlin data + cache exist locally and on Leonardo.

**Files:**
- Test: `tests/data/multiregion/test_berlin_integration.py`

- [ ] **Step 1: Write the integration test** (`@pytest.mark.slow`): build a Berlin `StageContext` against the real `data/processed/.../berlin` dirs; if `_SUCCESS`/`_PHASE1_VALIDATED` already present and code unchanged, assert `run_city` returns `validated` and runs **zero** stages (resume path). Then simulate a fix: monkeypatch `_head_sha` to a sha where a sub_e source differs, assert `stages_to_run` includes `sub_e`+downstream. This exercises the real state machine on real artifacts without re-running the heavy chain.

```python
# tests/data/multiregion/test_berlin_integration.py
from __future__ import annotations

from pathlib import Path

import pytest

from cfm.data.multiregion import driver, stages, state

REL = "2026-04-15.0"
ROOT = Path(__file__).resolve().parents[3]


def _berlin_ctx() -> stages.StageContext:
    base = ROOT / "data" / "processed"
    return stages.StageContext(
        region="berlin", release=REL, repo_root=ROOT, commit_sha="HEAD",
        sub_c_dir=base / "sub_c" / REL / "berlin",
        sub_d_dir=base / "sub_d" / REL / "berlin",
        sub_e_dir=base / "sub_e" / REL / "berlin",
        sub_f_dir=base / "sub_f" / REL / "berlin",
        sub_g_dir=base / "sub_g" / REL / "berlin")


@pytest.mark.slow
def test_berlin_resume_runs_nothing_when_complete_and_unchanged():
    ctx = _berlin_ctx()
    if not (ctx.sub_g_dir / "_PHASE1_VALIDATED").exists():
        pytest.skip("Berlin not materialized locally; run on Leonardo")
    head = driver._head_sha(ROOT)
    cs = state.CityState(region="berlin", completions={
        s.name: state.StageCompletion(stage=s.name, sha=head) for s in stages.STAGE_ORDER})
    assert state.stages_to_run(cs, head, stages.STAGE_ORDER, ROOT) == []
```

- [ ] **Step 2: Run locally** (skips gracefully if Berlin not materialized).
  Run: `uv run pytest tests/data/multiregion/test_berlin_integration.py -o addopts="" -v`
  Expected: PASS or SKIP (never error).

- [ ] **Step 3: Full fast-suite regression check.**
  Run: `uv run pytest -q`
  Expected: all pass (orchestrator added no fast-suite regressions).

- [ ] **Step 4: Commit.**

```bash
git add tests/data/multiregion/test_berlin_integration.py
git commit -m "test(multiregion): Berlin orchestrator integration (resume path)"
```

---

## Phase G — Operational extract (canary → gate → batch 2)

> These are operational runs on Leonardo, not TDD code tasks. Each has explicit entry/exit gates. Budget: process on a CPU partition only (free/cheap); no GPU-hours. Halt and report to the PI at each gate.

### Task G1: Generate + PI-ratify the canary list (~5 cities, max axis-span)

- [ ] **Step 1:** Using `selection.single_utm_zone_ok`, assemble ~5 candidate cities that maximally span the morphology axes (one each: medieval-organic, planned-grid, modernist-sprawl, sparse, +1 mixed), each passing the single-UTM filter and not cross-border-admin. Record `(name, country, level, bbox, morphology, density, projected_crs)` per city.
- [ ] **Step 2:** Present the named candidate list + axis assignments to the PI. **HALT for PI ratification** (spec §4 — list is PI-ratified, not guessed).
- [ ] **Step 3:** On ratification, write each city's `configs/data/regions/<name>.yaml` via `selection.write_region_config`; commit: `data(multiregion): canary region configs (PI-ratified)`.

### Task G2: Run the canary batch

- [ ] **Step 1:** Fetch each canary city on a Leonardo login node in tmux (`load_region(name, confirm=True)`), timing each ([[feedback_use_tmux_on_leonardo]]).
- [ ] **Step 2:** Submit per-city `multiregion_process.sbatch` (CPU partition). Collect `CityResult` per city into a `RollUp`; write the roll-up.
- [ ] **Step 3:** For each failed-needs-attention city, diagnose the regime and fix it as a **§9 construction-identity guard with a must-distinguish twin** (the guard fires on the EU regime AND on a synthetic version lacking the construction identity). Re-run via the sha-invalidation path. Halt-and-report each regime to the PI (do not improvise patches).

### Task G3: The three-part proceed-to-batch-2 gate

- [ ] **Step 1 — Regime gate:** `rollup.assert_ready_for_next_batch` passes (zero unaddressed failures) AND every regime fix has its proven must-distinguish guard test green.
- [ ] **Step 2 — Sizing gate:** Measure the redundancy proxy on the **canary** corpus (`proxy.compression_redundancy` over the canary token stream + the language baseline + Singapore); run `proxy.proxy_decision`. If `size_up`, raise the tile budget. If `ambiguous_unresolved`, hold r=20 sizing AND record the `r-unresolved-until-bakeoff` flag in the roll-up. **Finalize the batch-2 tile budget here** (not before).
- [ ] **Step 3 — Cost-model gate:** From the canary roll-up, compute tiles/city and fetch-seconds **per morphology**; use them (not Berlin's single point) to size the batch-2 city count to the finalized tile budget.
- [ ] **Step 4 — Composition gate (§5.1):** Confirm the canary is fully green under the **final post-fix shas** (re-run `stages_to_run` for each canary city at current HEAD → empty). Batch 2 launches only when canary and batch 2 share one validated sha baseline. **HALT-and-report the gate verdict + finalized batch-2 plan to the PI.**

### Task G4: Run batch 2 + final roll-up + DoD check

- [ ] **Step 1:** Generate + PI-ratify the batch-2 city list (spanning the remaining axis gaps shown by the canary's axis-coverage matrix). Write region configs.
- [ ] **Step 2:** Fetch (login/tmux) + process (CPU Slurm) batch-2 cities; merge into the roll-up. Address any new regime via the §9 discipline (expected near-zero — morphology space mostly covered).
- [ ] **Step 3 — DoD check:** Assert `rollup.total_validated_tokens >= 600_000_000` (30M × r=20) — or `>= recommended_tile_budget × 29_150` if the proxy sized up — and `assert_ready_for_next_batch` passes. Write the final roll-up with `axis_coverage`, `tile-budget vs achieved`, and the proxy verdict.
- [ ] **Step 4:** Write `reports/2026-MM-DD-phase-2-multiregion-extract.md` (the sub-project close-out: cities, axis coverage, tokens, proxy verdict + any r-unresolved flag, regimes fixed with their guards, validation status). Commit.

---

## Self-Review

**Spec coverage (each spec section → task):**
- §1 DoD → G4 (token-count check, roll-up, close-out). §2 per-city pipeline → B1 stage table + D2 driver. §3.1 topology → D1 + F2. §3.2 non-boost assertion → D1. §3.3 invalidate-on-fix + mandatory test → B2. §3.4 continue-but-loud → D2 + E2 gate. §4 selection (tile budget, axes, filters, PI-ratified list, validation tail) → E1 + G1 + G2-step3. §5 canary→remainder + 3-part gate → G2/G3. §5.1 composition → G3-step4. §6 roll-up + axis matrix → E2. §7 proxy (relative, pre-committed rule, ambiguous→conservative+flag) → E3 + G3-step2. §8 sub_f region_crs fix → A1. §9 testing (unit/integration/Gate-6) → B2/D1/E2/E3 (unit), F3 (integration), C1 (Gate 6). §11 deferrals → resolved: proxy X/Y (E3 pinned), source globs (B1 traced), canary list (G1 PI-ratified), Slurm array-vs-per-city (F2 per-city).
- **Gap check:** all spec sections map to a task. The "geometry-r compute-optimal" item is correctly out-of-scope (§10) — no task, by design.

**Placeholder scan:** no "TBD/TODO"; the only deferred-by-design values are the proxy X/Y (pinned to 0.10/0.5 in E3 with rationale) and the canary/batch-2 named lists (PI-ratified operational tasks G1/G4, which cannot be pre-named in code). Seed `source_globs` in B1 are explicitly replaced in B1-step5 by the trace and guarded by the C1 Gate-6 test.

**Type consistency:** `StageContext`/`Stage` (stages.py) consumed unchanged by state.py, driver.py, C1 test. `CityState`/`StageCompletion` (state.py) used in driver.py + tests. `CityRecord`/`RollUp` (rollup.py) used in E2 + G4. `ProxyVerdict.recommended_tile_budget`/`r_unresolved_flag` used in G3. `single_utm_zone_ok` returns `(bool, str|None)` consistently. No signature drift found.

---

## Execution Handoff

See the skill's execution-options prompt after this plan is saved.

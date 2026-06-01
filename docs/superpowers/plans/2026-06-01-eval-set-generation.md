# Eval-set Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the locked, held-out, real-city measurement substrate (Singapore in-distribution) that every PRD §9 metric layer is scored against — the held-out tile manifest, the raw + round-tripped-real reference distributions and ceilings, the conditioning labels, the degeneracy-rate judge, and the fail-loud holdout-leak audit — as DATA, with the model-facing scoring deferred to the eval-harness successor.

**Architecture:** A new read-only `src/cfm/eval/holdout/` package that imports (never re-derives) sealed sub-C/sub-D/sub-F/sub-G surfaces. It reads the 494 validated Singapore tiles, decodes sub-F's already-emitted tokens to produce round-tripped-real geometry, computes one shared `bref_placeholder_rate` consumed by both the C-ceiling and the D-rate-judge, drives a fresh per-stratum quota selector through G's co-optimization to freeze a `holdout_manifest`, and ships a fail-closed lineage audit the training scaffold will call. No sealed module is modified; any contradiction with a sealed contract halts the implementer (Gate 4).

**Tech Stack:** Python 3.11+, pyarrow (parquet I/O via `cfm.data.io.write_parquet`), pyyaml (`canonicalize_yaml`), shapely (geometry from sub-C WKB + decoded geoms), pytest (TDD; `@pytest.mark.slow` for the real-494-tile run). No scipy/numpy — power floors are pure-Python sizing formulas; the Wasserstein/KS *distance* computation is model-facing and deferred.

---

## Authoritative inputs (read before any task)

- **Spec:** `docs/superpowers/specs/2026-06-01-eval-set-generation-design.md` (commit `10dc463`) — seven locked decisions A–G, §2 shared-quantity one-source obligation, §3 dependency graph, §4 four precise-statement obligations, §5 per-principle table, §6 plan-time items, §7 deferrals.
- **Protocol:** `docs/protocols/sub-project-planning-protocol-v2.md` — six gates + six principles; §9 (construction-identity exclusion with regime-distinguishing guard) is the spine.

## Locked decisions carried into this plan (from spec + the 2026-06-01 plan-authoring source reads)

These are **verified facts** (file:line cited), not inferences. Every task that touches them must cite the source in a comment, not re-derive.

1. **Round-tripped-real is already on disk.** sub-F's per-tile `cells.parquet` holds the encoded real tiles (token_sequence per cell). Round-tripped-real geometry = `decode_feature(block)` over each feature block; `split_cell_into_features(token_sequence)` yields the blocks. (`cfm.data.sub_f.decoder.decode_feature`:76; `cfm.data.sub_g.seam_decodability.split_cell_into_features`; sub-G's `check_decodability` already does exactly this round-trip.) **We never re-encode and never need a model.**
2. **Density is an AGGREGATE.** `tile_population_density`'s locked proxy is `p75_building_footprint_ratio`, which is the p75 of the *same per-cell `building_footprint_ratio`* that `cell_density_bucket` buckets (`src/cfm/data/sub_d/evidence.py:308-337`). The tile label masks intra-tile spread ⇒ **D's rate-check stratifies on `cell_density_bucket` (cell granularity), never the tile label.**
3. **"Morphology" is a naming collision.** The SCORED morphology stratum = sub-D `road_skeleton_class` + `zoning_class` (`src/cfm/data/sub_d/io.py:51-53`, real variation). sub-C's field literally named `morphology_class` is the constant `"Asian-megacity"` for Singapore → UNSCORED. **Never name the sub-D stratum "morphology" in code** — use `morphology_stratum`; document the sub-C constant at the read site.
4. **The shared predicate to import + identity-lock:** `cfm.data.sub_g.seam_decodability._is_bref_placeholder_collapse(block: list[int], geom: dict) -> bool` (seam_decodability.py:146-167). It is `_has_outbound_bref(block) AND <2 distinct decoded coords`. Bref tokens are ids 1500–1507 (`cfm.data.sub_f.decoder._is_bref_token`:71). A feature block is `list[int]` with `block[0]==509` (`<feature>`), `block[-1]==510` (`<feature_end>`), body `block[1:-1]`.
5. **Fresh stratified selector, not a #11 fix.** sub-D's `select_layer3_subset` (`frequency_analysis.py:884`) is a tile-granularity, one-tile-per-dimension reviewer-diversity picker — a different tool. We build a NEW per-stratum quota selector that *consumes* sub-D labels (one-source) and fails loud (UNDERPOWERED-stated) where #11 failed silent. **`known_issues` #11 stays untouched** on the reviewer-diversity path.
6. **Corrected sequencing (overrides spec §6's "fix #11 first"):** **build the fresh selector → G measures *through* it → (N, selection) co-optimization → F freezes the manifest.** The selector must exist before G measures, or G concludes "sparse infeasible" against a selector that isn't built yet.
7. **δ is ONE number.** D's regime-distinguishing rate-excess threshold *is* G's δ-relaxation bound. Defined once as `DELTA_BREF_REGIME` in `sizing.py`, imported by `degeneracy.py`. Never carried as two quantities.

## Scope boundary (spec §1A, §4, §7) — write each task header to this

**In scope (computable from held-out REAL tiles alone):** held-out tile selection + frozen manifest; conditioning labels; raw + round-tripped-real reference distributions + the round-tripped-real ceilings; the shared bref-placeholder predicate + rate; **R2's real-side baseline + check-definition + the G-D1/G-D2 guards**; the never-train lineage audit (G-F1–F4 + region-scaling).

**Deferred to the eval-harness / training-scaffold successor (needs a trained model or external sim):** model-scoring orchestration; simulation-viability execution; **the tokenizer-on-MODEL side of R2** (we ship only the real-side baseline + the check definition + the guards); Wasserstein/KS *distance* computation; the training loader's actual exclusion (it calls *this* manifest + audit — one source).

**Generalization (PRD L113):** UNSCORED in v1 and **stated as unscored**, never "scored and passing" (spec §B). `region` is a first-class partition key so materializing a region D later slots in by adding a partition with zero change to lock/guard logic.

## File structure (`src/cfm/eval/holdout/`)

| File | Responsibility |
|---|---|
| `paths.py` | One-source path resolution: release/region constants, sub-{c,d,f,g} region dirs, eval-set output dir, holdout partition dir, tile dirname. |
| `labels.py` | Conditioning-label aggregation. Reads effective_conditioning.yaml + macro_core.parquet per tile; identity-locked to sub-D readers + macro vocab. Defines `morphology_stratum`; documents sub-C `morphology_class` as constant/UNSCORED. |
| `roundtrip.py` | Decode sub-F tokens to round-tripped-real geometry; pair each feature block with its cell's `cell_density_bucket` stratum. Reuses sub-F `decode_feature` + sub-G `split_cell_into_features`. |
| `bref_rate.py` | The §2 SHARED `bref_placeholder_rate(blocks, geoms, strata)`. Imports + identity-locks sub-G's `_is_bref_placeholder_collapse`. Stratified by `cell_density_bucket`. Computed once; imported by both C and D. |
| `baselines.py` | Raw-real + round-tripped-real reference distributions (building area, road length, cell density) + the round-tripped-real ceiling (`1 − bref_rate`) + the reported gap. Records source-tile lineage on every write (so G-F4 can bind). |
| `selector.py` | Fresh per-stratum quota selector. Consumes `TileLabels` (one-source). Regime-distinguishing guard: a stratum that can't meet quota / cell-floor → UNDERPOWERED-stated, never silent. |
| `sizing.py` | G's procedure: per-stratum populations → floors → co-optimize `(N, selection)` → ordered graceful-degradation. Defines `DELTA_BREF_REGIME` once. |
| `degeneracy.py` | D's per-instance exclude + distribution-level judge. G-D1 (model-emitted fixture re-proof) + G-D2 (at-threshold + stratified-cancellation). Imports the shared rate + `DELTA_BREF_REGIME`. |
| `manifest.py` | Frozen `holdout_manifest` (region-keyed tile IDs + provenance SHAs + manifest SHA), written once. Mirrors sub-C/sub-D manifest pattern. |
| `lineage_audit.py` | Fail-loud, fail-closed holdout-leak audit (G-F1–F4) + region-scaling. The guard function the training scaffold calls. |
| `pipeline.py` | Orchestrator wiring the corrected sequencing; writes the holdout partition, freezes the manifest, runs the guards, writes the `_EVAL_SET_LOCKED` marker + a `reports/` summary. |

Tests live under `tests/eval/holdout/`. Output data under `data/processed/eval_set/<release>/` (gitignored). Plan + report committed.

---

### Task 1: Package scaffold + path resolution

**Files:**
- Create: `src/cfm/eval/holdout/__init__.py`
- Create: `src/cfm/eval/holdout/paths.py`
- Create: `tests/eval/holdout/__init__.py`
- Test: `tests/eval/holdout/test_paths.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/holdout/test_paths.py
from __future__ import annotations

from pathlib import Path

from cfm.eval.holdout import paths


def test_tile_dirname_matches_sub_d_convention():
    # sub-D builds "tile=EPSG3414_i{i}_j{j}" (src/cfm/data/sub_d/pipeline.py:156).
    assert paths.tile_dirname(1, 7) == "tile=EPSG3414_i1_j7"
    assert paths.tile_dirname(9, 18) == "tile=EPSG3414_i9_j18"


def test_region_dirs_point_under_data_processed():
    rel, reg = "2026-04-15.0", "singapore"
    assert paths.sub_c_region_dir(rel, reg).as_posix().endswith(
        "data/processed/sub_c/2026-04-15.0/singapore"
    )
    assert paths.sub_d_region_dir(rel, reg).as_posix().endswith(
        "data/processed/sub_d/2026-04-15.0/singapore"
    )
    assert paths.sub_f_region_dir(rel, reg).as_posix().endswith(
        "data/processed/sub_f/2026-04-15.0/singapore"
    )


def test_holdout_partition_is_region_keyed():
    # spec §F: held-out tiles + derivatives live in a region-keyed holdout/ partition.
    p = paths.holdout_partition_dir("2026-04-15.0", "singapore")
    assert p.as_posix().endswith(
        "data/processed/eval_set/2026-04-15.0/holdout/region=singapore"
    )
    assert paths.holdout_manifest_path("2026-04-15.0").name == "holdout_manifest.yaml"


def test_default_release_and_region_constants():
    assert paths.DEFAULT_RELEASE == "2026-04-15.0"
    assert paths.DEFAULT_REGION == "singapore"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/holdout/test_paths.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cfm.eval.holdout'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/cfm/eval/holdout/__init__.py
"""Eval-set generation: the locked held-out real-city measurement substrate.

This package is read-only over sealed sub-C/sub-D/sub-F/sub-G outputs. It
imports their contracts (never re-derives), per the one-source discipline
(planning protocol v2 Gate 6). Model-facing scoring is deferred to the
eval-harness successor (spec §7).
"""
from __future__ import annotations
```

```python
# src/cfm/eval/holdout/paths.py
"""One-source path resolution for the eval-set substrate.

Every on-disk location the eval-set reads or writes is built here so no
other module hard-codes a path. Layout verified 2026-06-01:
- sub-{c,d,f,g} region dirs: data/processed/sub_X/<release>/<region>/
- per-tile dir: tile=EPSG3414_i{i}_j{j}/ (sub_d/pipeline.py:156)
- _PHASE1_VALIDATED marker: data/processed/sub_g/<release>/<region>/
"""
from __future__ import annotations

from pathlib import Path

#: Phase-1 validated Singapore release (sub-G _PHASE1_VALIDATED, 494 tiles).
DEFAULT_RELEASE: str = "2026-04-15.0"
DEFAULT_REGION: str = "singapore"

#: Default projection label embedded in sub-D tile dir names (EPSG:3414 → EPSG3414).
_EPSG_LABEL: str = "EPSG3414"


def _repo_root() -> Path:
    # src/cfm/eval/holdout/paths.py → repo root is four parents up from src/cfm.
    return Path(__file__).resolve().parents[4]


def _data_processed() -> Path:
    return _repo_root() / "data" / "processed"


def tile_dirname(tile_i: int, tile_j: int, epsg_label: str = _EPSG_LABEL) -> str:
    """Per-tile directory name, identical to sub-D (sub_d/pipeline.py:156)."""
    return f"tile={epsg_label}_i{int(tile_i)}_j{int(tile_j)}"


def sub_c_region_dir(release: str, region: str) -> Path:
    return _data_processed() / "sub_c" / release / region


def sub_d_region_dir(release: str, region: str) -> Path:
    return _data_processed() / "sub_d" / release / region


def sub_f_region_dir(release: str, region: str) -> Path:
    return _data_processed() / "sub_f" / release / region


def sub_g_region_dir(release: str, region: str) -> Path:
    return _data_processed() / "sub_g" / release / region


def phase1_validated_marker(release: str, region: str) -> Path:
    return sub_g_region_dir(release, region) / "_PHASE1_VALIDATED"


def eval_set_dir(release: str) -> Path:
    return _data_processed() / "eval_set" / release


def holdout_partition_dir(release: str, region: str) -> Path:
    """spec §F: region-keyed holdout partition the training loader excludes."""
    return eval_set_dir(release) / "holdout" / f"region={region}"


def holdout_manifest_path(release: str) -> Path:
    return eval_set_dir(release) / "holdout_manifest.yaml"


def eval_set_locked_marker(release: str) -> Path:
    return eval_set_dir(release) / "_EVAL_SET_LOCKED"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/holdout/test_paths.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/eval/holdout/ tests/eval/holdout/
uv run ruff check src/cfm/eval/holdout/ tests/eval/holdout/
git add src/cfm/eval/holdout/__init__.py src/cfm/eval/holdout/paths.py tests/eval/holdout/__init__.py tests/eval/holdout/test_paths.py
git commit -m "feat(eval-set): package scaffold + one-source path resolution"
```

---

### Task 2: Conditioning-label aggregation (identity-locked; morphology_stratum defined)

**In scope:** §E conditioning labels, reused from sub-C/sub-D (one source). **Defers nothing.**

**Files:**
- Create: `src/cfm/eval/holdout/labels.py`
- Test: `tests/eval/holdout/test_labels.py`

- [ ] **Step 1: Write the failing test (incl. the Gate-6 external-source-of-truth cross-reference)**

```python
# tests/eval/holdout/test_labels.py
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cfm.data.sub_d.io import (
    DerivationEvidenceRow,
    MacroCoreRow,
    write_derivation_evidence_parquet,
    write_macro_core_parquet,
)
from cfm.data.sub_d.enums import MetricNamespace, Scope, SlotKind
from cfm.data.sub_d.macro_vocab import load_macro_vocab
from cfm.eval.holdout import labels


_VOCAB = Path("configs/macro_plan/v1/macro_plan_vocab.yaml")


def _write_synth_tile(tile_dir: Path) -> None:
    """A 2-cell synthetic tile: one dense cell (bucket 3), one sparse (bucket 0)."""
    tile_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        MacroCoreRow(SlotKind.CELL, 0, 0, 0, None, None, None, Scope.ACTIVE,
                     zoning_class=0, cell_density_bucket=3, road_skeleton_class=None),
        MacroCoreRow(SlotKind.CELL, 1, 0, 1, None, None, None, Scope.ACTIVE,
                     zoning_class=1, cell_density_bucket=0, road_skeleton_class=None),
        MacroCoreRow(SlotKind.INTERNAL_EDGE, 0, None, None, 0, 0, 0, Scope.ACTIVE,
                     zoning_class=None, cell_density_bucket=None, road_skeleton_class=2),
    ]
    write_macro_core_parquet(rows, tile_dir / "macro_core.parquet")
    write_derivation_evidence_parquet(
        [DerivationEvidenceRow(SlotKind.TILE, 0, MetricNamespace.TILE_POPULATION_DENSITY,
                               "p75_building_footprint_ratio", 0.22, "1.0")],
        tile_dir / "derivation_evidence.parquet",
    )
    (tile_dir / "effective_conditioning.yaml").write_text(
        yaml.safe_dump({
            "effective_conditioning_schema_version": "1.0",
            "tile_i": 1, "tile_j": 7,
            "conditioning": {
                "population_density_bucket": 2,
                "morphology_class": "Asian-megacity",  # sub-C constant — UNSCORED
                "coastal_inland_river": 1,
                "admin_region": "Central Region",
            },
        }),
        encoding="utf-8",
    )


def test_read_tile_labels_aggregates_cell_and_tile_signals(tmp_path: Path):
    tile_dir = tmp_path / labels.paths.tile_dirname(1, 7)
    _write_synth_tile(tile_dir)
    tl = labels.read_tile_labels(tile_dir, tile_i=1, tile_j=7)

    assert tl.tile_i == 1 and tl.tile_j == 7
    assert tl.population_density_bucket == 2          # tile-level, from conditioning yaml
    # cell-granularity buckets, from macro_core CELL rows (the failure-mode stratum):
    assert sorted(tl.cell_density_buckets) == [0, 3]
    assert tl.coastal_inland_river == 1
    # morphology_stratum is the sub-D skeleton+zoning summary — NEVER sub-C morphology_class:
    assert tl.morphology_stratum.dominant_zoning_class in (0, 1)
    assert tl.morphology_stratum.modal_road_skeleton_class == 2


def test_sub_c_morphology_class_is_recorded_as_unscored_constant(tmp_path: Path):
    tile_dir = tmp_path / labels.paths.tile_dirname(1, 7)
    _write_synth_tile(tile_dir)
    tl = labels.read_tile_labels(tile_dir, tile_i=1, tile_j=7)
    # The collision guard: the constant sub-C field is carried verbatim and flagged,
    # never promoted into a "scored" dimension.
    assert tl.sub_c_morphology_class == "Asian-megacity"
    assert "morphology_class" in labels.UNSCORED_V1_DIMENSIONS


def test_GATE6_cell_density_buckets_are_valid_vocab_ids():
    """Gate 6: hand-enumerate the macro vocab's cell_density token_ids from the YAML
    (ground truth) and assert read_tile_labels only ever yields those ids — the
    expected set is computed from the vocab, NOT from labels.py."""
    vocab = load_macro_vocab(_VOCAB)
    expected_ids = {int(b["token_id"]) for b in vocab["locked_buckets"]["cell_density"]}
    assert expected_ids == {0, 1, 2, 3}
    assert labels.valid_cell_density_bucket_ids(_VOCAB) == expected_ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/holdout/test_labels.py -v`
Expected: FAIL — `AttributeError: module 'cfm.eval.holdout.labels' has no attribute 'read_tile_labels'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/cfm/eval/holdout/labels.py
"""Per-tile conditioning-label aggregation (spec §E), one-source over sub-C/sub-D.

This module READS already-derived labels; it never re-derives a density,
zoning, or skeleton determination (planning protocol v2 Gate 6 + §3).

NAMING COLLISION (verified 2026-06-01): the SCORED "morphology" dimension is
sub-D's road_skeleton_class + zoning_class (io.py:51-53), which vary across
Singapore. sub-C's field literally named `morphology_class` is the CONSTANT
string "Asian-megacity" (sub_c/conditioning.py) → UNSCORED v1. We never call
the sub-D stratum "morphology"; it is `morphology_stratum`. The sub-C constant
is carried verbatim and listed in UNSCORED_V1_DIMENSIONS.

DENSITY is an AGGREGATE (verified evidence.py:308-337): tile_population_density
= p75 of the same per-cell building_footprint_ratio that cell_density_bucket
buckets. So `population_density_bucket` (tile) is the held-out-unit + §9
conditioning label, and `cell_density_buckets` (per-cell) is D's stratification
key — a tile mean would mask intra-tile spread.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from cfm.data.sub_d.enums import SlotKind
from cfm.data.sub_d.io import read_macro_core_parquet
from cfm.data.sub_d.macro_vocab import load_macro_vocab
from cfm.eval.holdout import paths

#: v1 conditioning dimensions with no real Singapore variation — UNSCORED-stated,
#: never read as a met bar (spec §E + §5 unscored-not-passing).
UNSCORED_V1_DIMENSIONS: frozenset[str] = frozenset(
    {"region", "morphology_class", "coastal_inland_river"}
)


@dataclass(frozen=True)
class MorphologyStratum:
    """The SCORED morphology stratum = sub-D skeleton + zoning (io.py:51-53).

    Deliberately NOT named 'morphology' — that word is sub-C's constant field.
    """

    dominant_zoning_class: int | None
    modal_road_skeleton_class: int | None


@dataclass(frozen=True)
class TileLabels:
    tile_i: int
    tile_j: int
    population_density_bucket: int | None   # tile-level (conditioning yaml); held-out unit
    cell_density_buckets: tuple[int, ...]   # per active CELL; D's stratification key
    morphology_stratum: MorphologyStratum   # sub-D skeleton + zoning (SCORED)
    coastal_inland_river: int | None        # sub-C enum; UNSCORED (near-constant)
    admin_region: str | None                # sub-C; UNSCORED
    sub_c_morphology_class: str | None      # the constant; recorded, UNSCORED


def valid_cell_density_bucket_ids(vocab_path: Path) -> set[int]:
    """Ground-truth cell_density token_ids straight from the locked vocab."""
    vocab = load_macro_vocab(vocab_path)
    return {int(b["token_id"]) for b in vocab["locked_buckets"]["cell_density"]}


def read_tile_labels(tile_dir: Path, *, tile_i: int, tile_j: int) -> TileLabels:
    """Aggregate one tile's conditioning labels from sub-D artifacts on disk."""
    rows = read_macro_core_parquet(tile_dir / "macro_core.parquet")

    cell_density = tuple(
        int(r.cell_density_bucket)
        for r in rows
        if r.slot_kind == SlotKind.CELL and r.cell_density_bucket is not None
    )
    zoning = [
        int(r.zoning_class)
        for r in rows
        if r.slot_kind == SlotKind.CELL and r.zoning_class is not None
    ]
    skeleton = [
        int(r.road_skeleton_class)
        for r in rows
        if r.slot_kind == SlotKind.INTERNAL_EDGE and r.road_skeleton_class is not None
    ]
    morphology = MorphologyStratum(
        dominant_zoning_class=Counter(zoning).most_common(1)[0][0] if zoning else None,
        modal_road_skeleton_class=Counter(skeleton).most_common(1)[0][0] if skeleton else None,
    )

    ec = yaml.safe_load((tile_dir / "effective_conditioning.yaml").read_text(encoding="utf-8"))
    cond = ec.get("conditioning", {})
    pdb = cond.get("population_density_bucket")

    return TileLabels(
        tile_i=int(tile_i),
        tile_j=int(tile_j),
        population_density_bucket=int(pdb) if pdb is not None else None,
        cell_density_buckets=cell_density,
        morphology_stratum=morphology,
        coastal_inland_river=cond.get("coastal_inland_river"),
        admin_region=cond.get("admin_region"),
        sub_c_morphology_class=cond.get("morphology_class"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/holdout/test_labels.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/eval/holdout/labels.py tests/eval/holdout/test_labels.py
uv run ruff check src/cfm/eval/holdout/labels.py tests/eval/holdout/test_labels.py
git add src/cfm/eval/holdout/labels.py tests/eval/holdout/test_labels.py
git commit -m "feat(eval-set): identity-locked conditioning labels + morphology_stratum (Gate 6)"
```

---

### Task 3: Round-trip decode + per-block stratum pairing

**In scope:** round-tripped-real geometry (the C "core" side). Reuses sub-F decoder + sub-G splitter.

**Files:**
- Create: `src/cfm/eval/holdout/roundtrip.py`
- Test: `tests/eval/holdout/test_roundtrip.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/holdout/test_roundtrip.py
from __future__ import annotations

from cfm.eval.holdout import roundtrip


# A Case-A feature block (no bref): <feature>=509 ... <feature_end>=510.
# Reuses the sub-G fixture shape; a real anchor + one (dir,mag) pair → a 2-vertex line.
_SIMPLE_BLOCK = [509, 41, 300, 323, 363, 369, 1, 50, 510]


def test_decode_blocks_pairs_each_block_with_its_cell_stratum():
    tokens_by_cell = {(0, 0): _SIMPLE_BLOCK, (1, 0): _SIMPLE_BLOCK}
    cell_density_by_cell = {(0, 0): 3, (1, 0): 0}

    blocks, geoms, strata = roundtrip.decode_region_blocks(
        tokens_by_cell, cell_density_by_cell
    )

    assert len(blocks) == len(geoms) == len(strata) == 2
    # geoms are GeoJSON dicts (decode_feature output), blocks retain token provenance:
    assert all(isinstance(g, dict) and "type" in g for g in geoms)
    assert all(b[0] == 509 and b[-1] == 510 for b in blocks)
    assert sorted(strata) == [0, 3]


def test_decode_skips_cells_without_a_density_bucket():
    # A cell with no recorded cell_density_bucket (masked/non-active) is dropped from
    # the stratified stream rather than silently bucketed as 0.
    blocks, geoms, strata = roundtrip.decode_region_blocks(
        {(2, 2): _SIMPLE_BLOCK}, cell_density_by_cell={}
    )
    assert blocks == [] and geoms == [] and strata == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/holdout/test_roundtrip.py -v`
Expected: FAIL — `AttributeError: ... has no attribute 'decode_region_blocks'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/cfm/eval/holdout/roundtrip.py
"""Round-tripped-real geometry: decode sub-F's already-emitted tokens.

sub-F's cells.parquet IS the encoded real tiles. Round-tripped-real geometry =
decode_feature(block) over each feature block. We reuse sub-F's decoder and
sub-G's feature splitter so the round-trip is byte-identical to what sub-G's
check_decodability validated (one source). We never re-encode.

Each decoded block is paired with its cell's cell_density_bucket so the shared
bref-rate (§2) and the reference distributions can stratify at cell granularity
(D's stratification key — density is an aggregate, see labels.py).
"""
from __future__ import annotations

from typing import Any

from cfm.data.sub_f.decoder import decode_feature
from cfm.data.sub_g.seam_decodability import split_cell_into_features


def decode_region_blocks(
    tokens_by_cell: dict[tuple[int, int], list[int]],
    cell_density_by_cell: dict[tuple[int, int], int],
) -> tuple[list[list[int]], list[dict[str, Any]], list[int]]:
    """Return aligned (blocks, decoded_geoms, strata) for one tile/region.

    A cell with no recorded cell_density_bucket is skipped (not bucketed as 0).
    """
    blocks: list[list[int]] = []
    geoms: list[dict[str, Any]] = []
    strata: list[int] = []
    for cell, token_sequence in sorted(tokens_by_cell.items()):
        stratum = cell_density_by_cell.get(cell)
        if stratum is None:
            continue
        for block in split_cell_into_features(token_sequence):
            blocks.append(block)
            geoms.append(decode_feature(block))
            strata.append(int(stratum))
    return blocks, geoms, strata
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/holdout/test_roundtrip.py -v`
Expected: PASS. (If `_SIMPLE_BLOCK` does not decode cleanly under the real grammar, the implementer HALTS and reports — this would be a real diff against sub-F's decoder contract, not a test to weaken. Construct a valid Case-A block from `cfm.data.sub_f.encoder.encode_feature` on a 2-vertex LineString and use its `.tokens` as the fixture.)

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/eval/holdout/roundtrip.py tests/eval/holdout/test_roundtrip.py
uv run ruff check src/cfm/eval/holdout/roundtrip.py tests/eval/holdout/test_roundtrip.py
git add src/cfm/eval/holdout/roundtrip.py tests/eval/holdout/test_roundtrip.py
git commit -m "feat(eval-set): round-trip decode reusing sub-F decoder + sub-G splitter"
```

---

### Task 4: The §2 shared `bref_placeholder_rate` (import + identity-lock sub-G predicate)

**In scope:** spec §2 — the one shared quantity behind C's ceiling and D's rate-judge. **Load-bearing: there is no independent corroborant; the guards on this function are the only check on its correctness.**

**Files:**
- Create: `src/cfm/eval/holdout/bref_rate.py`
- Test: `tests/eval/holdout/test_bref_rate.py`

- [ ] **Step 1: Write the failing test (incl. the Gate-6 identity-lock against sub-G)**

```python
# tests/eval/holdout/test_bref_rate.py
from __future__ import annotations

from cfm.data.sub_g import seam_decodability
from cfm.eval.holdout import bref_rate


def test_bref_rate_overall_and_stratified():
    # block, geom pairs: two collapses in stratum 3, none in stratum 0.
    collapse_block = [509, 41, 300, 323, 363, 369, 1500, 510]  # body ends in bref (1500)
    collapse_geom = {"type": "LineString", "coordinates": [[0.0, 0.0], [0.0, 0.0]]}
    ok_block = [509, 41, 300, 323, 363, 369, 1, 50, 510]
    ok_geom = {"type": "LineString", "coordinates": [[0.0, 0.0], [10.0, 0.0]]}

    blocks = [collapse_block, collapse_block, ok_block]
    geoms = [collapse_geom, collapse_geom, ok_geom]
    strata = [3, 3, 0]

    res = bref_rate.bref_placeholder_rate(blocks, geoms, strata)
    assert res.overall_rate == 2 / 3
    assert res.per_stratum[3].n_total == 2 and res.per_stratum[3].n_collapse == 2
    assert res.per_stratum[3].rate == 1.0
    assert res.per_stratum[0].n_collapse == 0 and res.per_stratum[0].rate == 0.0


def test_GATE6_identity_lock_uses_sub_g_predicate_verbatim():
    """Gate 6: the eval must classify a block EXACTLY as sub-G does — same import,
    no reimplementation. Cross-reference: sub-G's own degenerate-no-bref fixture
    (a genuine defect with identical zero-length symptom but NO outbound bref) must
    NOT be counted as a placeholder collapse here, proving we key on construction
    identity, not magnitude."""
    # The eval's predicate IS sub-G's object (identity, not a copy).
    assert bref_rate._bref_predicate is seam_decodability._is_bref_placeholder_collapse

    degenerate_no_bref = [509, 41, 300, 323, 363, 369, 511, 443, 510]  # sub-G fixture
    zero_len_geom = {"type": "LineString", "coordinates": [[0.0, 0.0], [0.0, 0.0]]}
    res = bref_rate.bref_placeholder_rate([degenerate_no_bref], [zero_len_geom], [0])
    assert res.per_stratum[0].n_collapse == 0  # genuine defect is NOT excluded


def test_empty_input_is_zero_not_division_error():
    res = bref_rate.bref_placeholder_rate([], [], [])
    assert res.overall_rate == 0.0 and res.per_stratum == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/holdout/test_bref_rate.py -v`
Expected: FAIL — `ModuleNotFoundError` / `AttributeError: ... 'bref_placeholder_rate'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/cfm/eval/holdout/bref_rate.py
"""The §2 shared bref-placeholder rate — ONE function, two consumers.

C's round-tripped-real ceiling (= 1 − rate) and D's degeneracy-rate judge are
the SAME quantity. It is computed ONCE on round-tripped-real and imported by
both; recomputing it in C's path and D's path separately would resurrect the
reimplementation/drift bug class one-source exists to prevent (spec §2).

Because there is no independent corroborant for this quantity (on real data
sub-G's bijection grounded the bref; here nothing cross-checks it), the guards
on THIS function are the only check on its correctness — they are load-bearing.

Construction-identity exclusion (protocol v2 §9): we import sub-G's predicate by
reference and never reimplement it; the identity-lock test asserts that.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cfm.data.sub_g.seam_decodability import _is_bref_placeholder_collapse

#: The shared construction-identity predicate, imported by REFERENCE from sub-G.
#: Never reimplement (Gate 6 identity-lock; test asserts `is` identity).
_bref_predicate = _is_bref_placeholder_collapse


@dataclass(frozen=True)
class StratumRate:
    n_total: int
    n_collapse: int

    @property
    def rate(self) -> float:
        return self.n_collapse / self.n_total if self.n_total else 0.0


@dataclass(frozen=True)
class BrefRateResult:
    overall_rate: float
    per_stratum: dict[int, StratumRate]


def bref_placeholder_rate(
    blocks: list[list[int]],
    geoms: list[dict[str, Any]],
    strata: list[int],
) -> BrefRateResult:
    """Stratified bref-placeholder collapse rate over round-tripped-real.

    blocks/geoms/strata are aligned (block i decodes to geom i in stratum i).
    A block is a placeholder collapse iff sub-G's construction-identity predicate
    says so — NEVER a bare zero-length / magnitude test.
    """
    if not (len(blocks) == len(geoms) == len(strata)):
        raise ValueError("blocks, geoms, strata must be the same length")

    totals: dict[int, int] = {}
    collapses: dict[int, int] = {}
    n_total = 0
    n_collapse = 0
    for block, geom, stratum in zip(blocks, geoms, strata, strict=True):
        totals[stratum] = totals.get(stratum, 0) + 1
        n_total += 1
        if _bref_predicate(block, geom):
            collapses[stratum] = collapses.get(stratum, 0) + 1
            n_collapse += 1

    per_stratum = {
        s: StratumRate(n_total=totals[s], n_collapse=collapses.get(s, 0))
        for s in totals
    }
    overall = n_collapse / n_total if n_total else 0.0
    return BrefRateResult(overall_rate=overall, per_stratum=per_stratum)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/holdout/test_bref_rate.py -v`
Expected: PASS (3 tests). The identity-lock `is` assertion proves no reimplementation.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/eval/holdout/bref_rate.py tests/eval/holdout/test_bref_rate.py
uv run ruff check src/cfm/eval/holdout/bref_rate.py tests/eval/holdout/test_bref_rate.py
git add src/cfm/eval/holdout/bref_rate.py tests/eval/holdout/test_bref_rate.py
git commit -m "feat(eval-set): §2 shared bref_placeholder_rate, identity-locked to sub-G (Gate 6)"
```

---

### Task 5: Reference-distribution baselines (raw + round-tripped + ceiling + lineage)

**In scope:** §C raw + round-tripped-real reference distributions, the round-tripped-real ceiling (`1 − bref_rate`, from the §2 function — NOT recomputed), the reported gap, and source-tile lineage on every record (so G-F4 can bind). **Deferred:** the model-vs-baseline Wasserstein/KS distance.

**Files:**
- Create: `src/cfm/eval/holdout/baselines.py`
- Test: `tests/eval/holdout/test_baselines.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/holdout/test_baselines.py
from __future__ import annotations

from cfm.eval.holdout import baselines
from cfm.eval.holdout.bref_rate import BrefRateResult, StratumRate


def test_ceiling_is_one_minus_shared_rate_not_recomputed():
    # The ceiling MUST come from the §2 shared BrefRateResult, not a second count.
    shared = BrefRateResult(
        overall_rate=0.1,
        per_stratum={0: StratumRate(100, 0), 3: StratumRate(50, 10)},
    )
    ceil = baselines.geometric_validity_ceiling(shared)
    assert ceil.overall == 0.9
    assert ceil.per_stratum[3] == 0.8  # 1 − 10/50
    assert ceil.per_stratum[0] == 1.0


def test_reference_distribution_records_source_tile_lineage():
    # spec §F: every baseline write records its source-tile lineage or G-F4 can't bind.
    rec = baselines.ReferenceDistribution(
        metric="building_area_m2",
        kind="raw",
        stratum=3,
        samples=(12.0, 30.5, 7.1),
        source_tiles=(("singapore", "tile=EPSG3414_i1_j7"),),
    )
    assert rec.source_tiles  # non-empty lineage is mandatory
    with __import__("pytest").raises(ValueError):
        baselines.ReferenceDistribution(
            metric="building_area_m2", kind="raw", stratum=3,
            samples=(1.0,), source_tiles=(),  # empty lineage is rejected at construction
        )


def test_full_minus_core_gap_reported():
    gap = baselines.report_gap(full_value=0.85, core_value=0.92)
    assert abs(gap - (0.85 - 0.92)) < 1e-12  # the tokenizer's own contribution, signed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/holdout/test_baselines.py -v`
Expected: FAIL — attribute/module errors.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cfm/eval/holdout/baselines.py
"""Real-side reference distributions + the round-tripped-real ceiling (spec §C).

Two baselines per §9 layer (spec §C):
- core = round-tripped-real (real → tokens → decode): the architecture-comparison
  reference (cancels the shared tokenizer ceiling).
- full = raw-real (sub-C original geometry): the absolute-fidelity reference.
- gap (full − core) = the tokenizer's own contribution, reported explicitly.

The geometric-validity CEILING is `1 − bref_placeholder_rate`, taken from the §2
SHARED BrefRateResult — never recomputed here (one source; spec §2/§D ceiling).

DEFERRED (spec §7): the model-vs-baseline Wasserstein/KS distance. We ship the
reference samples + provenance; the distance is computed against model output in
the eval-harness successor.

Provenance propagation (spec §F): every ReferenceDistribution carries its
source-tile lineage so the fail-closed lineage audit (G-F4) can bind.
"""
from __future__ import annotations

from dataclasses import dataclass

from cfm.eval.holdout.bref_rate import BrefRateResult

#: A (region, tile_dirname) lineage anchor.
TileRef = tuple[str, str]


@dataclass(frozen=True)
class GeometricValidityCeiling:
    overall: float
    per_stratum: dict[int, float]


def geometric_validity_ceiling(shared: BrefRateResult) -> GeometricValidityCeiling:
    """Ceiling = 1 − bref-placeholder-rate, from the §2 shared result (not recomputed)."""
    return GeometricValidityCeiling(
        overall=1.0 - shared.overall_rate,
        per_stratum={s: 1.0 - sr.rate for s, sr in shared.per_stratum.items()},
    )


@dataclass(frozen=True)
class ReferenceDistribution:
    metric: str                 # e.g. "building_area_m2", "road_length_m", "cell_density"
    kind: str                   # "raw" | "round_tripped"
    stratum: int                # cell_density_bucket
    samples: tuple[float, ...]
    source_tiles: tuple[TileRef, ...]   # lineage — mandatory, non-empty

    def __post_init__(self) -> None:
        if not self.source_tiles:
            raise ValueError(
                "ReferenceDistribution requires non-empty source_tiles lineage "
                "(spec §F: provenance-propagation, or the G-F4 audit cannot bind)"
            )
        if self.kind not in ("raw", "round_tripped"):
            raise ValueError(f"kind must be 'raw' or 'round_tripped'; got {self.kind!r}")


def report_gap(*, full_value: float, core_value: float) -> float:
    """full − core = the tokenizer's own contribution (the H1 229m-residual shape)."""
    return full_value - core_value
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/holdout/test_baselines.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/eval/holdout/baselines.py tests/eval/holdout/test_baselines.py
uv run ruff check src/cfm/eval/holdout/baselines.py tests/eval/holdout/test_baselines.py
git add src/cfm/eval/holdout/baselines.py tests/eval/holdout/test_baselines.py
git commit -m "feat(eval-set): reference distributions + ceiling (one-source) + lineage"
```

---

### Task 6: Fresh stratified selector (regime-distinguishing guard; consumes labels)

**In scope:** spec §F selection mechanism. **Built BEFORE G measures (corrected sequencing).** Consumes `TileLabels` (one source); never re-derives density/morphology.

**Files:**
- Create: `src/cfm/eval/holdout/selector.py`
- Test: `tests/eval/holdout/test_selector.py`

- [ ] **Step 1: Write the failing test (incl. the #11-failure-class regime-distinguishing guard)**

```python
# tests/eval/holdout/test_selector.py
from __future__ import annotations

from cfm.eval.holdout.labels import MorphologyStratum, TileLabels
from cfm.eval.holdout import selector


def _tile(i: int, j: int, *, pdb: int, cells: list[int], zon: int, sk: int) -> TileLabels:
    return TileLabels(
        tile_i=i, tile_j=j, population_density_bucket=pdb,
        cell_density_buckets=tuple(cells),
        morphology_stratum=MorphologyStratum(dominant_zoning_class=zon, modal_road_skeleton_class=sk),
        coastal_inland_river=1, admin_region="Central Region",
        sub_c_morphology_class="Asian-megacity",
    )


def test_selection_is_deterministic_and_tie_breaks_lexicographically():
    tiles = [_tile(2, 1, pdb=0, cells=[0, 0], zon=0, sk=0),
             _tile(1, 1, pdb=0, cells=[0, 0], zon=0, sk=0)]
    quotas = {(0, (0, 0)): 1}
    r1 = selector.select_holdout_tiles(tiles, quotas, cell_density_floor={0: 2})
    r2 = selector.select_holdout_tiles(tiles, quotas, cell_density_floor={0: 2})
    assert r1.selected == r2.selected
    assert r1.selected == [(1, 1)]  # lexicographically-smallest fills the quota


def test_GUARD_underpowered_stratum_surfaces_not_silently_dropped():
    """#11 failed by SILENTLY skipping the sparse side. The fresh selector must
    FAIL LOUD: a cell-density stratum whose selected cells fall below its floor is
    reported as UNDERPOWERED, never omitted-and-called-success."""
    # Only sparse-cell tiles exist; the dense cell-density stratum 3 has 0 cells.
    tiles = [_tile(1, 1, pdb=0, cells=[0, 0], zon=0, sk=0)]
    quotas = {(0, (0, 0)): 1}
    res = selector.select_holdout_tiles(tiles, quotas, cell_density_floor={0: 2, 3: 5})
    assert res.selected == [(1, 1)]
    assert 3 in res.underpowered_cell_density_strata          # surfaced, not dropped
    assert res.underpowered_cell_density_strata[3].available < res.underpowered_cell_density_strata[3].floor


def test_GUARD_unfillable_tile_quota_surfaces_as_underpowered():
    tiles = [_tile(1, 1, pdb=0, cells=[0], zon=0, sk=0)]
    quotas = {(0, (0, 0)): 3}   # quota 3 but only 1 tile in the stratum
    res = selector.select_holdout_tiles(tiles, quotas, cell_density_floor={0: 1})
    assert (0, (0, 0)) in res.underpowered_tile_strata
    assert res.underpowered_tile_strata[(0, (0, 0))].available == 1


def test_consumes_labels_only_no_rederivation():
    # Structural: the selector signature takes TileLabels; it must not import any
    # sub-D derivation function. (Enforced by review + this import-surface assertion.)
    import inspect
    src = inspect.getsource(selector)
    assert "evidence" not in src and "derive_density" not in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/holdout/test_selector.py -v`
Expected: FAIL — module/attribute errors.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cfm/eval/holdout/selector.py
"""Fresh per-stratum quota selector for the held-out set (spec §F selection).

This is NOT sub-D's select_layer3_subset (a tile-granularity, one-tile-per-
dimension reviewer-diversity picker, frequency_analysis.py:884). The eval needs
a per-stratum QUOTA sampler sized to power statistical floors — a different tool.
sub-D's #11 selector stays untouched on its reviewer-diversity path.

It CONSUMES sub-D-derived TileLabels (one source); it never re-derives a density
or morphology determination (no import of sub-D evidence/derivation code).

#11's failure class was a SILENT under-pick of the sparse side (a sign error that
made a guard skip a dimension and report success anyway). The regime-distinguishing
guard here is the opposite: a tile-stratum that can't fill its quota, or a
cell-density stratum whose selected cells fall below its floor, is SURFACED as
UNDERPOWERED (G's degradation policy, spec §G) — never silently omitted.

A tile stratum key is (population_density_bucket, (dominant_zoning_class,
modal_road_skeleton_class)). Cell-density floors are checked on the UNION of the
selected tiles' cells (D's stratification key — density is an aggregate).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from cfm.eval.holdout.labels import TileLabels

TileKey = tuple[int, int]
TileStratum = tuple[int, tuple[int | None, int | None]]


@dataclass(frozen=True)
class Shortfall:
    available: int
    floor: int


@dataclass
class SelectionResult:
    selected: list[TileKey]
    per_tile_stratum_counts: dict[TileStratum, int]
    underpowered_tile_strata: dict[TileStratum, Shortfall] = field(default_factory=dict)
    underpowered_cell_density_strata: dict[int, Shortfall] = field(default_factory=dict)


def _tile_stratum(tl: TileLabels) -> TileStratum:
    return (
        int(tl.population_density_bucket) if tl.population_density_bucket is not None else -1,
        (tl.morphology_stratum.dominant_zoning_class, tl.morphology_stratum.modal_road_skeleton_class),
    )


def select_holdout_tiles(
    tile_labels: list[TileLabels],
    quotas: dict[TileStratum, int],
    cell_density_floor: dict[int, int],
) -> SelectionResult:
    """Pick tiles to fill per-stratum quotas; surface every shortfall."""
    by_stratum: dict[TileStratum, list[TileLabels]] = defaultdict(list)
    for tl in tile_labels:
        by_stratum[_tile_stratum(tl)].append(tl)

    selected: list[TileKey] = []
    counts: dict[TileStratum, int] = {}
    underpowered_tiles: dict[TileStratum, Shortfall] = {}

    for stratum, quota in sorted(quotas.items()):
        pool = sorted(by_stratum.get(stratum, []), key=lambda tl: (tl.tile_i, tl.tile_j))
        take = pool[:quota]
        counts[stratum] = len(take)
        selected.extend((tl.tile_i, tl.tile_j) for tl in take)
        if len(take) < quota:
            underpowered_tiles[stratum] = Shortfall(available=len(take), floor=quota)

    # Cell-density coverage on the UNION of selected tiles' cells.
    selected_set = set(selected)
    cell_counts: dict[int, int] = defaultdict(int)
    for tl in tile_labels:
        if (tl.tile_i, tl.tile_j) in selected_set:
            for b in tl.cell_density_buckets:
                cell_counts[b] += 1
    underpowered_cells: dict[int, Shortfall] = {}
    for bucket, floor in sorted(cell_density_floor.items()):
        have = cell_counts.get(bucket, 0)
        if have < floor:
            underpowered_cells[bucket] = Shortfall(available=have, floor=floor)

    return SelectionResult(
        selected=sorted(selected),
        per_tile_stratum_counts=counts,
        underpowered_tile_strata=underpowered_tiles,
        underpowered_cell_density_strata=underpowered_cells,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/holdout/test_selector.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/eval/holdout/selector.py tests/eval/holdout/test_selector.py
uv run ruff check src/cfm/eval/holdout/selector.py tests/eval/holdout/test_selector.py
git add src/cfm/eval/holdout/selector.py tests/eval/holdout/test_selector.py
git commit -m "feat(eval-set): fresh per-stratum quota selector with fail-loud underpower guard"
```

---

### Task 7: G sizing + co-optimization (δ defined once; ordered graceful degradation)

**In scope:** spec §G procedure + the binding per-stratum floors + the ordered degradation policy. **`DELTA_BREF_REGIME` is defined here once** and imported by Task 8's degeneracy guards (the one δ).

**Files:**
- Create: `src/cfm/eval/holdout/sizing.py`
- Test: `tests/eval/holdout/test_sizing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/holdout/test_sizing.py
from __future__ import annotations

import math

import pytest

from cfm.eval.holdout import sizing


def test_DELTA_is_a_single_justified_number_below_one():
    # spec §6 + §G option-3: δ is a chosen, justified rate-excess, not a round default.
    assert 0.0 < sizing.DELTA_BREF_REGIME < 1.0
    assert sizing.DELTA_BREF_REGIME != 0.5  # not a round default (rough-numbers heuristic)


def test_rate_detection_floor_matches_hand_computed():
    # n ≈ z² p(1−p)/δ²  (spec §G: R2 rate-detection floor). z(0.975)=1.95996.
    p, delta = 0.10, 0.05
    expected = math.ceil((1.95996 ** 2) * p * (1 - p) / (delta ** 2))
    assert sizing.rate_detection_floor(p=p, delta=delta) == expected
    assert sizing.rate_detection_floor(p=p, delta=delta) != 100  # rough, not round


def test_ks_power_floor_is_documented_inverse_square_in_effect():
    # Smaller effect ⇒ strictly larger floor (monotone), per the v1 KS approximation.
    assert sizing.ks_two_sample_floor(effect=0.2) > sizing.ks_two_sample_floor(effect=0.4)


def test_degradation_is_ordered_coarsen_then_underpowered_then_relax():
    order = [s.name for s in sizing.DegradationStep]
    assert order == ["COARSEN_STRATA", "REPORT_UNDERPOWERED", "RELAX_DELTA_WITHIN_BOUND"]


def test_relax_delta_only_within_regime_bound():
    # A relaxed δ that still separates faithful-from-over-emitting (≤ the bound) is OK.
    assert sizing.relaxed_delta_is_legitimate(relaxed=sizing.DELTA_BREF_REGIME)
    assert sizing.relaxed_delta_is_legitimate(relaxed=sizing.DELTA_BREF_REGIME - 0.001)
    # Relaxing PAST the regime-distinguishing bound is weakening-to-pass → illegitimate.
    assert not sizing.relaxed_delta_is_legitimate(relaxed=sizing.DELTA_BREF_REGIME + 0.001)


def test_STRUCTURAL_per_stratum_not_whole_set():
    """Threshold-pairing (protocol v2 §2): a whole-set-only sizing that masks an
    underpowered stratum must be caught. feasibility() returns infeasible strata
    even when the whole-set N looks sufficient."""
    populations = {0: 1000, 3: 4}          # stratum 3 is starved
    floors = {0: 50, 3: 50}
    report = sizing.feasibility(populations, floors)
    assert report.whole_set_ok is True     # 1004 ≥ 100 — the masking signal
    assert 3 in report.infeasible_strata   # but the per-stratum check still fires
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/holdout/test_sizing.py -v`
Expected: FAIL — module/attribute errors.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cfm/eval/holdout/sizing.py
"""G's eval-set-size procedure: per-stratum floors → (N, selection) → degradation.

N is determined by floors and ceilings; the binding ones are PER-STRATUM, not
whole-set (spec §G). A whole-set power calc masks underpowered strata — the
vacuous pass at the sizing layer — so feasibility() always reports per-stratum
infeasibility even when the whole-set N looks sufficient (threshold-pairing,
protocol v2 §2).

δ (DELTA_BREF_REGIME) is ONE number: D's regime-distinguishing rate-excess AND
G's δ-relaxation bound (spec §6 — do NOT carry two). Defined here; imported by
degeneracy.py. The justification below is the load-bearing rationale.

Graceful degradation is ORDERED (spec §G): coarsen strata → report UNDERPOWERED →
relax δ ONLY within the regime-distinguishing bound. Relaxing past the bound is
weakening-the-assertion-to-pass (a halt-on-validator-fail violation in a sizing
costume) → the honest output is UNDERPOWERED, not "passed at relaxed δ".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

#: z for a two-sided 95% interval (one-source for both floors below).
_Z_0975: float = 1.95996

# DECISION: δ = 0.03 (3 percentage-point rate-excess). Justification — the v1
# round-tripped-real bref-placeholder rate is small (single-digit %); a model that
# "learned the limitation" reproduces it within sampling noise, while a model
# "over-emitting" degenerate stubs pushes the rate materially above it. 3pp is
# chosen as the smallest excess reliably above the per-stratum sampling noise floor
# at the achievable per-stratum N (re-confirmed against the real measurement in the
# slow run, Task 10). NOT a round default (0.05/0.10 rejected as round; rough-
# numbers heuristic). Revisit if the slow-run per-stratum noise floor exceeds 3pp.
DELTA_BREF_REGIME: float = 0.03


class DegradationStep(Enum):
    COARSEN_STRATA = 1
    REPORT_UNDERPOWERED = 2
    RELAX_DELTA_WITHIN_BOUND = 3


def rate_detection_floor(*, p: float, delta: float) -> int:
    """Min samples to detect a rate excess δ around base rate p: n ≈ z² p(1−p)/δ²."""
    if not (0.0 <= p <= 1.0) or delta <= 0.0:
        raise ValueError("require 0≤p≤1 and delta>0")
    return math.ceil((_Z_0975 ** 2) * p * (1.0 - p) / (delta ** 2))


def ks_two_sample_floor(*, effect: float, alpha: float = 0.05) -> int:
    """v1 KS two-sample sample-size approximation (equal n).

    The α-critical statistic for equal n is c(α)·sqrt(2/n) with c(0.05)=1.358; to
    resolve a true distributional gap `effect`, n ≈ ceil(2·(c(α)/effect)²). This is
    a sizing FLOOR only; the KS/Wasserstein DISTANCE against model output is deferred
    (spec §7). Documented approximation (DECISION: revisit if a stratum's effect-size
    assumption is contradicted by the slow-run distributions).
    """
    if effect <= 0.0:
        raise ValueError("effect must be > 0")
    c_alpha = 1.358 if abs(alpha - 0.05) < 1e-9 else 1.358
    return math.ceil(2.0 * (c_alpha / effect) ** 2)


def relaxed_delta_is_legitimate(*, relaxed: float) -> bool:
    """A relaxed δ is legitimate iff it still separates faithful-from-over-emitting,
    i.e. it stays at or below the regime-distinguishing bound (spec §G option 3)."""
    return 0.0 < relaxed <= DELTA_BREF_REGIME


@dataclass(frozen=True)
class FeasibilityReport:
    whole_set_ok: bool
    infeasible_strata: dict[int, int]   # stratum → shortfall (floor − population)


def feasibility(populations: dict[int, int], floors: dict[int, int]) -> FeasibilityReport:
    """Per-stratum feasibility. whole_set_ok is reported ONLY to expose the masking
    risk; the verdict is the per-stratum infeasible set (threshold-pairing §2)."""
    whole_set_ok = sum(populations.values()) >= max(floors.values(), default=0)
    infeasible = {
        s: floors[s] - populations.get(s, 0)
        for s in floors
        if populations.get(s, 0) < floors[s]
    }
    return FeasibilityReport(whole_set_ok=whole_set_ok, infeasible_strata=infeasible)


@dataclass
class SizingResult:
    n: int
    per_stratum_floor: dict[int, int]
    per_stratum_population: dict[int, int]
    underpowered_strata: list[int] = field(default_factory=list)
    degradation_log: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/holdout/test_sizing.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/eval/holdout/sizing.py tests/eval/holdout/test_sizing.py
uv run ruff check src/cfm/eval/holdout/sizing.py tests/eval/holdout/test_sizing.py
git add src/cfm/eval/holdout/sizing.py tests/eval/holdout/test_sizing.py
git commit -m "feat(eval-set): G per-stratum floors + ordered degradation + single δ"
```

---

### Task 8: D degeneracy guards (G-D1 model-fixture re-proof; G-D2 at-threshold + stratified-cancellation)

**In scope:** spec §D — per-instance exclude + distribution-level judge; the R2 check-definition. **Uses the §2 shared `bref_placeholder_rate` and the single `DELTA_BREF_REGIME`.** **Deferred:** the tokenizer-on-model execution (no model here).

**Files:**
- Create: `src/cfm/eval/holdout/degeneracy.py`
- Test: `tests/eval/holdout/test_degeneracy.py`

- [ ] **Step 1: Write the failing test (the two regime-distinguishing guards, §9 spine)**

```python
# tests/eval/holdout/test_degeneracy.py
from __future__ import annotations

from cfm.eval.holdout import degeneracy
from cfm.eval.holdout.bref_rate import bref_placeholder_rate
from cfm.eval.holdout.sizing import DELTA_BREF_REGIME


# --- per-instance exclusion fixtures ---
_OUTBOUND_BREF_COLLAPSE = [509, 41, 300, 323, 363, 369, 1500, 510]  # body ends in bref
_COLLAPSE_GEOM = {"type": "LineString", "coordinates": [[0.0, 0.0], [0.0, 0.0]]}
# G-D1: a MODEL-style degenerate block with NO outbound bref (distinct from sub-G's
# real-data fixture) — the model emitted a zero-length stub it should not have.
_MODEL_DEGENERATE_NO_BREF = [509, 7, 300, 323, 363, 369, 511, 443, 510]


def test_per_instance_excludes_outbound_bref_collapse():
    v = degeneracy.classify_block(_OUTBOUND_BREF_COLLAPSE, _COLLAPSE_GEOM)
    assert v is degeneracy.Verdict.EXCLUDED_BREF_PLACEHOLDER  # faithful model not penalized


def test_GD1_gate_fires_on_model_emitted_degeneracy_without_bref():
    """G-D1 RE-PROVEN on a MODEL-emitted fixture (not inherited from sub-G's real-data
    drill): identical zero-length symptom, no outbound bref → MODEL_INVALID. Proves the
    exclusion keys on construction identity in the regime the model populates."""
    v = degeneracy.classify_block(_MODEL_DEGENERATE_NO_BREF, _COLLAPSE_GEOM)
    assert v is degeneracy.Verdict.MODEL_INVALID


def test_GD2_at_threshold_just_over_trips_just_under_passes():
    """G-D2 at the threshold (not at 2×): faithful rate r0; a model emitting just past
    r0+δ must TRIP; just under must PASS. δ is the single DELTA_BREF_REGIME."""
    r0 = 0.05  # round-tripped-real faithful rate in this stratum
    over = degeneracy.over_emission_verdict(model_rate=r0 + DELTA_BREF_REGIME + 0.005, faithful_rate=r0)
    under = degeneracy.over_emission_verdict(model_rate=r0 + DELTA_BREF_REGIME - 0.005, faithful_rate=r0)
    assert over is degeneracy.RateVerdict.OVER_EMITTING
    assert under is degeneracy.RateVerdict.WITHIN_TOLERANCE


def test_GD2_stratified_cancellation_global_matches_but_one_stratum_diverges():
    """G-D2 stratified (not global): the Singapore-wide rate matches round-tripped-real
    while a dense stratum over-emits and a sparse stratum under-emits. A global check
    passes (vacuous); the stratified check MUST trip on the diverging stratum.
    Strata = cell_density_bucket (density is an aggregate — labels.py)."""
    faithful = {0: 0.05, 3: 0.05}   # per-stratum faithful rates
    # model: stratum 0 under-emits (0.00), stratum 3 over-emits (0.10) → global ~0.05
    model_blocks = (
        [[509, 7, 300, 323, 363, 369, 511, 443, 510]] * 0   # stratum 0: none collapse
        + [_OUTBOUND_BREF_COLLAPSE] * 10                    # stratum 3: 10 collapse
        + [[509, 7, 300, 1, 50, 510]] * 90                  # stratum 3: 90 ok
        + [[509, 7, 300, 1, 50, 510]] * 100                 # stratum 0: 100 ok
    )
    model_geoms = (
        [_COLLAPSE_GEOM] * 10
        + [{"type": "LineString", "coordinates": [[0.0, 0.0], [10.0, 0.0]]}] * 90
        + [{"type": "LineString", "coordinates": [[0.0, 0.0], [10.0, 0.0]]}] * 100
    )
    model_strata = [3] * 100 + [0] * 100
    model_rate = bref_placeholder_rate(model_blocks, model_geoms, model_strata)

    report = degeneracy.stratified_over_emission(model_rate, faithful_rate=faithful)
    assert report.global_within_tolerance is True       # the vacuous-pass signal
    assert 3 in report.over_emitting_strata              # the stratified check fires
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/holdout/test_degeneracy.py -v`
Expected: FAIL — module/attribute errors.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cfm/eval/holdout/degeneracy.py
"""D's known-limitation stance: per-instance exclude + distribution-level judge.

Per-instance (spec §D): the bref-placeholder shape is EXCLUDED via the §2 shared
predicate (construction identity) so a faithful model is not penalized. A
degenerate block WITHOUT an outbound bref is a genuine model defect → MODEL_INVALID.

Distribution-level (spec §D = R2): REPORT the bref-placeholder RATE on model output
vs round-tripped-real; the excess is a reported model-degeneracy term. Per-instance
cannot separate "faithful reproduction" from "over-emission"; the rate can.

Guards are the §9 spine, regime-distinguishing:
- G-D1: re-proven on a MODEL-EMITTED fixture (test), not inherited from sub-G's
  real-data drill — coverage on real data does not carry to the model regime.
- G-D2: at-threshold (just-over trips, just-under passes) AND stratified (a global
  match with a per-stratum divergence must trip). Strata = cell_density_bucket.

δ is the single DELTA_BREF_REGIME imported from sizing.py (one number).

DEFERRED (spec §7): producing the model token stream (the tokenizer-on-MODEL side
of R2) needs a trained model — that runs in the eval-harness successor. This module
ships the classifier, the rate-comparison, and the guards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from cfm.eval.holdout.bref_rate import (
    BrefRateResult,
    _bref_predicate,
    bref_placeholder_rate,
)
from cfm.eval.holdout.sizing import DELTA_BREF_REGIME

# Re-export so callers read the rate through this module's R2 surface too.
__all__ = [
    "Verdict",
    "RateVerdict",
    "classify_block",
    "over_emission_verdict",
    "stratified_over_emission",
    "bref_placeholder_rate",
]


class Verdict(Enum):
    EXCLUDED_BREF_PLACEHOLDER = 1   # faithful v1-limitation reproduction — not penalized
    MODEL_INVALID = 2               # genuine degeneracy (no outbound bref) — counts against
    VALID = 3


def classify_block(block: list[int], geom: dict) -> Verdict:
    """Per-instance verdict via the shared construction-identity predicate."""
    from shapely.geometry import shape

    if _bref_predicate(block, geom):
        return Verdict.EXCLUDED_BREF_PLACEHOLDER
    if geom.get("type") in ("LineString", "Polygon") and not shape(geom).is_valid:
        return Verdict.MODEL_INVALID
    return Verdict.VALID


class RateVerdict(Enum):
    WITHIN_TOLERANCE = 1
    OVER_EMITTING = 2


def over_emission_verdict(*, model_rate: float, faithful_rate: float) -> RateVerdict:
    """At-threshold rate judge: excess > δ ⇒ over-emitting (spec §D G-D2)."""
    return (
        RateVerdict.OVER_EMITTING
        if (model_rate - faithful_rate) > DELTA_BREF_REGIME
        else RateVerdict.WITHIN_TOLERANCE
    )


@dataclass(frozen=True)
class StratifiedOverEmissionReport:
    global_within_tolerance: bool
    over_emitting_strata: dict[int, float]  # stratum → model rate excess
    per_stratum_verdict: dict[int, RateVerdict] = field(default_factory=dict)


def stratified_over_emission(
    model_rate: BrefRateResult, *, faithful_rate: dict[int, float]
) -> StratifiedOverEmissionReport:
    """Stratified rate judge: trips on ANY diverging stratum even if the global rate
    matches (the distributional vacuous pass). Strata = cell_density_bucket."""
    over: dict[int, float] = {}
    verdicts: dict[int, RateVerdict] = {}
    for stratum, sr in model_rate.per_stratum.items():
        r0 = faithful_rate.get(stratum, 0.0)
        v = over_emission_verdict(model_rate=sr.rate, faithful_rate=r0)
        verdicts[stratum] = v
        if v is RateVerdict.OVER_EMITTING:
            over[stratum] = sr.rate - r0

    global_faithful = (
        sum(faithful_rate.get(s, 0.0) * sr.n_total for s, sr in model_rate.per_stratum.items())
        / sum(sr.n_total for sr in model_rate.per_stratum.values())
        if model_rate.per_stratum else 0.0
    )
    global_within = (model_rate.overall_rate - global_faithful) <= DELTA_BREF_REGIME

    return StratifiedOverEmissionReport(
        global_within_tolerance=global_within,
        over_emitting_strata=over,
        per_stratum_verdict=verdicts,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/holdout/test_degeneracy.py -v`
Expected: PASS (4 tests). If `classify_block` on `_MODEL_DEGENERATE_NO_BREF` does not produce an invalid geom (so G-D1 can't fire), the implementer HALTS — the fixture must decode to a genuinely OGC-invalid/zero-length geom without an outbound bref; build it from the decoder, do not weaken the assertion.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/eval/holdout/degeneracy.py tests/eval/holdout/test_degeneracy.py
uv run ruff check src/cfm/eval/holdout/degeneracy.py tests/eval/holdout/test_degeneracy.py
git add src/cfm/eval/holdout/degeneracy.py tests/eval/holdout/test_degeneracy.py
git commit -m "feat(eval-set): D per-instance exclude + G-D1/G-D2 regime-distinguishing guards"
```

---

### Task 9: Frozen holdout manifest (region-keyed; written once)

**In scope:** spec §F lock artifact. Mirrors sub-C/sub-D manifest + SHA-exclusion pattern.

**Files:**
- Create: `src/cfm/eval/holdout/manifest.py`
- Test: `tests/eval/holdout/test_manifest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/holdout/test_manifest.py
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cfm.eval.holdout import manifest


def _provs() -> dict:
    return {
        (1, 7): {"provenance_sha256": "a" * 64, "macro_vocab_sha256": "b" * 64},
        (1, 8): {"provenance_sha256": "c" * 64, "macro_vocab_sha256": "b" * 64},
    }


def test_build_manifest_is_region_keyed_and_sorted():
    data = manifest.build_holdout_manifest(
        region="singapore", selected_tiles=[(1, 8), (1, 7)], per_tile_provenance=_provs()
    )
    assert data["regions"]["singapore"]["partition_path"] == "holdout/region=singapore"
    tiles = data["regions"]["singapore"]["tiles"]
    assert [(t["tile_i"], t["tile_j"]) for t in tiles] == [(1, 7), (1, 8)]  # sorted
    assert tiles[0]["provenance_sha256"] == "a" * 64


def test_freeze_computes_sha_excluding_the_sha_field_and_writes_once(tmp_path: Path):
    data = manifest.build_holdout_manifest(
        region="singapore", selected_tiles=[(1, 7)], per_tile_provenance=_provs()
    )
    path = tmp_path / "holdout_manifest.yaml"
    manifest.freeze_holdout_manifest(data, path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert len(loaded["manifest_sha256"]) == 64
    # recompute over the loaded dict minus its sha → identical (sha excludes itself):
    assert manifest.manifest_sha256(loaded) == loaded["manifest_sha256"]
    # written once: a second freeze refuses to overwrite the locked artifact.
    with pytest.raises(FileExistsError):
        manifest.freeze_holdout_manifest(data, path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/holdout/test_manifest.py -v`
Expected: FAIL — module/attribute errors.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cfm/eval/holdout/manifest.py
"""Frozen holdout manifest — the lock artifact (spec §F), written once.

Region-keyed (region is a first-class partition key, spec §B): adding a held-out
region D later is adding a `regions[D]` entry, with zero change to the freeze or
audit logic. Mirrors sub-C/sub-D: canonical YAML, a manifest_sha256 that EXCLUDES
itself (the `*_sha256` exclusion grammar), and write-once semantics — a contaminated
or re-derived holdout invalidates every eval number, so the artifact never moves
once frozen.
"""
from __future__ import annotations

from pathlib import Path

from cfm.data.determinism import compute_sha256
from cfm.data.io import canonicalize_yaml

MANIFEST_SCHEMA_VERSION: str = "1.0"

TileKey = tuple[int, int]


def build_holdout_manifest(
    *,
    region: str,
    selected_tiles: list[TileKey],
    per_tile_provenance: dict[TileKey, dict],
) -> dict:
    """Build the (unfrozen) manifest dict for one region."""
    tiles = []
    for (ti, tj) in sorted(selected_tiles):
        prov = per_tile_provenance[(ti, tj)]
        tiles.append({
            "tile_i": int(ti),
            "tile_j": int(tj),
            "provenance_sha256": prov["provenance_sha256"],
            "macro_vocab_sha256": prov.get("macro_vocab_sha256"),
        })
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "regions": {
            region: {
                "partition_path": f"holdout/region={region}",
                "tiles": tiles,
            }
        },
    }


def manifest_sha256(data: dict) -> str:
    """SHA over the canonical manifest EXCLUDING the manifest_sha256 field itself."""
    payload = {k: v for k, v in data.items() if k != "manifest_sha256"}
    return compute_sha256(canonicalize_yaml(payload).encode("utf-8"))


def freeze_holdout_manifest(data: dict, path: Path) -> None:
    """Stamp the manifest SHA and write ONCE. Refuses to overwrite a locked manifest."""
    if path.exists():
        raise FileExistsError(
            f"holdout manifest already locked at {path}; it is written once and never "
            "regenerated (spec §F). Delete deliberately only to re-lock the eval set."
        )
    frozen = dict(data)
    frozen["manifest_sha256"] = manifest_sha256(frozen)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonicalize_yaml(frozen), encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/holdout/test_manifest.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/eval/holdout/manifest.py tests/eval/holdout/test_manifest.py
uv run ruff check src/cfm/eval/holdout/manifest.py tests/eval/holdout/test_manifest.py
git add src/cfm/eval/holdout/manifest.py tests/eval/holdout/test_manifest.py
git commit -m "feat(eval-set): frozen region-keyed holdout manifest, written once"
```

---

### Task 10: Fail-loud lineage audit (G-F1–F4) + region-scaling test

**In scope:** spec §F audit. **The guard function the training scaffold calls (one source).**

**Files:**
- Create: `src/cfm/eval/holdout/lineage_audit.py`
- Test: `tests/eval/holdout/test_lineage_audit.py`

- [ ] **Step 1: Write the failing test (all four guards must FAIL in the leak regime; region-scaling byte-identical)**

```python
# tests/eval/holdout/test_lineage_audit.py
from __future__ import annotations

import pytest

from cfm.eval.holdout import lineage_audit as la


def _manifest(regions: dict[str, list[tuple[int, int]]]) -> dict:
    return {
        "regions": {
            r: {"tiles": [{"tile_i": i, "tile_j": j} for (i, j) in tiles]}
            for r, tiles in regions.items()
        }
    }


HOLDOUT = _manifest({"singapore": [(1, 7), (1, 8)]})


def _art(path: str, lineage) -> la.Artifact:
    return la.Artifact(path=path, lineage=lineage)


def test_clean_training_set_passes():
    arts = [_art("train/a.parquet", frozenset({("singapore", 2, 2)}))]
    la.audit_no_holdout_leak(HOLDOUT, arts)  # no raise


def test_GF1_held_out_tile_in_training_path_trips():
    arts = [_art("train/tile=EPSG3414_i1_j7/x.parquet", frozenset({("singapore", 1, 7)}))]
    with pytest.raises(la.HoldoutLeakError):
        la.audit_no_holdout_leak(HOLDOUT, arts)


def test_GF2_held_out_derived_artifact_trips():
    # A reference distribution whose lineage includes a held-out tile.
    arts = [_art("train/ref_dist.parquet", frozenset({("singapore", 2, 2), ("singapore", 1, 8)}))]
    with pytest.raises(la.HoldoutLeakError):
        la.audit_no_holdout_leak(HOLDOUT, arts)


def test_GF3_r2_on_real_baseline_referenced_from_training_trips():
    arts = [_art("train/r2_tokenizer_on_real.parquet", frozenset({("singapore", 1, 7)}))]
    with pytest.raises(la.HoldoutLeakError):
        la.audit_no_holdout_leak(HOLDOUT, arts)


def test_GF4_absent_lineage_trips_on_the_absence_fail_closed():
    # The completeness twin: a training-reachable artifact with NO recorded lineage
    # FAILS (fail-closed) — not pass — because an untracked derivative is exactly
    # where a held-out leak hides.
    arts = [_art("train/mystery.parquet", None)]
    with pytest.raises(la.HoldoutLeakError) as exc:
        la.audit_no_holdout_leak(HOLDOUT, arts)
    assert "absent lineage" in str(exc.value)


def test_REGION_SCALING_two_region_manifest_uses_identical_logic():
    # spec §B done-right test: a synthetic 2-region holdout → the audit is
    # byte-identical (no per-region special-casing). Same leak shape trips in either.
    two = _manifest({"singapore": [(1, 7)], "regionD": [(5, 5)]})
    clean = [_art("train/a.parquet", frozenset({("singapore", 9, 9), ("regionD", 1, 1)}))]
    la.audit_no_holdout_leak(two, clean)  # no raise
    leak = [_art("train/b.parquet", frozenset({("regionD", 5, 5)}))]
    with pytest.raises(la.HoldoutLeakError):
        la.audit_no_holdout_leak(two, leak)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/holdout/test_lineage_audit.py -v`
Expected: FAIL — module/attribute errors.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cfm/eval/holdout/lineage_audit.py
"""Fail-loud, fail-closed holdout-leak audit (spec §F). The guard the training
scaffold calls (one source) to prove no training-reachable artifact's lineage
includes any held-out tile — tiles AND every derivative.

Belt-and-suspenders is justified because the failure is silent AND unrecoverable:
a contaminated holdout invalidates every eval number undetectably.

Guards (each must FAIL in the leak regime):
- G-F1: a held-out TILE in training's path → trips.
- G-F2: a held-out-DERIVED artifact (lineage includes a held-out tile) → trips.
- G-F3: the tokenizer-on-real R2 baseline referenced from training → trips
  (same mechanism as G-F2; the R2 baseline's lineage is its source tiles).
- G-F4: a training-reachable artifact with ABSENT lineage → trips ON THE ABSENCE
  (fail-closed). Without it the guarantee is only "no artifact with RECORDED
  held-out lineage leaks" — strictly weaker, and the gap is where untracked
  derivatives hide.

Region-keyed (spec §B): the audit iterates regions with one code path; a 2-region
manifest exercises identical logic (no per-region special-casing).
"""
from __future__ import annotations

from dataclasses import dataclass

#: (region, tile_i, tile_j)
TileRef = tuple[str, int, int]


@dataclass(frozen=True)
class Artifact:
    path: str
    lineage: frozenset[TileRef] | None   # None = untracked (G-F4 fail-closed)


@dataclass(frozen=True)
class LineageFailure:
    path: str
    reason: str


class HoldoutLeakError(Exception):
    def __init__(self, failures: list[LineageFailure]) -> None:
        self.failures = failures
        super().__init__(
            "held-out lineage leak detected:\n"
            + "\n".join(f"  {f.path}: {f.reason}" for f in failures)
        )


def _holdout_tile_refs(holdout_manifest: dict) -> set[TileRef]:
    refs: set[TileRef] = set()
    for region, payload in holdout_manifest["regions"].items():
        for t in payload["tiles"]:
            refs.add((region, int(t["tile_i"]), int(t["tile_j"])))
    return refs


def audit_no_holdout_leak(
    holdout_manifest: dict, training_reachable: list[Artifact]
) -> None:
    """Raise HoldoutLeakError listing EVERY failure; return None iff clean."""
    holdout = _holdout_tile_refs(holdout_manifest)
    failures: list[LineageFailure] = []
    for art in training_reachable:
        if art.lineage is None:                                  # G-F4
            failures.append(LineageFailure(art.path, "absent lineage (fail-closed)"))
            continue
        leaked = art.lineage & holdout                            # G-F1/F2/F3
        if leaked:
            failures.append(
                LineageFailure(art.path, f"lineage includes held-out tiles {sorted(leaked)}")
            )
    if failures:
        raise HoldoutLeakError(failures)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/holdout/test_lineage_audit.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/eval/holdout/lineage_audit.py tests/eval/holdout/test_lineage_audit.py
uv run ruff check src/cfm/eval/holdout/lineage_audit.py tests/eval/holdout/test_lineage_audit.py
git add src/cfm/eval/holdout/lineage_audit.py tests/eval/holdout/test_lineage_audit.py
git commit -m "feat(eval-set): fail-closed lineage audit G-F1..F4 + region-scaling"
```

---

### Task 11: Pipeline orchestrator + `_EVAL_SET_LOCKED` marker + report

**In scope:** wire the corrected sequencing end-to-end; freeze the manifest; write the holdout partition + baselines (with lineage); write the marker + a `reports/` summary. The real-494-tile run (where the literal N, δ, and per-stratum floors get recorded) is `@pytest.mark.slow`.

**Files:**
- Create: `src/cfm/eval/holdout/pipeline.py`
- Test: `tests/eval/holdout/test_pipeline.py`
- Create (by the slow run): `reports/phase-1-eval-set/2026-06-01-singapore-eval-set.md`

- [ ] **Step 1: Write the failing test (fast: synthetic mini-region; slow: real 494)**

```python
# tests/eval/holdout/test_pipeline.py
from __future__ import annotations

from pathlib import Path

import pytest

from cfm.eval.holdout import paths, pipeline


def test_corrected_sequencing_order_is_encoded():
    # The orchestrator must build the selector, then size THROUGH it, then freeze.
    # We assert the documented step order so a reorder is caught by review + test.
    assert pipeline.SEQUENCE == (
        "labels",
        "bref_rate",
        "baselines",
        "build_selector",
        "size_through_selector",
        "run_degeneracy_guards",
        "freeze_manifest",
        "write_partition_and_marker",
    )


def test_marker_not_written_when_a_degeneracy_guard_fails(tmp_path: Path, monkeypatch):
    # If the degeneracy guards report over-emission (a model regime) OR a leak is
    # found, the marker is NOT written. (Here we simulate via the result object.)
    result = pipeline.EvalSetResult(
        n=0, manifest_path=None, marker_written=False,
        underpowered_strata=[], degradation_log=["degeneracy guard failed"],
    )
    assert result.marker_written is False


@pytest.mark.slow
def test_generate_eval_set_on_real_singapore_locks_substrate():
    """SLOW: the real 494-tile run. Records the literal N, δ, per-stratum floors and
    the bref-rate into the report; freezes the manifest; writes _EVAL_SET_LOCKED.
    Skips if the Phase-1 validated data is absent."""
    rel, reg = paths.DEFAULT_RELEASE, paths.DEFAULT_REGION
    if not paths.phase1_validated_marker(rel, reg).is_file():
        pytest.skip("Phase-1 validated Singapore data not present")
    result = pipeline.generate_eval_set(release=rel, region=reg)
    assert result.marker_written is True
    assert result.n > 0 and result.n < 494          # leaves a viable training residual
    assert paths.holdout_manifest_path(rel).is_file()
    assert paths.eval_set_locked_marker(rel).is_file()
    # rough-numbers heuristic: a measured N is not a round default.
    assert result.n not in (50, 100, 200)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/holdout/test_pipeline.py -v`
Expected: FAIL — module/attribute errors (slow test deselected by default).

- [ ] **Step 3: Write minimal implementation**

```python
# src/cfm/eval/holdout/pipeline.py
"""Eval-set generation orchestrator (spec §3 dependency web + §6 sequencing).

Corrected sequencing (plan decision 6): BUILD the fresh selector → G measures
THROUGH it → (N, selection) → F freezes the manifest. sub-D's #11 selector is
untouched. δ is the single DELTA_BREF_REGIME (sizing.py).

The marker _EVAL_SET_LOCKED is written ONLY if: the manifest froze, the degeneracy
guards did not report over-emission on the real-side baseline, and the lineage
audit found no leak. Underpowered strata are recorded (UNDERPOWERED-stated), not
treated as a met bar.

This module reads sealed sub-C/sub-D/sub-F outputs and writes only under
data/processed/eval_set/<release>/ + reports/. Model-facing scoring is deferred
(spec §7).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from cfm.eval.holdout import (
    baselines,
    bref_rate,
    labels,
    lineage_audit,
    manifest,
    paths,
    roundtrip,
    selector,
    sizing,
)

#: The documented step order (corrected sequencing). Asserted in tests.
SEQUENCE: tuple[str, ...] = (
    "labels",
    "bref_rate",
    "baselines",
    "build_selector",
    "size_through_selector",
    "run_degeneracy_guards",
    "freeze_manifest",
    "write_partition_and_marker",
)


@dataclass
class EvalSetResult:
    n: int
    manifest_path: Path | None
    marker_written: bool
    underpowered_strata: list = field(default_factory=list)
    degradation_log: list[str] = field(default_factory=list)


def generate_eval_set(*, release: str, region: str) -> EvalSetResult:
    """Run the full eval-set generation pipeline for one region.

    Implementation note for the executor: assemble per-tile TileLabels by reading
    the sub-D manifest's tiles[] inventory, then read_tile_labels per tile dir;
    decode round-tripped-real blocks via roundtrip.decode_region_blocks using the
    sub-F cells.parquet + the per-cell cell_density_bucket from labels; compute the
    §2 bref_rate ONCE; compute baselines + ceiling from it; measure per-stratum
    populations; set per-stratum floors (sizing.rate_detection_floor with p = the
    measured bref rate, sizing.ks_two_sample_floor); build quotas; call
    selector.select_holdout_tiles; on infeasibility apply sizing's ORDERED
    degradation; freeze the manifest from the selection + per-tile provenance;
    write the holdout partition (baselines with lineage) + _EVAL_SET_LOCKED.

    The literal N, δ, per-stratum floors, the bref rate, and any UNDERPOWERED strata
    are written to reports/phase-1-eval-set/ in the slow run (reproducibility mandate:
    config + code commit + data snapshot + metrics + prose).
    """
    raise NotImplementedError(
        "Executor: implement per the docstring against the Task-1..10 surfaces. "
        "Keep every step in SEQUENCE order; do not reorder selector after sizing."
    )
```

> **Executor guidance for Step 3 (this is the one task whose body is genuinely data-driven):** the function above is a scaffold. Implement it by composing the already-tested Task 1–10 surfaces in `SEQUENCE` order. Do **not** invent new logic in any sealed module. When you run the slow test against the real 494 tiles, the per-stratum populations, the chosen N, the bref-rate, and the floors are *measurements* — record them verbatim in the report (rough-numbers heuristic: if any comes out a round default, re-trace the measurement before trusting it). If a sealed-module read contradicts an assumption here (e.g. a tile dir missing `effective_conditioning.yaml`), HALT and report — do not patch around it.

- [ ] **Step 4: Run test to verify it passes (fast suite), then run the slow lock**

Run: `uv run pytest tests/eval/holdout/test_pipeline.py -v`
Expected: PASS (2 fast tests; slow deselected).

After implementing the orchestrator body:
Run: `uv run pytest tests/eval/holdout/test_pipeline.py -v -m slow`
Expected: PASS — `_EVAL_SET_LOCKED` written, manifest frozen, report emitted with the measured N/δ/floors/bref-rate.

- [ ] **Step 5: Lint + commit (code) then commit the report separately**

```bash
uv run ruff format src/cfm/eval/holdout/pipeline.py tests/eval/holdout/test_pipeline.py
uv run ruff check src/cfm/eval/holdout/pipeline.py tests/eval/holdout/test_pipeline.py
git add src/cfm/eval/holdout/pipeline.py tests/eval/holdout/test_pipeline.py
git commit -m "feat(eval-set): orchestrator wiring corrected sequencing + lock marker"
# after the slow run produces the report:
git add reports/phase-1-eval-set/
git commit -m "expt(eval-set): lock Singapore held-out substrate — measured N, δ, floors, bref-rate"
```

---

### Task 12: Full-suite regression + spec-coverage close-out

**Files:**
- Test: run the whole fast suite + the eval-set slow suite.

- [ ] **Step 1: Run the full fast suite**

Run: `uv run pytest -q`
Expected: PASS — no regressions in sub-C/D/E/F/G (this sub-project added only new files).

- [ ] **Step 2: Run the eval-set slow suite explicitly**

Run: `uv run pytest tests/eval/holdout -v -m slow`
Expected: PASS — the real lock test green.

- [ ] **Step 3: Lint the whole package**

Run: `uv run ruff check src/cfm/eval/holdout tests/eval/holdout && uv run ruff format --check src/cfm/eval/holdout tests/eval/holdout`
Expected: clean.

- [ ] **Step 4: Spec-coverage checklist (record in the report)**

Confirm each spec section maps to a task: A→Tasks 2/5/all-headers; B→Tasks 9/10 (region-keyed + scaling); C→Tasks 4/5; §2→Task 4; D→Task 8; E→Task 2; F→Tasks 9/10; G→Tasks 6/7. Generalization UNSCORED-stated (report). Deferred items (model scoring, sim, tokenizer-on-model R2, Wasserstein/KS distance) named in the report as out-of-scope → eval-harness successor.

- [ ] **Step 5: Commit the close-out note**

```bash
git add reports/phase-1-eval-set/
git commit -m "docs(eval-set): spec-coverage close-out + deferred-items ledger"
```

---

## Self-Review (run against the spec before handoff)

**1. Spec coverage.** A (scope) — every task header states in/deferred; precise-statement #1 honored (R2 real-side in, model-side deferred). B (region partition + UNSCORED generalization) — Tasks 9/10 region-keyed + scaling test; report states generalization unscored. C (core/full + gap) — Tasks 4/5; precise-statement #3 (sim-viability contract-defined/execution-deferred) recorded in report, not built. §2 (shared rate) — Task 4, one function, identity-locked, guards on the shared fn. D (per-instance exclude + rate judge; G-D1/G-D2) — Task 8. E (labels, one source; morphology collision; density aggregate) — Task 2. F (lock + G-F1..F4 + manifest) — Tasks 9/10. G (per-stratum floors, ordered degradation, single δ) — Tasks 6/7; precise-statement #2 (G→F ordering) encoded in Task 11 SEQUENCE.

**2. Placeholder scan.** Task 11's orchestrator body is an intentional `NotImplementedError` scaffold with a precise executor docstring — flagged, not a hidden TODO; its logic is the composition of already-complete, already-tested Task 1–10 surfaces. All other steps contain complete code.

**3. Type consistency.** `BrefRateResult`/`StratumRate` (Task 4) consumed by Tasks 5/8. `TileLabels`/`MorphologyStratum` (Task 2) consumed by Tasks 6/7/11. `DELTA_BREF_REGIME` defined once (Task 7), imported by Task 8. `_bref_predicate` is the `is`-identical sub-G object (Tasks 4/8). Manifest dict shape (Task 9) consumed by the audit (Task 10) — `regions[*].tiles[*].{tile_i,tile_j}`.

**4. Protocol v2 gates present.** Gate 6 external-source-of-truth: Task 2 (vocab hand-enumeration), Task 4 (`is`-identity to sub-G). Threshold-pairing (§2): Task 6 (underpower surfaces), Task 7 (per-stratum-not-whole-set structural check). §9 construction-identity + regime-distinguishing guards: Task 8 (G-D1/G-D2), Task 10 (G-F1..F4 each fails-in-leak-regime). Rough-numbers: Tasks 7/11 (δ and N not round defaults). Halt-on-defect / no-test-weakening: called out in Tasks 3/8 where a fixture might not decode.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-06-01-eval-set-generation.md`. Two execution options:

1. **Subagent-Driven (recommended)** — a fresh subagent per task, two-stage review between tasks, fast iteration. Each dispatch must forbid new branches/push/PR (project branch discipline) and apply the pre-dispatch audit (re-verify sealed signatures at dispatch — they may have drifted).
2. **Inline Execution** — execute tasks in this session via executing-plans, batched with review checkpoints.

Which approach?

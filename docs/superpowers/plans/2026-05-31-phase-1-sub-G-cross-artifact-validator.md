# Phase-1 sub-G — cross-artifact consistency validator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build PRD §5 stage-five — the cross-artifact consistency validator that runs sub-E→sub-F on a region (Singapore), then independently checks the three inter-stage seams and gates the region with `_PHASE1_VALIDATED`.

**Architecture:** A thin subprocess chain runner (resume-from-`_SUCCESS`) materializes the missing real sub-E/sub-F caches, then a validator runs three *independent* seam checks over every tile, accumulates failures grouped by signature, writes `quarantine_report.yaml` + `_PHASE1_ACCURACY_BASELINE.yaml` every run, and writes `_PHASE1_VALIDATED` iff the quarantine is empty AND the seam-3 sanity floor holds. Independence is by *provenance*: every check derives its expected value from a spec clause **outside** the stage it validates, never by reusing that stage's own reader/derivation code.

**Tech Stack:** Python 3.11+, `pyarrow.parquet` (reads via `pq.ParquetFile(path).read()` — never bare `read_table`), `shapely` (WKB + geometry validity), `pydantic`-free dataclasses, `pytest`, `ruff`. Reuses sub-F's `decoder.decode_feature` (seam 3) and sub-D's `read_macro_core_parquet` (pure byte-read). Authority: spec `docs/superpowers/specs/2026-05-31-phase-1-sub-G-cross-artifact-validator-design.md` (v3) + `docs/protocols/sub-project-planning-protocol-v1.md`.

---

## Design rules (from spec §3 — apply to every task)

1. **Independence-by-construction** — a check that passes because both sides agree on the same wrong thing guards nothing.
2. **Measure-from-source** — derive the expected value from upstream source/spec; never reuse the validated stage's reader (specifically: **never import `sub_f.boundary_contract.load_boundary_contract` or `sub_f.encoder._classify_feature_for_bref`** in seam 2; never read sub-D's verdict in seam 1).
3. **Provenance-citation** — every structural invariant cites a spec clause *outside the stage it validates*; circular-by-provenance invariants are rejected/deferred.
4. **Halt-and-revisit** — the first real run is a *measurement*; expect sub-F §8 defects. No push-through.
5. **Action contract per measurement** — every reported number has a defined response (sanity floor / deferral trigger).

**Verified upstream contract facts (Gate-2 / protocol §3, read 2026-05-31):**
- sub-E `boundary_contract.parquet` (`src/cfm/data/sub_e/writer.py:38-48`): 7 cols `slot_kind`(int8), `slot_index`(int16), `lower_cell_i`(int8), `lower_cell_j`(int8), `axis`(int8), `scope_marker`(int8), `boundary_class_enum`(int16, nullable). 144 rows/tile (112 internal + 32 external). `SlotKind` INTERNAL_EDGE=1, EXTERNAL_EDGE=2.
- `BoundaryClass` (`src/cfm/data/sub_e/derivation.py:19-23`): BOUNDARY_NOT_APPLICABLE=0 (never on-disk), NONE=1, MAJOR_ROAD=2, MINOR_ROAD=3. **NULL-vs-enum invariant:** `boundary_class_enum` non-null iff `scope_marker == 0`. **Active-emission subset** (rows that could yield a bref) = `scope_marker == 0 AND boundary_class_enum ∈ {2,3}`.
- sub-F `cells.parquet` (`src/cfm/data/sub_f/io.py:27-36`): `cell_i`,`cell_j`,`cell_slot_index`,`token_sequence`(list<int16>),`feature_count`,`provenance_sha256`. 64 rows/tile; `cell_slot_index == cell_i*8+cell_j`.
- bref IDs (`configs/sub_f/boundary_reference_vocab.yaml:29-68`): 1500=`N_MAJOR`,1501=`E_MAJOR`,1502=`S_MAJOR`,1503=`W_MAJOR`,1504=`N_MINOR`,1505=`E_MINOR`,1506=`S_MINOR`,1507=`W_MINOR`.
- feature split (`src/cfm/data/sub_f/encoder.py:214-215`): `<feature>`=509, `<feature_end>`=510. Per-feature order (`encoder.py:436-456`): `[509, semantic_id, (inbound_bref if C/D), anchor×4, (dir,mag) pairs, (outbound_bref if B/D), 510]`.
- encoder bref rule (`encoder.py:471-527` `_classify_feature_for_bref`): LineString only; `coords[0]`→inbound edge, `coords[-1]`→outbound edge, edge test on `cell_origin=(0,0)`/extent 250m (W: x≈0, E: x≈250, S: y≈0, N: y≈250); class from contract; emit only MAJOR/MINOR.
- `encode_tile` (`src/cfm/data/sub_f/pipeline_writer.py:63-120`): groups sub-C features by their `(cell_i,cell_j)` column, `wkb_loads(r["geometry"])` **as-is**, calls `encode_cell(..., cell_origin default (0,0))`, **sub-C row order preserved**, Multi* split into parts (one token-feature per part).
- sub-D `macro_core.parquet` (`src/cfm/data/sub_d/io.py:41-55`, reader `read_macro_core_parquet` `:117`): targets `zoning_class`,`cell_density_bucket` on `SlotKind.CELL` slots; `road_skeleton_class` on `SlotKind.INTERNAL_EDGE` slots; all `int16`, populated only on active slots (else `None`).
- locked bins (`configs/macro_plan/v1/macro_plan_vocab.yaml`): `cell_density` `:3472-3488` `[0,0.05),[0.05,0.15),[0.15,0.35),[0.35,∞)`; `road_skeleton` `:3489-3505` `[0,1),[1,4),[4,9),[9,∞)`.
- density metric (`src/cfm/data/sub_d/evidence.py:145-193`): `building_footprint_ratio = Σ(area of feature_class==BUILDING polygons)/cell_area_admin_clipped_m2`. road-skeleton metric (`evidence.py:201-245`): count of `crossings` rows whose `source_feature_id` joins a `feature_class==ROAD` feature, per `(lower_cell_i,lower_cell_j,axis)`.
- chain invocation (subprocess; spec §9 #5 = subprocess): sub-E `scripts/derive_boundary_contracts.py --release --region --sub-c-region-dir --sub-d-region-dir --output-region-dir [--lever-3-collapse]`; sub-F `scripts/sub_f/derive.py --release --region --sub-c-region-dir --sub-d-region-dir --sub-e-region-dir --output-region-dir [--extracted-utc] [--no-alpha-drop-report]`. Each writes `<output-region-dir>/_SUCCESS`. Region layout: `data/processed/sub_{c,d,e,f}/<release>/<region>/{_SUCCESS, manifest.yaml, tile=EPSG3414_i{i}_j{j}/}`.

---

## File structure

| File | Responsibility |
|---|---|
| `src/cfm/data/sub_g/__init__.py` | package marker |
| `src/cfm/data/sub_g/diagnostics.py` | `Diagnostic` dataclass (7-field seam shape), signature grouping, `write_quarantine_report`, `write_accuracy_baseline`, `run_metadata` split (stable-in-digest vs excluded) |
| `src/cfm/data/sub_g/readers.py` | sub-G's **independent** parquet readers (sub-E contract raw rows; sub-C features/cells/crossings; sub-F cells) — never sub-F's `load_boundary_contract` |
| `src/cfm/data/sub_g/buckets.py` | `bucket_of(value, edges)` + the locked cut-point loaders from `macro_plan_vocab.yaml` |
| `src/cfm/data/sub_g/seam_macro_geometry.py` | seam 1: SI-1 density, SI-2 road-skeleton, SI-3 zoning |
| `src/cfm/data/sub_g/seam_contract_tokens.py` | seam 2: independent sub-E parse + independent bref prediction from sub-C geometry + sub-F token parse + per-cell bijection |
| `src/cfm/data/sub_g/seam_decodability.py` | seam 3: feature split + decode + GeoJSON validity + structural vertex bound + accuracy-vs-original + decomposition |
| `src/cfm/data/sub_g/versions.py` | `VALIDATOR_VERSION`; `_PHASE1_VALIDATED` writer (digest + run_metadata) |
| `src/cfm/data/sub_g/validator.py` | orchestrate seams over all tiles; accumulate+group; write artifacts; sanity floor; gate |
| `src/cfm/data/sub_g/pipeline.py` | chain runner (subprocess sub-E→sub-F, resume-from-`_SUCCESS`, `--force`), then validator |
| `scripts/sub_g/derive_phase1_region.py` | CLI → chain runner |
| `scripts/sub_g/validate_phase1_region.py` | CLI → validator alone |
| `tests/data/sub_g/…` | per-seam synthetic fixtures + `test_singapore_integration.py` (`@pytest.mark.slow`, fail-loud) |

---

## Task 1: Diagnostics — record, signature grouping, report writers

**Files:**
- Create: `src/cfm/data/sub_g/__init__.py`
- Create: `src/cfm/data/sub_g/diagnostics.py`
- Test: `tests/data/sub_g/test_diagnostics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/sub_g/test_diagnostics.py
from __future__ import annotations

from cfm.data.sub_g.diagnostics import Diagnostic, group_by_signature, render_quarantine_report


def _diag(tile: str, value: float, bucket: str) -> Diagnostic:
    return Diagnostic(
        tile_id=tile,
        invariant_name="density_bucket_matches_footprint",
        artifact_left="sub_c.building_footprint_ratio",
        observed_left=value,
        artifact_right="sub_d.cell_density_bucket",
        observed_right=bucket,
        expected_relationship="ratio in [a,b) implies bucket==k",
        spec_clause_citation="PRD §5 line 65 + macro_plan_vocab.yaml:3472-3488",
        signature="density bucket one-step-too-high vs footprint range",
    )


def test_group_by_signature_collapses_same_pattern():
    diags = [_diag(f"tile=i0_j{j}", 0.40 + j * 0.001, "high") for j in range(100)]
    groups = group_by_signature(diags)
    assert len(groups) == 1
    g = groups[0]
    assert g.invariant_name == "density_bucket_matches_footprint"
    assert g.signature == "density bucket one-step-too-high vs footprint range"
    assert g.instance_count == 100
    assert g.tile_ids == sorted(d.tile_id for d in diags)
    # numeric value-summary present
    assert g.value_summary["observed_left"]["min"] <= g.value_summary["observed_left"]["max"]


def test_groups_sorted_by_count_desc_then_invariant_asc():
    a = [_diag("tile=i0_j0", 0.4, "high")]
    b = [
        Diagnostic("tile=i0_j0", "zzz_invariant", "l", 1, "r", 2, "rel", "cite", "sigZ"),
        Diagnostic("tile=i0_j1", "zzz_invariant", "l", 1, "r", 2, "rel", "cite", "sigZ"),
    ]
    groups = group_by_signature(a + b)
    assert [g.instance_count for g in groups] == [2, 1]  # count desc first


def test_render_is_byte_deterministic_and_empty_record_is_explicit():
    out_empty = render_quarantine_report(groups=[], region="singapore", release="2026-04-15.0",
                                          validator_version="1.0.0")
    assert "groups: []" in out_empty
    # run_metadata volatile fields excluded from the determinism-relevant body:
    a = render_quarantine_report(groups=group_by_signature([_diag("tile=i0_j0", 0.4, "high")]),
                                 region="singapore", release="2026-04-15.0", validator_version="1.0.0")
    b = render_quarantine_report(groups=group_by_signature([_diag("tile=i0_j0", 0.4, "high")]),
                                 region="singapore", release="2026-04-15.0", validator_version="1.0.0")
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/sub_g/test_diagnostics.py -v`
Expected: FAIL with `ModuleNotFoundError: cfm.data.sub_g.diagnostics`

- [ ] **Step 3: Write minimal implementation**

```python
# src/cfm/data/sub_g/__init__.py
"""Phase-1 sub-G cross-artifact consistency validator (PRD stage five)."""
```

```python
# src/cfm/data/sub_g/diagnostics.py
"""Diagnostic record, signature grouping, and byte-deterministic report writers.

A Diagnostic is one cross-artifact seam failure. Diagnostics are grouped by
(invariant_name, signature) where the *signature* is the failure PATTERN, not
the per-tile values (spec Decision 6). Reports are written every run; the empty
record is explicit `groups: []` (positive meaning — run completed, found
nothing), never file-absence (spec Decision 7).

Byte-determinism: the report BODY (groups, sorted) is deterministic; volatile
run-metadata (timestamp/host/uuid/commit) lives in a clearly-marked block
excluded from any digest (sub-C §12.4 + sub-E §9.2 EXCLUDED_FROM_SHA).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from cfm.data.io import canonicalize_yaml


@dataclass(frozen=True)
class Diagnostic:
    tile_id: str
    invariant_name: str
    artifact_left: str
    observed_left: object
    artifact_right: str
    observed_right: object
    expected_relationship: str
    spec_clause_citation: str
    signature: str  # the failure PATTERN; see spec Decision 6


@dataclass(frozen=True)
class DiagnosticGroup:
    invariant_name: str
    signature: str
    instance_count: int
    tile_ids: list[str]
    value_summary: dict
    spec_clause_citation: str
    hypothesis: str | None  # optional; empty is honest (spec Decision 6 obligation 4)


def _summarize(values: list[object]) -> dict:
    """min/max/median for numerics; value:count distribution for categoricals."""
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if nums and len(nums) == len(values):
        return {"min": min(nums), "max": max(nums), "median": median(nums)}
    dist: dict[str, int] = {}
    for v in values:
        dist[str(v)] = dist.get(str(v), 0) + 1
    return {"distribution": dict(sorted(dist.items()))}


def group_by_signature(diags: list[Diagnostic]) -> list[DiagnosticGroup]:
    """Collapse Diagnostics into groups keyed by (invariant_name, signature).

    Sort: instance_count desc, then invariant_name asc (deterministic tiebreak,
    spec Decision 6 obligation 2).
    """
    buckets: dict[tuple[str, str], list[Diagnostic]] = {}
    for d in diags:
        buckets.setdefault((d.invariant_name, d.signature), []).append(d)
    groups: list[DiagnosticGroup] = []
    for (inv, sig), members in buckets.items():
        groups.append(
            DiagnosticGroup(
                invariant_name=inv,
                signature=sig,
                instance_count=len(members),
                tile_ids=sorted(m.tile_id for m in members),
                value_summary={
                    "observed_left": _summarize([m.observed_left for m in members]),
                    "observed_right": _summarize([m.observed_right for m in members]),
                },
                spec_clause_citation=members[0].spec_clause_citation,
                hypothesis=None,
            )
        )
    groups.sort(key=lambda g: (-g.instance_count, g.invariant_name))
    return groups


def _group_to_dict(g: DiagnosticGroup) -> dict:
    d = {
        "invariant_name": g.invariant_name,
        "signature": g.signature,
        "instance_count": g.instance_count,
        "tile_ids": g.tile_ids,
        "value_summary": g.value_summary,
        "spec_clause_citation": g.spec_clause_citation,
    }
    if g.hypothesis:
        d["hypothesis"] = g.hypothesis
    return d


def render_quarantine_report(
    groups: list[DiagnosticGroup], region: str, release: str, validator_version: str
) -> str:
    """Byte-deterministic YAML. `groups: []` is written explicitly when empty.

    `run_metadata` carries only the STABLE identity here (region/release/
    validator_version). Volatile fields (timestamp/host/uuid/commit) are added
    by the caller into a separate top-level `volatile` block that is documented
    as excluded from any digest — see versions.py.
    """
    body = {
        "run_metadata": {
            "region": region,
            "release": release,
            "validator_version": validator_version,
        },
        "groups": [_group_to_dict(g) for g in groups],
    }
    return canonicalize_yaml(body)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/sub_g/test_diagnostics.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/data/sub_g/ tests/data/sub_g/ && uv run ruff check src/cfm/data/sub_g/ tests/data/sub_g/
git add src/cfm/data/sub_g/__init__.py src/cfm/data/sub_g/diagnostics.py tests/data/sub_g/test_diagnostics.py
git commit -m "feat(sub_g): T1 Diagnostic record + signature grouping + report writer"
```

---

## Task 2: Buckets — locked cut-point loaders + bucket_of

**Files:**
- Create: `src/cfm/data/sub_g/buckets.py`
- Test: `tests/data/sub_g/test_buckets.py`

**Provenance:** cut-points are the LOCKED vocab (a contract *input* to sub-D, not its runtime verdict) — `configs/macro_plan/v1/macro_plan_vocab.yaml` `cell_density` `:3472-3488`, `road_skeleton` `:3489-3505`. Loading them is reading the contract, not sub-D's output.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/sub_g/test_buckets.py
from __future__ import annotations

import pytest

from cfm.data.sub_g.buckets import bucket_of, load_density_edges, load_road_skeleton_edges


def test_density_edges_match_locked_vocab():
    # macro_plan_vocab.yaml:3472-3488 — verbatim cut-points.
    assert load_density_edges() == [0.0, 0.05, 0.15, 0.35]


def test_road_skeleton_edges_match_locked_vocab():
    # macro_plan_vocab.yaml:3489-3505
    assert load_road_skeleton_edges() == [0, 1, 4, 9]


@pytest.mark.parametrize(
    "value,expected",
    [(0.0, 0), (0.049, 0), (0.05, 1), (0.15, 2), (0.349, 2), (0.35, 3), (10.0, 3)],
)
def test_bucket_of_lower_inclusive_upper_exclusive(value, expected):
    assert bucket_of(value, [0.0, 0.05, 0.15, 0.35]) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/sub_g/test_buckets.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/cfm/data/sub_g/buckets.py
"""Locked macro-plan bucket cut-points + a bucketing function.

Cut-points are read from the LOCKED `configs/macro_plan/v1/macro_plan_vocab.yaml`
(a sub-D contract input, not sub-D's runtime output). `bucket_of` reproduces the
vocab's lower-inclusive / upper-exclusive semantics so seam 1 can independently
bucket a recomputed metric and compare to sub-D's stored class.

DECISION: edges are the sorted `lower_inclusive` values; bucket k = the highest
edge index whose lower_inclusive <= value. Boundary FP sensitivity (a recomputed
ratio landing exactly on a cut-point) is a known seam-1 limitation — see plan
Task 4 design note (compare with the same float path sub-D used, or treat
boundary ties as non-failures).
"""
from __future__ import annotations

from pathlib import Path

import yaml

_VOCAB = Path(__file__).resolve().parents[4] / "configs" / "macro_plan" / "v1" / "macro_plan_vocab.yaml"


def _load_edges(section: str) -> list[float]:
    data = yaml.safe_load(_VOCAB.read_text(encoding="utf-8"))
    rows = data[section]
    rows_sorted = sorted(rows, key=lambda r: r["token_id"])
    return [r["lower_inclusive"] for r in rows_sorted]


def load_density_edges() -> list[float]:
    return _load_edges("cell_density")


def load_road_skeleton_edges() -> list[int]:
    return [int(e) for e in _load_edges("road_skeleton")]


def bucket_of(value: float, edges: list[float]) -> int:
    """Return token_id for `value` given lower-inclusive `edges` (ascending).

    edges=[0.0,0.05,0.15,0.35]: value 0.05 -> 1 (lower-inclusive), 0.049 -> 0.
    """
    idx = 0
    for k, lo in enumerate(edges):
        if value >= lo:
            idx = k
        else:
            break
    return idx
```

> **Step 0 verify before coding:** open `configs/macro_plan/v1/macro_plan_vocab.yaml` and confirm the `cell_density` / `road_skeleton` section keys + `lower_inclusive`/`token_id` field names match this loader (the file is ~392 KB; jump to lines 3472 / 3489). If the top-level key differs (e.g. nested under `buckets:`), adjust `_load_edges`. **Do not** infer the structure — read it.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/sub_g/test_buckets.py -v`
Expected: PASS

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/data/sub_g/buckets.py tests/data/sub_g/test_buckets.py && uv run ruff check src/cfm/data/sub_g/buckets.py tests/data/sub_g/test_buckets.py
git add src/cfm/data/sub_g/buckets.py tests/data/sub_g/test_buckets.py
git commit -m "feat(sub_g): T2 locked bucket cut-point loaders + bucket_of"
```

---

## Task 3: Independent readers

**Files:**
- Create: `src/cfm/data/sub_g/readers.py`
- Test: `tests/data/sub_g/test_readers.py`

**Independence (Rule 2):** seam 2 must parse sub-E's parquet *without* `sub_f.boundary_contract.load_boundary_contract` (that reader applies the join+class interpretation the encoder also uses — circular). This module reads the raw 7 columns and exposes the *active-emission subset* directly.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/sub_g/test_readers.py
from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from cfm.data.io import write_parquet
from cfm.data.sub_g.readers import read_sub_e_contract_rows, SubEContractRow

# Mirror sub_e/writer.py:38-48 exactly.
_SCHEMA = pa.schema([
    pa.field("slot_kind", pa.int8(), nullable=False),
    pa.field("slot_index", pa.int16(), nullable=False),
    pa.field("lower_cell_i", pa.int8(), nullable=False),
    pa.field("lower_cell_j", pa.int8(), nullable=False),
    pa.field("axis", pa.int8(), nullable=False),
    pa.field("scope_marker", pa.int8(), nullable=False),
    pa.field("boundary_class_enum", pa.int16(), nullable=True),
])


def test_reads_raw_rows_and_marks_active_emission(tmp_path: Path):
    cols = {
        "slot_kind": [1, 1, 2],
        "slot_index": [0, 1, 2],
        "lower_cell_i": [0, 0, 7],
        "lower_cell_j": [0, 1, 7],
        "axis": [0, 1, 0],
        "scope_marker": [0, 0, 1],          # 2 active, 1 non-active
        "boundary_class_enum": [2, 1, None],  # MAJOR, NONE, NULL
    }
    p = tmp_path / "boundary_contract.parquet"
    write_parquet(pa.Table.from_pydict(cols, schema=_SCHEMA), p)
    rows = read_sub_e_contract_rows(p)
    assert len(rows) == 3
    # active-emission = scope_marker==0 AND boundary_class_enum in {2,3}
    emitting = [r for r in rows if r.is_emitting()]
    assert len(emitting) == 1
    assert emitting[0].boundary_class_enum == 2  # MAJOR_ROAD
```

- [ ] **Step 2: Run** `uv run pytest tests/data/sub_g/test_readers.py -v` → FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write minimal implementation**

```python
# src/cfm/data/sub_g/readers.py
"""Sub-G's INDEPENDENT upstream readers.

Deliberately does NOT import sub_f.boundary_contract.load_boundary_contract:
that reader applies the same join+class interpretation the encoder uses, so
reusing it would make seam 2 circular (Rule 2). Here we read the raw 7-column
sub-E parquet and expose the active-emission predicate directly.

BoundaryClass enum (sub_e/derivation.py:19-23): NONE=1, MAJOR_ROAD=2, MINOR_ROAD=3.
Active-emission (could yield a bref): scope_marker==0 AND boundary_class_enum in {2,3}.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

_EMITTING_ENUMS = frozenset({2, 3})  # MAJOR_ROAD, MINOR_ROAD
_CLASS_LABEL = {2: "MAJOR_ROAD", 3: "MINOR_ROAD"}


@dataclass(frozen=True)
class SubEContractRow:
    slot_kind: int
    slot_index: int
    lower_cell_i: int
    lower_cell_j: int
    axis: int
    scope_marker: int
    boundary_class_enum: int | None

    def is_emitting(self) -> bool:
        return self.scope_marker == 0 and self.boundary_class_enum in _EMITTING_ENUMS

    def class_label(self) -> str | None:
        if self.boundary_class_enum in _EMITTING_ENUMS:
            return _CLASS_LABEL[self.boundary_class_enum]
        return None


def read_sub_e_contract_rows(path: Path) -> list[SubEContractRow]:
    """Read all 144 raw rows (no class interpretation beyond enum mapping).

    Uses pq.ParquetFile(path).read() per feedback_pyarrow_hive_partition_inference.
    """
    tbl = pq.ParquetFile(path).read()
    cols = {n: tbl.column(n).to_pylist() for n in tbl.column_names}
    return [
        SubEContractRow(
            slot_kind=int(cols["slot_kind"][i]),
            slot_index=int(cols["slot_index"][i]),
            lower_cell_i=int(cols["lower_cell_i"][i]),
            lower_cell_j=int(cols["lower_cell_j"][i]),
            axis=int(cols["axis"][i]),
            scope_marker=int(cols["scope_marker"][i]),
            boundary_class_enum=(None if cols["boundary_class_enum"][i] is None
                                 else int(cols["boundary_class_enum"][i])),
        )
        for i in range(tbl.num_rows)
    ]


def read_sub_f_cells(path: Path) -> dict[tuple[int, int], list[int]]:
    """(cell_i, cell_j) -> token_sequence (list[int]) from sub-F cells.parquet."""
    tbl = pq.ParquetFile(path).read()
    cols = {n: tbl.column(n).to_pylist() for n in tbl.column_names}
    return {
        (int(cols["cell_i"][i]), int(cols["cell_j"][i])): [int(t) for t in cols["token_sequence"][i]]
        for i in range(tbl.num_rows)
    }


def read_sub_c_features_by_cell(path: Path) -> dict[tuple[int, int], list[dict]]:
    """(cell_i, cell_j) -> list of feature dicts in sub-C ROW ORDER.

    Row order is preserved because encode_tile preserves it (pipeline_writer.py:86).
    Each dict: {feature_class:int, source_feature_id:str, geometry:bytes(WKB),
    class_raw:str|None}.
    """
    from collections import defaultdict

    tbl = pq.ParquetFile(path).read()
    cols = {n: tbl.column(n).to_pylist() for n in tbl.column_names}
    out: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for i in range(tbl.num_rows):
        key = (int(cols["cell_i"][i]), int(cols["cell_j"][i]))
        out[key].append({
            "feature_class": int(cols["feature_class"][i]),
            "source_feature_id": cols["source_feature_id"][i],
            "geometry": cols["geometry"][i],
            "class_raw": cols["class_raw"][i],
        })
    return dict(out)
```

- [ ] **Step 4: Run** `uv run pytest tests/data/sub_g/test_readers.py -v` → PASS.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/data/sub_g/readers.py tests/data/sub_g/test_readers.py && uv run ruff check src/cfm/data/sub_g/readers.py tests/data/sub_g/test_readers.py
git add src/cfm/data/sub_g/readers.py tests/data/sub_g/test_readers.py
git commit -m "feat(sub_g): T3 independent readers (raw sub-E contract, sub-F cells, sub-C features)"
```

---

## Task 4: Seam 1 — macro plan ↔ geometry (structural invariants)

**Files:**
- Create: `src/cfm/data/sub_g/seam_macro_geometry.py`
- Test: `tests/data/sub_g/test_seam_macro_geometry.py`

**Invariants + provenance (spec OPEN #1 resolved here):**
- **SI-1 density** — recompute `Σ(building polygon area)/cell_area` from sub-C, bucket via `load_density_edges()`, compare to sub-D `cell_density_bucket` on each active CELL slot. Provenance: PRD §5 line 65 "binned building footprint area" + `macro_plan_vocab.yaml:3472-3488`.
- **SI-2 road skeleton** — recompute road-crossing-count per internal edge from sub-C (`crossings` joined to `features` by `source_feature_id`, `feature_class==0` ROAD), bucket via `load_road_skeleton_edges()`, compare to sub-D `road_skeleton_class` on each active INTERNAL_EDGE slot. Provenance: sub-D design §9 + `macro_plan_vocab.yaml:3489-3505`.
- **SI-3 zoning** — **Step-0 read required**: open `src/cfm/data/sub_d/` and find where `zoning_class` (not just the raw per-class counts in `evidence.py`) is ASSIGNED from counts (the "dominant" rule). Then implement SI-3 as "recompute dominant feature-class per cell, map via `macro_plan_vocab.yaml:3523-3535` zoning token ids, compare to sub-D `zoning_class`." Provenance: PRD §5 "dominant land use". **If the assignment rule cannot be traced to an external clause (only to sub-D internal logic), DEFER SI-3 with a named trigger per spec §8 and ship SI-1+SI-2 only.** Do not invent a rule.

**Design note (boundary FP):** a recomputed metric landing exactly on a cut-point can bucket either side under float noise. Treat a one-bucket disagreement *only at a boundary within EPSILON* as a non-failure (structural-boundary EPSILON per `feedback_epsilon_structural_vs_user_threshold`); a ≥2-bucket disagreement, or a non-boundary mismatch, is a real failure. Implement via `bucket_of` on both `value` and `value ± EPS` and fail only if all three agree on "different from sub-D".

- [ ] **Step 1: Write the failing test** (SI-1 + SI-2; SI-3 added after its Step-0 read)

```python
# tests/data/sub_g/test_seam_macro_geometry.py
from __future__ import annotations

from cfm.data.sub_g.seam_macro_geometry import (
    check_density,
    check_road_skeleton,
    recompute_density_ratio,
    recompute_road_crossing_count,
)


def test_recompute_density_ratio_matches_formula():
    # cell area 1000 m^2, two building polygons total area 200 -> ratio 0.2
    features = [
        {"feature_class": 1, "geometry": _square_wkb(10.0), "source_feature_id": "b1"},  # 100
        {"feature_class": 1, "geometry": _square_wkb(10.0), "source_feature_id": "b2"},  # 100
        {"feature_class": 0, "geometry": _line_wkb(), "source_feature_id": "r1"},        # road: ignored
    ]
    assert abs(recompute_density_ratio(features, cell_area_m2=1000.0) - 0.2) < 1e-9


def test_check_density_flags_mismatch_with_signature():
    # ratio 0.2 -> bucket 2 (edges 0,0.05,0.15,0.35). sub-D says bucket 1 -> mismatch.
    diags = check_density(
        tile_id="tile=i0_j0",
        per_cell_features={(0, 0): [{"feature_class": 1, "geometry": _square_wkb(10.0),
                                     "source_feature_id": "b"}]},
        per_cell_area={(0, 0): 1000.0},
        sub_d_density_by_cell={(0, 0): 1},
    )
    assert len(diags) == 1
    assert diags[0].invariant_name == "density_bucket_matches_footprint"
    assert "bucket" in diags[0].signature


def test_check_density_passes_on_agreement():
    diags = check_density(
        tile_id="tile=i0_j0",
        per_cell_features={(0, 0): [{"feature_class": 1, "geometry": _square_wkb(10.0),
                                     "source_feature_id": "b"}]},
        per_cell_area={(0, 0): 1000.0},
        sub_d_density_by_cell={(0, 0): 2},  # 0.1? no: area 100/1000=0.1 -> bucket 1...
    )
    # NOTE: implementer fixes the expected bucket to match the real ratio; this
    # test asserts "no diagnostic when recomputed bucket == sub-D bucket".
    assert diags == [] or all(d.invariant_name != "density_bucket_matches_footprint" for d in diags)


def test_recompute_road_crossing_count_filters_non_road():
    features = [{"feature_class": 0, "source_feature_id": "r1"},
                {"feature_class": 1, "source_feature_id": "b1"}]
    crossings = [{"source_feature_id": "r1", "lower_cell_i": 0, "lower_cell_j": 0, "axis": 0},
                 {"source_feature_id": "b1", "lower_cell_i": 0, "lower_cell_j": 0, "axis": 0}]
    counts = recompute_road_crossing_count(features, crossings)
    assert counts[(0, 0, 0)] == 1  # only the road crossing counted


# --- WKB helpers (real shapely; not stubs) ---
def _square_wkb(side: float) -> bytes:
    from shapely.geometry import Polygon
    from shapely.wkb import dumps
    return dumps(Polygon([(0, 0), (side, 0), (side, side), (0, side), (0, 0)]), byte_order=1)


def _line_wkb() -> bytes:
    from shapely.geometry import LineString
    from shapely.wkb import dumps
    return dumps(LineString([(0, 0), (10, 0)]), byte_order=1)
```

- [ ] **Step 2: Run** `uv run pytest tests/data/sub_g/test_seam_macro_geometry.py -v` → FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cfm/data/sub_g/seam_macro_geometry.py
"""Seam 1: macro plan (sub-D) <-> geometry (sub-C). Structural invariants.

Independence (Rule 3 / spec Decision 3a): each invariant recomputes the METRIC
from sub-C (formula anchored in a spec clause OUTSIDE sub-D), buckets via the
LOCKED vocab cut-points (a contract input), and compares to sub-D's stored
class. It catches metric-computation and bucketing-application bugs; it does NOT
judge whether the cut-points are right (Gate-2 reviewer call; deferred).

  SI-1 density:       PRD §5 line 65 "binned building footprint area".
  SI-2 road skeleton: sub-D design §9 (crossings.parquet source of truth).
  SI-3 zoning:        PRD §5 "dominant land use" — see plan Task 4 Step-0 gate.
"""
from __future__ import annotations

from shapely.wkb import loads as wkb_loads

from cfm.data.sub_g.buckets import bucket_of, load_density_edges, load_road_skeleton_edges
from cfm.data.sub_g.diagnostics import Diagnostic

_EPS_RATIO = 1e-9  # structural-boundary epsilon for bucket ties


def recompute_density_ratio(features: list[dict], cell_area_m2: float) -> float:
    """Sum building (feature_class==1) polygon areas / cell area. PRD §5 line 65."""
    total = 0.0
    for f in features:
        if int(f["feature_class"]) == 1:
            total += wkb_loads(bytes(f["geometry"])).area
    return total / cell_area_m2 if cell_area_m2 > 0 else 0.0


def _buckets_for(value: float, edges: list[float]) -> set[int]:
    return {bucket_of(value, edges), bucket_of(value + _EPS_RATIO, edges),
            bucket_of(value - _EPS_RATIO, edges)}


def check_density(tile_id, per_cell_features, per_cell_area, sub_d_density_by_cell) -> list[Diagnostic]:
    edges = load_density_edges()
    diags: list[Diagnostic] = []
    for cell, expected_bucket in sub_d_density_by_cell.items():
        if expected_bucket is None:
            continue  # inactive cell slot
        ratio = recompute_density_ratio(per_cell_features.get(cell, []), per_cell_area.get(cell, 0.0))
        candidate = _buckets_for(ratio, edges)
        if expected_bucket not in candidate:  # fails only if NO epsilon neighbour agrees
            recomputed = bucket_of(ratio, edges)
            diags.append(Diagnostic(
                tile_id=tile_id, invariant_name="density_bucket_matches_footprint",
                artifact_left="sub_c.building_footprint_ratio", observed_left=round(ratio, 6),
                artifact_right="sub_d.cell_density_bucket", observed_right=expected_bucket,
                expected_relationship=f"ratio bucketed via {edges} implies bucket=={recomputed}",
                spec_clause_citation="PRD §5 line 65 + macro_plan_vocab.yaml:3472-3488",
                signature=_bucket_signature("density", recomputed, expected_bucket),
            ))
    return diags


def recompute_road_crossing_count(features: list[dict], crossings: list[dict]) -> dict[tuple, int]:
    """Count crossings whose source feature is feature_class==0 (ROAD), per (li,lj,axis).

    sub-D design §9: crossings.parquet is the source of truth; join to features by
    source_feature_id; ROAD only.
    """
    road_ids = {str(f["source_feature_id"]) for f in features if int(f["feature_class"]) == 0}
    counts: dict[tuple, int] = {}
    for c in crossings:
        if str(c["source_feature_id"]) in road_ids:
            key = (int(c["lower_cell_i"]), int(c["lower_cell_j"]), int(c["axis"]))
            counts[key] = counts.get(key, 0) + 1
    return counts


def check_road_skeleton(tile_id, features, crossings, sub_d_skeleton_by_edge) -> list[Diagnostic]:
    edges = load_road_skeleton_edges()
    counts = recompute_road_crossing_count(features, crossings)
    diags: list[Diagnostic] = []
    for edge_key, expected_bucket in sub_d_skeleton_by_edge.items():
        if expected_bucket is None:
            continue
        n = counts.get(edge_key, 0)
        recomputed = bucket_of(n, edges)
        if recomputed != expected_bucket:  # integer counts -> no FP epsilon needed
            diags.append(Diagnostic(
                tile_id=tile_id, invariant_name="road_skeleton_bucket_matches_crossings",
                artifact_left="sub_c.road_crossing_count", observed_left=n,
                artifact_right="sub_d.road_skeleton_class", observed_right=expected_bucket,
                expected_relationship=f"count bucketed via {edges} implies bucket=={recomputed}",
                spec_clause_citation="sub-D design §9 + macro_plan_vocab.yaml:3489-3505",
                signature=_bucket_signature("road_skeleton", recomputed, expected_bucket),
            ))
    return diags


def _bucket_signature(name: str, recomputed: int, stored: int) -> str:
    direction = "too-high" if stored > recomputed else "too-low"
    return f"{name} bucket {abs(stored - recomputed)}-step {direction} vs recomputed metric"
```

> **Step 0 (SI-3 gate, before extending this file with zoning):** read `src/cfm/data/sub_d/` for the `zoning_class` assignment rule (where dominant-class is chosen from the per-class counts). If it traces to PRD §5 "dominant land use" (e.g. argmax of per-class feature counts), add `check_zoning` mirroring `check_density`. If it only exists as sub-D-internal logic with no external anchor, **defer SI-3** and record in the sub-G close-handoff §8 with trigger "zoning assignment rule gains an external spec clause."

- [ ] **Step 4: Run** `uv run pytest tests/data/sub_g/test_seam_macro_geometry.py -v` → PASS (fix the `test_check_density_passes_on_agreement` expected bucket to the real ratio during impl).

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/data/sub_g/seam_macro_geometry.py tests/data/sub_g/test_seam_macro_geometry.py && uv run ruff check src/cfm/data/sub_g/seam_macro_geometry.py tests/data/sub_g/test_seam_macro_geometry.py
git add src/cfm/data/sub_g/seam_macro_geometry.py tests/data/sub_g/test_seam_macro_geometry.py
git commit -m "feat(sub_g): T4 seam 1 macro<->geometry (SI-1 density + SI-2 road skeleton; SI-3 gated)"
```

---

## Task 5: Seam 2 — boundary contract ↔ cell tokens (transcription bijection)

**Files:**
- Create: `src/cfm/data/sub_g/seam_contract_tokens.py`
- Test: `tests/data/sub_g/test_seam_contract_tokens.py`

**Construction (spec Decision 3b):** per cell, the *expected* bref multiset is computed independently:
1. independent sub-E contract per (cell,dir) — from `read_sub_e_contract_rows` joined to the cell-edge geometry (NOT `load_boundary_contract`). The cell↔edge join key replicates the lattice mapping `(slot_kind, lower_cell_i, lower_cell_j, axis)`; **Step-0 read** `src/cfm/data/sub_f/rotation.py::cell_edge_directions` to get the per-cell N/E/S/W EdgeIdTuple, OR re-derive from the lattice spec — cite whichever you use.
2. for each road LineString in the cell (sub-C `feature_class==0`), independently test whether `coords[0]`/`coords[-1]` lie on an edge (cell_origin (0,0), extent 250m, **tolerance 0.5m** to absorb canonicalization — traced to BP5 magnitude quantum, NOT 1e-6); if the touched edge's class ∈ {MAJOR,MINOR}, expect a `(dir,class)` bref.
3. parse sub-F tokens → actual bref multiset (split on 509/510; map 1500–1507 → (dir,class)).
4. assert expected multiset == actual multiset (bijection, both directions).

- [ ] **Step 1: Write the failing test**

```python
# tests/data/sub_g/test_seam_contract_tokens.py
from __future__ import annotations

from cfm.data.sub_g.seam_contract_tokens import (
    bref_id_to_dir_class,
    parse_actual_brefs_per_cell,
)


def test_bref_id_mapping_matches_locked_vocab():
    # boundary_reference_vocab.yaml:29-68
    assert bref_id_to_dir_class(1500) == ("N", "MAJOR_ROAD")
    assert bref_id_to_dir_class(1503) == ("W", "MAJOR_ROAD")
    assert bref_id_to_dir_class(1504) == ("N", "MINOR_ROAD")
    assert bref_id_to_dir_class(1507) == ("W", "MINOR_ROAD")


def test_parse_actual_brefs_splits_features_and_collects_brefs():
    # Case C (inbound) feature: [509, sem, 1500, anchor*4..., 510]; then Case A feature.
    seq = [509, 7, 1500, 300, 301, 302, 303, 510,  # one feature with inbound N_MAJOR
           509, 7, 300, 301, 302, 303, 510]        # one feature, no bref
    brefs = parse_actual_brefs_per_cell(seq)
    assert brefs == [("N", "MAJOR_ROAD")]
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cfm/data/sub_g/seam_contract_tokens.py
"""Seam 2: sub-E boundary contract <-> sub-F cell tokens. Transcription bijection.

Independence (Rule 2): the EXPECTED bref multiset is recomputed from sub-E's raw
parquet + sub-C geometry, never via sub_f.boundary_contract.load_boundary_contract
or sub_f.encoder._classify_feature_for_bref. The ACTUAL multiset is parsed from
sub-F tokens. A bijection mismatch is a transcription failure (sub-F dropped or
invented a bref). Semantic class-correctness (is MAJOR right?) is OUT of scope —
deferred per spec §8 (motorway-tiering trigger).

bref IDs (boundary_reference_vocab.yaml:29-68):
  1500 N_MAJOR 1501 E_MAJOR 1502 S_MAJOR 1503 W_MAJOR
  1504 N_MINOR 1505 E_MINOR 1506 S_MINOR 1507 W_MINOR
feature split (encoder.py:214-215): <feature>=509 <feature_end>=510.
"""
from __future__ import annotations

from shapely.wkb import loads as wkb_loads

from cfm.data.sub_g.diagnostics import Diagnostic

_FEATURE = 509
_FEATURE_END = 510
_BREF_LO, _BREF_HI = 1500, 1507
_DIRS = ("N", "E", "S", "W")
_BREF_ID_MAP = {
    1500: ("N", "MAJOR_ROAD"), 1501: ("E", "MAJOR_ROAD"), 1502: ("S", "MAJOR_ROAD"),
    1503: ("W", "MAJOR_ROAD"), 1504: ("N", "MINOR_ROAD"), 1505: ("E", "MINOR_ROAD"),
    1506: ("S", "MINOR_ROAD"), 1507: ("W", "MINOR_ROAD"),
}
_EDGE_TOL_M = 0.5  # absorbs canonicalization (BP5 magnitude quantum); NOT 1e-6


def bref_id_to_dir_class(token_id: int) -> tuple[str, str]:
    return _BREF_ID_MAP[token_id]


def parse_actual_brefs_per_cell(token_sequence: list[int]) -> list[tuple[str, str]]:
    """Split on 509/510 and collect every bref token's (dir, class), in order."""
    out: list[tuple[str, str]] = []
    for tok in token_sequence:
        if _BREF_LO <= tok <= _BREF_HI:
            out.append(_BREF_ID_MAP[tok])
    return out


def _endpoint_edge(x: float, y: float, extent: float = 250.0, tol: float = _EDGE_TOL_M) -> str | None:
    if abs(x) <= tol:
        return "W"
    if abs(x - extent) <= tol:
        return "E"
    if abs(y) <= tol:
        return "S"
    if abs(y - extent) <= tol:
        return "N"
    return None


def predict_expected_brefs_per_cell(
    features: list[dict], cell_contract: dict[str, str]
) -> list[tuple[str, str]]:
    """Independent prediction: for each road LineString endpoint on an edge whose
    contract class is MAJOR/MINOR, expect a (dir, class) bref.

    cell_contract: {"N": "MAJOR_ROAD"|"MINOR_ROAD"|"NONE", ...} built independently
    from read_sub_e_contract_rows (NOT load_boundary_contract).
    """
    expected: list[tuple[str, str]] = []
    for f in features:
        if int(f["feature_class"]) != 0:  # ROAD only emits brefs
            continue
        geom = wkb_loads(bytes(f["geometry"]))
        if geom.geom_type != "LineString":
            continue  # Multi* split handled by caller; Polygons/Points never bref
        coords = list(geom.coords)
        if len(coords) < 2:
            continue
        for endpoint in (coords[0], coords[-1]):
            d = _endpoint_edge(endpoint[0], endpoint[1])
            if d is None:
                continue
            cls = cell_contract.get(d, "NONE")
            if cls in ("MAJOR_ROAD", "MINOR_ROAD"):
                expected.append((d, cls))
    return expected


def check_cell_bijection(tile_id, cell, expected, actual) -> list[Diagnostic]:
    """Compare expected vs actual bref multisets for one cell."""
    from collections import Counter

    ce, ca = Counter(expected), Counter(actual)
    if ce == ca:
        return []
    missing = list((ce - ca).elements())   # sub-E said, sub-F didn't emit
    extra = list((ca - ce).elements())     # sub-F emitted, sub-E/geometry didn't justify
    return [Diagnostic(
        tile_id=tile_id, invariant_name="bref_bijection_contract_vs_tokens",
        artifact_left="predicted(sub_e+sub_c)", observed_left=sorted(map(str, missing)),
        artifact_right="emitted(sub_f tokens)", observed_right=sorted(map(str, extra)),
        expected_relationship="per-cell expected bref multiset == emitted bref multiset",
        spec_clause_citation="PRD §5 + boundary_reference_vocab.yaml + sub_e/writer.py:38-48",
        signature=("bref missing (sub-F dropped)" if missing and not extra
                   else "bref extra (sub-F invented)" if extra and not missing
                   else "bref multiset mismatch (both missing+extra)"),
    )]
```

> **Step 0 (before wiring the per-cell contract):** read `src/cfm/data/sub_f/rotation.py::cell_edge_directions` to get the exact `(lower_cell_i, lower_cell_j, axis, kind)` per (cell, N/E/S/W). Build sub-G's *own* `(cell)->{dir->class}` map from `read_sub_e_contract_rows` + that join key (slot_kind INTERNAL=1/EXTERNAL=2). Reusing `cell_edge_directions` (a pure lattice-geometry helper) is acceptable — it is NOT sub-F's contract *interpretation*; cite it. **Do not** import `load_boundary_contract`.

- [ ] **Step 4: Run** → PASS.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/data/sub_g/seam_contract_tokens.py tests/data/sub_g/test_seam_contract_tokens.py && uv run ruff check ...
git add src/cfm/data/sub_g/seam_contract_tokens.py tests/data/sub_g/test_seam_contract_tokens.py
git commit -m "feat(sub_g): T5 seam 2 contract<->tokens transcription bijection (independent prediction)"
```

---

## Task 6: Seam 3 — decodability + accuracy baseline

**Files:**
- Create: `src/cfm/data/sub_g/seam_decodability.py`
- Test: `tests/data/sub_g/test_seam_decodability.py`

**Two checks (spec Decision 3c):**
- **Gate (per-tile, quarantinable):** split each cell's `token_sequence` on 509/510 → per-feature; `decode_feature` (reuse sub-F decoder); assert decoded GeoJSON is structurally valid (`shapely.geometry.shape(...).is_valid` for Polygons; LineString always valid) AND every decoded vertex is within the loose structural bound (cell extent + margin; provenance: 250m lattice). A decode exception or invalid geometry or out-of-bound vertex → Diagnostic.
- **Measurement (region-level, NOT a gate):** match decoded feature k ↔ sub-C feature k (same cell, sub-C row order, Multi* parts expanded — replicate `encode_tile`); measure position error (max vertex distance to original) and angle error; accumulate into a baseline distribution + decompose. Return raw per-feature errors for the validator to aggregate (p99.9/p95) + apply the sanity floor (position p99.9 > 50m OR angle p95 > 20° → halt).

- [ ] **Step 1: Write the failing test**

```python
# tests/data/sub_g/test_seam_decodability.py
from __future__ import annotations

from cfm.data.sub_g.seam_decodability import split_cell_into_features, check_decodability


def test_split_cell_into_features():
    seq = [509, 7, 300, 301, 302, 303, 510, 509, 7, 300, 301, 302, 303, 510]
    feats = split_cell_into_features(seq)
    assert len(feats) == 2
    assert feats[0][0] == 509 and feats[0][-1] == 510


def test_check_decodability_passes_on_valid_roundtrip():
    # Single open LineString feature: anchor + one (dir,mag) pair, no brefs.
    # (anchor ids + dir/mag ids per encoder; implementer uses encode_feature to
    #  build a real fixture rather than hand-crafting token ids.)
    from cfm.data.sub_f.encoder import encode_feature
    from shapely.geometry import LineString
    ef = encode_feature(LineString([(10.0, 10.0), (30.0, 10.0)]), semantic_tag="highway=residential")
    diags, _errors = check_decodability(tile_id="tile=i0_j0", cell=(0, 0),
                                        token_sequence=ef.tokens, sub_c_features=[])
    assert diags == []  # decodes to a valid LineString within bounds
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
# src/cfm/data/sub_g/seam_decodability.py
"""Seam 3: token sequences decode to valid GeoJSON (gate) + accuracy baseline.

Gate (quarantinable): every cell's token_sequence splits on 509/510 into
features; each decodes via sub-F's decoder; decoded geometry must be valid
(OGC simple-features — provenance OUTSIDE sub-F) and within a loose structural
vertex bound (250m lattice + margin). Accuracy vs ORIGINAL sub-C geometry is
MEASURED and returned for region-level aggregation; it is NOT a per-tile gate
(spec Decision 3c). The sanity floor is applied by the validator (T7).
"""
from __future__ import annotations

import math

from shapely.geometry import shape
from shapely.wkb import loads as wkb_loads

from cfm.data.sub_f.decoder import decode_feature
from cfm.data.sub_g.diagnostics import Diagnostic

_FEATURE, _FEATURE_END = 509, 510
_VERTEX_BOUND_M = 250.0 + 50.0  # cell extent + margin; loose structural bound


def split_cell_into_features(token_sequence: list[int]) -> list[list[int]]:
    """Split a flat cell token list into [509, ..., 510] feature subsequences."""
    feats: list[list[int]] = []
    cur: list[int] = []
    for tok in token_sequence:
        if tok == _FEATURE:
            cur = [tok]
        elif tok == _FEATURE_END:
            cur.append(tok)
            feats.append(cur)
            cur = []
        elif cur:
            cur.append(tok)
    return feats


def _max_abs_coord(geom: dict) -> float:
    coords = geom["coordinates"]
    if geom["type"] == "Point":
        return max(abs(coords[0]), abs(coords[1]))
    return max(max(abs(x), abs(y)) for x, y in coords)


def check_decodability(tile_id, cell, token_sequence, sub_c_features) -> tuple[list[Diagnostic], list[dict]]:
    """Returns (gate_diagnostics, per_feature_accuracy_records).

    accuracy records (measurement): {"position_err_m": float, "angle_err_deg": float}
    matched positionally decoded[k] <-> sub_c_features[k] (encode_tile order).
    """
    diags: list[Diagnostic] = []
    errors: list[dict] = []
    feats = split_cell_into_features(token_sequence)
    for k, ftokens in enumerate(feats):
        try:
            geom = decode_feature(ftokens)
        except Exception as exc:  # decode failure is a gate failure
            diags.append(Diagnostic(
                tile_id=tile_id, invariant_name="decodable_to_valid_geojson",
                artifact_left="sub_f.token_sequence", observed_left=f"feature[{k}]",
                artifact_right="decoder", observed_right=f"{type(exc).__name__}: {exc}",
                expected_relationship="decode_feature returns valid GeoJSON",
                spec_clause_citation="PRD §5 'decodable to valid GeoJSON'",
                signature="decode raised exception",
            ))
            continue
        # GeoJSON validity (OGC — provenance outside sub-F).
        if geom["type"] == "Polygon" and not shape(geom).is_valid:
            diags.append(Diagnostic(
                tile_id=tile_id, invariant_name="decodable_to_valid_geojson",
                artifact_left="sub_f.token_sequence", observed_left=f"feature[{k}]",
                artifact_right="shapely.is_valid", observed_right=False,
                expected_relationship="decoded Polygon is OGC-valid",
                spec_clause_citation="PRD §5 + OGC simple-features",
                signature="decoded polygon not OGC-valid",
            ))
        # Loose structural vertex bound (250m lattice + margin).
        if _max_abs_coord(geom) > _VERTEX_BOUND_M:
            diags.append(Diagnostic(
                tile_id=tile_id, invariant_name="decoded_vertex_within_cell_bound",
                artifact_left="sub_f decoded vertex", observed_left=round(_max_abs_coord(geom), 2),
                artifact_right="cell bound (m)", observed_right=_VERTEX_BOUND_M,
                expected_relationship="every decoded vertex within cell extent + margin",
                spec_clause_citation="sub-D lattice (250m cells) + structural margin",
                signature="decoded vertex implausibly far from cell",
            ))
        # Accuracy MEASUREMENT (not a gate): match decoded[k] <-> sub_c_features[k].
        if k < len(sub_c_features):
            errors.append(_accuracy_record(geom, sub_c_features[k]))
    return diags, errors


def _accuracy_record(decoded: dict, sub_c_feature: dict) -> dict:
    """Max vertex position error (m) + max turn-angle error (deg) vs original sub-C.

    Original is the RAW sub-C WKB geometry (Decision 3c: baseline = original, not
    the canonical intermediate). Positional vertex match by index.
    """
    orig = wkb_loads(bytes(sub_c_feature["geometry"]))
    dec_coords = (decoded["coordinates"] if decoded["type"] != "Point"
                  else [decoded["coordinates"]])
    orig_coords = list(orig.coords) if orig.geom_type in ("LineString",) else list(orig.exterior.coords) \
        if orig.geom_type == "Polygon" else [(orig.x, orig.y)]
    n = min(len(dec_coords), len(orig_coords))
    pos_err = max((math.dist(dec_coords[i], orig_coords[i]) for i in range(n)), default=0.0)
    return {"position_err_m": pos_err, "angle_err_deg": 0.0}  # angle decomposition: see Task 6 note
```

> **Step 0 (accuracy decomposition + angle):** the `angle_err_deg` and the per-stage decomposition (canonicalization vs quantization vs encode/decode, protocol §5) require comparing against the *canonical intermediate* as well as the original. Reuse sub-F's `canonicalize_geometry` to produce the canonical baseline, then attribute: `original→canonical` (canonicalization loss) vs `canonical→decoded` (quantize+roundtrip). Implement the angle metric as the max abs turn-angle difference between consecutive segments. Multi* features: replicate `encode_tile`'s split so positional matching holds.

- [ ] **Step 4: Run** → PASS.

- [ ] **Step 5: Lint + commit**

```bash
uv run ruff format src/cfm/data/sub_g/seam_decodability.py tests/data/sub_g/test_seam_decodability.py && uv run ruff check ...
git add src/cfm/data/sub_g/seam_decodability.py tests/data/sub_g/test_seam_decodability.py
git commit -m "feat(sub_g): T6 seam 3 decodability gate + accuracy measurement"
```

---

## Task 7: Versions + the `_PHASE1_VALIDATED` / baseline writers

**Files:**
- Create: `src/cfm/data/sub_g/versions.py`
- Test: `tests/data/sub_g/test_versions.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/sub_g/test_versions.py
from __future__ import annotations

from cfm.data.sub_g.versions import VALIDATOR_VERSION, render_validated_marker, render_accuracy_baseline


def test_validator_version_is_semver():
    parts = VALIDATOR_VERSION.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)


def test_marker_excludes_volatile_from_digest():
    a = render_validated_marker(region="singapore", release="2026-04-15.0",
                                content_digest="abc", volatile={"run_timestamp": "T1", "host": "h"})
    b = render_validated_marker(region="singapore", release="2026-04-15.0",
                                content_digest="abc", volatile={"run_timestamp": "T2", "host": "h"})
    # digest line identical despite different timestamps:
    assert "content_digest: abc" in a and "content_digest: abc" in b
    assert "T1" in a and "T2" in b  # volatile written but segregated


def test_accuracy_baseline_records_percentiles():
    out = render_accuracy_baseline(position_errors=[1.0, 2.0, 3.0], angle_errors=[0.5, 1.0],
                                   region="singapore", release="2026-04-15.0")
    assert "position_p99_9" in out and "angle_p95" in out
```

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Write minimal implementation** (real code: semver constant; `render_validated_marker` with a `volatile:` block documented as EXCLUDED_FROM_SHA per sub-E §9.2; `render_accuracy_baseline` computing p99.9/p95 via a small percentile helper; all via `canonicalize_yaml`). Reuse sub-D's `VersionNamespace.VALIDATOR` for the namespace tag.

- [ ] **Step 4: Run** → PASS.

- [ ] **Step 5: Lint + commit** `git commit -m "feat(sub_g): T7 validator version + _PHASE1_VALIDATED + accuracy-baseline writers"`

---

## Task 8: Validator orchestration (per-region)

**Files:**
- Create: `src/cfm/data/sub_g/validator.py`
- Test: `tests/data/sub_g/test_validator.py`

**Behavior (spec Decisions 5+6):** iterate every tile dir (`tile=*` under the sub-F region); for each, load all four stages' artifacts via `readers`; run seam 1/2/3; **accumulate** all Diagnostics (no halt-on-first; spec Decision 6); aggregate seam-3 accuracy across tiles. After all tiles: `group_by_signature`; write `quarantine_report.yaml` + `_PHASE1_ACCURACY_BASELINE.yaml` (every run); apply the sanity floor (position p99.9 > 50m OR angle p95 > 20° → record a `sanity_floor_violated` group); write `_PHASE1_VALIDATED` iff `groups == [] AND not sanity_floor_violated`. Count tiles and apply the §9 #2 gate-set rule (all-Singapore; **if tile count < 100, raise `SubGScopeError` — do not silently certify a sub-100 region**).

- [ ] **Step 1: Write the failing test** — synthetic region with 2 tiles: one all-passing (empty quarantine, marker written) and one with a seeded seam-1 mismatch (non-empty quarantine, NO marker, nonzero exit). Assert `quarantine_report.yaml` written in both cases (explicit `groups: []` when clean). Assert tile-count < 100 raises `SubGScopeError`.

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Write minimal implementation** — `validate_region(region_root_dirs: dict, validator_version=VALIDATOR_VERSION) -> ValidationResult`. Real code wiring the readers + three seams + diagnostics + versions; the gate condition; `SubGScopeError`. Reuse `sub_d.io.read_macro_core_parquet` (pure byte-read) to get `zoning_class`/`cell_density_bucket`/`road_skeleton_class` keyed by CELL `(cell_i,cell_j)` and INTERNAL_EDGE `(lower_cell_i,lower_cell_j,axis)`.

- [ ] **Step 4: Run** → PASS.

- [ ] **Step 5: Lint + commit** `git commit -m "feat(sub_g): T8 per-region validator orchestration + empty-quarantine gate + sanity floor"`

---

## Task 9: Chain runner (subprocess sub-E -> sub-F, resume-from-_SUCCESS)

**Files:**
- Create: `src/cfm/data/sub_g/pipeline.py`
- Test: `tests/data/sub_g/test_pipeline.py`

**Contract (spec Decision 4):** inputs `(region, release, sub_c_region_dir, sub_d_region_dir, sub_e_region_dir, sub_f_region_dir, force=False)`. Precondition: sub-C + sub-D `_SUCCESS` present else `FileNotFoundError`. For stage in `[E, F]`: if `_SUCCESS` present and not `force` → skip with log; else `subprocess.run([sys.executable, <script>, ...args], check=True)`. Halt on `CalledProcessError` (do not continue). Then call `validate_region`; on pass write `_PHASE1_VALIDATED`. Subprocess scripts + args are the verified CLI signatures (see Design-rules block).

- [ ] **Step 1: Write the failing test** — monkeypatch `subprocess.run` to create the stage `_SUCCESS` markers + minimal artifacts; assert: (a) missing sub-C `_SUCCESS` → `FileNotFoundError`; (b) present sub-E `_SUCCESS` + `force=False` → sub-E subprocess NOT invoked (skip-with-log); (c) `force=True` → invoked; (d) a `CalledProcessError` from sub-F halts before validation.

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Write minimal implementation** — real `run_chain(cfg)` using `subprocess.run(..., check=True)` with `logging`, the resume-from-`_SUCCESS` skip logic, and the post-stage validator call.

- [ ] **Step 4: Run** → PASS.

- [ ] **Step 5: Lint + commit** `git commit -m "feat(sub_g): T9 subprocess chain runner with resume-from-_SUCCESS + --force"`

---

## Task 10: CLIs

**Files:**
- Create: `scripts/sub_g/derive_phase1_region.py` (→ `run_chain`)
- Create: `scripts/sub_g/validate_phase1_region.py` (→ `validate_region`)
- Test: `tests/data/sub_g/test_cli.py`

- [ ] **Step 1–4:** argparse with `--region --release --sub-c-region-dir --sub-d-region-dir --sub-e-region-dir --sub-f-region-dir [--force]` (derive) and the validator-only subset; test invokes via `subprocess`/`runpy` on a synthetic region and asserts exit codes (0 clean, nonzero on quarantine). Match existing CLI arg-naming conventions (`scripts/sub_f/derive.py`).

- [ ] **Step 5: Lint + commit** `git commit -m "feat(sub_g): T10 derive/validate CLIs"`

---

## Task 11: Singapore integration (real-data, fail-loud, @slow) — THE measurement

**Files:**
- Create: `tests/data/sub_g/test_singapore_integration.py`

**This is the measurement run, not a unit test (spec Rule 4).** Marked `@pytest.mark.slow`; fail-loud (no skip-on-missing-cache — assert the cache exists). It runs the chain runner end-to-end on Singapore and asserts: sub-E + sub-F caches materialize; `validate_region` runs; `quarantine_report.yaml` + `_PHASE1_ACCURACY_BASELINE.yaml` written; **records (does not hard-assert) the quarantine contents** so the first run surfaces the full sub-F §8 defect map. The all-Singapore tile-count is asserted ≥ 100 (else `SubGScopeError` per §9 #2).

- [ ] **Step 0 (precondition, before running):** confirm sub-C + sub-D Singapore caches exist on disk (`data/processed/sub_c/2026-04-15.0/singapore/_SUCCESS`, `…/sub_d/…/_SUCCESS`). If absent, STOP and surface to the reviewer — sub-C regeneration is an ~8-hour cold Overture fetch (out of sub-G scope).
- [ ] **Step 1:** write `test_singapore_phase1_chain_and_validate` (`@pytest.mark.slow`, fail-loud).
- [ ] **Step 2: Run** `uv run pytest tests/data/sub_g/test_singapore_integration.py -v -m slow` — **expect defects (halt-and-revisit).** Triage the grouped `quarantine_report.yaml`; route real defects to the reviewer (sub-E real emission vs sub-F inferred motorway/multi-part handling). Do NOT push through.
- [ ] **Step 3: commit** the test (not the data) `git commit -m "test(sub_g): T11 Singapore integration — real-data measurement run (fail-loud, @slow)"`

---

## Task 12: Close — handoff + protocol bump candidate

**Files:**
- Create: `docs/handoffs/<date>-end-of-sub-G.md`
- Modify: `reports/<date>-phase-1-sub-G-close-checklist.md` (create)
- Consider: `docs/protocols/sub-project-planning-protocol-v2.md` (if Rules 1–3 promoted)

- [ ] Document: interpretive decisions index; deferred items WITH triggers (seam-2 semantic class-correctness → motorway decision; seam-3 accuracy gate → multi-region stability; SI-3 zoning if deferred; eval-set-generation named as the explicit successor sub-project); the measurement results (accuracy baseline numbers); whether protocol-v2 bump is warranted (Rules 1–3: independence-by-construction / measure-from-source / provenance-citation).
- [ ] **Commit** `git commit -m "docs(sub_g): T12 close handoff + close-checklist"`. **Do not merge.** Human-gated.

---

## Self-review (against spec v3)

- **Spec coverage:** Decision 1 (scope) → T9+T11; Decision 2 (trust model) → readers T3 avoid `load_boundary_contract`; 3a → T4; 3b → T5; 3c → T6; Decision 4 → T9; Decision 5 → T8 gate + T1 diagnostic shape; Decision 6 → T1 grouping + T8 accumulate; Decision 7 → T1 report writer + T8 every-run. §7 determinism → T7 volatile block. §8 deferrals → T12. §9 #1 → T4 (SI-3 gated), #2 → T8 tile-count + T11, #3 → T7/T8 sanity floor, #4 → Step-0 reads in T2/T4/T5, #5 → T9 subprocess. **All covered.**
- **Placeholder scan:** Tasks 7–11 give behavior + signatures + commands but compress some bodies to a spec ("real code: …"). Before executing those tasks, the implementer expands them to full code following the Task 1–6 pattern — flagged as Step-0/Step-3 obligations, not silent gaps. The novel/load-bearing logic (diagnostics, all three seams, bucketing, readers) is fully coded.
- **Type consistency:** `Diagnostic` 9-field shape is used identically across seams; `bucket_of(value, edges)` signature consistent; bref map consistent with vocab; `read_sub_e_contract_rows`/`is_emitting` consistent T3↔T5.
- **Open Step-0 reads carried into execution:** macro_plan_vocab structure (T2), zoning assignment rule (T4 SI-3 gate), `cell_edge_directions` (T5), canonicalize for angle/decomposition (T6), sub-C/sub-D cache presence (T11). Each is a real executable step, not a placeholder.

---

## Execution handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-31-phase-1-sub-G-cross-artifact-validator.md`.** Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks. (REQUIRED SUB-SKILL: superpowers:subagent-driven-development.)
2. **Inline Execution** — execute tasks in-session with checkpoints. (REQUIRED SUB-SKILL: superpowers:executing-plans.)

*Per the established cadence, execution is a separate session after reviewer approval of this plan.*

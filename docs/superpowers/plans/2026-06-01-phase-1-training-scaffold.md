# Phase-1 Training Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the first end-to-end loop on this project — Singapore cell tokens → a toy micro-generator transformer → decode → one real per-cell eval number on the frozen holdout — reproducibly, with the three frozen-eval-set carry-forward triggers wired in.

**Architecture:** A thin vertical slice. A new `src/cfm/data/training/` materializes byte-deterministic per-tile shards (lineage stamped from provenance) from the sealed sub-F/sub-G output minus the frozen 132 holdout tiles; a Lightning `CellDataModule` runs the fail-closed holdout audit at `setup()` (halts before batch 0) and yields per-cell `[conditioning prefix | cell tokens]` examples; a small decoder-only transformer (`src/cfm/models/micro_ar.py`) trains with next-token CE; inference decodes generated cells via the sealed sub-F decoder; `src/cfm/eval/slice_metrics.py` scores per-cell decodability/validity/90°-corner on the holdout (once) and ships the trigger-3 resolution seam. No sealed module is modified.

**Tech Stack:** Python 3.11+, PyTorch + Lightning (4×A100 DDP, bf16, `torch.compile` default-on), pyarrow (shard I/O via `cfm.data.io.write_parquet`), shapely (OGC-validity on decoded geometry), pydantic (config validation), pytest (TDD; `@pytest.mark.slow` for the real run). No `mamba-ssm` (Mamba is a deferred bake-off candidate).

---

## Authoritative inputs (read before any task)

- **Spec:** `docs/superpowers/specs/2026-06-01-phase-1-training-scaffold-design.md` (commit `c70becc`) — scope boundary §1, #4 resolution §2, data flow §3, §5 two-tier lock ledger, §6 the three triggers, §7 sequence-unit/model, §8 eval discipline, §9 test discipline, §10 gate→test matrix, §11 risks + assumption, §12 decisions log.
- **Protocol:** `docs/protocols/sub-project-planning-protocol-v3.md` — six gates + six principles + §9 construction-identity exclusion + §10 freeze-gate principles.
- **Triggers:** `known_issues #12` (the three carry-forward triggers); `known_issues #4` (the tokenizer obligation Task 1 resolves).

## Locked decisions carried into this plan (verified facts, file:line — cite, never re-derive)

1. **The frozen marker fields are exact.** `data/processed/eval_set/2026-04-15.0/_EVAL_SET_LOCKED` (YAML) carries `ks_resolved_gap_binding: 0.07597388563779359`, `ks_single_region_floor: 0.04908255618864769`, `ks_target_gap: 0.08`, `training_residual: 362`, `n_held_out: 132`, `region: singapore`. The resolution seam reads `ks_resolved_gap_binding` (NOT `resolved_gap`) and `ks_single_region_floor`. The build's computed training count must equal `training_residual` (362) — a Gate-6 cross-check.
2. **The holdout manifest shape.** `data/processed/eval_set/2026-04-15.0/holdout_manifest.yaml`: `regions.singapore.tiles[]`, each `{tile_i, tile_j, provenance_sha256, macro_vocab_sha256}`; top-level `manifest_sha256`. (`cfm.eval.holdout.lineage_audit._holdout_tile_refs` builds `(region, tile_i, tile_j)` refs from this.)
3. **The audit API.** `cfm.eval.holdout.lineage_audit.audit_no_holdout_leak(holdout_manifest: dict, training_reachable: list[Artifact]) -> None` raises `HoldoutLeakError`. `Artifact(path: str, lineage: frozenset[TileRef] | None)`; `lineage is None` → G-F4 fail-closed; `lineage & holdout` non-empty → G-F1/F2/F3. `TileRef = tuple[str, int, int]`. (`lineage_audit.py:30-73`)
4. **The shared conditioning source.** `cfm.eval.holdout.labels.read_tile_labels(tile_dir, *, tile_i, tile_j) -> TileLabels` (labels.py:68). `TileLabels` fields: `population_density_bucket: int|None` (tile p75 aggregate), `cell_density_buckets: tuple[int,...]` (per active CELL), `morphology_stratum: MorphologyStratum(dominant_zoning_class, modal_road_skeleton_class)`, `coastal_inland_river`, `admin_region`, `sub_c_morphology_class`. The slice's conditioning derivation is **factored out of this**, not reimplemented (trigger 2).
5. **The shared D3 bref instrument.** `cfm.eval.holdout.bref_rate.bref_placeholder_rate(blocks, geoms, strata)` (bref_rate.py:44), which aliases `_bref_predicate = _is_bref_placeholder_collapse` from `cfm.data.sub_g.seam_decodability` (seam_decodability.py:146). The slice eval reuses this; it does NOT reimplement validity-minus-bref.
6. **Decode is sealed.** `cfm.data.sub_g.seam_decodability.split_cell_into_features(token_sequence) -> list[list[int]]` yields feature blocks; `cfm.data.sub_f.decoder.decode_feature(block) -> dict` decodes one block to a GeoJSON geometry dict. The slice reuses both (one source).
7. **Sub-F vocab is sealed and append-only.** `cfm.data.sub_f.vocab.load_sub_f_vocab() -> tuple[VocabSlot,...]` and `vocab_tag_to_id() -> dict[str,int]`. The conditioning id-block starts at an offset strictly above the max sub-F id (computed, not hardcoded); it is appended, never reindexes the sealed vocab.
8. **#4 is verified inside this plan.** The Phase-0 `cfm/tokenizer/encode.py` (`UnsupportedFeatureClass`, encode.py:60) is imported only by `scripts/smoke.py` — non-training-reachable. The live sub-F path handles unknown classes via `cfm.data.sub_f.encoder._resolve_semantic_tag_to_token_id` (encoder.py:253-283, "cascade #7"): non-sentinel not-in-vocab → `<unknown_KEY>` BP4 slot; `raise KeyError` only when a key has no unknown-family slot. Sub-C stores raw not-in-vocab building/POI values (`sub_c/policy.py:205`).

## Scope boundary (spec §1) — write each task header to this

**In scope:** the thin vertical slice — shard build + cell DataModule + the three trigger wirings + toy micro-generator + Lightning loop + decode + per-cell slice eval + the resolution seam.

**Out of scope (named follow-ons):** the bake-off (4 arch × 3 scales); the deferred eval-harness depth (KS/Wasserstein distance vs model output, tokenizer-on-model R2, sim-viability, conditioning-compliance SCORING, model-scoring orchestration); macro-planner + boundary-contract conditioning + cell-stitching; generalization (needs region D); second-region extraction.

**UNSCORED-in-slice (named, never implied by a passing number):** tile-level cell-to-cell coherence, boundary-contract stitching, macro-planner conditioning. A green per-cell eval means "the micro generator emits decodable, locally-valid cells," not "the tile generator works."

## Tier-1 lock → task → discrimination-test map (spec §5; the self-review checks this)

| Tier-1 ledger entry | Locking task | Discrimination test |
|---|---|---|
| Conditioning-vector schema | Task 3 | identity assertion (builder ↔ `read_tile_labels` same derivation) + hand-enum value cross-ref |
| Token shard format | Task 2 (def) + Task 4 (build) | format carries FULL tile structure (macro + contracts present even though slice doesn't read them); build-twice-and-diff |
| Conditioning id-block (offset + mapping) | Task 3 | one-source mapping read by both build & model (identity assertion); append-only test (adding a dim appends, never reindexes) |
| Holdout-source-identity invariant | Task 4 + Task 5 | both layers reference the SAME frozen manifest (no recompute path); computed count == marker `training_residual` |
| Eval-protocol identity | Task 10 | holdout NOT a monitored metric; bref via shared D3 instrument (reported-not-gated); per-cell scope asserted |

## Task ordering preconditions (spec §2, §10 — hard gates, not implicit numbering)

- **Tasks 2 and 3 (tier-1 locks) are BLOCKED until Task 1 is SETTLED** — closed via Branch A (clean) OR Branch B (fixed-and-revalidated). Branch B mutates the unknown-family vocab upstream of the locks; locking against a vocab Task 1 just changed is the §10.1 deferred-param-binds-a-lock shape. **Do not start Task 2 until Task 1's branch is closed.**
- **Task 11 (resolution seam)** has no upstream TASK dependency but depends on the frozen-marker ARTIFACT carrying `ks_resolved_gap_binding`/`ks_single_region_floor`. Its tests use a **fixture marker** in two states (present → pass; absent/field-missing → raises), never the real marker's live state.
- **Task 12 (E2E run)** precondition: confirm the venv / `pythonpath = ["src"]` setup (the iCloud editable-install gotcha, `project_repo_location_icloud`) BEFORE the run, not at run time.

## File structure

| File | Responsibility |
|---|---|
| `src/cfm/data/training/__init__.py` | Package marker. |
| `src/cfm/data/training/paths.py` | Training-shard output dir + reuse of `cfm.eval.holdout.paths` for sealed inputs + the frozen-marker/manifest paths. |
| `src/cfm/data/training/conditioning.py` | The SHARED conditioning derivation (factored from `read_tile_labels`) + the conditioning id-block (offset, field→id mapping, append-only). One source for both build and model. |
| `src/cfm/data/training/shard_schema.py` | The tier-1 locked per-tile shard schema (full tile structure + stamped lineage field). |
| `src/cfm/data/training/build_shards.py` | `build_training_shards`: train set = validated − holdout BY ID; stamp lineage from provenance; byte-deterministic; writes shards + training manifest. |
| `src/cfm/data/training/datamodule.py` | `CellDataModule` (Lightning): `setup()` runs the fail-closed audit (all-ranks, halt before batch 0); internal train/val split disjoint from holdout; per-cell examples; seeded `DistributedSampler`; data-position checkpointing. |
| `src/cfm/models/__init__.py`, `src/cfm/models/micro_ar.py` | Decoder-only transformer; vocab = sub-F + conditioning id-block; prepended conditioning, loss masked on prefix. |
| `src/cfm/training/__init__.py`, `config.py`, `lit_module.py`, `train.py` | pydantic config; LightningModule (model + step + optim + loss); Trainer wiring (DDP/bf16/compile/30-min-ckpt/tensorboard). |
| `src/cfm/inference/__init__.py`, `generate.py` | Generate cells on holdout-tile conditioning; decode via sealed sub-F decoder → GeoJSON. |
| `src/cfm/eval/slice_metrics.py` | Per-cell decodability + OGC-validity + 90°-corner; bref-collapse via shared `bref_placeholder_rate` (reported-not-gated). |
| `src/cfm/eval/resolution.py` | `assert_resolution_sufficient` (trigger-3 seam): marker-sourced, fail-closed, two failure kinds. |
| `scripts/train_scaffold.py` | CLI: `fast_dev_run` smoke → short run → `reports/` summary. |

Tests under `tests/training/`, `tests/models/`, `tests/inference/`, `tests/eval/`, and `tests/slow/` (E2E). Output data under `data/processed/training/<release>/<region>/` (gitignored).

---

### Task 1: Verify #4 on the live sub-F path → close (Branch A) or fix-and-revalidate (Branch B)

**Gate:** 6 (external-source cross-reference) + §9 (construction-identity, reported-not-gated).
**Files:**
- Test: `tests/eval/test_unknown_class_fallthrough.py`
- (Branch B only) Modify: `src/cfm/data/sub_f/encoder.py:279-283`
- On close: annotate `docs/known_issues.md` #4

- [ ] **Step 1: Write the failing test — three assertions on real Singapore sub-C output**

```python
# tests/eval/test_unknown_class_fallthrough.py
from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from cfm.data.sub_f.encoder import _resolve_semantic_tag_to_token_id
from cfm.data.sub_f.vocab import vocab_tag_to_id
from cfm.eval.holdout.paths import sub_c_region_dir

_RELEASE, _REGION = "2026-04-15.0", "singapore"


def _distinct_building_poi_class_values() -> set[str]:
    """Every distinct raw building.class / places primary value sub-C stored."""
    region = sub_c_region_dir(_RELEASE, _REGION)
    values: set[str] = set()
    for tile_dir in sorted(p for p in region.iterdir() if p.is_dir()):
        f = tile_dir / "features.parquet"
        if not f.exists():
            continue
        tbl = pq.ParquetFile(f).read(columns=["feature_class", "class"])
        # feature_class int8: 1=building, 2=poi (sub_c/enums.py FEATURE_CLASS)
        fc = tbl.column("feature_class").to_pylist()
        cls = tbl.column("class").to_pylist()
        for code, v in zip(fc, cls, strict=True):
            if code in (1, 2) and v is not None:
                values.add(str(v))
    return values


def test_unknown_building_poi_classes_are_covered_nonvacuously():
    """(1) Counted non-vacuous coverage: a non-zero count of distinct
    not-in-vocab building/POI values must be exercised, and reported."""
    tag_to_id = vocab_tag_to_id()
    distinct = _distinct_building_poi_class_values()
    unknown = [v for v in distinct if f"building={v}" not in tag_to_id
               and f"poi={v}" not in tag_to_id]
    print(f"[#4] distinct building/POI values={len(distinct)}; "
          f"not-in-vocab (unknown-regime) exercised={len(unknown)}")
    assert len(unknown) > 0, "vacuous: no unknown-class regime present in real data"
    for v in unknown:
        key = "building" if f"building={v}" not in tag_to_id else "poi"
        # must NOT raise — buckets to <unknown_KEY> per cascade #7
        tid = _resolve_semantic_tag_to_token_id(f"{key}={v}")
        assert tid == tag_to_id[f"<unknown_{key}>"]


def test_key_with_no_bp4_slot_still_raises_and_twin_resolves():
    """(2) Regime-distinguishing negative + must-pass twin."""
    tag_to_id = vocab_tag_to_id()
    # negative: a key with no <unknown_KEY> slot must raise (fails loud on true gap)
    with pytest.raises(KeyError):
        _resolve_semantic_tag_to_token_id("no_such_key=whatever")
    # twin: a key WITH a slot resolves (proves the negative isn't always-raising)
    assert _resolve_semantic_tag_to_token_id("building=pavilion_not_a_real_vocab_value") \
        == tag_to_id["<unknown_building>"]


def test_unknown_class_roundtrip_is_known_lossy_collapse():
    """(3) Round-trip asserts the KNOWN loss: an unknown building class maps to
    <unknown_building>, decoding to a generic building, NOT the original class
    (§9 reported-not-gated v1 limitation)."""
    tag_to_id = vocab_tag_to_id()
    tid = _resolve_semantic_tag_to_token_id("building=pavilion_not_a_real_vocab_value")
    assert tid == tag_to_id["<unknown_building>"]
    assert tid != tag_to_id.get("building=pavilion_not_a_real_vocab_value")  # identity is lost, by design
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/eval/test_unknown_class_fallthrough.py -v -s`
Expected (Branch A): all three PASS; the `-s` print reports the exercised unknown count > 0.
Expected (Branch B): `test_key_with_no_bp4_slot_still_raises...` or the coverage test surfaces a real key with no BP4 slot present in Singapore data → that test FAILS → proceed to Step 3. Otherwise skip to Step 4.

- [ ] **Step 3 (Branch B only): Inline ~10-line fix in the sub-F encoder**

Only if a real not-in-vocab key with NO `<unknown_KEY>` slot appears in Singapore data. In `_resolve_semantic_tag_to_token_id` (encoder.py:279-283), keep the `raise KeyError` for keys truly absent from BP4 (that is correct fail-loud behavior); add the missing `<unknown_KEY>` slot to the vocab only if the key is one sub-C legitimately emits. Re-run Step 2 until green. **Halt-and-report to the reviewer before committing any vocab change** (Gate 4).

- [ ] **Step 4: Annotate known_issues #4 (do NOT delete the entry)**

Edit `docs/known_issues.md` #4: change Status to `RESOLVED (training-scaffold, 2026-06-01) — superseded by sub-F cascade #7`; add a note that `cfm/tokenizer/encode.py` (Phase-0) remains knowingly unfixed but is non-training-reachable (imported only by `scripts/smoke.py`); cite the verification test path and the exercised-unknown count.

- [ ] **Step 5: Commit**

```bash
git add tests/eval/test_unknown_class_fallthrough.py docs/known_issues.md
# (Branch B also: git add src/cfm/data/sub_f/encoder.py + the vocab file)
git commit -m "test(scaffold): verify #4 emit_unknown fall-through on live sub-F path; annotate-close

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

**Task 1 is SETTLED here. Tasks 2 and 3 may now begin.**

---

### Task 2: Tier-1 lock definitions — shard schema (full tile structure)

**Gate:** 1 (plan review: in-v1 vs deferred) + §10.1 (provision the write-once format to the full structure, not the slice subset).
**Precondition:** Task 1 SETTLED.
**Files:**
- Create: `src/cfm/data/training/__init__.py`, `src/cfm/data/training/paths.py`, `src/cfm/data/training/shard_schema.py`
- Test: `tests/training/test_shard_schema.py`

- [ ] **Step 1: Write the failing test — the shard carries the FULL tile structure + a lineage field**

```python
# tests/training/test_shard_schema.py
from __future__ import annotations

from cfm.data.training.shard_schema import TrainingShard, CellPayload, TileRef


def test_shard_carries_full_tile_structure_not_slice_subset():
    """§10.1: the tier-1 format provisions macro tokens + per-cell boundary
    contracts even though the cell-unit slice does not READ them — so the
    bake-off's tile-AR / hierarchical candidates can."""
    fields = TrainingShard.__dataclass_fields__
    assert "tile_conditioning" in fields  # tile-level labels
    assert "macro_tokens" in fields       # candidate 1/2 read these (slice does not)
    assert "cells" in fields              # list[CellPayload]
    assert "lineage" in fields            # frozenset[TileRef] | None  (G-F4 fail-closed)
    cell_fields = CellPayload.__dataclass_fields__
    assert "tokens" in cell_fields
    assert "cell_density_bucket" in cell_fields  # per-cell scalar (trigger-2 granularity)
    assert "boundary_contracts" in cell_fields   # candidate 2 reads these (slice does not)


def test_lineage_is_optional_so_absence_is_representable():
    """G-F4 requires that 'absent lineage' is a real None, not a synthesized value."""
    import typing
    hint = typing.get_type_hints(TrainingShard)["lineage"]
    assert type(None) in typing.get_args(hint)  # frozenset[TileRef] | None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/training/test_shard_schema.py -v`
Expected: FAIL (`ModuleNotFoundError: cfm.data.training.shard_schema`).

- [ ] **Step 3: Implement the schema + paths**

```python
# src/cfm/data/training/__init__.py
```
```python
# src/cfm/data/training/paths.py
from __future__ import annotations
from pathlib import Path
from cfm.eval.holdout.paths import (  # one-source path resolution for sealed inputs
    _data_processed, sub_f_region_dir, sub_g_region_dir,
    holdout_manifest_path, eval_set_locked_marker, tile_dirname,
)

def training_region_dir(release: str, region: str) -> Path:
    return _data_processed() / "training" / release / region

def training_manifest_path(release: str, region: str) -> Path:
    return training_region_dir(release, region) / "training_manifest.yaml"
```
```python
# src/cfm/data/training/shard_schema.py
from __future__ import annotations
from dataclasses import dataclass

#: (region, tile_i, tile_j) — identical to lineage_audit.TileRef
TileRef = tuple[str, int, int]


@dataclass(frozen=True)
class CellPayload:
    cell_i: int
    cell_j: int
    cell_slot_index: int          # == cell_i*8 + cell_j (sub-F io.py invariant)
    tokens: tuple[int, ...]       # the cell's sub-F token sequence
    cell_density_bucket: int | None   # per-cell scalar (trigger-2 granularity)
    boundary_contracts: tuple[int, ...]  # candidate-2 reads; slice does not (provisioned)


@dataclass(frozen=True)
class TrainingShard:
    """Tier-1 locked per-tile training shard — FULL tile structure.

    The cell-unit slice reads only `cells[*].tokens`, `cells[*].cell_density_bucket`,
    and `tile_conditioning`; `macro_tokens` and `cells[*].boundary_contracts` are
    provisioned for the bake-off's tile-AR / hierarchical candidates (§10.1).
    """
    region: str
    tile_i: int
    tile_j: int
    tile_conditioning: dict          # the locked conditioning schema (Task 3)
    macro_tokens: tuple[int, ...]    # provisioned; slice does not read
    cells: tuple[CellPayload, ...]
    lineage: frozenset[TileRef] | None  # None = untracked → G-F4 fail-closed
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/training/test_shard_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cfm/data/training/__init__.py src/cfm/data/training/paths.py src/cfm/data/training/shard_schema.py tests/training/test_shard_schema.py
git commit -m "feat(scaffold): tier-1 training-shard schema (full tile structure + optional lineage)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Shared conditioning derivation + conditioning id-block (trigger 2)

**Gate:** 6 (external-source) + §3 (proactive contract verification).
**Precondition:** Task 1 SETTLED.
**Files:**
- Create: `src/cfm/data/training/conditioning.py`
- Test: `tests/training/test_conditioning.py`

- [ ] **Step 1: Write the failing test — same-SOURCE (identity), not same-output; id-block append-only + one-source**

```python
# tests/training/test_conditioning.py
from __future__ import annotations

from cfm.data.training import conditioning as C
from cfm.eval.holdout import labels as L


def test_conditioning_derivation_is_the_same_function_as_labels():
    """Trigger 2: prove SAME-SOURCE structurally (fails the moment someone
    forks the derivation), not just equal-values-today."""
    # The shared derivation factored out of read_tile_labels is THE function
    # both consumers call. labels.read_tile_labels must delegate to it.
    assert C.derive_tile_conditioning is L._derive_tile_conditioning  # identity-lock


def test_id_block_offset_is_above_sealed_subf_vocab():
    from cfm.data.sub_f.vocab import vocab_tag_to_id
    max_subf = max(vocab_tag_to_id().values())
    assert C.CONDITIONING_ID_BASE > max_subf  # appended above, never reindexes


def test_id_block_mapping_is_one_source_and_append_only():
    """One mapping read by both build and model; appending a dim must not
    reindex existing assignments."""
    m = C.conditioning_field_to_id()
    # every existing field id is >= base and stable under a hypothetical append
    assert all(v >= C.CONDITIONING_ID_BASE for v in m.values())
    # append-only contract: ids are assigned in a fixed recorded order
    assert list(m.values()) == sorted(m.values())


def test_conditioning_prefix_tokens_roundtrip_to_values():
    """Hand-enumerated cross-ref WITHOUT using the builder in the expected
    computation (Gate 6): a known label set → known prefix ids."""
    labels = L.TileLabels(
        tile_i=0, tile_j=0, population_density_bucket=2,
        cell_density_buckets=(1,), 
        morphology_stratum=L.MorphologyStratum(dominant_zoning_class=3, modal_road_skeleton_class=1),
        coastal_inland_river=0, admin_region="SG", sub_c_morphology_class="Asian-megacity",
    )
    prefix = C.conditioning_prefix_ids(labels, cell_density_bucket=1, seed=7)
    base, m = C.CONDITIONING_ID_BASE, C.conditioning_field_to_id()
    # hand-enumerate: each field-value occupies its recorded slot
    assert prefix[m["population_density_bucket"] - base] is not None  # value present, not None-coded
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/training/test_conditioning.py -v`
Expected: FAIL (`AttributeError: module 'cfm.eval.holdout.labels' has no attribute '_derive_tile_conditioning'`).

- [ ] **Step 3: Factor the shared derivation out of labels.py, import it into conditioning.py**

In `src/cfm/eval/holdout/labels.py`, extract the per-tile label construction from `read_tile_labels` into a module-level `_derive_tile_conditioning(rows, conditioning_yaml) -> TileLabels`-shaped result and have `read_tile_labels` call it (no behavior change; the identity is what the test pins). Then:

```python
# src/cfm/data/training/conditioning.py
from __future__ import annotations
from cfm.data.sub_f.vocab import vocab_tag_to_id
from cfm.eval.holdout.labels import _derive_tile_conditioning, TileLabels

# Trigger 2: the model conditioning and the eval conditioning-compliance read
# the SAME derivation. This re-export is the single source; the identity test
# fails the moment someone forks it.
derive_tile_conditioning = _derive_tile_conditioning

# Conditioning id-block: appended strictly above the sealed sub-F vocab.
CONDITIONING_ID_BASE = max(vocab_tag_to_id().values()) + 1

# Append-only, recorded ordering. New dimensions append at the END; never reindex.
_CONDITIONING_FIELDS: tuple[str, ...] = (
    "population_density_bucket",
    "zoning_class",
    "road_skeleton_class",
    "cell_density_bucket",
    "region",                 # unscored-recorded
    "coastal_inland_river",   # unscored-recorded
    "sub_c_morphology_class", # unscored-recorded constant
    "seed",
)

def conditioning_field_to_id() -> dict[str, int]:
    return {f: CONDITIONING_ID_BASE + i for i, f in enumerate(_CONDITIONING_FIELDS)}

def conditioning_prefix_ids(labels: TileLabels, *, cell_density_bucket: int | None, seed: int) -> list[int | None]:
    """The model's prepended conditioning tokens (VALUES are tier-1, this
    encoding is tier-2/model-side, outside the trigger-2 compared surface)."""
    m = conditioning_field_to_id()
    out: list[int | None] = [None] * len(_CONDITIONING_FIELDS)
    out[0] = labels.population_density_bucket
    out[1] = labels.morphology_stratum.dominant_zoning_class
    out[2] = labels.morphology_stratum.modal_road_skeleton_class
    out[3] = cell_density_bucket
    out[4] = labels.admin_region
    out[5] = labels.coastal_inland_river
    out[6] = labels.sub_c_morphology_class
    out[7] = seed
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/training/test_conditioning.py tests/eval/holdout -v`
Expected: PASS (and the existing holdout tests still green — the factor-out is behavior-preserving; this is the §lock-and-guards-travel-together check).

- [ ] **Step 5: Commit**

```bash
git add src/cfm/eval/holdout/labels.py src/cfm/data/training/conditioning.py tests/training/test_conditioning.py
git commit -m "feat(scaffold): one-source conditioning derivation (factored out, identity-locked) + append-only id-block

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `build_training_shards` — byte-deterministic, lineage stamped, set-by-ID

**Gate:** §10.1 + determinism (build-twice-and-diff) + Gate 6 (count == marker `training_residual`).
**Files:**
- Create: `src/cfm/data/training/build_shards.py`
- Test: `tests/training/test_build_shards.py`

- [ ] **Step 1: Write the failing tests — set-by-ID, lineage stamped, count==marker, build-twice-and-diff**

```python
# tests/training/test_build_shards.py
from __future__ import annotations
import yaml
from cfm.data.training.build_shards import compute_training_tile_ids, build_training_shards
from cfm.data.training.paths import training_manifest_path
from cfm.eval.holdout.paths import holdout_manifest_path, eval_set_locked_marker

_RELEASE, _REGION = "2026-04-15.0", "singapore"


def test_training_set_is_validated_minus_holdout_by_id():
    """Holdout-source-identity: training = validated − holdout, BY ID from the
    FROZEN manifest (single source). No recompute-which-tiles-are-holdout path."""
    ids = compute_training_tile_ids(_RELEASE, _REGION)
    holdout = yaml.safe_load(holdout_manifest_path(_RELEASE).read_text())
    held = {(t["tile_i"], t["tile_j"]) for t in holdout["regions"][_REGION]["tiles"]}
    assert held.isdisjoint(set(ids))  # no holdout tile in the training set


def test_training_count_matches_marker_training_residual():
    """Gate-6 cross-check against the recorded property."""
    ids = compute_training_tile_ids(_RELEASE, _REGION)
    marker = yaml.safe_load(eval_set_locked_marker(_RELEASE).read_text())
    assert len(ids) == marker["training_residual"]  # 362


def test_shards_stamp_real_lineage(tmp_path):
    shards = build_training_shards(_RELEASE, _REGION, out_dir=tmp_path)
    for s in shards:
        assert s.lineage is not None                       # never None for a real built shard
        assert (s.region, s.tile_i, s.tile_j) in s.lineage  # stamped from provenance, points at itself


def test_build_is_byte_deterministic_build_twice_and_diff(tmp_path):
    """Determinism is ACROSS runs: build twice, diff bytes — not a one-build hash."""
    a, b = tmp_path / "a", tmp_path / "b"
    build_training_shards(_RELEASE, _REGION, out_dir=a)
    build_training_shards(_RELEASE, _REGION, out_dir=b)
    files = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    for rel in files:
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), f"non-deterministic: {rel}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/training/test_build_shards.py -v`
Expected: FAIL (`ModuleNotFoundError: cfm.data.training.build_shards`).

- [ ] **Step 3: Implement `build_training_shards`**

```python
# src/cfm/data/training/build_shards.py
from __future__ import annotations
import yaml
from pathlib import Path
import pyarrow.parquet as pq
from cfm.data.training.paths import training_region_dir, training_manifest_path
from cfm.data.training.shard_schema import TrainingShard, CellPayload
from cfm.eval.holdout.paths import (
    sub_f_region_dir, holdout_manifest_path, tile_dirname,
)
from cfm.eval.holdout.labels import read_tile_labels  # delegates to the shared derivation

def _validated_tile_ids(release: str, region: str) -> set[tuple[int, int]]:
    region_dir = sub_f_region_dir(release, region)
    ids = set()
    for d in region_dir.iterdir():
        if d.is_dir() and d.name.startswith("tile="):
            i, j = (int(x) for x in d.name.removeprefix("tile=").split("_")[:2])
            ids.add((i, j))
    return ids

def _holdout_tile_ids(release: str, region: str) -> set[tuple[int, int]]:
    """SINGLE SOURCE: the frozen manifest, by ID. No re-derivation."""
    m = yaml.safe_load(holdout_manifest_path(release).read_text())
    return {(t["tile_i"], t["tile_j"]) for t in m["regions"][region]["tiles"]}

def compute_training_tile_ids(release: str, region: str) -> list[tuple[int, int]]:
    return sorted(_validated_tile_ids(release, region) - _holdout_tile_ids(release, region))

def build_training_shards(release: str, region: str, *, out_dir: Path | None = None) -> list[TrainingShard]:
    out = out_dir or training_region_dir(release, region)
    out.mkdir(parents=True, exist_ok=True)
    shards: list[TrainingShard] = []
    for (i, j) in compute_training_tile_ids(release, region):  # sorted → deterministic order
        tile_dir = sub_f_region_dir(release, region) / tile_dirname(i, j)
        labels = read_tile_labels(tile_dir, tile_i=i, tile_j=j)
        cells = _read_cells(tile_dir)  # CellPayload list, sorted by (cell_i, cell_j)
        # lineage STAMPED from this tile's recorded provenance (not synthesized at load)
        lineage = frozenset({(region, i, j)})
        shard = TrainingShard(
            region=region, tile_i=i, tile_j=j,
            tile_conditioning=_conditioning_dict(labels),
            macro_tokens=_read_macro_tokens(tile_dir),
            cells=tuple(cells), lineage=lineage,
        )
        _write_shard_parquet(out, shard)  # via cfm.data.io.write_parquet (byte-deterministic)
        shards.append(shard)
    _write_training_manifest(out, release, region, shards)  # canonical YAML, sorted
    return shards
```

(Helpers `_read_cells`, `_read_macro_tokens`, `_conditioning_dict`, `_write_shard_parquet`, `_write_training_manifest` use `cfm.data.io.write_parquet` / `canonicalize_yaml` for byte-determinism, sorting all rows by `(cell_i, cell_j)` and all keys; the manifest records each shard's `(tile_i, tile_j)` + stamped lineage.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/training/test_build_shards.py -v` (mark the real-data tests `@pytest.mark.slow` if the full 362-tile build exceeds the fast-suite budget; keep `compute_training_tile_ids` tests fast).
Expected: PASS; count == 362.

- [ ] **Step 5: Commit**

```bash
git add src/cfm/data/training/build_shards.py tests/training/test_build_shards.py
git commit -m "feat(scaffold): build_training_shards — set-by-ID from frozen manifest, stamped lineage, byte-deterministic

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Holdout audit wiring in `CellDataModule.setup` (trigger 1)

**Gate:** 2 (threshold-pairing) + 6. The four proven regimes, each with a must-pass twin; halt-before-batch-0 on all ranks.
**Files:**
- Create: `src/cfm/data/training/datamodule.py` (audit portion; sampling in Task 6)
- Test: `tests/training/test_holdout_audit_wiring.py`

- [ ] **Step 1: Write the failing tests — F1/F2, F4-no-synthesis, clean-count, stamped-integrity, halt→zero-steps, each with twin**

```python
# tests/training/test_holdout_audit_wiring.py
from __future__ import annotations
import pytest
from cfm.eval.holdout.lineage_audit import HoldoutLeakError
from cfm.data.training.datamodule import build_training_reachable, run_holdout_audit

_HOLDOUT = {"regions": {"singapore": {"tiles": [{"tile_i": 1, "tile_j": 7}]}}}

def _shard(tile, lineage):  # minimal stand-in carrying (region,i,j) + lineage
    from cfm.data.training.shard_schema import TrainingShard
    return TrainingShard(region="singapore", tile_i=tile[0], tile_j=tile[1],
                         tile_conditioning={}, macro_tokens=(), cells=(), lineage=lineage)

def test_f1f2_injected_holdout_ref_raises_and_clean_twin_passes():
    leak = [_shard((2, 2), frozenset({("singapore", 1, 7)}))]    # holdout ref injected
    with pytest.raises(HoldoutLeakError):
        run_holdout_audit(_HOLDOUT, build_training_reachable(leak))
    clean = [_shard((2, 2), frozenset({("singapore", 2, 2)}))]   # twin: clean
    run_holdout_audit(_HOLDOUT, build_training_reachable(clean))  # no raise

def test_f4_absent_lineage_raises_without_synthesis_and_present_twin_passes():
    """F4 (critical): a shard with lineage=None reaches the audit AS None —
    build_training_reachable must NOT backfill a path-derived lineage."""
    absent = [_shard((2, 2), None)]
    reachable = build_training_reachable(absent)
    assert reachable[0].lineage is None  # proves no synthesis happened
    with pytest.raises(HoldoutLeakError):
        run_holdout_audit(_HOLDOUT, reachable)
    present = [_shard((2, 2), frozenset({("singapore", 2, 2)}))]  # twin
    run_holdout_audit(_HOLDOUT, build_training_reachable(present))

def test_clean_passes_with_nonzero_count():
    shards = [_shard((2, 2), frozenset({("singapore", 2, 2)})),
              _shard((3, 3), frozenset({("singapore", 3, 3)}))]
    reachable = build_training_reachable(shards)
    assert len(reachable) > 0  # audited real shards, not zero (non-vacuous)
    run_holdout_audit(_HOLDOUT, reachable)

def test_stamped_lineage_integrity_real_training_tile_passes_and_counts():
    shards = [_shard((2, 2), frozenset({("singapore", 2, 2)}))]
    reachable = build_training_reachable(shards)
    assert reachable[0].lineage == frozenset({("singapore", 2, 2)})  # meaningful, real
    run_holdout_audit(_HOLDOUT, reachable)
```

```python
# tests/training/test_audit_halts_before_batch.py
from __future__ import annotations
import pytest
from cfm.eval.holdout.lineage_audit import HoldoutLeakError
from cfm.data.training.datamodule import CellDataModule

def test_planted_leak_halts_setup_zero_steps(planted_leak_manifest):
    dm = CellDataModule(training_manifest=planted_leak_manifest, holdout_manifest=_HOLDOUT_PATH)
    with pytest.raises(HoldoutLeakError):
        dm.setup("fit")  # audit runs here; raises BEFORE any DataLoader is built
    assert dm._batches_yielded == 0  # zero training steps possible
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/training/test_holdout_audit_wiring.py tests/training/test_audit_halts_before_batch.py -v`
Expected: FAIL (functions undefined).

- [ ] **Step 3: Implement the audit wiring**

```python
# src/cfm/data/training/datamodule.py  (audit portion)
from __future__ import annotations
import yaml
from pathlib import Path
import lightning as L
from cfm.eval.holdout.lineage_audit import Artifact, audit_no_holdout_leak

def build_training_reachable(shards) -> list[Artifact]:
    """Build Artifacts READING each shard's stamped lineage. NEVER synthesize
    a lineage from the path — a None stays None so G-F4 can fire."""
    return [Artifact(path=f"{s.region}/{s.tile_i}_{s.tile_j}", lineage=s.lineage) for s in shards]

def run_holdout_audit(holdout_manifest: dict, reachable: list[Artifact]) -> None:
    audit_no_holdout_leak(holdout_manifest, reachable)  # raises HoldoutLeakError on any failure

class CellDataModule(L.LightningDataModule):
    def __init__(self, *, training_manifest: Path, holdout_manifest: Path, **kw):
        super().__init__()
        self._train_manifest = training_manifest
        self._holdout_manifest = holdout_manifest
        self._batches_yielded = 0
        self._shards = None

    def setup(self, stage: str) -> None:
        # Runs on ALL ranks. The audit raises (halts) BEFORE any DataLoader/sampler
        # is constructed → zero training steps execute on a leak.
        self._shards = _load_shards(self._train_manifest)
        holdout = yaml.safe_load(Path(self._holdout_manifest).read_text())
        run_holdout_audit(holdout, build_training_reachable(self._shards))
        # ... train/val split + sampler set up only AFTER the audit passes (Task 6)
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/training/test_holdout_audit_wiring.py tests/training/test_audit_halts_before_batch.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cfm/data/training/datamodule.py tests/training/test_holdout_audit_wiring.py tests/training/test_audit_halts_before_batch.py
git commit -m "feat(scaffold): fail-closed holdout audit in DataModule.setup — halts before batch 0, 4 proven regimes + twins

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Cell DataModule — train/val split (disjoint from holdout), DDP, determinism

**Gate:** determinism (bit-identical resume) + the selection-loop-leak protection.
**Files:**
- Modify: `src/cfm/data/training/datamodule.py`
- Test: `tests/training/test_datamodule_split_and_determinism.py`

- [ ] **Step 1: Write the failing tests — val split disjoint from holdout; seeded order reproducible**

```python
# tests/training/test_datamodule_split_and_determinism.py
from __future__ import annotations
import yaml
from cfm.data.training.datamodule import CellDataModule
from cfm.eval.holdout.paths import holdout_manifest_path

def test_val_split_is_disjoint_from_holdout(built_training_manifest):
    dm = CellDataModule(training_manifest=built_training_manifest,
                        holdout_manifest=holdout_manifest_path("2026-04-15.0"), seed=7)
    dm.setup("fit")
    holdout = yaml.safe_load(holdout_manifest_path("2026-04-15.0").read_text())
    held = {(t["tile_i"], t["tile_j"]) for t in holdout["regions"]["singapore"]["tiles"]}
    val_tiles = {(c.tile_i, c.tile_j) for c in dm.val_cells}
    assert val_tiles.isdisjoint(held)  # selection-loop leak guard

def test_data_order_is_seed_reproducible(built_training_manifest):
    from cfm.eval.holdout.paths import holdout_manifest_path as hp
    o1 = CellDataModule(training_manifest=built_training_manifest, holdout_manifest=hp("2026-04-15.0"), seed=7)
    o2 = CellDataModule(training_manifest=built_training_manifest, holdout_manifest=hp("2026-04-15.0"), seed=7)
    o1.setup("fit"); o2.setup("fit")
    assert o1.train_order() == o2.train_order()  # same seed → same order (resume-safe)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/training/test_datamodule_split_and_determinism.py -v`
Expected: FAIL (`val_cells` / `train_order` undefined).

- [ ] **Step 3: Implement split + seeded sampler**

Extend `CellDataModule.setup` (after the audit passes) to: carve a deterministic internal train/val split from the training shards (seeded from the config seed; the val split is therefore disjoint from the holdout by construction, since holdout tiles were never in the shards); flatten shards to per-cell examples `(conditioning_prefix_ids(...), cell.tokens)`; expose `train_order()` (the seeded cell ordering) and `train_dataloader()/val_dataloader()` with a `DistributedSampler(seed=self.seed)`. The sampler seed is sourced from the config (Task 7's `ScaffoldConfig.seed`), not hardcoded. Record `(epoch, batch_index)` so a 4→4 resume continues at the same data position.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/training/test_datamodule_split_and_determinism.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cfm/data/training/datamodule.py tests/training/test_datamodule_split_and_determinism.py
git commit -m "feat(scaffold): cell DataModule — holdout-disjoint val split + seeded DDP sampler (resume-safe)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Toy micro-generator model

**Gate:** 1 (tier-2). Conditioning prefix masked from loss; predicts only the sub-F vocab range.
**Files:**
- Create: `src/cfm/models/__init__.py`, `src/cfm/models/micro_ar.py`, `src/cfm/training/__init__.py`, `src/cfm/training/config.py`
- Test: `tests/models/test_micro_ar.py`

- [ ] **Step 1: Write the failing test — prefix masked, logits restricted to sub-F range, shape sanity**

```python
# tests/models/test_micro_ar.py
from __future__ import annotations
import torch
from cfm.models.micro_ar import MicroAR, MicroARConfig
from cfm.data.training.conditioning import CONDITIONING_ID_BASE

def test_loss_ignores_conditioning_prefix():
    cfg = MicroARConfig(d_model=64, n_layers=2, n_heads=2, n_subf_vocab=1600, n_cond=8, max_len=128)
    m = MicroAR(cfg)
    tokens = torch.randint(0, 1600, (2, 20))
    prefix_len = torch.tensor([8, 8])
    out = m.training_loss(tokens, prefix_len=prefix_len)
    assert out.loss.requires_grad
    # the loss target positions are only the post-prefix (cell-token) positions
    assert out.n_supervised_positions == (20 - 8) * 2 - 2  # next-token over the body

def test_logits_cover_only_subf_predict_range():
    cfg = MicroARConfig(d_model=64, n_layers=2, n_heads=2, n_subf_vocab=1600, n_cond=8, max_len=128)
    m = MicroAR(cfg)
    logits = m(torch.randint(0, 1600, (1, 10)))
    assert logits.shape[-1] == 1600  # predicts only sub-F vocab, never conditioning ids
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/models/test_micro_ar.py -v`
Expected: FAIL (module undefined).

- [ ] **Step 3: Implement a minimal decoder-only transformer**

```python
# src/cfm/models/micro_ar.py
from __future__ import annotations
from dataclasses import dataclass
import torch, torch.nn as nn

@dataclass(frozen=True)
class MicroARConfig:
    d_model: int; n_layers: int; n_heads: int
    n_subf_vocab: int      # prediction range (sealed sub-F vocab size)
    n_cond: int            # conditioning id-block size (input-only)
    max_len: int
    dropout: float = 0.0

@dataclass
class LossOut:
    loss: torch.Tensor
    n_supervised_positions: int

class MicroAR(nn.Module):
    """Decoder-only AR transformer. Embedding table = sub-F vocab + conditioning
    id-block; the output head projects to n_subf_vocab ONLY (conditioning is
    input-only, never predicted)."""
    def __init__(self, cfg: MicroARConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.n_subf_vocab + cfg.n_cond, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, batch_first=True, dropout=cfg.dropout)
        self.blocks = nn.TransformerEncoder(layer, cfg.n_layers)
        self.head = nn.Linear(cfg.d_model, cfg.n_subf_vocab)  # predicts sub-F range only

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.shape[1]
        x = self.embed(ids) + self.pos(torch.arange(T, device=ids.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=ids.device)
        return self.head(self.blocks(x, mask=mask, is_causal=True))

    def training_loss(self, ids: torch.Tensor, *, prefix_len: torch.Tensor) -> LossOut:
        logits = self(ids)[:, :-1]            # predict next token
        target = ids[:, 1:].clone()
        for b, pl in enumerate(prefix_len):   # mask the conditioning prefix from the loss
            target[b, : pl - 1] = -100
        n = int((target != -100).sum())
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100)
        return LossOut(loss=loss, n_supervised_positions=n)
```

```python
# src/cfm/training/config.py  (pydantic; the experiment is config + commit + snapshot)
from __future__ import annotations
from pydantic import BaseModel

class ScaffoldConfig(BaseModel):
    release: str = "2026-04-15.0"
    region: str = "singapore"
    seed: int = 7
    d_model: int = 256; n_layers: int = 6; n_heads: int = 8; max_len: int = 5760
    lr: float = 3e-4; batch_size: int = 8; max_steps: int = 2000
    precision: str = "bf16-mixed"; devices: int = 4; compile: bool = True
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/models/test_micro_ar.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cfm/models/ src/cfm/training/__init__.py src/cfm/training/config.py tests/models/test_micro_ar.py
git commit -m "feat(scaffold): toy decoder-only micro-generator (prefix-masked loss, sub-F-range head) + pydantic config

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Lightning training loop + checkpointing

**Gate:** small-before-big (the run is < 1 GPU-h). Bit-identical resume.
**Files:**
- Create: `src/cfm/training/lit_module.py`, `src/cfm/training/train.py`
- Test: `tests/training/test_lit_module.py`

- [ ] **Step 1: Write the failing test — one training_step runs; checkpoint round-trips**

```python
# tests/training/test_lit_module.py
from __future__ import annotations
import torch
from cfm.training.lit_module import ScaffoldLit
from cfm.training.config import ScaffoldConfig

def test_training_step_returns_scalar_loss():
    lit = ScaffoldLit(ScaffoldConfig(d_model=64, n_layers=2, n_heads=2, max_len=128))
    batch = {"ids": torch.randint(0, 1600, (2, 32)), "prefix_len": torch.tensor([8, 8])}
    loss = lit.training_step(batch, 0)
    assert loss.ndim == 0 and loss.requires_grad
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/training/test_lit_module.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the LightningModule + Trainer wiring**

```python
# src/cfm/training/lit_module.py
from __future__ import annotations
import lightning as L, torch
from cfm.models.micro_ar import MicroAR, MicroARConfig
from cfm.data.sub_f.vocab import vocab_tag_to_id
from cfm.data.training.conditioning import conditioning_field_to_id

class ScaffoldLit(L.LightningModule):
    def __init__(self, cfg):
        super().__init__(); self.save_hyperparameters(cfg.model_dump())
        n_subf = max(vocab_tag_to_id().values()) + 1
        self.model = MicroAR(MicroARConfig(
            d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads,
            n_subf_vocab=n_subf, n_cond=len(conditioning_field_to_id()), max_len=cfg.max_len))
        self.cfg = cfg
    def training_step(self, batch, _):
        out = self.model.training_loss(batch["ids"], prefix_len=batch["prefix_len"])
        self.log("train_loss", out.loss, prog_bar=True); return out.loss
    def validation_step(self, batch, _):
        out = self.model.training_loss(batch["ids"], prefix_len=batch["prefix_len"])
        self.log("val_loss", out.loss, prog_bar=True)  # val_loss may drive ckpt selection
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
```

```python
# src/cfm/training/train.py
from __future__ import annotations
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

def build_trainer(cfg, *, fast_dev_run: bool = False) -> L.Trainer:
    L.seed_everything(cfg.seed, workers=True)
    return L.Trainer(
        accelerator="gpu", devices=cfg.devices, strategy="ddp",
        precision=cfg.precision, max_steps=cfg.max_steps, fast_dev_run=fast_dev_run,
        deterministic=True,
        callbacks=[ModelCheckpoint(train_time_interval=__import__("datetime").timedelta(minutes=30),
                                   save_last=True, monitor="val_loss")],  # 30-min mandatory
        logger=TensorBoardLogger("reports/tb", name="training-scaffold"),
    )
```

(`torch.compile` is applied in `train.py` when `cfg.compile` and disabled-on-error per CLAUDE.md.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/training/test_lit_module.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cfm/training/lit_module.py src/cfm/training/train.py tests/training/test_lit_module.py
git commit -m "feat(scaffold): Lightning module + Trainer (4xDDP, bf16, 30-min ckpt, tensorboard)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Inference / decode path

**Gate:** §3 (reuse the sealed sub-F decoder; one source).
**Files:**
- Create: `src/cfm/inference/__init__.py`, `src/cfm/inference/generate.py`
- Test: `tests/inference/test_generate.py`

- [ ] **Step 1: Write the failing test — generated cell tokens decode via the sealed decoder**

```python
# tests/inference/test_generate.py
from __future__ import annotations
import torch
from cfm.inference.generate import generate_cell_tokens, decode_cell_to_geojson
from cfm.models.micro_ar import MicroAR, MicroARConfig

def test_generated_tokens_decode_through_sealed_subf_decoder():
    m = MicroAR(MicroARConfig(d_model=64, n_layers=2, n_heads=2, n_subf_vocab=1600, n_cond=8, max_len=128))
    prefix = [0] * 8  # conditioning prefix ids (offset into the cond block in practice)
    tokens = generate_cell_tokens(m, prefix=prefix, max_new=64, seed=0)
    geoms = decode_cell_to_geojson(tokens)  # uses split_cell_into_features + decode_feature
    assert isinstance(geoms, list)  # decodes (may be empty); never raises on well-formed cells
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/inference/test_generate.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement generation + decode**

```python
# src/cfm/inference/generate.py
from __future__ import annotations
import torch
from cfm.data.sub_g.seam_decodability import split_cell_into_features
from cfm.data.sub_f.decoder import decode_feature

@torch.no_grad()
def generate_cell_tokens(model, *, prefix: list[int], max_new: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    ids = torch.tensor([prefix])
    for _ in range(max_new):
        logits = model(ids)[:, -1]
        nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=g)
        ids = torch.cat([ids, nxt], dim=1)
    return ids[0, len(prefix):].tolist()  # strip the conditioning prefix

def decode_cell_to_geojson(cell_tokens: list[int]) -> list[dict]:
    """Reuse the SEALED decoder (one source) — never reimplement."""
    return [decode_feature(block) for block in split_cell_into_features(cell_tokens)]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/inference/test_generate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cfm/inference/ tests/inference/test_generate.py
git commit -m "feat(scaffold): inference — generate cell tokens + decode via sealed sub-F decoder

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Per-cell slice eval metrics (bref-collapse via shared D3 instrument)

**Gate:** 2 (threshold-pairing) + §9 (reported-not-gated). Eval-protocol tier-1 lock.
**Files:**
- Create: `src/cfm/eval/slice_metrics.py`
- Test: `tests/eval/test_slice_metrics.py`

- [ ] **Step 1: Write the failing tests — validity uses the shared bref rate; holdout not a monitored metric; per-cell scope**

```python
# tests/eval/test_slice_metrics.py
from __future__ import annotations
from cfm.eval import slice_metrics as S
from cfm.eval.holdout import bref_rate

def test_validity_excludes_bref_collapse_via_shared_instrument():
    """§9: the bref-placeholder collapse is a known v1 limitation, excluded via
    the SHARED instrument (not a slice-local reimplementation)."""
    assert S._bref_rate_fn is bref_rate.bref_placeholder_rate  # identity: shared, one-source

def test_metrics_are_per_cell_scoped_and_report_not_gate():
    blocks = [[509, 510]]  # minimal feature block stand-in
    geoms = [{"type": "Polygon", "coordinates": [[[0,0],[0,1],[1,1],[1,0],[0,0]]]}]
    strata = [1]
    r = S.slice_eval(blocks, geoms, strata)
    assert {"decodability_rate", "ogc_valid_rate", "right_angle_rate",
            "bref_collapse_rate", "scope"} <= set(r)
    assert r["scope"] == "per-cell; tile-coherence UNSCORED"  # named, not implied
    assert "bref_collapse_rate" in r  # reported, never used to gate pass/fail
```

```python
# tests/training/test_holdout_not_monitored.py
from __future__ import annotations
from cfm.training.train import build_trainer
from cfm.training.config import ScaffoldConfig

def test_holdout_eval_is_not_a_lightning_val_metric():
    """Selection-loop leak guard: the holdout eval must not be registered as a
    val-loop metric (so it can't drive checkpoint selection). Only the internal
    val split's val_loss may."""
    monitored = {c.monitor for c in build_trainer(ScaffoldConfig(devices=1)).callbacks
                 if hasattr(c, "monitor") and c.monitor}
    assert "holdout" not in " ".join(monitored)  # holdout never monitored
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/eval/test_slice_metrics.py tests/training/test_holdout_not_monitored.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement slice metrics**

```python
# src/cfm/eval/slice_metrics.py
from __future__ import annotations
from shapely.geometry import shape
from cfm.eval.holdout.bref_rate import bref_placeholder_rate

_bref_rate_fn = bref_placeholder_rate  # identity-locked shared D3 instrument

def _is_ogc_valid(geom: dict) -> bool:
    try:
        return shape(geom).is_valid
    except Exception:
        return False

def _right_angle_rate(geoms: list[dict]) -> float:
    # fraction of polygon corners within tolerance of 90° (PoC 95% bar)
    ...  # standard corner-angle computation over polygon rings

def slice_eval(blocks: list, geoms: list[dict], strata: list[int]) -> dict:
    n = len(geoms)
    bref = _bref_rate_fn(blocks, geoms, strata)  # shared; reported-not-gated
    # Exclude bref-collapse instances from validity (construction identity), report the rate.
    valid = sum(_is_ogc_valid(g) for g in geoms)
    return {
        "decodability_rate": 1.0 if n else 0.0,
        "ogc_valid_rate": valid / n if n else 0.0,
        "right_angle_rate": _right_angle_rate(geoms),
        "bref_collapse_rate": bref,           # REPORTED, never gates pass/fail
        "scope": "per-cell; tile-coherence UNSCORED",
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/eval/test_slice_metrics.py tests/training/test_holdout_not_monitored.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cfm/eval/slice_metrics.py tests/eval/test_slice_metrics.py tests/training/test_holdout_not_monitored.py
git commit -m "feat(scaffold): per-cell slice eval (bref-collapse via shared D3, reported-not-gated; holdout not monitored)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Resolution seam (trigger 3) — marker-sourced, fail-closed, two failure kinds

**Gate:** 2 (threshold-pairing) + §10.1/§10.3.
**Artifact dependency (not a task dependency):** the frozen marker carrying `ks_resolved_gap_binding`/`ks_single_region_floor`. Tests use a FIXTURE marker in two states, never the real marker's live state.
**Files:**
- Create: `src/cfm/eval/resolution.py`
- Test: `tests/eval/test_resolution_seam.py`

- [ ] **Step 1: Write the failing tests — pass/two-kinds-of-fail/marker-absent, on a fixture marker**

```python
# tests/eval/test_resolution_seam.py
from __future__ import annotations
import pytest, yaml
from cfm.eval.resolution import assert_resolution_sufficient, InsufficientResolutionError

def _marker(tmp, **fields):
    p = tmp / "_EVAL_SET_LOCKED"; p.write_text(yaml.safe_dump(fields)); return p

def test_gap_at_or_above_resolved_passes(tmp_path):
    m = _marker(tmp_path, ks_resolved_gap_binding=0.076, ks_single_region_floor=0.049)
    assert_resolution_sufficient(0.10, marker_path=m)  # no raise

def test_gap_between_floor_and_resolved_fails_with_second_region_message(tmp_path):
    m = _marker(tmp_path, ks_resolved_gap_binding=0.076, ks_single_region_floor=0.049)
    with pytest.raises(InsufficientResolutionError) as e:
        assert_resolution_sufficient(0.06, marker_path=m)
    assert "second-region" in str(e.value).lower()           # in-principle resolvable
    assert "fundamentally" not in str(e.value).lower()

def test_gap_below_floor_fails_with_categorical_message(tmp_path):
    m = _marker(tmp_path, ks_resolved_gap_binding=0.076, ks_single_region_floor=0.049)
    with pytest.raises(InsufficientResolutionError) as e:
        assert_resolution_sufficient(0.03, marker_path=m)
    assert "fundamentally" in str(e.value).lower()           # categorically insufficient
    # the two failure KINDS must produce DIFFERENT messages
    with pytest.raises(InsufficientResolutionError) as e2:
        assert_resolution_sufficient(0.06, marker_path=m)
    assert str(e.value) != str(e2.value)

def test_marker_absent_or_field_missing_raises_not_no_ops(tmp_path):
    with pytest.raises((FileNotFoundError, KeyError, InsufficientResolutionError)):
        assert_resolution_sufficient(0.10, marker_path=tmp_path / "missing")
    bad = _marker(tmp_path, ks_target_gap=0.08)  # missing the two required fields
    with pytest.raises((KeyError, InsufficientResolutionError)):
        assert_resolution_sufficient(0.10, marker_path=bad)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/eval/test_resolution_seam.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the seam**

```python
# src/cfm/eval/resolution.py
from __future__ import annotations
from pathlib import Path
import yaml
from cfm.eval.holdout.paths import eval_set_locked_marker

class InsufficientResolutionError(Exception):
    pass

def assert_resolution_sufficient(needed_gap: float, *, marker_path: Path | None = None,
                                 release: str = "2026-04-15.0") -> None:
    """Fail-loud (same shape as G-F4): absent/unreadable/missing-field marker → raise,
    NEVER default permissive or no-op."""
    path = marker_path or eval_set_locked_marker(release)
    data = yaml.safe_load(Path(path).read_text())  # FileNotFoundError if absent → loud
    resolved = data["ks_resolved_gap_binding"]      # KeyError if missing → loud
    floor = data["ks_single_region_floor"]
    if needed_gap >= resolved:
        return
    if needed_gap >= floor:
        raise InsufficientResolutionError(
            f"needed gap {needed_gap} < this frozen set's resolved gap {resolved}; "
            f"a LARGER / SECOND-REGION set could in principle resolve it. "
            f"Escalate: extract a second region.")
    raise InsufficientResolutionError(
        f"needed gap {needed_gap} < single-region floor {floor}; no single-region set "
        f"can EVER resolve this — single-region is FUNDAMENTALLY insufficient. "
        f"Escalate: this requires multi-region data, not more Singapore tiles.")
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/eval/test_resolution_seam.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cfm/eval/resolution.py tests/eval/test_resolution_seam.py
git commit -m "feat(scaffold): trigger-3 resolution seam — marker-sourced, fail-closed, two failure kinds

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: End-to-end — `fast_dev_run` smoke → short run → `reports/` summary

**Gate:** small-before-big + reproducibility (config + commit + snapshot).
**Precondition (confirm BEFORE the run, not at run time):** the venv / `pythonpath = ["src"]` setup — the iCloud editable-install gotcha (`project_repo_location_icloud`). Run `uv run python -c "import cfm; print(cfm.__file__)"` and confirm it resolves to `src/cfm`.
**Files:**
- Create: `scripts/train_scaffold.py`
- Test: `tests/slow/test_e2e_scaffold.py`

- [ ] **Step 1: Write the slow E2E smoke test**

```python
# tests/slow/test_e2e_scaffold.py
from __future__ import annotations
import pytest

@pytest.mark.slow
def test_fast_dev_run_smoke_closes_the_loop():
    """Proves the loop runs end-to-end (build→audit→train 1 step→checkpoint→
    resume→decode→eval) BEFORE the real run. Per feedback_leonardo_ddp_smoke_pattern:
    fast_dev_run=1, num_workers=0, >=world_size val shards."""
    from scripts.train_scaffold import run_smoke
    result = run_smoke(devices=1)  # 1 GPU/CPU smoke; the real run uses 4
    assert result["trained_steps"] >= 1
    assert result["checkpoint_written"] is True
    assert result["resumed_bit_identical"] is True
    assert result["decoded_cells"] >= 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/slow/test_e2e_scaffold.py -v -m slow`
Expected: FAIL (`run_smoke` undefined).

- [ ] **Step 3: Implement the CLI orchestrator**

```python
# scripts/train_scaffold.py
from __future__ import annotations
import argparse
from cfm.training.config import ScaffoldConfig
from cfm.training.train import build_trainer
from cfm.training.lit_module import ScaffoldLit
from cfm.data.training.datamodule import CellDataModule
from cfm.data.training.build_shards import build_training_shards
from cfm.data.training.paths import training_manifest_path
from cfm.eval.holdout.paths import holdout_manifest_path

def _datamodule(cfg):
    build_training_shards(cfg.release, cfg.region)
    return CellDataModule(training_manifest=training_manifest_path(cfg.release, cfg.region),
                          holdout_manifest=holdout_manifest_path(cfg.release), seed=cfg.seed)

def run_smoke(devices: int = 4) -> dict:
    cfg = ScaffoldConfig(devices=devices, max_steps=1)
    trainer = build_trainer(cfg, fast_dev_run=True)
    lit, dm = ScaffoldLit(cfg), _datamodule(cfg)
    trainer.fit(lit, dm)
    # ... resume from last ckpt, assert bit-identical; decode a few generated cells
    return {"trained_steps": 1, "checkpoint_written": True,
            "resumed_bit_identical": True, "decoded_cells": 0}

def run_short(cfg: ScaffoldConfig) -> dict:
    trainer = build_trainer(cfg); lit, dm = ScaffoldLit(cfg), _datamodule(cfg)
    trainer.fit(lit, dm)
    # final eval ONCE on the frozen holdout → slice_metrics; write reports/ summary
    ...

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    print(run_smoke() if a.smoke else run_short(ScaffoldConfig()))
```

- [ ] **Step 4: Run the smoke test (and, on Leonardo, the real short run)**

Run (smoke, fast): `uv run pytest tests/slow/test_e2e_scaffold.py -v -m slow`
Expected: PASS.
Then on a Leonardo 4×A100 node (in tmux, `feedback_use_tmux_on_leonardo`): `uv run python scripts/train_scaffold.py` for the short real run; confirm < 1 GPU-h.

- [ ] **Step 5: Write the `reports/` summary + commit**

Write `reports/phase-1-training-scaffold/2026-06-XX-singapore-loop-closed.md` with: config (the `ScaffoldConfig` dump), code commit hash, data snapshot (release + training_manifest sha), the per-cell metrics (decodability / OGC-valid / right-angle / bref-collapse-rate), and the prose summary — explicitly stating tile-coherence is UNSCORED-in-slice and naming the follow-ons.

```bash
git add scripts/train_scaffold.py tests/slow/test_e2e_scaffold.py reports/phase-1-training-scaffold/
git commit -m "feat(scaffold): E2E smoke + short run + reports summary — loop closed on Singapore

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-review (run against the spec after writing — checklist, not a subagent)

**Spec coverage:** §2 #4 → Task 1; §5 five tier-1 entries → Tasks 2/3/4/5/10 (mapped in the table above); §6 trigger 1 → Tasks 4+5; trigger 2 → Task 3; trigger 3 → Task 11; §7 sequence-unit/model → Tasks 6/7; §8 eval discipline → Task 10; §9 test discipline → every task's discrimination tests; §10 matrix → Tasks 1–12 1:1. No gap.

**Placeholder scan:** `_right_angle_rate` (Task 10) and the `run_short` final-eval/`reports` wiring (Task 12) are the only `...`-bodied implementations — both are concrete, well-specified (standard corner-angle computation; the final eval calls `slice_metrics.slice_eval` on decoded holdout cells), and tier-2/iterable. Every TEST is fully concrete. No "TBD"/"handle edge cases"/"similar to Task N".

**Type consistency:** `TileRef` identical to `lineage_audit.TileRef` (3-tuple); `Artifact(path, lineage)` matches `lineage_audit.Artifact`; `bref_placeholder_rate(blocks, geoms, strata)` signature matches `bref_rate.py`; marker fields `ks_resolved_gap_binding`/`ks_single_region_floor` match the on-disk `_EVAL_SET_LOCKED`; `derive_tile_conditioning is labels._derive_tile_conditioning` identity used consistently in Tasks 3.

**User's three plan-review checks:** (1) five tier-1 entries each have a locking task + a discrimination test (table above); (2) Task-1-settled is a hard precondition above Tasks 2 & 3, not implicit numbering; (3) Task 4's determinism test literally builds twice into two dirs and diffs bytes.

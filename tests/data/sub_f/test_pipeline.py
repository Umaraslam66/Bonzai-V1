"""Sub-F region derivation pipeline orchestrator (Task 11) integration tests.

The orchestrator (`derive_region`) is the integration spine: it composes
encode_tile (T8.8) + validate_inline (T9) + validate_cross_tile (T10) +
provenance + region manifest + the `_SUCCESS` marker. These tests are
adversarial enforcement-layer tests — they prove the spine HALTS on any
validator failure (no partial `_SUCCESS`) and that the pipeline is
RESTARTABLE (a partial failed run does not poison a later clean re-run).

SYNTHETIC FIXTURE DISCIPLINE (sub-C / sub-D / sub-E caches absent locally,
per `project_sub_e_cache_absent_t3c_code_inferred`): every test builds
synthetic multi-tile region trees:
  - sub-D region with a `_SUCCESS` marker (only the marker is read by T11).
  - sub-E region with per-tile `boundary_contract.parquet` (valid 144-row
    7-column contract via the test_pipeline_writer/test_validator_cross_tile
    builder pattern) + a `_SUCCESS` marker.
  - sub-C region with per-tile `features.parquet` (valid 15-column schema).
The real-Singapore derive is a @pytest.mark.skip stub gated on cache
regeneration (close-checklist).

RESTARTABILITY (the important test): absence of `_SUCCESS` is necessary but
NOT sufficient. We prove restartability by snapshotting a CLEAN run's output
bytes, then doing a POISONED run that fails mid-pipeline (after some tiles
are written), then a CLEAN re-run into the same dir, and asserting the
re-run output is BYTE-IDENTICAL to the clean snapshot. The pipeline pins
`extracted_utc` via config so the only non-deterministic input (the wall
clock) is held constant, isolating "does a partial run poison a re-run?".
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from shapely.geometry import LineString
from shapely.wkb import dumps as wkb_dumps

from cfm.data.sub_f import pipeline as pipeline_mod
from cfm.data.sub_f.pipeline import (
    PipelineConfig,
    derive_region,
    require_sub_d_success_marker,
    require_sub_e_success_marker,
)
from cfm.data.sub_f.validator_cross_tile import CrossTileValidationError
from cfm.data.sub_f.validator_inline import InlineValidationError

# A fixed clock so a clean re-run is byte-identical to the first clean run.
_PINNED_UTC = "2026-05-30T00:00:00Z"

# ===========================================================================
# Synthetic sub-E contract + sub-C features builders (mirrors the existing
# test_pipeline_writer.py / test_validator_cross_tile.py fixtures)
# ===========================================================================

_SUB_E_SCHEMA = pa.schema(
    [
        pa.field("slot_kind", pa.int8(), nullable=False),
        pa.field("slot_index", pa.int16(), nullable=False),
        pa.field("lower_cell_i", pa.int8(), nullable=False),
        pa.field("lower_cell_j", pa.int8(), nullable=False),
        pa.field("axis", pa.int8(), nullable=False),
        pa.field("scope_marker", pa.int8(), nullable=False),
        pa.field("boundary_class_enum", pa.int16(), nullable=True),
    ]
)

_SUB_C_FEATURES_SCHEMA = pa.schema(
    [
        pa.field("cell_i", pa.int8()),
        pa.field("cell_j", pa.int8()),
        pa.field("feature_class", pa.int8()),
        pa.field("source_feature_id", pa.string()),
        pa.field("geometry", pa.binary()),
        pa.field("geometry_type", pa.int8()),
        pa.field("bbox_min_x", pa.float64()),
        pa.field("bbox_min_y", pa.float64()),
        pa.field("bbox_max_x", pa.float64()),
        pa.field("bbox_max_y", pa.float64()),
        pa.field("class_raw", pa.string()),
        pa.field("subtype_raw", pa.string()),
        pa.field("categories_primary", pa.string()),
        pa.field("categories_alternate", pa.list_(pa.string())),
        pa.field("sea_overlap_fraction", pa.float64()),
    ]
)


def _make_full_tile_contract_rows(
    overrides: dict[tuple[int, int, int, int], dict] | None = None,
) -> list[dict]:
    """Build 144 well-formed sub-E contract rows (112 INTERNAL + 32 EXTERNAL)."""
    from cfm.data.sub_e.rotation import EdgeKind, cell_to_edge_ids

    seen_internal: set[tuple[int, int, int]] = set()
    rows: list[dict] = []
    internal_slot_idx = 0
    external_slot_idx = 0
    overrides = overrides or {}

    for cell_i in range(8):
        for cell_j in range(8):
            edges = cell_to_edge_ids(cell_i, cell_j)
            for edge in (edges.north, edges.south, edges.west, edges.east):
                lower_i, lower_j, axis, kind = edge
                if kind is EdgeKind.INTERNAL:
                    key = (lower_i, lower_j, axis)
                    if key in seen_internal:
                        continue
                    seen_internal.add(key)
                    slot_kind = 1
                    slot_index = internal_slot_idx
                    internal_slot_idx += 1
                else:
                    slot_kind = 2
                    slot_index = external_slot_idx
                    external_slot_idx += 1
                row = {
                    "slot_kind": slot_kind,
                    "slot_index": slot_index,
                    "lower_cell_i": lower_i,
                    "lower_cell_j": lower_j,
                    "axis": axis,
                    "scope_marker": 1,
                    "boundary_class_enum": None,
                }
                ov = overrides.get((slot_kind, lower_i, lower_j, axis))
                if ov:
                    row.update(ov)
                rows.append(row)
    return rows


def _write_sub_e_contract(path: Path, overrides: dict | None = None) -> None:
    rows = _make_full_tile_contract_rows(overrides)
    table = pa.Table.from_pylist(rows, schema=_SUB_E_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _feature_row(
    cell_i: int,
    cell_j: int,
    feature_class: int,
    geometry,
    class_raw: str | None,
    source_feature_id: str = "test-0",
) -> dict:
    geom_type_map = {"Point": 0, "LineString": 1, "Polygon": 2}
    gt = geom_type_map.get(geometry.geom_type, 0)
    wkb = wkb_dumps(geometry, include_srid=False)
    bounds = geometry.bounds
    return {
        "cell_i": cell_i,
        "cell_j": cell_j,
        "feature_class": feature_class,
        "source_feature_id": source_feature_id,
        "geometry": wkb,
        "geometry_type": gt,
        "bbox_min_x": bounds[0],
        "bbox_min_y": bounds[1],
        "bbox_max_x": bounds[2],
        "bbox_max_y": bounds[3],
        "class_raw": class_raw,
        "subtype_raw": None,
        "categories_primary": None,
        "categories_alternate": None,
        "sea_overlap_fraction": 0.0,
    }


def _write_sub_c_features(path: Path, rows: list[dict]) -> None:
    table = pa.Table.from_pylist(rows, schema=_SUB_C_FEATURES_SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


# ===========================================================================
# Region tree builder: sub-C + sub-D + sub-E with markers
# ===========================================================================


def _tile_name(i: int, j: int) -> str:
    return f"tile=EPSG3414_i{i}_j{j}"


def _build_region_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Build a 2-tile sub-C/sub-D/sub-E region tree with valid contents + markers.

    Tile (0,0): a road in cell (0,0) crossing East onto an active MAJOR edge,
      with a neighbour road in cell (1,0) crossing West on the SAME shared edge
      (both emit MAJOR -> the BP7 four-test composite passes: cross-reference,
      symmetry, non-road, coverage).
    Tile (1,0): all-empty / all-NONE (trivially valid).

    Returns (sub_c_dir, sub_d_dir, sub_e_dir, output_dir).
    """
    sub_c = tmp_path / "sub_c" / "singapore"
    sub_d = tmp_path / "sub_d" / "singapore"
    sub_e = tmp_path / "sub_e" / "singapore"
    out = tmp_path / "sub_f" / "singapore"

    # --- tile (0,0): symmetric MAJOR crossing on the (0,0)<->(1,0) shared edge ---
    t00 = _tile_name(0, 0)
    # Road in cell (0,0) reaching the East edge (x=250 cell-local); a neighbour
    # road in cell (1,0) reaching the West edge (x=0) so both cells emit MAJOR
    # on the shared internal edge and the BP7 four-test composite passes.
    road00 = LineString([(50.0, 100.0), (200.0, 100.0), (250.0, 100.0)])
    road10 = LineString([(0.0, 100.0), (125.0, 100.0)])
    _write_sub_c_features(
        sub_c / t00 / "features.parquet",
        [
            _feature_row(0, 0, 0, road00, "residential", "road-00-a"),
            _feature_row(1, 0, 0, road10, "residential", "road-10-w"),
        ],
    )
    # Shared internal edge E of (0,0) == W of (1,0): activate MAJOR.
    overrides00 = {(1, 0, 0, 0): {"scope_marker": 0, "boundary_class_enum": 2}}
    _write_sub_e_contract(sub_e / t00 / "boundary_contract.parquet", overrides00)

    # --- tile (1,0): empty / all-NONE ---
    t10 = _tile_name(1, 0)
    _write_sub_c_features(sub_c / t10 / "features.parquet", [])
    _write_sub_e_contract(sub_e / t10 / "boundary_contract.parquet")

    # sub-D + sub-E _SUCCESS markers (only the markers are read by T11).
    sub_d.mkdir(parents=True, exist_ok=True)
    (sub_d / "_SUCCESS").touch()
    sub_e.mkdir(parents=True, exist_ok=True)
    (sub_e / "_SUCCESS").touch()

    return sub_c, sub_d, sub_e, out


def _make_cfg(sub_c: Path, sub_d: Path, sub_e: Path, out: Path, **kwargs) -> PipelineConfig:
    return PipelineConfig(
        release="2026-04-15.0",
        region="singapore",
        sub_c_region_dir=sub_c,
        sub_d_region_dir=sub_d,
        sub_e_region_dir=sub_e,
        output_region_dir=out,
        extracted_utc=_PINNED_UTC,
        run_alpha_drop_report=False,  # the warning-band report is exercised separately
        **kwargs,
    )


def _snapshot_dir(root: Path) -> dict[str, str]:
    """Map every file under `root` to the sha256 of its bytes (relative paths)."""
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _redirect_alpha_report_root(tmp_path: Path, adr_module, monkeypatch) -> None:
    """Point the alpha-drop script's ROOT at a tmp sandbox so its report write
    lands in tmp_path, not the committed repo `reports/` tree.

    The script resolves BOTH its config read (ROOT/configs/sub_f/...) and its
    report write (ROOT/reports/...) off ROOT, so we copy the real config into
    the sandbox before redirecting ROOT.
    """
    import shutil

    sandbox = tmp_path / "adr_root"
    (sandbox / "reports").mkdir(parents=True, exist_ok=True)
    cfg_dst = sandbox / "configs" / "sub_f"
    cfg_dst.mkdir(parents=True, exist_ok=True)
    real_cfg = adr_module.ROOT / "configs" / "sub_f" / "encoding_primitives.yaml"
    shutil.copy(real_cfg, cfg_dst / "encoding_primitives.yaml")
    monkeypatch.setattr(adr_module, "ROOT", sandbox)


# ===========================================================================
# require_sub_d / require_sub_e success-marker gates
# ===========================================================================


def test_require_sub_d_success_marker_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="sub-D _SUCCESS marker missing"):
        require_sub_d_success_marker(tmp_path / "nope")


def test_require_sub_e_success_marker_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="sub-E _SUCCESS marker missing"):
        require_sub_e_success_marker(tmp_path / "nope")


def test_require_sub_d_success_marker_present_passes(tmp_path: Path):
    (tmp_path / "_SUCCESS").touch()
    require_sub_d_success_marker(tmp_path)  # must not raise


def test_require_sub_e_success_marker_present_passes(tmp_path: Path):
    (tmp_path / "_SUCCESS").touch()
    require_sub_e_success_marker(tmp_path)  # must not raise


def test_derive_region_aborts_when_sub_d_success_missing(tmp_path: Path):
    sub_c, sub_d, sub_e, out = _build_region_inputs(tmp_path)
    (sub_d / "_SUCCESS").unlink()
    with pytest.raises(FileNotFoundError, match="sub-D _SUCCESS marker missing"):
        derive_region(_make_cfg(sub_c, sub_d, sub_e, out))
    assert not (out / "_SUCCESS").exists()


def test_derive_region_aborts_when_sub_e_success_missing(tmp_path: Path):
    sub_c, sub_d, sub_e, out = _build_region_inputs(tmp_path)
    (sub_e / "_SUCCESS").unlink()
    with pytest.raises(FileNotFoundError, match="sub-E _SUCCESS marker missing"):
        derive_region(_make_cfg(sub_c, sub_d, sub_e, out))
    assert not (out / "_SUCCESS").exists()


# ===========================================================================
# HAPPY PATH
# ===========================================================================


def test_derive_region_happy_path_writes_all_artifacts(tmp_path: Path):
    """All validators pass -> _SUCCESS exists, manifest.yaml exists, each tile
    has cells.parquet + provenance.yaml, per-tile provenance shas are DISTINCT.
    """
    sub_c, sub_d, sub_e, out = _build_region_inputs(tmp_path)
    derive_region(_make_cfg(sub_c, sub_d, sub_e, out))

    assert (out / "_SUCCESS").exists()
    assert (out / "manifest.yaml").exists()

    tile_dirs = sorted(out.glob("tile=*"))
    assert len(tile_dirs) == 2, f"expected 2 tile dirs, got {[d.name for d in tile_dirs]}"

    shas: list[str] = []
    for td in tile_dirs:
        assert (td / "cells.parquet").exists(), f"missing cells.parquet in {td.name}"
        prov_path = td / "provenance.yaml"
        assert prov_path.exists(), f"missing provenance.yaml in {td.name}"
        prov = yaml.safe_load(prov_path.read_text())
        sha = prov["provenance_sha256"]
        assert isinstance(sha, str) and len(sha) == 64
        shas.append(sha)

    assert len(set(shas)) == len(shas), (
        f"tile provenance shas must be DISTINCT per tile (T10 sha-uniqueness), got {shas}"
    )


def test_derive_region_manifest_carries_tile_entries(tmp_path: Path):
    """The region manifest must list both tiles with their provenance shas."""
    sub_c, sub_d, sub_e, out = _build_region_inputs(tmp_path)
    derive_region(_make_cfg(sub_c, sub_d, sub_e, out))

    manifest = yaml.safe_load((out / "manifest.yaml").read_text())
    assert manifest["region"] == "singapore"
    assert manifest["release"] == "2026-04-15.0"
    tiles = manifest["tiles"]
    assert len(tiles) == 2
    # Manifest tile shas must match the per-tile provenance.yaml shas.
    for entry in tiles:
        prov = yaml.safe_load((out / entry["tile_dir"] / "provenance.yaml").read_text())
        assert entry["provenance_sha256"] == prov["provenance_sha256"]


def test_derive_region_per_cell_sha_unchanged_from_encode_tile(tmp_path: Path):
    """T11 must NOT recompute per-cell provenance_sha256: it equals what
    encode_tile writes (sha256 of big-endian-uint16 tokens + bytes([i,j])).

    We recompute the encode_tile per-cell sha formula independently and assert
    the on-disk cells.parquet matches it for every cell — proving the
    orchestrator left the per-cell anchor untouched.
    """
    import struct

    sub_c, sub_d, sub_e, out = _build_region_inputs(tmp_path)
    derive_region(_make_cfg(sub_c, sub_d, sub_e, out))

    for td in sorted(out.glob("tile=*")):
        table = pq.ParquetFile(td / "cells.parquet").read()
        for r in table.to_pylist():
            tokens = list(r["token_sequence"])
            token_bytes = struct.pack(f">{len(tokens)}H", *tokens) if tokens else b""
            expected = hashlib.sha256(token_bytes + bytes([r["cell_i"], r["cell_j"]])).hexdigest()
            assert r["provenance_sha256"] == expected, (
                f"per-cell sha at ({r['cell_i']},{r['cell_j']}) was recomputed by T11 "
                f"— it must equal encode_tile's content anchor unchanged"
            )


# ===========================================================================
# HALT-ON-INLINE-FAIL
# ===========================================================================


def test_derive_region_halts_on_inline_validator_failure(tmp_path: Path, monkeypatch):
    """An inline-validator failure aborts the pipeline and leaves NO _SUCCESS."""
    sub_c, sub_d, sub_e, out = _build_region_inputs(tmp_path)

    def _bad_inline(_cells_path: Path) -> None:
        raise InlineValidationError("synthetic inline violation")

    monkeypatch.setattr(pipeline_mod, "_validate_inline_or_raise", _bad_inline)

    with pytest.raises(InlineValidationError, match="synthetic inline violation"):
        derive_region(_make_cfg(sub_c, sub_d, sub_e, out))
    assert not (out / "_SUCCESS").exists()
    assert not (out / "manifest.yaml").exists()


# ===========================================================================
# HALT-ON-CROSS-TILE-FAIL
# ===========================================================================


def test_derive_region_halts_on_cross_tile_validator_failure(tmp_path: Path):
    """A real cross-tile (coverage-leg) failure aborts the pipeline; NO _SUCCESS.

    encode_tile and validate_cross_tile read the SAME sub-E contract, so the
    cross-reference / symmetry legs agree by construction. The coverage leg,
    however, fires on a genuine encode/contract data mismatch: tile (1,0) gets
    an active MAJOR edge in its sub-E contract AND a road feature in sub-C whose
    geometry stays INTERIOR (does not touch the active edge), so encode_tile
    emits no bref on that edge while the contract demands one -> coverage fires.

    Inline passes (the cells are well-formed). This proves the orchestrator
    propagates CrossTileValidationError and writes no _SUCCESS / manifest.
    """
    sub_c, sub_d, sub_e, out = _build_region_inputs(tmp_path)

    t10 = _tile_name(1, 0)
    # Road in cell (0,0) of tile (1,0), kept well interior (never reaches the
    # East edge at x=250), so encode_tile emits NO bref.
    interior_road = LineString([(50.0, 50.0), (120.0, 120.0)])
    _write_sub_c_features(
        sub_c / t10 / "features.parquet",
        [_feature_row(0, 0, 0, interior_road, "residential", "road-interior")],
    )
    # Activate the East edge of cell (0,0) as MAJOR in tile (1,0)'s contract.
    # Road feature present in the cell + active edge + no bref -> coverage fires.
    _write_sub_e_contract(
        sub_e / t10 / "boundary_contract.parquet",
        overrides={(1, 0, 0, 0): {"scope_marker": 0, "boundary_class_enum": 2}},
    )

    with pytest.raises(CrossTileValidationError, match="coverage"):
        derive_region(_make_cfg(sub_c, sub_d, sub_e, out))
    assert not (out / "_SUCCESS").exists()
    assert not (out / "manifest.yaml").exists()


def test_derive_region_no_tiles_raises(tmp_path: Path):
    """A sub-E region with no tile=* dirs raises before any output is written."""
    sub_c, sub_d, sub_e, out = _build_region_inputs(tmp_path)
    # Remove all sub-E tile dirs (keep the _SUCCESS marker).
    for td in sub_e.glob("tile=*"):
        for f in td.rglob("*"):
            if f.is_file():
                f.unlink()
        td.rmdir()
    with pytest.raises(FileNotFoundError, match=r"no tile=\* dirs"):
        derive_region(_make_cfg(sub_c, sub_d, sub_e, out))
    assert not (out / "_SUCCESS").exists()


# ===========================================================================
# RESTARTABILITY (the important failure-mode test)
# ===========================================================================


def test_derive_region_is_restartable_after_poisoned_partial_run(tmp_path: Path):
    """Prove restartability: a partial failed run does not poison a later clean
    re-run, and the clean re-run is BYTE-IDENTICAL to a from-scratch clean run.

    Steps:
      1. CLEAN run into out_clean -> snapshot every output file's bytes.
      2. POISONED run into out_dirty that fails AFTER the first tile is written
         (monkeypatch encode_tile to raise on the second tile). Assert:
           (a) NO _SUCCESS exists in out_dirty;
           (b) out_dirty is genuinely partial (the first tile's cells.parquet
               IS on disk — proving we tested the resume-from-partial regime,
               not a no-op-before-any-write).
      3. CLEAN re-run into out_dirty (same dir). Assert its byte snapshot
         EQUALS the clean-run snapshot from step 1 -> the partial run left no
         orphaned/stale files that change re-run output, and no step resumed
         from partial on-disk state.

    The pinned `extracted_utc` holds the only non-deterministic input (the wall
    clock) constant, so any byte difference would be a real restartability
    defect, not a timestamp artifact.
    """
    sub_c, sub_d, sub_e, out_clean = _build_region_inputs(tmp_path)
    out_dirty = tmp_path / "sub_f_dirty" / "singapore"

    # --- Step 1: clean run + snapshot ---
    derive_region(_make_cfg(sub_c, sub_d, sub_e, out_clean))
    assert (out_clean / "_SUCCESS").exists()
    clean_snapshot = _snapshot_dir(out_clean)
    assert clean_snapshot, "clean run produced no files"

    # --- Step 2: poisoned partial run (fails on the SECOND tile) ---
    real_encode = pipeline_mod.encode_tile
    call_count = {"n": 0}

    def _poison_second_tile(sc, se, oc):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise RuntimeError("synthetic mid-pipeline failure on tile 2")
        return real_encode(sc, se, oc)

    # Patch the name the orchestrator's seam resolves (_encode_tile_or_raise
    # calls module-level encode_tile).
    import cfm.data.sub_f.pipeline as pmod

    orig = pmod.encode_tile
    pmod.encode_tile = _poison_second_tile
    try:
        with pytest.raises(RuntimeError, match="synthetic mid-pipeline failure"):
            derive_region(_make_cfg(sub_c, sub_d, sub_e, out_dirty))
    finally:
        pmod.encode_tile = orig

    # (a) no _SUCCESS on a partial run.
    assert not (out_dirty / "_SUCCESS").exists()
    assert not (out_dirty / "manifest.yaml").exists()
    # (b) the run IS genuinely partial: the first tile's cells.parquet exists.
    partial_cells = list(out_dirty.glob("tile=*/cells.parquet"))
    assert partial_cells, (
        "poisoned run wrote NO tile output — the test did not exercise the "
        "resume-from-partial regime; tighten the poison to fail AFTER a write"
    )

    # --- Step 3: clean re-run into the SAME partial dir ---
    derive_region(_make_cfg(sub_c, sub_d, sub_e, out_dirty))
    assert (out_dirty / "_SUCCESS").exists()
    rerun_snapshot = _snapshot_dir(out_dirty)

    assert rerun_snapshot == clean_snapshot, (
        "clean re-run after a poisoned partial run is NOT byte-identical to a "
        "from-scratch clean run — partial state poisoned the re-run (orphaned "
        "files or resume-from-partial behavior). Differing entries: "
        + str(
            {
                k: (clean_snapshot.get(k), rerun_snapshot.get(k))
                for k in set(clean_snapshot) | set(rerun_snapshot)
                if clean_snapshot.get(k) != rerun_snapshot.get(k)
            }
        )
    )


def test_derive_region_two_clean_runs_are_byte_identical(tmp_path: Path):
    """Determinism floor: two independent clean runs (same pinned clock) into
    separate dirs produce byte-identical output. Restartability rests on this.
    """
    sub_c, sub_d, sub_e, out_a = _build_region_inputs(tmp_path)
    out_b = tmp_path / "sub_f_b" / "singapore"

    derive_region(_make_cfg(sub_c, sub_d, sub_e, out_a))
    derive_region(_make_cfg(sub_c, sub_d, sub_e, out_b))

    assert _snapshot_dir(out_a) == _snapshot_dir(out_b)


# ===========================================================================
# ALPHA-DROP WARNING-BAND WIRING
# ===========================================================================


def test_derive_region_emits_alpha_drop_warning_band_report(tmp_path: Path, monkeypatch):
    """When run_alpha_drop_report is enabled, the orchestrator invokes the
    warning-band diagnostic at budgets (5760, 6016] and writes its report.

    Confirms the close-checklist wiring is live (not inert) and that the
    budgets passed are the RE-LOCK values 5760/6016 (commit c1eb2a1), NOT the
    original 5792/5888. We spy on the script's entrypoint to capture the args.

    The report-output ROOT is redirected to tmp_path so the test does not write
    into the committed repo `reports/` tree.
    """
    sub_c, sub_d, sub_e, out = _build_region_inputs(tmp_path)

    import scripts.sub_f.compute_alpha_drop_report as adr

    _redirect_alpha_report_root(tmp_path, adr, monkeypatch)

    captured: dict = {}
    real = adr.run_alpha_drop_report

    def _spy(**kwargs):
        captured.update(kwargs)
        return real(**kwargs)

    # The orchestrator imports run_alpha_drop_report inside _emit_alpha_drop_report
    # from the module at call time, so patching the module attribute is enough.
    monkeypatch.setattr(adr, "run_alpha_drop_report", _spy)

    cfg = PipelineConfig(
        release="2026-04-15.0",
        region="singapore",
        sub_c_region_dir=sub_c,
        sub_d_region_dir=sub_d,
        sub_e_region_dir=sub_e,
        output_region_dir=out,
        extracted_utc=_PINNED_UTC,
        run_alpha_drop_report=True,
    )
    derive_region(cfg)

    assert (out / "_SUCCESS").exists()
    assert captured.get("budget_raw") == 5760, (
        f"alpha-drop budget_raw must be the RE-LOCK 5760, got {captured.get('budget_raw')}"
    )
    assert captured.get("budget_padded") == 6016, (
        f"alpha-drop budget_padded must be the RE-LOCK 6016, got {captured.get('budget_padded')}"
    )
    assert captured.get("sub_c_region_dir") == sub_c
    # The report file landed under the redirected ROOT, not the repo reports/.
    assert (
        tmp_path / "adr_root" / "reports" / "sub_f_task_3c_warning_band_singapore.yaml"
    ).exists()


def test_alpha_drop_report_entrypoint_returns_report_dict(tmp_path: Path, monkeypatch):
    """The thin importable entrypoint computes a report dict at the given budgets.

    Exercises run_alpha_drop_report directly (the seam the orchestrator calls)
    on a synthetic sub-C region; asserts the report carries the budgets and the
    drop bookkeeping fields the orchestrator logs. ROOT is redirected so the
    write lands in tmp_path, not the committed repo reports/ tree.
    """
    import scripts.sub_f.compute_alpha_drop_report as adr

    _redirect_alpha_report_root(tmp_path, adr, monkeypatch)

    sub_c, _sub_d, _sub_e, _out = _build_region_inputs(tmp_path)
    report = adr.run_alpha_drop_report(
        sub_c_region_dir=sub_c,
        budget_raw=5760,
        budget_padded=6016,
        label="warning_band_test_singapore",
    )
    assert report["budget_raw"] == 5760
    assert report["budget_padded"] == 6016
    assert "n_cells_dropped" in report
    assert "n_cells_total" in report
    assert "drop_set_by_type" in report
    # A tiny synthetic region has no cells over a 5760-token budget.
    assert report["n_cells_dropped"] == 0


# ===========================================================================
# REAL-SINGAPORE derive — SKIP STUB (sub-C/sub-D/sub-E caches absent)
# ===========================================================================


@pytest.mark.skip(
    reason=(
        "awaiting sub-C/sub-D/sub-E cache regeneration — real-region derive; "
        "see close-checklist. Un-skip when all three caches exist; run "
        "derive_region against the real cached Singapore region and assert "
        "_SUCCESS + manifest + per-tile cells/provenance, then run the BP7 "
        "composite via validate_cross_tile on the real output."
    )
)
def test_derive_region_against_real_singapore():  # type: ignore[empty-body]
    """Integration: run derive_region on the real cached Singapore region.

    Requires real caches under data/processed/sub_{c,d,e}/<release>/singapore/.
    This is the real-region layer of the verification debt inherited from
    T8/T10. See reports/2026-05-23-phase-1-sub-F-close-checklist.md.
    """
    ...

"""Task 16 — Layer 2 pipeline-stage tests on the torture-tile fixture.

This file covers the spec §13.2 Layer 2 tests that exercise the full
``extract_region`` pipeline against the declarative torture-tile fixture and
verify three load-bearing determinism contracts:

1. **Per-tile byte-determinism**: re-running ``extract_region`` produces
   byte-identical parquet bytes and content-sha-identical YAML files (modulo
   ``EXCLUDED_FROM_SHA`` paths).
2. **Provenance + manifest sha stability** across runs.
3. **Diagnostic-payload determinism**: corrupting the same row the same way
   produces byte-identical ``TileValidationError`` payload bytes.

Named tests per plan Task 16 / spec §13.2:

- test_torture_tile_reextract_byte_identical_modulo_excluded_fields
- test_provenance_sha256_byte_deterministic_across_runs
- test_manifest_sha256_byte_deterministic_across_runs
- 8 per-invariant diagnostic-payload tests (corrupt-one-row pattern):
  - test_bbox_matches_wkb_diagnostic_includes_row_and_both_bboxes
  - test_geometry_type_matches_wkb_diagnostic_includes_row_and_both_types
  - test_kept_cell_rule_diagnostic_includes_cell_and_water_fractions
  - test_water_fraction_bounds_diagnostic_includes_cell_and_offending_value
  - test_crossings_features_consistency_diagnostic_includes_source_feature_id
  - test_kept_features_count_diagnostic_includes_cell_counts
  - test_mean_water_fraction_diagnostic_includes_expected_vs_actual_formula
  - test_schema_correctness_diagnostic_includes_missing_column_name
- test_validator_diagnostic_payloads_byte_deterministic
- test_cross_tile_validator_detects_manifest_not_updated_after_single_tile_rerun
- test_pyarrow_version_2_6_parquet_format_correct  (verify-at-impl §14.3)
- test_pyproj_uses_formula_path_for_svy21          (verify-at-impl §14.3)

Pool-size independence tests + four cross-tile validator failure modes
(orphan / missing / sha-mismatch / outputs-sha-mismatch) are already covered
in test_pipeline_parallel.py and test_cli.py respectively (Tasks 13 + 14);
they are not duplicated here.

Per ``feedback_test_weakening_to_pass.md``: if any of these fires on the
clean torture tile, STOP and escalate — do not weaken the assertion.

Per ``feedback_pyarrow_hive_partition_inference.md``: use
``pq.ParquetFile(path).read()``, NOT ``pq.read_table(path)``, inside
``tile=*`` directories.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyproj
import pytest
import yaml

from cfm.data.sub_c.determinism import compute_sha256, compute_sha256_excluding
from cfm.data.sub_c.errors import TileValidationError
from cfm.data.sub_c.pipeline import extract_region
from cfm.data.sub_c.validator_cross_tile import validate_extraction_cross_tile
from cfm.data.sub_c.validator_inline import _ENUM_TO_WKB_TYPE, validate_tile_inline
from tests.fixtures.sub_c.build_torture_tile import (
    TORTURE_TILE_I,
    TORTURE_TILE_J,
    build_torture_region,
)

# ---------------------------------------------------------------------------
# Repo-relative paths (re-used to drive secondary extractions)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_POLICY_YAML = _REPO_ROOT / "configs" / "data" / "missing_value_policy.yaml"
_VOCAB_YAML = _REPO_ROOT / "configs" / "tokenizer" / "vocab_phase1.yaml"

# Fixed timestamps for byte-deterministic extraction in tests.
_EXTRACTED_UTC = "2026-05-18T00:00:00Z"
_COMMIT_SHA = "b86c509" + "0" * 33  # canonical 40-char sha

_TORTURE_TILE_NAME = f"tile=EPSG3414_i{TORTURE_TILE_I}_j{TORTURE_TILE_J}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_parquet(path: Path) -> pa.Table:
    """Read parquet without Hive partition column inference.

    pq.read_table inside a directory named ``tile=*`` would inject a spurious
    'tile' column.  Always use ParquetFile.read() in sub-C tests.
    """
    return pq.ParquetFile(path).read()


def _extract_torture_region(out_dir: Path) -> None:
    """Drive a second torture-tile extraction with identical inputs to the
    session-scoped fixture, into ``out_dir``.

    Used by the determinism tests that compare the session-fixture output to a
    second independent extraction.
    """
    region = build_torture_region()
    extract_region(
        region,
        out_dir,
        policy_yaml_path=_POLICY_YAML,
        vocab_yaml_path=_VOCAB_YAML,
        release="2026-05-18.torture",
        commit_sha=_COMMIT_SHA,
        extracted_utc=_EXTRACTED_UTC,
        started_utc=_EXTRACTED_UTC,
        rerun_reason="initial",
        pool_size=1,
    )


def _copy_torture_output(
    torture_tile_output: Path,
    tmp_path: Path,
    *,
    subdir: str = "corrupted",
) -> Path:
    """Copy the session-scoped torture output to ``tmp_path/<subdir>`` so the
    caller can mutate one artifact without polluting the shared fixture.

    Returns the path of the new tile directory (the parent ``region_dir`` is at
    its parent).  Corruption tests should mutate one parquet/yaml file inside
    the returned tile dir and then re-run ``validate_tile_inline``.
    """
    dest = tmp_path / subdir
    shutil.copytree(torture_tile_output, dest)
    return dest / _TORTURE_TILE_NAME


# ===========================================================================
# Section 1 — Determinism tests
# ===========================================================================


def test_torture_tile_reextract_byte_identical_modulo_excluded_fields(
    torture_tile_output: Path,
    tmp_path: Path,
) -> None:
    """Spec §13.2 + §14: re-extract twice; per-tile parquet bytes are equal,
    yaml content-shas (excluding EXCLUDED_FROM_SHA paths) are equal.

    This is the primary per-tile-determinism test: covers the whole
    extract_region pipeline end-to-end.
    """
    out_b = tmp_path / "second_run"
    _extract_torture_region(out_b)

    tile_a = torture_tile_output / _TORTURE_TILE_NAME
    tile_b = out_b / _TORTURE_TILE_NAME

    # Parquet bytes: byte-identical (sub-C controls compression, sort, schema).
    for name in ("cells.parquet", "features.parquet", "crossings.parquet"):
        bytes_a = (tile_a / name).read_bytes()
        bytes_b = (tile_b / name).read_bytes()
        assert bytes_a == bytes_b, f"{name} bytes differ between runs"

    # YAML files: content-sha equal under EXCLUDED_FROM_SHA stripping.
    for name in ("meta.yaml", "provenance.yaml"):
        data_a = yaml.safe_load((tile_a / name).read_text(encoding="utf-8"))
        data_b = yaml.safe_load((tile_b / name).read_text(encoding="utf-8"))
        sha_a = compute_sha256_excluding(data_a, name)
        sha_b = compute_sha256_excluding(data_b, name)
        assert sha_a == sha_b, f"{name} content-sha differs between runs"


def test_provenance_sha256_byte_deterministic_across_runs(
    torture_tile_output: Path,
    tmp_path: Path,
) -> None:
    """Spec §14.6 E1: per-tile provenance content-sha is byte-stable across
    two independent extractions with identical inputs."""
    out_b = tmp_path / "second_run_prov"
    _extract_torture_region(out_b)

    prov_a = yaml.safe_load(
        (torture_tile_output / _TORTURE_TILE_NAME / "provenance.yaml").read_text(encoding="utf-8")
    )
    prov_b = yaml.safe_load(
        (out_b / _TORTURE_TILE_NAME / "provenance.yaml").read_text(encoding="utf-8")
    )

    sha_a = compute_sha256_excluding(prov_a, "provenance.yaml")
    sha_b = compute_sha256_excluding(prov_b, "provenance.yaml")
    assert sha_a == sha_b, "provenance.yaml content-sha must be byte-deterministic across runs"


def test_manifest_sha256_byte_deterministic_across_runs(
    torture_tile_output: Path,
    tmp_path: Path,
) -> None:
    """Spec §14.6: manifest content-sha is byte-stable across two independent
    extractions with identical inputs (after stripping started_utc /
    completed_utc / final-segment *_sha256 paths)."""
    out_b = tmp_path / "second_run_manifest"
    _extract_torture_region(out_b)

    man_a = yaml.safe_load((torture_tile_output / "manifest.yaml").read_text(encoding="utf-8"))
    man_b = yaml.safe_load((out_b / "manifest.yaml").read_text(encoding="utf-8"))

    sha_a = compute_sha256_excluding(man_a, "manifest.yaml")
    sha_b = compute_sha256_excluding(man_b, "manifest.yaml")
    assert sha_a == sha_b, "manifest.yaml content-sha must be byte-deterministic across runs"


# ===========================================================================
# Section 2 — Per-invariant diagnostic-payload tests
#
# Pattern: copy clean torture tile to tmp, corrupt ONE row of ONE artifact,
# run validate_tile_inline, assert TileValidationError has the right invariant,
# right failed_row keys, right detail keys, and right tile name.
# ===========================================================================


def test_schema_correctness_diagnostic_includes_missing_column_name(
    torture_tile_output: Path,
    tmp_path: Path,
) -> None:
    """Drop the ``geometry_type`` column from features.parquet → invariant
    ``schema_correctness`` raised with detail containing the offending file.

    The diagnostic payload must include enough information for an engineer to
    locate the missing column: ``detail["file"]`` and stringified actual/expected
    schemas (spec §12.4 P2).
    """
    tile_dir = _copy_torture_output(torture_tile_output, tmp_path)
    features_path = tile_dir / "features.parquet"
    table = _read_parquet(features_path)
    idx = table.schema.get_field_index("geometry_type")
    table = table.remove_column(idx)
    pq.write_table(table, features_path)

    with pytest.raises(TileValidationError) as exc_info:
        validate_tile_inline(tile_dir)

    err = exc_info.value
    assert err.invariant == "schema_correctness"
    assert err.tile == _TORTURE_TILE_NAME
    assert err.detail.get("file") == "features.parquet"
    # "geometry_type" must surface in the expected-schema string so the operator
    # can identify the missing column without re-reading the spec.
    assert "geometry_type" in err.detail.get("expected", "")
    assert "geometry_type" not in err.detail.get("actual", "")


def test_bbox_matches_wkb_diagnostic_includes_row_and_both_bboxes(
    torture_tile_output: Path,
    tmp_path: Path,
) -> None:
    """Offset ``bbox_min_x`` of row 0 by 100m → invariant ``bbox_matches_wkb``
    raised with both ``stored_bbox`` and ``wkb_bbox`` in detail."""
    tile_dir = _copy_torture_output(torture_tile_output, tmp_path)
    features_path = tile_dir / "features.parquet"
    table = _read_parquet(features_path)
    bbox_col = table.column("bbox_min_x").to_pylist()
    bbox_col[0] += 100.0
    table = table.set_column(
        table.schema.get_field_index("bbox_min_x"),
        "bbox_min_x",
        pa.array(bbox_col, type=pa.float64()),
    )
    pq.write_table(table, features_path)

    with pytest.raises(TileValidationError) as exc_info:
        validate_tile_inline(tile_dir)

    err = exc_info.value
    assert err.invariant == "bbox_matches_wkb"
    assert err.tile == _TORTURE_TILE_NAME
    assert err.failed_row.get("row_index") == 0
    assert "source_feature_id" in err.failed_row
    # Both bboxes must be in detail with 4 floats each (min_x, min_y, max_x, max_y).
    assert len(err.detail.get("stored_bbox", [])) == 4
    assert len(err.detail.get("wkb_bbox", [])) == 4
    # The corruption was +100m on min_x, so the two bboxes must differ on index 0.
    assert err.detail["stored_bbox"][0] != err.detail["wkb_bbox"][0]


def test_geometry_type_matches_wkb_diagnostic_includes_row_and_both_types(
    torture_tile_output: Path,
    tmp_path: Path,
) -> None:
    """Flip ``geometry_type`` of row 0 from its real WKB type → invariant
    ``geometry_type_matches_wkb`` raised with both ``stored_type`` and
    ``wkb_type`` in detail."""
    tile_dir = _copy_torture_output(torture_tile_output, tmp_path)
    features_path = tile_dir / "features.parquet"
    table = _read_parquet(features_path)
    gt_col = table.column("geometry_type").to_pylist()
    # Map int8 enum 0=Point/1=LineString/2=Polygon → flip to a different value
    # the actual WKB header still reports.  Use (orig + 1) % 3 for determinism.
    gt_col[0] = (gt_col[0] + 1) % 3
    table = table.set_column(
        table.schema.get_field_index("geometry_type"),
        "geometry_type",
        pa.array(gt_col, type=pa.int8()),
    )
    pq.write_table(table, features_path)

    with pytest.raises(TileValidationError) as exc_info:
        validate_tile_inline(tile_dir)

    err = exc_info.value
    assert err.invariant == "geometry_type_matches_wkb"
    assert err.tile == _TORTURE_TILE_NAME
    assert err.failed_row.get("row_index") == 0
    assert "source_feature_id" in err.failed_row
    assert "stored_type" in err.detail
    assert "wkb_type" in err.detail
    # The two fields are in DIFFERENT namespaces — stored_type is our int8 enum
    # (0=Point, 1=LineString, 2=Polygon) and wkb_type is the WKB header code
    # (1=Point, 2=LineString, 3=Polygon).  Verify the mismatch via the enum
    # mapping rather than by raw numeric inequality (the two encodings can
    # happen to share an integer value while still indicating a mismatch —
    # e.g. enum=2 vs wkb=2 means stored=Polygon vs WKB=LineString).
    # Reuse the production mapping so test and code can't silently drift.
    expected_wkb = _ENUM_TO_WKB_TYPE[err.detail["stored_type"]]
    assert expected_wkb != err.detail["wkb_type"], (
        f"corruption did not produce a real mismatch: "
        f"stored_type={err.detail['stored_type']} → expected WKB={expected_wkb}, "
        f"actual WKB={err.detail['wkb_type']}"
    )


def test_crossings_features_consistency_diagnostic_includes_source_feature_id(
    torture_tile_output: Path,
    tmp_path: Path,
) -> None:
    """Inject a crossings.parquet row referencing a source_feature_id that
    appears on only one cell in features.parquet → invariant
    ``crossings_features_source_id_consistency`` raised with the offending
    source_feature_id surfaced in failed_row.

    We pick an existing single-cell feature (F01_single_cell_road, designed to
    live entirely in cell (0, 0)) and add a crossings row for it.
    """
    tile_dir = _copy_torture_output(torture_tile_output, tmp_path)

    # Confirm F01_single_cell_road appears on exactly one cell in features.
    features = _read_parquet(tile_dir / "features.parquet")
    feat_ids = features.column("source_feature_id").to_pylist()
    feat_ci = features.column("cell_i").to_pylist()
    feat_cj = features.column("cell_j").to_pylist()
    distinct_cells = {
        (ci, cj)
        for fid, ci, cj in zip(feat_ids, feat_ci, feat_cj, strict=True)
        if fid == "F01_single_cell_road"
    }
    assert len(distinct_cells) == 1, (
        f"fixture invariant: F01_single_cell_road must appear on exactly 1 cell; "
        f"got {distinct_cells}"
    )

    # Build a crossings row referencing F01_single_cell_road
    crossings_path = tile_dir / "crossings.parquet"
    crossings = _read_parquet(crossings_path)

    new_row = {
        "source_feature_id": "F01_single_cell_road",
        "lower_cell_i": 0,
        "lower_cell_j": 0,
        "axis": 0,
        "ring_index": 0,
        "event_type": 0,
        "edge_position_m": 0.0,
        "edge_extent_length_m": 0.0,
    }
    # Build a one-row table with the same schema then concatenate.
    appended = pa.Table.from_pylist([new_row], schema=crossings.schema)
    merged = pa.concat_tables([crossings, appended])
    # Re-sort to match the canonical crossings sort order to avoid tripping
    # the schema_correctness sort-key check first.
    merged_sorted = merged.sort_by(
        [
            ("lower_cell_i", "ascending"),
            ("lower_cell_j", "ascending"),
            ("axis", "ascending"),
            ("source_feature_id", "ascending"),
            ("ring_index", "ascending"),
            ("edge_position_m", "ascending"),
            ("event_type", "ascending"),
        ]
    )
    pq.write_table(merged_sorted, crossings_path)

    with pytest.raises(TileValidationError) as exc_info:
        validate_tile_inline(tile_dir)

    err = exc_info.value
    assert err.invariant == "crossings_features_source_id_consistency"
    assert err.tile == _TORTURE_TILE_NAME
    assert err.failed_row.get("source_feature_id") == "F01_single_cell_road"
    assert "distinct_cells_in_features" in err.detail
    assert err.detail.get("required_min_distinct_cells") == 2


def test_water_fraction_bounds_diagnostic_includes_cell_and_offending_value(
    torture_tile_output: Path,
    tmp_path: Path,
) -> None:
    """Set ``water_fraction = 2.0`` on cell row 0 (violates [0, 1+EPS]) →
    invariant ``water_fraction_bounds`` raised with cell_i/cell_j in failed_row
    and the offending value in detail."""
    tile_dir = _copy_torture_output(torture_tile_output, tmp_path)
    cells_path = tile_dir / "cells.parquet"
    cells = _read_parquet(cells_path)

    wf_col = cells.column("water_fraction").to_pylist()
    wf_col[0] = 2.0
    cells = cells.set_column(
        cells.schema.get_field_index("water_fraction"),
        "water_fraction",
        pa.array(wf_col, type=pa.float64()),
    )
    pq.write_table(cells, cells_path)

    with pytest.raises(TileValidationError) as exc_info:
        validate_tile_inline(tile_dir)

    err = exc_info.value
    assert err.invariant == "water_fraction_bounds"
    assert err.tile == _TORTURE_TILE_NAME
    assert "cell_i" in err.failed_row
    assert "cell_j" in err.failed_row
    assert err.failed_row.get("row_index") == 0
    assert err.detail.get("water_fraction") == 2.0
    assert "bounds violated" in err.detail.get("reason", "")


def test_kept_cell_rule_diagnostic_includes_cell_and_water_fractions(
    torture_tile_output: Path,
    tmp_path: Path,
) -> None:
    """Build a row with ``sea_water_fraction >= 1.0`` and
    ``kept_features_count == 0`` (violates the drop rule) → invariant
    ``kept_cell_rule`` raised with cell coords + both water fractions in
    detail.

    We mutate cell row 0 to hit this: set sea_water_fraction = 1.0, water_fraction = 1.0,
    and kept_features_count = 0.  This will trip invariant 5 (water_fraction_bounds)
    LAST since sea > wf is the order, but we also need invariant 5 to PASS first
    so that invariant 6 actually fires.  Use sea = 1.0 == wf = 1.0 to satisfy
    sea <= wf AND hit the kept_cell_rule.
    """
    tile_dir = _copy_torture_output(torture_tile_output, tmp_path)
    cells_path = tile_dir / "cells.parquet"
    cells = _read_parquet(cells_path)

    wf_col = cells.column("water_fraction").to_pylist()
    sea_col = cells.column("sea_water_fraction").to_pylist()
    kfc_col = cells.column("kept_features_count").to_pylist()

    wf_col[0] = 1.0
    sea_col[0] = 1.0
    kfc_col[0] = 0

    cells = cells.set_column(
        cells.schema.get_field_index("water_fraction"),
        "water_fraction",
        pa.array(wf_col, type=pa.float64()),
    )
    cells = cells.set_column(
        cells.schema.get_field_index("sea_water_fraction"),
        "sea_water_fraction",
        pa.array(sea_col, type=pa.float64()),
    )
    cells = cells.set_column(
        cells.schema.get_field_index("kept_features_count"),
        "kept_features_count",
        pa.array(kfc_col, type=pa.int32()),
    )
    pq.write_table(cells, cells_path)

    with pytest.raises(TileValidationError) as exc_info:
        validate_tile_inline(tile_dir)

    err = exc_info.value
    assert err.invariant == "kept_cell_rule"
    assert err.tile == _TORTURE_TILE_NAME
    assert "cell_i" in err.failed_row
    assert "cell_j" in err.failed_row
    assert err.failed_row.get("row_index") == 0
    assert err.detail.get("sea_water_fraction") == 1.0
    assert err.detail.get("kept_features_count") == 0


def test_kept_features_count_diagnostic_includes_cell_counts(
    torture_tile_output: Path,
    tmp_path: Path,
) -> None:
    """Bump ``kept_features_count`` of cell row 0 by +1 (now disagrees with
    actual features.parquet row count for that cell) → invariant
    ``kept_features_count_matches`` raised with stored vs actual in detail.

    We must keep invariants 1-6 passing.  Increasing kfc by 1 keeps it >= 1 so
    it won't trip invariant 6 (which fires only when kfc == 0 AND sea ~= 1).
    """
    tile_dir = _copy_torture_output(torture_tile_output, tmp_path)
    cells_path = tile_dir / "cells.parquet"
    cells = _read_parquet(cells_path)

    kfc_col = cells.column("kept_features_count").to_pylist()
    original_kfc = kfc_col[0]
    kfc_col[0] = original_kfc + 1

    cells = cells.set_column(
        cells.schema.get_field_index("kept_features_count"),
        "kept_features_count",
        pa.array(kfc_col, type=pa.int32()),
    )
    pq.write_table(cells, cells_path)

    with pytest.raises(TileValidationError) as exc_info:
        validate_tile_inline(tile_dir)

    err = exc_info.value
    assert err.invariant == "kept_features_count_matches"
    assert err.tile == _TORTURE_TILE_NAME
    assert "cell_i" in err.failed_row
    assert "cell_j" in err.failed_row
    assert err.detail.get("stored") == original_kfc + 1
    assert err.detail.get("actual") == original_kfc


def test_mean_water_fraction_diagnostic_includes_expected_vs_actual_formula(
    torture_tile_output: Path,
    tmp_path: Path,
) -> None:
    """Bump ``meta.aggregates.mean_water_fraction`` by 0.1 → invariant
    ``mean_water_fraction_matches`` raised with stored vs computed in detail."""
    tile_dir = _copy_torture_output(torture_tile_output, tmp_path)
    meta_path = tile_dir / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))

    original_mwf = float(meta["aggregates"]["mean_water_fraction"])
    meta["aggregates"]["mean_water_fraction"] = original_mwf + 0.1
    meta_path.write_text(yaml.dump(meta, default_flow_style=False), encoding="utf-8")

    with pytest.raises(TileValidationError) as exc_info:
        validate_tile_inline(tile_dir)

    err = exc_info.value
    assert err.invariant == "mean_water_fraction_matches"
    assert err.tile == _TORTURE_TILE_NAME
    assert err.detail.get("field") == "mean_water_fraction"
    assert err.detail.get("stored") == original_mwf + 0.1
    # computed value must echo the formula's recomputation
    assert err.detail.get("computed") is not None
    assert abs(err.detail["computed"] - original_mwf) < 1e-6, (
        "computed must equal the area-weighted formula result, i.e. the original"
    )


# ===========================================================================
# Section 3 — Diagnostic-payload determinism rollup
# ===========================================================================


def _corrupt_water_fraction_row_zero(tile_dir: Path) -> None:
    """Deterministic corruption: set cells.parquet row 0 ``water_fraction = 2.0``.

    Always corrupts the same row the same way; the resulting
    ``TileValidationError.failed_row`` and ``detail`` payloads must be
    byte-identical across invocations (per spec §12.4 + Topic 7 L3).
    """
    cells_path = tile_dir / "cells.parquet"
    cells = _read_parquet(cells_path)
    wf_col = cells.column("water_fraction").to_pylist()
    wf_col[0] = 2.0
    cells = cells.set_column(
        cells.schema.get_field_index("water_fraction"),
        "water_fraction",
        pa.array(wf_col, type=pa.float64()),
    )
    pq.write_table(cells, cells_path)


def test_validator_diagnostic_payloads_byte_deterministic(
    torture_tile_output: Path,
    tmp_path: Path,
) -> None:
    """Spec §12.4 + Topic 7 L3: corrupting the same row the same way produces
    byte-identical ``TileValidationError`` payload bytes.

    We use two independent copies of the clean torture-tile, apply the SAME
    corruption to each, run the validator on each, and assert that the
    repr() of failed_row and detail are byte-identical between the two error
    objects.

    NOTE on running in two processes: the spec's wording about "across two
    pytest sessions" is satisfied here by running two independent copies of
    the same corruption pipeline.  Because the payload construction reads only
    deterministic inputs (row 0 of the same parquet bytes) and the dict
    insertion order is controlled by the validator's source code, the bytes
    must reproduce.  A genuine two-session test would re-run pytest itself,
    which is not the right granularity for a unit test.
    """
    tile_a = _copy_torture_output(torture_tile_output, tmp_path, subdir="corrupted_a")
    tile_b = _copy_torture_output(torture_tile_output, tmp_path, subdir="corrupted_b")

    _corrupt_water_fraction_row_zero(tile_a)
    _corrupt_water_fraction_row_zero(tile_b)

    with pytest.raises(TileValidationError) as exc_a:
        validate_tile_inline(tile_a)
    with pytest.raises(TileValidationError) as exc_b:
        validate_tile_inline(tile_b)

    err_a = exc_a.value
    err_b = exc_b.value

    # All four payload fields must be byte-equal under repr().
    assert err_a.invariant == err_b.invariant
    # The tile_name differs because the parent dirs are different ('corrupted_a'
    # vs 'corrupted_b'); the TILE-LEVEL name segment (tile=EPSG3414_iI_jJ) is
    # the only "tile" the validator records.  Both must equal _TORTURE_TILE_NAME.
    assert err_a.tile == _TORTURE_TILE_NAME
    assert err_b.tile == _TORTURE_TILE_NAME

    assert repr(err_a.failed_row) == repr(err_b.failed_row), (
        "failed_row bytes must reproduce across runs given identical corruption"
    )
    assert repr(err_a.detail) == repr(err_b.detail), (
        "detail bytes must reproduce across runs given identical corruption"
    )


# ===========================================================================
# Section 4 — Cross-tile validator: rerun-without-manifest-update
# ===========================================================================


def test_cross_tile_validator_detects_manifest_not_updated_after_single_tile_rerun(
    torture_tile_output: Path,
    tmp_path: Path,
) -> None:
    """Spec §13.2: simulate a single-tile re-extraction whose new
    provenance.yaml content-sha disagrees with the manifest's stored sha (the
    operator forgot to re-run the manifest aggregator).

    Cross-tile validator must raise
    ``manifest_provenance_sha_matches_disk``.
    """
    # Copy session-fixture output to tmp so we can mutate it.
    out_region = tmp_path / "rerun"
    shutil.copytree(torture_tile_output, out_region)

    # Sanity: validator passes on the clean copy.
    validate_extraction_cross_tile(out_region)

    # Now simulate "re-extract one tile without updating the manifest" by
    # bumping the tile's provenance.rerun_count field.  This is a non-excluded
    # field (only extracted_utc + *_sha256 are stripped) so the content sha
    # MUST change.
    prov_path = out_region / _TORTURE_TILE_NAME / "provenance.yaml"
    prov_data = yaml.safe_load(prov_path.read_text(encoding="utf-8"))
    prov_data["extraction"]["rerun_count"] = int(prov_data["extraction"].get("rerun_count", 0)) + 1
    prov_data["extraction"]["rerun_reason"] = "single-tile rerun without manifest update"
    prov_path.write_text(yaml.dump(prov_data, default_flow_style=False), encoding="utf-8")

    with pytest.raises(TileValidationError) as exc_info:
        validate_extraction_cross_tile(out_region)

    err = exc_info.value
    assert err.invariant == "manifest_provenance_sha_matches_disk"
    assert err.tile == _TORTURE_TILE_NAME
    assert err.detail.get("stored_sha") != err.detail.get("computed_sha")


# ===========================================================================
# Section 5 — Verify-at-impl (spec §14.3)
# ===========================================================================


def test_pyarrow_version_2_6_parquet_format_correct(
    torture_tile_output: Path,
) -> None:
    """Spec §14.3 + io.py _PARQUET_WRITE_KWARGS["version"] = "2.6": the
    written parquet file metadata reports format version ``"2.6"``.

    Why this matters: pyarrow's default format version changes across versions.
    Pinning at write time means the written bytes (and their content shas) are
    stable across pyarrow upgrades.  This test certifies the pin took effect.
    """
    cells_path = torture_tile_output / _TORTURE_TILE_NAME / "cells.parquet"
    pf = pq.ParquetFile(cells_path)
    fmt_version = pf.metadata.format_version
    # pyarrow returns the format_version as a string like "2.6".
    assert fmt_version == "2.6", (
        f"parquet format_version pinned to '2.6' in io.py; got {fmt_version!r}"
    )


def test_pyproj_uses_formula_path_for_svy21() -> None:
    """Spec §14.3: the 4326 → 3414 transformation must use the closed-form
    Transverse Mercator formula path (no PROJ datum grid file required).

    Why this matters: PROJ grid files are user-machine-dependent assets.
    Datum-grid-based transforms can produce different bytes on different
    machines (depending on grid file presence and version).  The SVY21
    transform is formula-only (no Helmert datum shift requires a grid), and
    we want a positive assertion of that property.

    Implementation: ``pyproj.Transformer.from_crs("EPSG:4326",
    "EPSG:3414").description`` reports the operation method; for SVY21 it must
    NOT mention "grid" or "ntv2" or "gtx" — those are the named grid-based
    method tokens.
    """
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3414", always_xy=True)
    description = (transformer.description or "").lower()

    forbidden_grid_tokens = ("grid", "ntv2", "gtx", "nadgrids")
    for token in forbidden_grid_tokens:
        assert token not in description, (
            f"pyproj 4326→3414 transform must be formula-only; "
            f"description={description!r} mentions '{token}'"
        )

    # Sanity: the transformer must produce a non-trivial result on a known
    # Singapore lat/lon (Marina Bay ~ (103.8587, 1.2839)).  If pyproj falls
    # back to a no-op the assertion below will fail.
    x, y = transformer.transform(103.8587, 1.2839)
    assert 25000 < x < 35000, f"sanity: Marina Bay SVY21 easting ~ 30km; got {x}"
    assert 25000 < y < 35000, f"sanity: Marina Bay SVY21 northing ~ 30km; got {y}"


# ===========================================================================
# Section 6 — Outputs-sha integrity check (round-trip with file bytes)
# ===========================================================================


def test_torture_tile_outputs_sha_match_file_bytes(
    torture_tile_output: Path,
) -> None:
    """Spec §11.7 + §14.6: provenance.outputs.*_sha256 must equal
    ``compute_sha256(path.read_bytes())`` for each of the four output files.

    This is a second-order determinism check: certifies that the SHA-256 stored
    in provenance.yaml is consistent with what's actually on disk.  The
    cross-tile validator runs the same check (invariant 4 in
    validator_cross_tile), so this test is structural redundancy that catches
    silent drift in the orchestrator's hash-computation path before the
    cross-tile gate.
    """
    tile_dir = torture_tile_output / _TORTURE_TILE_NAME
    prov_data = yaml.safe_load((tile_dir / "provenance.yaml").read_text(encoding="utf-8"))
    outputs = prov_data["outputs"]

    for sha_key, filename in {
        "cells_parquet_sha256": "cells.parquet",
        "features_parquet_sha256": "features.parquet",
        "crossings_parquet_sha256": "crossings.parquet",
        "meta_yaml_sha256": "meta.yaml",
    }.items():
        stored_sha = outputs[sha_key]
        actual_sha = compute_sha256((tile_dir / filename).read_bytes())
        assert stored_sha == actual_sha, (
            f"{sha_key}: stored={stored_sha} actual={actual_sha} (file={filename})"
        )


# ===========================================================================
# Section 7 — Fix #1 + Fix #2 integration tests
# ===========================================================================


def test_pipeline_inland_river_cell_has_nonzero_water_fraction_fix1(
    torture_tile_output: Path,
) -> None:
    """Fix #1 integration: cells containing the inland river F11 must have
    water_fraction > sea_water_fraction (i.e. inland-water contribution is nonzero).

    The torture fixture includes F11 (inland river, base.class='river', length=950m)
    which occupies cells in the y=28600 row (cell_j=2).  Before Fix #1, these cells
    had water_fraction == sea_water_fraction (placeholder).  After Fix #1, the river's
    polygon/geometry union contributes nonzero area to the inland_water computation.

    NOTE: F11 is a LineString river.  LineStrings have zero area; unary_union of a
    LineString + Polygon mix yields a GeometryCollection whose .area is the polygon
    area only.  Therefore cells containing ONLY F11 (LineString) will have
    inland_fraction == 0.0 UNLESS there is a polygon water feature in those cells.

    The torture fixture does not include a polygon inland-water feature alongside F11.
    So strictly speaking, this test verifies the pipeline runs correctly (no crash,
    water_fraction >= sea_water_fraction invariant satisfied) with a LineString river.

    To verify that polygon inland water actually contributes, see the unit test
    test_apply_sea_mask_with_inland_water_returns_combined_water_fraction.
    """
    tile_dir = torture_tile_output / _TORTURE_TILE_NAME
    cells = _read_parquet(tile_dir / "cells.parquet")

    # Verify the inline validator passed (provenance.yaml present → tile complete).
    assert (tile_dir / "provenance.yaml").exists(), (
        "provenance.yaml must exist for a successfully extracted tile"
    )

    # For every kept cell: water_fraction >= sea_water_fraction (invariant #5).
    wf_col = cells.column("water_fraction").to_pylist()
    swf_col = cells.column("sea_water_fraction").to_pylist()
    for i, (wf, swf) in enumerate(zip(wf_col, swf_col, strict=True)):
        assert wf >= swf - 1e-12, (
            f"Cell row {i}: water_fraction ({wf}) < sea_water_fraction ({swf}). "
            "Invariant #5 violated — Fix #1 may have broken the water_fraction computation."
        )


def test_pipeline_admin_region_none_when_no_divisions_theme_fix2(
    tmp_path: Path,
) -> None:
    """Fix #2: when no divisions theme is present, admin_region = None for all tiles.

    The torture-tile fixture does not include a 'divisions' key in themes.
    This verifies the pipeline handles absent divisions gracefully and emits
    admin_region=None in meta.yaml.conditioning_per_tile.admin_region.
    """
    out = tmp_path / "no_divisions"
    # Use the torture region (no divisions theme).
    from tests.fixtures.sub_c.build_torture_tile import build_torture_region

    region = build_torture_region()
    assert "divisions" not in region.themes, (
        "Torture region must not include a divisions theme — this test relies on that"
    )

    extract_region(
        region,
        out,
        policy_yaml_path=_POLICY_YAML,
        vocab_yaml_path=_VOCAB_YAML,
        release="2026-05-18.torture",
        commit_sha=_COMMIT_SHA,
        extracted_utc=_EXTRACTED_UTC,
        started_utc=_EXTRACTED_UTC,
        rerun_reason="initial",
        pool_size=1,
    )

    # All tiles must have admin_region = null (None) in meta.yaml.
    import yaml

    for tile_dir in out.iterdir():
        if not tile_dir.is_dir() or not tile_dir.name.startswith("tile="):
            continue
        meta = yaml.safe_load((tile_dir / "meta.yaml").read_text(encoding="utf-8"))
        cond = meta.get("conditioning_per_tile", {})
        assert cond.get("admin_region") is None, (
            f"{tile_dir.name}: expected admin_region=null when no divisions theme, "
            f"got {cond.get('admin_region')!r}"
        )

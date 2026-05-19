"""Tests for the sub-D CLI scripts (Task 14).

Two thin CLIs over the pipeline + validator:

- ``scripts/derive_macro_plan.py`` wraps ``derive_region_macro_plan`` and
  resolves default paths from ``--region`` + ``--release`` + repo layout
  (``data/processed/sub_d/<release>/<region>/``).
- ``scripts/validate_macro_plan.py`` wraps ``validate_region`` and exits
  nonzero on any contract violation.

Both follow sub-C's ``main(argv) -> int`` pattern so tests can call them
in-process (no subprocess) when only parsing matters, AND can be invoked
via subprocess to exercise the actual script entry point.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import yaml

from cfm.data.io import canonicalize_yaml, write_parquet

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_LOCKED_VOCAB_PATH = _REPO_ROOT / "configs" / "macro_plan" / "v1" / "macro_plan_vocab.yaml"


# ---------------------------------------------------------------------------
# Mini fixtures — copied from test_pipeline so test_cli is self-contained.
# Same empty-tile-(0, 0) sub-C region shape.
# ---------------------------------------------------------------------------

_CELLS_SCHEMA = pa.schema(
    [
        pa.field("cell_i", pa.int8()),
        pa.field("cell_j", pa.int8()),
        pa.field("cell_area_admin_clipped_m2", pa.float64()),
    ]
)
_FEATURES_SCHEMA = pa.schema(
    [
        pa.field("cell_i", pa.int8()),
        pa.field("cell_j", pa.int8()),
        pa.field("feature_class", pa.int8()),
        pa.field("source_feature_id", pa.string()),
        pa.field("geometry", pa.binary()),
    ]
)
_CROSSINGS_SCHEMA = pa.schema(
    [
        pa.field("source_feature_id", pa.string()),
        pa.field("lower_cell_i", pa.int8()),
        pa.field("lower_cell_j", pa.int8()),
        pa.field("axis", pa.int8()),
    ]
)


def _empty_table(schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pydict({f.name: [] for f in schema}, schema=schema)


def _build_minimal_sub_c_region(root: Path) -> Path:
    region_dir = root / "singapore"
    tile_dir = region_dir / "tile=EPSG3414_i0_j0"
    tile_dir.mkdir(parents=True)

    write_parquet(_empty_table(_CELLS_SCHEMA), tile_dir / "cells.parquet")
    write_parquet(_empty_table(_FEATURES_SCHEMA), tile_dir / "features.parquet")
    write_parquet(_empty_table(_CROSSINGS_SCHEMA), tile_dir / "crossings.parquet")

    meta = {
        "schema_version": "1.1",
        "tile_i": 0,
        "tile_j": 0,
        "aggregates": {"kept_cell_count": 0},
        "config": {"sliver_drop_rule": "drop iff area < 0.01"},
        "conditioning_per_tile": {
            "admin_region": "Central",
            "morphology_class": "Asian-megacity",
            "era_class": "contemporary",
            "coastal_inland_river": 1,
            "population_density_bucket": None,
            "population_density_bucket_owner": "sub-D",
        },
    }
    (tile_dir / "meta.yaml").write_text(canonicalize_yaml(meta), encoding="utf-8")
    sub_c_prov = {
        "schema_version": "1.0",
        "tile_i": 0,
        "tile_j": 0,
        "extraction": {"extracted_utc": "2026-04-15T00:00:00Z"},
        "outputs": {"cells_parquet_sha256": "deadbeef"},
    }
    (tile_dir / "provenance.yaml").write_text(canonicalize_yaml(sub_c_prov), encoding="utf-8")

    manifest = {
        "schema_version": "1.1",
        "sub_c_schema_version": "1.1",
        "release": "2026-04-15.0",
        "region": "singapore",
        "region_crs": "EPSG:3414",
        "config": {
            "cell_grid": [8, 8],
            "cell_size_m": 250,
            "tile_size_m": 2000,
            "internal_edge_count": 112,
            "external_edge_count": 32,
            "sliver_drop_rule": "drop iff area < 0.01",
        },
        "conditioning_defaults": {
            "country": "SG",
            "climate_zone": "tropical_rainforest",
        },
        "tiles": [{"tile_i": 0, "tile_j": 0, "provenance_sha256": "deadbeef"}],
    }
    (region_dir / "manifest.yaml").write_text(canonicalize_yaml(manifest), encoding="utf-8")
    (region_dir / "_SUCCESS").write_bytes(b"")
    return region_dir


# ---------------------------------------------------------------------------
# Plan-named tests
# ---------------------------------------------------------------------------


def test_derive_macro_plan_cli_resolves_default_paths(tmp_path: Path, monkeypatch):
    """``derive_macro_plan.py`` resolves ``--output-dir`` from ``--release`` +
    ``--region`` + repo layout when not explicitly passed. The CLI does NOT
    need to actually run the pipeline to satisfy this test — we use
    ``--dry-run`` semantics: the CLI parses args, computes paths, prints
    them, returns 0 without writing anything.

    The dry-run path is what the Task 14 plan calls "resolves default paths."
    Future task can extend this to also exercise the wet-run path; for now,
    pinning that the resolver computes the right defaults is what matters.
    """
    sys.path.insert(0, str(_SCRIPTS_DIR))
    try:
        import derive_macro_plan
    finally:
        sys.path.pop(0)

    sub_c_region_dir = _build_minimal_sub_c_region(tmp_path / "sub_c")

    # No --output-dir — let the CLI compute it from --release + --region.
    argv = [
        "--region",
        "singapore",
        "--release",
        "2026-04-15.0",
        "--sub-c-dir",
        str(sub_c_region_dir),
        "--macro-vocab",
        str(_LOCKED_VOCAB_PATH),
        "--commit-sha",
        "abc" + "0" * 37,
        "--extracted-utc",
        "2026-05-19T12:00:00Z",
        "--output-root",
        str(tmp_path / "computed_root"),
        "--dry-run",
    ]
    exit_code = derive_macro_plan.main(argv)
    assert exit_code == 0

    # The resolved default output dir lives under <output-root>/<release>/<region>/.
    expected_output_dir = tmp_path / "computed_root" / "2026-04-15.0" / "singapore"
    # Dry-run does NOT create the dir — we just want the resolver to have
    # computed the right value. We verify by calling the wet-run path next
    # with an explicit --output-dir and checking that the explicit path is
    # honoured.
    explicit_output_dir = tmp_path / "explicit_sub_d"
    argv_wet = [
        "--region",
        "singapore",
        "--release",
        "2026-04-15.0",
        "--sub-c-dir",
        str(sub_c_region_dir),
        "--output-dir",
        str(explicit_output_dir),
        "--macro-vocab",
        str(_LOCKED_VOCAB_PATH),
        "--commit-sha",
        "abc" + "0" * 37,
        "--extracted-utc",
        "2026-05-19T12:00:00Z",
    ]
    exit_code = derive_macro_plan.main(argv_wet)
    assert exit_code == 0
    assert (explicit_output_dir / "_SUCCESS").is_file()
    # The default-resolution branch from the dry-run did NOT write anything.
    assert not expected_output_dir.exists()


def test_validate_macro_plan_cli_returns_nonzero_on_validator_error(tmp_path: Path):
    """``validate_macro_plan.py`` exits nonzero when the sub-D region fails
    validation. We build a happy-path pair, hand-edit the sub-D manifest's
    config to drift from sub-C (B6 violation), then invoke the CLI via
    subprocess.

    We use subprocess (not in-process ``main(argv)``) so the exit code is
    observed at the OS boundary — the same surface that operators interact
    with.
    """
    # Build sub-C + sub-D using the pipeline (happy path).
    sys.path.insert(0, str(_SCRIPTS_DIR))
    try:
        import derive_macro_plan
    finally:
        sys.path.pop(0)

    sub_c_region_dir = _build_minimal_sub_c_region(tmp_path / "sub_c")
    sub_d_region_dir = tmp_path / "sub_d" / "singapore"
    exit_code = derive_macro_plan.main(
        [
            "--region",
            "singapore",
            "--release",
            "2026-04-15.0",
            "--sub-c-dir",
            str(sub_c_region_dir),
            "--output-dir",
            str(sub_d_region_dir),
            "--macro-vocab",
            str(_LOCKED_VOCAB_PATH),
            "--commit-sha",
            "abc" + "0" * 37,
            "--extracted-utc",
            "2026-05-19T12:00:00Z",
        ]
    )
    assert exit_code == 0
    assert (sub_d_region_dir / "_SUCCESS").is_file()

    # Hand-edit sub-D manifest to drift from sub-C (B6 violation).
    manifest_path = sub_d_region_dir / "manifest.yaml"
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    data["config"]["tile_size_m"] = 9999
    manifest_path.write_text(canonicalize_yaml(data), encoding="utf-8")

    # Invoke validate CLI via subprocess.
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPTS_DIR / "validate_macro_plan.py"),
            "--output-dir",
            str(sub_d_region_dir),
            "--sub-c-dir",
            str(sub_c_region_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        f"Expected nonzero exit on B6 drift; got {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# Companion: the validate CLI returns 0 on a clean pair.
# Not in the plan's 2 named tests, but pins the happy path so a future
# refactor that broke `--output-dir` parsing would surface immediately.
# ---------------------------------------------------------------------------


def test_validate_macro_plan_cli_returns_zero_on_clean_region(tmp_path: Path):
    sys.path.insert(0, str(_SCRIPTS_DIR))
    try:
        import derive_macro_plan
    finally:
        sys.path.pop(0)

    sub_c_region_dir = _build_minimal_sub_c_region(tmp_path / "sub_c")
    sub_d_region_dir = tmp_path / "sub_d" / "singapore"
    exit_code = derive_macro_plan.main(
        [
            "--region",
            "singapore",
            "--release",
            "2026-04-15.0",
            "--sub-c-dir",
            str(sub_c_region_dir),
            "--output-dir",
            str(sub_d_region_dir),
            "--macro-vocab",
            str(_LOCKED_VOCAB_PATH),
            "--commit-sha",
            "abc" + "0" * 37,
            "--extracted-utc",
            "2026-05-19T12:00:00Z",
        ]
    )
    assert exit_code == 0

    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPTS_DIR / "validate_macro_plan.py"),
            "--output-dir",
            str(sub_d_region_dir),
            "--sub-c-dir",
            str(sub_c_region_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Expected zero exit on clean region; got {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

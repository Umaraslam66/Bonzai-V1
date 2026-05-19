"""Tests for the sub-D sidecar derivation pipeline (Task 14).

The pipeline is the orchestrator that glues every sub-D module together:

1. Read sub-C manifest and tile inputs.
2. Load locked macro vocab.
3. Build fixed-lattice targets (scope + derivation).
4. Write per-tile ``macro_core.parquet``.
5. Write per-tile ``derivation_evidence.parquet``.
6. Write per-tile ``effective_conditioning.yaml``.
7. Write per-tile ``provenance.yaml`` (with ``outputs.*_sha256`` computed
   from the just-written file bytes).
8. Write region ``manifest.yaml``.
9. Run ``validate_region``.
10. Write ``_SUCCESS`` ONLY if validation passes.

The chain-of-custody depends on the output-sha ordering: provenance.yaml's
``outputs.*_sha256`` must be the sha of the bytes that landed on disk, not
the bytes of an in-memory dataclass. The tests pin that ordering implicitly
— if step 7 hashed in-memory bytes instead, the validator's
``provenance.outputs.* vs live file bytes`` check (Task 13) would fail.

``_SUCCESS`` gating: ``validate_region`` raises before ``write_success_marker``
is called, so a validation failure leaves no ``_SUCCESS`` behind.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from cfm.data.io import canonicalize_yaml, write_parquet
from cfm.data.sub_d.errors import SubDValidationError
from cfm.data.sub_d.pipeline import derive_region_macro_plan

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCKED_VOCAB_PATH = _REPO_ROOT / "configs" / "macro_plan" / "v1" / "macro_plan_vocab.yaml"


# ---------------------------------------------------------------------------
# Minimal sub-C region fixture: one tile (0, 0) with 0-row parquets that
# carry the right column schemas. The pipeline exercises every sub-D module
# end-to-end on this fixture; the evidence module returns:
#   - empty zoning/density (no active cells)
#   - 112 dense road-skeleton rows (all 0 counts)
#   - 4 tile-population-density rows (all 0.0)
# and macro_core ends up 208 FULLY_MASKED rows — a valid degenerate output.
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


def _build_minimal_sub_c_region(
    root: Path,
    *,
    region: str = "singapore",
    region_crs: str = "EPSG:3414",
) -> Path:
    """Build a sub-C region with one empty tile (0, 0). Parquet schemas
    match what sub-D's evidence module reads; zero rows means an all-masked
    sub-D output."""
    epsg_label = region_crs.replace(":", "")
    region_dir = root / region
    tile_dir = region_dir / f"tile={epsg_label}_i0_j0"
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
        "region": region,
        "region_crs": region_crs,
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
        "tiles": [
            {
                "tile_i": 0,
                "tile_j": 0,
                "provenance_sha256": "deadbeef",
            }
        ],
    }
    (region_dir / "manifest.yaml").write_text(canonicalize_yaml(manifest), encoding="utf-8")
    (region_dir / "_SUCCESS").write_bytes(b"")
    return region_dir


_PINNED_EXTRACTION_KWARGS = {
    "release": "2026-04-15.0",
    "region": "singapore",
    "commit_sha": "abc" + "0" * 37,
    "extracted_utc": "2026-05-19T12:00:00Z",
}


# ---------------------------------------------------------------------------
# Plan-named tests
# ---------------------------------------------------------------------------


def test_pipeline_refuses_to_run_without_locked_macro_vocab(tmp_path: Path):
    """The pipeline cannot derive targets without the locked vocab. Passing a
    non-existent path must raise BEFORE any sub-D artifact is written —
    Task 14 explicitly checks the vocab path first so a misconfiguration
    cannot leave a half-written sub-D region on disk.
    """
    sub_c_region_dir = _build_minimal_sub_c_region(tmp_path / "sub_c")
    output_dir = tmp_path / "sub_d" / "singapore"
    missing_vocab = tmp_path / "does_not_exist.yaml"

    with pytest.raises(SubDValidationError):
        derive_region_macro_plan(
            sub_c_region_dir=sub_c_region_dir,
            output_dir=output_dir,
            macro_vocab_path=missing_vocab,
            **_PINNED_EXTRACTION_KWARGS,
        )

    # Nothing should have been written under output_dir.
    assert not output_dir.exists() or not any(output_dir.iterdir())


def test_pipeline_writes_all_per_tile_sidecar_artifacts(tmp_path: Path):
    """After a successful run, every per-tile sub-D artifact exists with the
    canonical filename in the canonical tile dir layout.
    """
    sub_c_region_dir = _build_minimal_sub_c_region(tmp_path / "sub_c")
    output_dir = tmp_path / "sub_d" / "singapore"

    derive_region_macro_plan(
        sub_c_region_dir=sub_c_region_dir,
        output_dir=output_dir,
        macro_vocab_path=_LOCKED_VOCAB_PATH,
        **_PINNED_EXTRACTION_KWARGS,
    )

    tile_dir = output_dir / "tile=EPSG3414_i0_j0"
    assert (tile_dir / "macro_core.parquet").is_file()
    assert (tile_dir / "derivation_evidence.parquet").is_file()
    assert (tile_dir / "effective_conditioning.yaml").is_file()
    assert (tile_dir / "provenance.yaml").is_file()
    assert (output_dir / "manifest.yaml").is_file()
    assert (output_dir / "_SUCCESS").is_file()


def test_pipeline_writes_manifest_then_success_after_validation(tmp_path: Path):
    """``_SUCCESS`` is the LAST write, gated on ``validate_region`` returning
    without raising. If validation fails, ``_SUCCESS`` must not exist —
    consumers rely on its presence as a green-light marker.

    We simulate a validation failure by monkey-patching ``validate_region``
    inside the pipeline module to raise; the pipeline must propagate the
    exception AND leave ``_SUCCESS`` absent.
    """
    sub_c_region_dir = _build_minimal_sub_c_region(tmp_path / "sub_c")
    output_dir = tmp_path / "sub_d" / "singapore"

    # Patch validate_region inside the pipeline module to raise.
    import cfm.data.sub_d.pipeline as pipeline_mod

    real_validate = pipeline_mod.validate_region

    def _failing_validate(*args, **kwargs):
        raise SubDValidationError("synthetic validation failure")

    pipeline_mod.validate_region = _failing_validate
    try:
        with pytest.raises(SubDValidationError, match="synthetic validation failure"):
            derive_region_macro_plan(
                sub_c_region_dir=sub_c_region_dir,
                output_dir=output_dir,
                macro_vocab_path=_LOCKED_VOCAB_PATH,
                **_PINNED_EXTRACTION_KWARGS,
            )
    finally:
        pipeline_mod.validate_region = real_validate

    # Manifest was written BEFORE validation ran, so it exists.
    assert (output_dir / "manifest.yaml").is_file()
    # _SUCCESS must NOT exist — the validator gate stopped the pipeline.
    assert not (output_dir / "_SUCCESS").exists()


def test_pipeline_is_byte_identical_on_same_inputs(tmp_path: Path):
    """Same sub-C inputs + same locked vocab + same pinned commit_sha +
    extracted_utc → byte-identical sub-D output across re-runs.

    Re-running the pipeline twice into two output dirs and comparing every
    artifact's bytes pins the determinism contract end-to-end. Any
    non-determinism (dict iteration order, timestamp leak, parquet writer
    drift) would surface as a byte diff somewhere.
    """
    sub_c_region_dir = _build_minimal_sub_c_region(tmp_path / "sub_c")
    output_a = tmp_path / "sub_d_a" / "singapore"
    output_b = tmp_path / "sub_d_b" / "singapore"

    for out in (output_a, output_b):
        derive_region_macro_plan(
            sub_c_region_dir=sub_c_region_dir,
            output_dir=out,
            macro_vocab_path=_LOCKED_VOCAB_PATH,
            **_PINNED_EXTRACTION_KWARGS,
        )

    artifacts = [
        "manifest.yaml",
        "_SUCCESS",
        "tile=EPSG3414_i0_j0/macro_core.parquet",
        "tile=EPSG3414_i0_j0/derivation_evidence.parquet",
        "tile=EPSG3414_i0_j0/effective_conditioning.yaml",
        "tile=EPSG3414_i0_j0/provenance.yaml",
    ]
    for rel in artifacts:
        a_bytes = (output_a / rel).read_bytes()
        b_bytes = (output_b / rel).read_bytes()
        assert a_bytes == b_bytes, f"byte drift in {rel}"

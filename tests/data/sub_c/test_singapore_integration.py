"""Layer 3 cached-Singapore integration tests (spec §13.3).

These tests run against sub-A's Singapore cache (2026-04-15.0 release).
They are marked ``slow`` and excluded from the default fast suite.

Tile selection rationale
------------------------
We scope the extraction to a rectangular sub-region in SVY21 that covers two
representative tiles:

  tile (15, 14) — Marina Bay area (coastal, high building density, mixed POI)
  tile (13, 17) — MacRitchie Reservoir area (inland, lower density, forested)

To avoid running the full 494-tile Singapore extraction inside CI, we clip the
Region's admin_polygon to a bounding box that contains both tiles:

  SVY21: x ∈ [26000, 32000],  y ∈ [28000, 36000]

This produces up to 12 tiles (a 3x4 grid inside the admin polygon), which
completes in < 60 s wall-clock on a MacBook Pro with pool_size=4.

Pre-condition: ``data/cache/overture/2026-04-15.0/singapore/`` must exist.
Run sub-A's load_region("singapore") first if the cache is absent (takes ~8 h).
"""

from __future__ import annotations

import types
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml
from pyproj import Transformer
from shapely.geometry import Polygon

from cfm.data.overture import load_region
from cfm.data.sub_c.pipeline import extract_region
from cfm.data.sub_c.validator_cross_tile import validate_extraction_cross_tile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_POLICY_YAML = _REPO_ROOT / "configs" / "data" / "missing_value_policy.yaml"
_VOCAB_YAML = _REPO_ROOT / "configs" / "tokenizer" / "vocab_phase1.yaml"

# Fixed timestamps so provenance SHA is stable across runs.
_EXTRACTED_UTC = "2026-05-18T00:00:00Z"
_COMMIT_SHA = "b86c509" + "0" * 33  # 40-char canonical SHA

# Sub-region bounding box in SVY21 (covers both target tiles plus neighbours).
# tile (15, 14): SVY21 x ∈ [30000, 32000], y ∈ [28000, 30000]
# tile (13, 17): SVY21 x ∈ [26000, 28000], y ∈ [34000, 36000]
_SUBREG_MIN_X = 26000
_SUBREG_MAX_X = 32000
_SUBREG_MIN_Y = 28000
_SUBREG_MAX_Y = 36000

# Canonical target tiles.
TILE_MARINA = (15, 14)
TILE_RESERVOIR = (13, 17)

# Expected schema version after Multi* enum extension (§14.9 bump from 1.0 → 1.1).
_EXPECTED_SUB_C_SCHEMA_VERSION = "1.1"


# ---------------------------------------------------------------------------
# Helper: build a scoped-down Region-like for the sub-region
# ---------------------------------------------------------------------------


def _build_sub_region(full_region):  # type: ignore[return]
    """Return a SimpleNamespace that looks like a Region but has a clipped
    admin_polygon covering only the sub-region bounding box.

    The themes are left full-Singapore — the orchestrator clips them to the
    admin polygon during ``_clip_features_to_admin``, so only features
    overlapping the sub-region box are processed.  This avoids touching the
    sub-A Region internals beyond ``.admin_polygon``.

    The sub-region box is in SVY21; we back-project to EPSG:4326 (which the
    Region exposes) and intersect with the real Singapore admin polygon so the
    result is valid geography (not a pure rectangle cut through the sea).
    """
    transformer_inv = Transformer.from_crs("EPSG:3414", "EPSG:4326", always_xy=True)

    corners_svy21 = [
        (_SUBREG_MIN_X, _SUBREG_MIN_Y),
        (_SUBREG_MAX_X, _SUBREG_MIN_Y),
        (_SUBREG_MAX_X, _SUBREG_MAX_Y),
        (_SUBREG_MIN_X, _SUBREG_MAX_Y),
    ]
    corners_4326 = [transformer_inv.transform(x, y) for x, y in corners_svy21]
    box_4326 = Polygon(corners_4326)

    # Intersect with the real admin polygon so the sub-region stays onshore.
    sub_admin_4326 = box_4326.intersection(full_region.admin_polygon)

    return types.SimpleNamespace(
        name=full_region.name,
        themes=full_region.themes,
        admin_polygon=sub_admin_4326,
    )


# ---------------------------------------------------------------------------
# Shared slow fixture: extract the sub-region once per session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def singapore_sub_region_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped fixture: extract the Singapore sub-region once.

    All four Layer 3 tests share this session-level extraction to avoid running
    the pipeline twice.  The fixture verifies the cache exists (fast load) and
    uses fixed timestamps for deterministic provenance SHAs.
    """
    full_region = load_region("singapore")
    sub_region = _build_sub_region(full_region)

    out = tmp_path_factory.mktemp("singapore_sub_region", numbered=False)
    extract_region(
        sub_region,
        out,
        policy_yaml_path=_POLICY_YAML,
        vocab_yaml_path=_VOCAB_YAML,
        release="2026-04-15.0",
        commit_sha=_COMMIT_SHA,
        extracted_utc=_EXTRACTED_UTC,
        started_utc=_EXTRACTED_UTC,
        rerun_reason="initial",
        pool_size=4,
    )
    return out


# ---------------------------------------------------------------------------
# Test 1: shape assertions on two specific tiles
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_singapore_two_tile_extraction_shape(singapore_sub_region_output: Path) -> None:
    """Extract a Singapore sub-region and assert SHAPE on two specific tiles.

    Tiles chosen:
      (15, 14) — Marina Bay area: coastal, mixed-use, high density.
      (13, 17) — MacRitchie Reservoir area: inland, forested, lower density.

    Shape assertions use loose bounds (existence + plausible ranges) rather than
    pinned counts — Overture release content changes between releases.  We
    verify:
      - The tile directory and all 5 required files exist.
      - cells.parquet has >=1 row and <=64 rows (max 8x8 cells per tile).
      - features.parquet has ≥ 1 row.
      - meta.yaml has the expected top-level keys.
    """
    out = singapore_sub_region_output

    for tile_i, tile_j in [TILE_MARINA, TILE_RESERVOIR]:
        tile_name = f"tile=EPSG3414_i{tile_i}_j{tile_j}"
        tile_dir = out / tile_name
        assert tile_dir.exists(), f"Expected tile dir missing: {tile_dir}"

        # All 5 required files must be present.
        for fname in (
            "cells.parquet",
            "features.parquet",
            "crossings.parquet",
            "meta.yaml",
            "provenance.yaml",
        ):
            assert (tile_dir / fname).exists(), f"{fname} missing in {tile_dir}"

        # cells.parquet: 1 <= rows <= 64  (8x8 grid; sea drops reduce the count)
        cells = pq.ParquetFile(tile_dir / "cells.parquet").read()
        assert 1 <= cells.num_rows <= 64, (
            f"tile {tile_name}: cells.parquet has unexpected row count {cells.num_rows}"
        )

        # features.parquet: ≥ 1 feature in a real-world non-empty tile
        features = pq.ParquetFile(tile_dir / "features.parquet").read()
        assert features.num_rows >= 1, (
            f"tile {tile_name}: features.parquet is empty — expected real-world features"
        )

        # meta.yaml structure check.
        meta = yaml.safe_load((tile_dir / "meta.yaml").read_text(encoding="utf-8"))
        for key in (
            "schema_version",
            "tile_i",
            "tile_j",
            "aggregates",
            "config",
            "conditioning_per_tile",
        ):
            assert key in meta, f"tile {tile_name}: meta.yaml missing key '{key}'"

        assert meta["tile_i"] == tile_i
        assert meta["tile_j"] == tile_j


# ---------------------------------------------------------------------------
# Test 2: re-extract → byte-identical parquet files (modulo excluded fields)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_singapore_tile_reextract_byte_identical_modulo_excluded_fields(
    tmp_path: Path,
) -> None:
    """Two independent extractions of the same sub-region produce byte-identical
    parquet files for the two target tiles.

    Per spec §14.5 determinism contract: byte output is invariant under identical
    inputs (same region data, same fixed timestamps, same policy/vocab yamls).
    We use ``pool_size=1`` for both runs to isolate the sequential determinism
    guarantee.

    We compare *parquet* bytes directly (parquet writer kwargs are pinned per
    §14.3).  YAML files (meta.yaml, provenance.yaml) are NOT compared byte-for-
    byte here — the cross-tile validator (Test 3) checks the integrity chain
    instead.  Timestamps in YAML fields are excluded from SHA per EXCLUDED_FROM_SHA
    (§14.6) but are included literally in the YAML bytes; since we pass a fixed
    ``extracted_utc`` both runs produce identical YAML bytes too.
    """
    full_region = load_region("singapore")
    sub_region = _build_sub_region(full_region)

    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"

    common_kwargs: dict = dict(
        policy_yaml_path=_POLICY_YAML,
        vocab_yaml_path=_VOCAB_YAML,
        release="2026-04-15.0",
        commit_sha=_COMMIT_SHA,
        extracted_utc=_EXTRACTED_UTC,
        started_utc=_EXTRACTED_UTC,
        rerun_reason="initial",
        pool_size=1,
    )

    extract_region(sub_region, run1, **common_kwargs)
    extract_region(sub_region, run2, **common_kwargs)

    # Compare parquet bytes for the two target tiles.
    for tile_i, tile_j in [TILE_MARINA, TILE_RESERVOIR]:
        tile_name = f"tile=EPSG3414_i{tile_i}_j{tile_j}"
        for fname in ("cells.parquet", "features.parquet", "crossings.parquet"):
            path1 = run1 / tile_name / fname
            path2 = run2 / tile_name / fname
            assert path1.exists(), f"run1 missing {tile_name}/{fname}"
            assert path2.exists(), f"run2 missing {tile_name}/{fname}"
            bytes1 = path1.read_bytes()
            bytes2 = path2.read_bytes()
            assert bytes1 == bytes2, (
                f"tile {tile_name}/{fname}: byte mismatch between run1 and run2 "
                f"(run1_size={len(bytes1)}, run2_size={len(bytes2)}). "
                "Determinism contract violated."
            )


# ---------------------------------------------------------------------------
# Test 3: cross-tile validator passes on the sub-region output
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_singapore_two_tile_cross_tile_validator_pass(
    singapore_sub_region_output: Path,
) -> None:
    """validate_extraction_cross_tile passes on the Singapore sub-region output.

    This exercises all 4 cross-tile invariants (spec §12.2):
      1. sub_c_schema_version_consistency — manifest version matches every tile's
         meta.yaml and provenance.yaml schema_version.
      2. manifest_tiles_match_filesystem — tile dirs on disk exactly match
         manifest.tiles[].
      3. manifest_provenance_sha_matches_disk — provenance SHA stored in manifest
         matches the SHA derived from the on-disk provenance.yaml content.
      4. provenance_outputs_sha_match_files — provenance.outputs.*_sha256 match
         the actual raw file bytes on disk.

    A TileValidationError from any invariant is a hard failure; per
    feedback_test_weakening_to_pass.md we do NOT catch or weaken the assertion.
    """
    # Raises TileValidationError on failure; no assertion needed.
    validate_extraction_cross_tile(singapore_sub_region_output)


# ---------------------------------------------------------------------------
# Test 4: manifest.sub_c_schema_version matches every tile's YAML versions
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_singapore_manifest_sub_c_schema_version_consistency(
    singapore_sub_region_output: Path,
) -> None:
    """manifest.sub_c_schema_version == "1.1" and matches every tile's YAML.

    After the Multi* GEOMETRY_TYPE enum extension (Task 17), the schema version
    is bumped from "1.0" to "1.1" per spec §14.9 (append-only enum change).

    We verify three things:
      (a) manifest.sub_c_schema_version == "1.1".
      (b) Every tile's meta.yaml.schema_version == "1.1".
      (c) Every tile's provenance.yaml.schema_version == "1.1".

    This is a stricter version of the cross-tile validator invariant #1 — we
    pin the expected version explicitly so that a future accidental version
    change will fail here before reaching the cross-tile validator.
    """
    out = singapore_sub_region_output
    manifest_path = out / "manifest.yaml"
    assert manifest_path.exists(), "manifest.yaml missing from sub-region output"

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    sub_c_ver = str(manifest.get("sub_c_schema_version", ""))

    assert sub_c_ver == _EXPECTED_SUB_C_SCHEMA_VERSION, (
        f"manifest.sub_c_schema_version={sub_c_ver!r}, expected {_EXPECTED_SUB_C_SCHEMA_VERSION!r}"
    )

    # Verify every tile dir in the manifest matches.
    for tile_entry in manifest.get("tiles", []):
        tile_i = tile_entry["tile_i"]
        tile_j = tile_entry["tile_j"]
        tile_name = f"tile=EPSG3414_i{tile_i}_j{tile_j}"
        tile_dir = out / tile_name

        meta = yaml.safe_load((tile_dir / "meta.yaml").read_text(encoding="utf-8"))
        assert str(meta.get("schema_version", "")) == _EXPECTED_SUB_C_SCHEMA_VERSION, (
            f"{tile_name}/meta.yaml.schema_version={meta.get('schema_version')!r}, "
            f"expected {_EXPECTED_SUB_C_SCHEMA_VERSION!r}"
        )

        prov = yaml.safe_load((tile_dir / "provenance.yaml").read_text(encoding="utf-8"))
        assert str(prov.get("schema_version", "")) == _EXPECTED_SUB_C_SCHEMA_VERSION, (
            f"{tile_name}/provenance.yaml.schema_version={prov.get('schema_version')!r}, "
            f"expected {_EXPECTED_SUB_C_SCHEMA_VERSION!r}"
        )

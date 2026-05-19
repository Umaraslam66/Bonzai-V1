"""Task 15 tests: torture-tile and cross-tile micro-fixture builders.

Named tests per plan Task 15:
- test_torture_tile_extraction_succeeds
- test_torture_tile_inline_validator_passes_on_clean_output
- test_torture_tile_contains_all_named_edge_case_tags (optional, per plan)
- test_cross_tile_micro_fixture_extracts_two_tiles

These tests verify the fixture builders produce well-formed synthetic regions
that the full extract_region pipeline can process successfully.  They exercise
the Layer 2 fixture path independently of Task 16's full torture-tile test suite
so that fixture regressions can be caught early.

Session-scoped extraction (spec §13.2 P5): the torture tile is extracted ONCE
per pytest session (``scope="session"``).  Subsequent tests share the output
directory.  Tests that require corruption should copy the session output to a
fresh tmp_path — that pattern is implemented in Task 16.
"""

from __future__ import annotations

from pathlib import Path

from cfm.data.sub_c.pipeline import extract_region
from cfm.data.sub_c.validator_inline import validate_tile_inline
from tests.fixtures.sub_c.build_cross_tile_fixture import (
    CROSS_TILE_J,
    CROSS_TILE_LEFT_I,
    CROSS_TILE_RIGHT_I,
    build_cross_tile_micro_region,
)
from tests.fixtures.sub_c.build_torture_tile import (
    TORTURE_TILE_I,
    TORTURE_TILE_J,
    TortureFeatureDef,
    torture_tile_features,
)

# ---------------------------------------------------------------------------
# Config paths (repo-root relative)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_POLICY_YAML = _REPO_ROOT / "configs" / "data" / "missing_value_policy.yaml"
_VOCAB_YAML = _REPO_ROOT / "configs" / "tokenizer" / "vocab_phase1.yaml"

# Fixed timestamps for byte-deterministic extraction in tests.
_EXTRACTED_UTC = "2026-05-18T00:00:00Z"
_COMMIT_SHA = "b86c509" + "0" * 33  # canonical 40-char sha


# ---------------------------------------------------------------------------
# Session-scoped fixture: torture_tile_output is defined in conftest.py
# (Task 16 promoted it from this module to conftest.py so other test modules
# in tests/data/sub_c/ can share the same one-per-session extraction).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# test_torture_tile_extraction_succeeds
# ---------------------------------------------------------------------------


def test_torture_tile_extraction_succeeds(torture_tile_output: Path) -> None:
    """extract_region on the torture tile must produce:
    - manifest.yaml at the output root
    - exactly 1 tile directory: tile=EPSG3414_i<TORTURE_TILE_I>_j<TORTURE_TILE_J>
    - all 5 per-tile artifacts present inside that directory

    Named test per plan Task 15.
    """
    out = torture_tile_output
    manifest_path = out / "manifest.yaml"
    assert manifest_path.exists(), "manifest.yaml must exist"

    import yaml

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    tiles = manifest.get("tiles", [])

    assert len(tiles) == 1, (
        f"torture-tile region must produce exactly 1 tile; got {len(tiles)}: {tiles}"
    )

    tile = tiles[0]
    assert tile["tile_i"] == TORTURE_TILE_I, (
        f"expected tile_i={TORTURE_TILE_I}, got {tile['tile_i']}"
    )
    assert tile["tile_j"] == TORTURE_TILE_J, (
        f"expected tile_j={TORTURE_TILE_J}, got {tile['tile_j']}"
    )

    tile_dir = out / f"tile=EPSG3414_i{TORTURE_TILE_I}_j{TORTURE_TILE_J}"
    assert tile_dir.is_dir(), f"tile directory must exist: {tile_dir}"

    for artifact in (
        "cells.parquet",
        "features.parquet",
        "crossings.parquet",
        "meta.yaml",
        "provenance.yaml",
    ):
        assert (tile_dir / artifact).exists(), f"artifact missing: {tile_dir / artifact}"

    # _SUCCESS must NOT be written by extract_region (Task 14's responsibility)
    assert not (out / "_SUCCESS").exists(), "_SUCCESS must not be written by extract_region"


# ---------------------------------------------------------------------------
# test_torture_tile_inline_validator_passes_on_clean_output
# ---------------------------------------------------------------------------


def test_torture_tile_inline_validator_passes_on_clean_output(
    torture_tile_output: Path,
) -> None:
    """validate_tile_inline on the torture tile output must not raise.

    This test verifies that the pipeline produces a tile whose parquet schemas,
    bbox derivations, water fractions, and cell-count aggregates all satisfy
    the 10 named invariants in validator_inline.py (spec §12.1).

    Named test per plan Task 15.
    """
    tile_dir = torture_tile_output / f"tile=EPSG3414_i{TORTURE_TILE_I}_j{TORTURE_TILE_J}"
    # If this raises TileValidationError, it is a pipeline bug (per
    # feedback_test_weakening_to_pass.md: STOP and escalate, do not weaken).
    validate_tile_inline(tile_dir)


# ---------------------------------------------------------------------------
# test_torture_tile_contains_all_named_edge_case_tags (optional)
# ---------------------------------------------------------------------------


def test_torture_tile_contains_all_named_edge_case_tags() -> None:
    """Assert that every TortureFeatureDef has a non-empty tag string and that
    all 12 expected feature IDs are present.

    Optional per plan Task 15; validates the declarative fixture itself rather
    than the pipeline output.
    """
    features = torture_tile_features()

    # Every feature must have a tag
    for f in features:
        assert isinstance(f, TortureFeatureDef), f"expected TortureFeatureDef, got {type(f)}"
        assert f.tag.strip(), f"feature {f.fid!r} has empty tag"
        assert f.fid.strip(), "feature has empty fid"

    fids = {f.fid for f in features}

    # All 12 named features from the spec §13.2 fixture design must be present
    expected_fids = {
        "F01_single_cell_road",
        "F02_multi_cell_road_3_cells",
        "F03_corner_crossing_road",
        "F04_polygon_interior_ring_crossing",
        "F05_colinear_entirety_road",
        "F06_touch_but_not_cross",
        "F07_partial_colinearity_polygon",
        "F08_zigzag_multi_crossing",
        "F09_inland_poi",
        "F10_coastal_poi",
        "F11_inland_river",
        "F12_sea_ocean_polygon",
    }
    missing = expected_fids - fids
    assert not missing, f"Missing feature IDs in torture_tile_features(): {missing}"

    # Tag coverage check — each known §8.3 edge case must appear in at least one tag
    all_tags = " ".join(f.tag for f in features)
    required_tag_substrings = [
        "single-cell",  # F01
        "multi-cell",  # F02
        "corner",  # F03
        "interior ring",  # F04
        "co-linear",  # F05
        "touch-but-not",  # F06
        "partial co-linearity",  # F07
        "alternating",  # F08
        "inland POI",  # F09
        "coastal POI",  # F10
        "river",  # F11
        "ocean",  # F12
    ]
    for substr in required_tag_substrings:
        assert substr in all_tags, (
            f"No feature tag covers {substr!r}; add a TortureFeatureDef with this edge case"
        )


# ---------------------------------------------------------------------------
# test_cross_tile_micro_fixture_extracts_two_tiles
# ---------------------------------------------------------------------------


def test_cross_tile_micro_fixture_extracts_two_tiles(tmp_path: Path) -> None:
    """extract_region on the cross-tile micro-fixture must produce exactly 2 tiles.

    Named test per plan Task 15.

    Verifies:
    - manifest contains exactly 2 tile entries
    - tile dirs are tile=EPSG3414_i<LEFT>_j<J> and tile=EPSG3414_i<RIGHT>_j<J>
    - all 5 per-tile artifacts present in each tile dir
    """
    out = tmp_path / "cross_tile"
    region = build_cross_tile_micro_region()
    manifest_obj = extract_region(
        region,
        out,
        policy_yaml_path=_POLICY_YAML,
        vocab_yaml_path=_VOCAB_YAML,
        release="2026-05-18.cross",
        commit_sha=_COMMIT_SHA,
        extracted_utc=_EXTRACTED_UTC,
        started_utc=_EXTRACTED_UTC,
        rerun_reason="initial",
        pool_size=1,
    )

    tiles = manifest_obj.tiles
    assert len(tiles) == 2, (
        f"cross-tile micro-fixture must produce exactly 2 tiles; got {len(tiles)}: {tiles}"
    )

    tile_ids = {(t["tile_i"], t["tile_j"]) for t in tiles}
    expected_ids = {
        (CROSS_TILE_LEFT_I, CROSS_TILE_J),
        (CROSS_TILE_RIGHT_I, CROSS_TILE_J),
    }
    assert tile_ids == expected_ids, f"expected tile IDs {expected_ids}, got {tile_ids}"

    for tile_i, tile_j in expected_ids:
        tile_dir = out / f"tile=EPSG3414_i{tile_i}_j{tile_j}"
        assert tile_dir.is_dir(), f"tile dir missing: {tile_dir}"
        for artifact in (
            "cells.parquet",
            "features.parquet",
            "crossings.parquet",
            "meta.yaml",
            "provenance.yaml",
        ):
            assert (tile_dir / artifact).exists(), f"artifact missing: {tile_dir / artifact}"

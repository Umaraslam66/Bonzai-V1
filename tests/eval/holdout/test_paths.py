from __future__ import annotations

from cfm.eval.holdout import paths


def test_tile_dirname_matches_sub_d_convention():
    # sub-D builds "tile=EPSG3414_i{i}_j{j}" (src/cfm/data/sub_d/pipeline.py:156).
    assert paths.tile_dirname(1, 7) == "tile=EPSG3414_i1_j7"
    assert paths.tile_dirname(9, 18) == "tile=EPSG3414_i9_j18"


def test_region_dirs_point_under_data_processed():
    rel, reg = "2026-04-15.0", "singapore"
    assert (
        paths.sub_c_region_dir(rel, reg)
        .as_posix()
        .endswith("data/processed/sub_c/2026-04-15.0/singapore")
    )
    assert (
        paths.sub_d_region_dir(rel, reg)
        .as_posix()
        .endswith("data/processed/sub_d/2026-04-15.0/singapore")
    )
    assert (
        paths.sub_f_region_dir(rel, reg)
        .as_posix()
        .endswith("data/processed/sub_f/2026-04-15.0/singapore")
    )


def test_holdout_partition_is_region_keyed():
    # spec §F: held-out tiles + derivatives live in a region-keyed holdout/ partition.
    p = paths.holdout_partition_dir("2026-04-15.0", "singapore")
    assert p.as_posix().endswith("data/processed/eval_set/2026-04-15.0/holdout/region=singapore")
    assert paths.holdout_manifest_path("2026-04-15.0").name == "holdout_manifest.yaml"


def test_default_release_and_region_constants():
    assert paths.DEFAULT_RELEASE == "2026-04-15.0"
    assert paths.DEFAULT_REGION == "singapore"

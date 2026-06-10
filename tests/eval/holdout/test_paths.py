from __future__ import annotations

import yaml

from cfm.eval.holdout import paths
from cfm.eval.holdout.paths import (
    _EU_HELD_OUT_CITIES,
    epsg_label_for_region,
    multiregion_eval_set_locked_marker,
    multiregion_holdout_manifest_path,
    tile_dirname,
)


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


def test_tile_dirname_uses_passed_label():
    assert tile_dirname(337, 2662, epsg_label="EPSG25832") == "tile=EPSG25832_i337_j2662"


def test_epsg_label_for_region_reads_region_config():
    # munich's region config has projected_crs: EPSG:25832 -> label EPSG25832
    assert epsg_label_for_region("munich") == "EPSG25832"
    assert epsg_label_for_region("krakow") == "EPSG25834"
    # Singapore round-trip: config has projected_crs: "EPSG:3414" (quoted in YAML);
    # guards both colon-strip and the de-SG generalization being behavior-preserving
    # against the legacy hardcoded _EPSG_LABEL value (spec §5).
    assert epsg_label_for_region("singapore") == "EPSG3414"


def test_multiregion_paths_distinct_from_sg():
    release = "2026-04-15.0"
    # multiregion manifest lives under .../multiregion/
    mr_manifest = multiregion_holdout_manifest_path(release)
    assert mr_manifest.as_posix().endswith(
        "eval_set/2026-04-15.0/multiregion/holdout_manifest.yaml"
    )
    # must be distinct from the frozen SG manifest
    sg_manifest = paths.holdout_manifest_path(release)
    assert mr_manifest != sg_manifest


def test_multiregion_eval_set_locked_marker():
    release = "2026-04-15.0"
    marker = multiregion_eval_set_locked_marker(release)
    assert marker.as_posix().endswith("eval_set/2026-04-15.0/multiregion/_EVAL_SET_LOCKED")
    # must be distinct from the frozen SG marker
    sg_marker = paths.eval_set_locked_marker(release)
    assert marker != sg_marker


def test_eu_held_out_cities_constant_matches_manifest():
    # DRIFT GUARD (finding M3 / Gate-6): the hardcoded `_EU_HELD_OUT_CITIES` selector
    # is a SECOND copy of the multiregion manifest's own `held_out_cities` field. The
    # copy is intentional (the selector must pick SG-vs-EU before opening any manifest),
    # but a hardcoded duplicate can DRIFT: if the manifest's held-out cities change and
    # this constant goes stale, a held-out city could leak into training and the
    # generalization test would silently break. Cross-reference the constant against the
    # manifest read DIRECTLY (the external source of truth), WITHOUT routing through any
    # paths.py helper in the assertion logic. GREEN on disk (both =
    # eisenhuttenstadt/glasgow/krakow/munich); RED the moment either side diverges.
    manifest = yaml.safe_load(
        multiregion_holdout_manifest_path("2026-04-15.0").read_text(encoding="utf-8")
    )
    assert _EU_HELD_OUT_CITIES == set(manifest["held_out_cities"])

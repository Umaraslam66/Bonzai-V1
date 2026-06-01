from __future__ import annotations

from pathlib import Path

import yaml

from cfm.data.sub_d.enums import MetricNamespace, Scope, SlotKind
from cfm.data.sub_d.io import (
    DerivationEvidenceRow,
    MacroCoreRow,
    write_derivation_evidence_parquet,
    write_macro_core_parquet,
)
from cfm.data.sub_d.macro_vocab import load_macro_vocab
from cfm.eval.holdout import labels, paths


def _write_synth_tile(tile_dir: Path) -> None:
    """A 2-cell synthetic tile: one dense cell (bucket 3), one sparse (bucket 0)."""
    tile_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        MacroCoreRow(
            SlotKind.CELL,
            0,
            0,
            0,
            None,
            None,
            None,
            Scope.ACTIVE,
            zoning_class=0,
            cell_density_bucket=3,
            road_skeleton_class=None,
        ),
        MacroCoreRow(
            SlotKind.CELL,
            1,
            0,
            1,
            None,
            None,
            None,
            Scope.ACTIVE,
            zoning_class=1,
            cell_density_bucket=0,
            road_skeleton_class=None,
        ),
        MacroCoreRow(
            SlotKind.INTERNAL_EDGE,
            0,
            None,
            None,
            0,
            0,
            0,
            Scope.ACTIVE,
            zoning_class=None,
            cell_density_bucket=None,
            road_skeleton_class=2,
        ),
    ]
    write_macro_core_parquet(rows, tile_dir / "macro_core.parquet")
    write_derivation_evidence_parquet(
        [
            DerivationEvidenceRow(
                SlotKind.TILE,
                0,
                MetricNamespace.TILE_POPULATION_DENSITY,
                "p75_building_footprint_ratio",
                0.22,
                "1.0",
            )
        ],
        tile_dir / "derivation_evidence.parquet",
    )
    (tile_dir / "effective_conditioning.yaml").write_text(
        yaml.safe_dump(
            {
                "effective_conditioning_schema_version": "1.0",
                "tile_i": 1,
                "tile_j": 7,
                "conditioning": {
                    "population_density_bucket": 2,
                    "morphology_class": "Asian-megacity",  # sub-C constant - UNSCORED
                    "coastal_inland_river": 1,
                    "admin_region": "Central Region",
                },
            }
        ),
        encoding="utf-8",
    )


def test_read_tile_labels_aggregates_cell_and_tile_signals(tmp_path: Path):
    tile_dir = tmp_path / paths.tile_dirname(1, 7)
    _write_synth_tile(tile_dir)
    tl = labels.read_tile_labels(tile_dir, tile_i=1, tile_j=7)

    assert tl.tile_i == 1 and tl.tile_j == 7
    assert tl.population_density_bucket == 2  # tile-level, from conditioning yaml
    # cell-granularity buckets, from macro_core CELL rows (the failure-mode stratum):
    assert sorted(tl.cell_density_buckets) == [0, 3]
    assert tl.coastal_inland_river == 1
    # morphology_stratum is the sub-D skeleton+zoning summary - NEVER sub-C morphology_class:
    assert tl.morphology_stratum.dominant_zoning_class in (0, 1)
    assert tl.morphology_stratum.modal_road_skeleton_class == 2


def test_sub_c_morphology_class_is_recorded_as_unscored_constant(tmp_path: Path):
    tile_dir = tmp_path / paths.tile_dirname(1, 7)
    _write_synth_tile(tile_dir)
    tl = labels.read_tile_labels(tile_dir, tile_i=1, tile_j=7)
    # The collision guard: the constant sub-C field is carried verbatim and flagged,
    # never promoted into a "scored" dimension.
    assert tl.sub_c_morphology_class == "Asian-megacity"
    assert "morphology_class" in labels.UNSCORED_V1_DIMENSIONS


def test_GATE6_cell_density_buckets_are_valid_vocab_ids():
    """Gate 6: hand-enumerate the macro vocab's cell_density token_ids from the YAML
    (ground truth) and assert read_tile_labels only ever yields those ids - the
    expected set is computed from the vocab, NOT from labels.py."""
    vocab = load_macro_vocab(paths.macro_vocab_path())
    expected_ids = {int(b["token_id"]) for b in vocab["locked_buckets"]["cell_density"]}
    assert expected_ids == {0, 1, 2, 3}
    assert labels.valid_cell_density_bucket_ids(paths.macro_vocab_path()) == expected_ids

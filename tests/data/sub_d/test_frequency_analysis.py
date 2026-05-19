"""Tests for sub-D frequency analysis artifacts (Task 6).

This is Layer-3 discipline applied to synthetic fixtures: the analysis must
be byte-deterministic, validate its own structure, record monotonic
marginal-cost-of-cut sequences, expose reviewer-facing proposal sections,
and pin a zoning/density orthogonality comparison.

Real-data subset selection and proposal generation lives in Task 7. Vocab
locking lives in Task 8. Task 6 is only the function/CLI shape.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pyarrow as pa
import pytest
import yaml
from shapely import wkb as shapely_wkb
from shapely.geometry import Polygon

from cfm.data.sub_d.errors import SubDValidationError
from cfm.data.sub_d.frequency_analysis import (
    build_frequency_analysis,
    validate_frequency_analysis,
    write_frequency_analysis,
)
from cfm.data.sub_d.sub_c_reader import SubCTileInputs, SubCTilePaths


def _wkb_square(side: float) -> bytes:
    polygon = Polygon([(0.0, 0.0), (side, 0.0), (side, side), (0.0, side)])
    return shapely_wkb.dumps(polygon, hex=False, byte_order=1)


def _make_tile_inputs(
    *,
    tile_i: int,
    tile_j: int,
    active_cells: list[tuple[int, int]],
    features_per_cell: dict[tuple[int, int], list[tuple[int, str, bytes]]],
    crossings_rows: list[tuple[str, int, int, int]],
    cell_area: float = 100.0,
) -> SubCTileInputs:
    """Hand-assemble a SubCTileInputs without going through disk.

    features_per_cell maps (cell_i, cell_j) -> list of (feature_class,
    source_feature_id, wkb_bytes). crossings_rows is a list of
    (source_feature_id, lower_cell_i, lower_cell_j, axis).
    """
    cells_table = pa.table(
        {
            "cell_i": [ci for ci, _ in active_cells],
            "cell_j": [cj for _, cj in active_cells],
            "cell_area_admin_clipped_m2": [cell_area] * len(active_cells),
        }
    )

    feat_i, feat_j, feat_cls, feat_sid, feat_geom = [], [], [], [], []
    for (ci, cj), rows in features_per_cell.items():
        for fc, sid, wkb in rows:
            feat_i.append(ci)
            feat_j.append(cj)
            feat_cls.append(fc)
            feat_sid.append(sid)
            feat_geom.append(wkb)
    features_table = pa.table(
        {
            "cell_i": feat_i,
            "cell_j": feat_j,
            "feature_class": feat_cls,
            "source_feature_id": feat_sid,
            "geometry": feat_geom,
        }
    )

    crossings_table = pa.table(
        {
            "source_feature_id": [r[0] for r in crossings_rows],
            "lower_cell_i": [r[1] for r in crossings_rows],
            "lower_cell_j": [r[2] for r in crossings_rows],
            "axis": [r[3] for r in crossings_rows],
        }
    )

    paths = SubCTilePaths(
        tile_i=tile_i,
        tile_j=tile_j,
        tile_dir=Path(f"/dev/null/tile_{tile_i}_{tile_j}"),
        cells=Path("/dev/null/cells.parquet"),
        features=Path("/dev/null/features.parquet"),
        crossings=Path("/dev/null/crossings.parquet"),
        meta=Path("/dev/null/meta.yaml"),
        provenance=Path("/dev/null/provenance.yaml"),
    )
    digests = {
        "cells_parquet_sha256": f"{tile_i:02x}{tile_j:02x}cells",
        "features_parquet_sha256": f"{tile_i:02x}{tile_j:02x}feats",
        "crossings_parquet_sha256": f"{tile_i:02x}{tile_j:02x}cross",
        "meta_yaml_sha256": f"{tile_i:02x}{tile_j:02x}meta_",
        "provenance_yaml_sha256": f"{tile_i:02x}{tile_j:02x}prov_",
    }
    return SubCTileInputs(
        paths=paths,
        cells=cells_table,
        features=features_table,
        crossings=crossings_table,
        meta={"schema_version": "1.0", "tile_i": tile_i, "tile_j": tile_j},
        provenance={"schema_version": "1.0", "tile_i": tile_i, "tile_j": tile_j},
        digests=digests,
    )


def _build_synthetic_inputs() -> list[SubCTileInputs]:
    """Two small tiles with varied zoning/density/road signals.

    Counts are picked so the marginal-cost-of-cut sequence has a non-trivial
    monotonic shape and so zoning vs density are not perfectly correlated.
    """
    big_building = _wkb_square(7.0)   # 49 m^2 in a 100 m^2 cell -> ratio 0.49
    small_building = _wkb_square(3.0)  # 9 m^2 -> ratio 0.09

    tile_a = _make_tile_inputs(
        tile_i=0,
        tile_j=0,
        active_cells=[(0, 0), (1, 1), (2, 2)],
        features_per_cell={
            (0, 0): [
                (0, "r1", b""),
                (0, "r2", b""),
                (0, "r3", b""),
                (1, "b1", big_building),
                (2, "p1", b""),
            ],
            (1, 1): [
                (0, "r4", b""),
                (1, "b2", small_building),
            ],
            (2, 2): [(1, "b3", small_building), (3, "base1", b"")],
        },
        crossings_rows=[
            ("r1", 0, 0, 0),
            ("r2", 0, 0, 0),
            ("r3", 1, 1, 1),
            ("r4", 2, 2, 0),
            # building-ring crossings — must not contribute to road skeleton.
            ("b1", 0, 0, 1),
            ("b2", 1, 1, 0),
        ],
    )
    tile_b = _make_tile_inputs(
        tile_i=0,
        tile_j=1,
        active_cells=[(0, 0), (4, 4)],
        features_per_cell={
            (0, 0): [(0, "r5", b""), (0, "r6", b"")],
            (4, 4): [(1, "b4", big_building), (2, "p2", b""), (2, "p3", b"")],
        },
        crossings_rows=[
            ("r5", 0, 0, 0),
            ("r6", 0, 0, 1),
        ],
    )
    return [tile_a, tile_b]


def test_frequency_analysis_output_is_byte_identical_for_same_inputs():
    inputs1 = _build_synthetic_inputs()
    inputs2 = _build_synthetic_inputs()
    yaml_a = yaml.safe_dump(build_frequency_analysis(inputs1), sort_keys=True)
    yaml_b = yaml.safe_dump(build_frequency_analysis(inputs2), sort_keys=True)
    assert yaml_a == yaml_b


def test_frequency_analysis_enforces_non_empty_locked_buckets():
    analysis = build_frequency_analysis(_build_synthetic_inputs())
    # Sanity: the unmodified analysis validates.
    validate_frequency_analysis(analysis)

    # Tamper: empty locked_buckets in the zoning proposal — must reject.
    tampered = copy.deepcopy(analysis)
    tampered["zoning_proposal"]["locked_buckets"] = []
    with pytest.raises(SubDValidationError, match="locked_buckets"):
        validate_frequency_analysis(tampered)

    # Same for cell_density.
    tampered = copy.deepcopy(analysis)
    tampered["cell_density_proposal"]["locked_buckets"] = []
    with pytest.raises(SubDValidationError, match="locked_buckets"):
        validate_frequency_analysis(tampered)

    # Same for road_skeleton.
    tampered = copy.deepcopy(analysis)
    tampered["road_skeleton_proposal"]["locked_buckets"] = []
    with pytest.raises(SubDValidationError, match="locked_buckets"):
        validate_frequency_analysis(tampered)


def test_frequency_analysis_records_marginal_cost_monotonicity():
    analysis = build_frequency_analysis(_build_synthetic_inputs())
    for section in ("zoning_proposal", "cell_density_proposal", "road_skeleton_proposal"):
        cs = analysis[section]["candidate_strategies"]
        assert isinstance(cs, list)
        assert len(cs) >= 2, f"{section} candidate_strategies too short: {cs}"
        # Entries are ordered from least-aggressive cut (most categories) to
        # most-aggressive cut (fewest categories). The first entry has
        # marginal_cost == None (no prior to compare to); the rest are
        # non-decreasing floats on synthetic data — each additional cut hurts
        # coverage more per category dropped. (Real bimodal data can violate
        # monotonicity; validate_frequency_analysis reports values without
        # enforcing this property.)
        assert cs[0]["marginal_cost"] is None
        prior = -1.0
        for entry in cs[1:]:
            assert isinstance(entry["marginal_cost"], float)
            assert entry["marginal_cost"] >= prior - 1e-9, (
                f"{section} candidate_strategies marginal_cost not monotonic: {cs}"
            )
            prior = entry["marginal_cost"]
        # Categories strictly decrease, coverage non-strictly decreases.
        cats = [e["categories"] for e in cs]
        covs = [e["coverage"] for e in cs]
        assert cats == sorted(cats, reverse=True)
        for i in range(1, len(covs)):
            assert covs[i] <= covs[i - 1] + 1e-9


def test_frequency_analysis_writes_reviewable_proposal_sections(tmp_path: Path):
    analysis = build_frequency_analysis(_build_synthetic_inputs())
    out_path = tmp_path / "macro_plan_proposal.yaml"
    write_frequency_analysis(analysis, out_path)

    # File is written and round-trips.
    loaded = yaml.safe_load(out_path.read_text(encoding="utf-8"))

    # Top-level reviewer-facing keys.
    for key in (
        "analysis_version",
        "derivation_version",
        "tile_count",
        "input_digests",
        "zoning_proposal",
        "cell_density_proposal",
        "road_skeleton_proposal",
        "zoning_orthogonality",
    ):
        assert key in loaded, f"missing reviewer-facing section: {key}"

    # Each proposal section exposes the locked_buckets + candidate_strategies
    # series so the reviewer can see what alternatives were considered.
    for section in ("zoning_proposal", "cell_density_proposal", "road_skeleton_proposal"):
        assert "locked_buckets" in loaded[section]
        assert "candidate_strategies" in loaded[section]
        assert isinstance(loaded[section]["locked_buckets"], list)
        assert len(loaded[section]["locked_buckets"]) >= 1
        # Each candidate entry carries its bucket definition so the reviewer
        # can hand-edit locked_buckets without consulting source code.
        for entry in loaded[section]["candidate_strategies"]:
            assert "strategy" in entry
            assert "categories" in entry
            assert "coverage" in entry
            assert "marginal_cost" in entry
            # Bucket-definition field varies by section but at least one of
            # the documented keys is present.
            assert any(
                k in entry
                for k in ("kept_tokens", "bucket_boundaries", "bucket_lower_bounds")
            ), f"{section} candidate entry missing bucket definition: {entry}"

    # Bytes are identical on a second write of the same dict.
    other = tmp_path / "macro_plan_proposal_2.yaml"
    write_frequency_analysis(analysis, other)
    assert out_path.read_bytes() == other.read_bytes()

    # input_digests sorted by (tile_i, tile_j) for determinism.
    entries = loaded["input_digests"]
    assert entries == sorted(entries, key=lambda e: (e["tile_i"], e["tile_j"]))


def test_frequency_analysis_records_zoning_orthogonality_comparison():
    analysis = build_frequency_analysis(_build_synthetic_inputs())
    assert "zoning_orthogonality" in analysis

    ortho = analysis["zoning_orthogonality"]
    # The comparison must be quantitative — at minimum a correlation and a
    # sample-size so the reviewer can judge whether zoning and density are
    # encoding redundant signals.
    assert "building_count_vs_density_ratio" in ortho
    comparison = ortho["building_count_vs_density_ratio"]
    assert "correlation" in comparison
    assert isinstance(comparison["correlation"], float)
    assert -1.0 - 1e-9 <= comparison["correlation"] <= 1.0 + 1e-9
    assert "sample_size" in comparison
    assert isinstance(comparison["sample_size"], int)
    assert comparison["sample_size"] >= 1

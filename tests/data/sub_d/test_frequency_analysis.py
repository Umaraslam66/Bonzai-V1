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

from cfm.data.determinism import compute_sha256
from cfm.data.sub_d.errors import SubDValidationError
from cfm.data.sub_d.frequency_analysis import (
    INDEX_ARTIFACT_FILENAME,
    NAMESPACE_ARTIFACT_FILENAMES,
    build_frequency_analysis,
    is_eligible_for_subset,
    select_layer3_subset,
    validate_frequency_analysis,
    validate_proposal_index,
    write_frequency_analysis,
    write_proposal_artifacts,
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
    coastal_inland_river: int = 0,
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
        meta={
            "schema_version": "1.0",
            "tile_i": tile_i,
            "tile_j": tile_j,
            "conditioning_per_tile": {
                "coastal_inland_river": coastal_inland_river,
            },
        },
        provenance={"schema_version": "1.0", "tile_i": tile_i, "tile_j": tile_j},
        digests=digests,
    )


def _build_synthetic_inputs() -> list[SubCTileInputs]:
    """Two small tiles with varied zoning/density/road signals.

    Counts are picked so the marginal-cost-of-cut sequence has a non-trivial
    monotonic shape and so zoning vs density are not perfectly correlated.
    """
    big_building = _wkb_square(7.0)  # 49 m^2 in a 100 m^2 cell -> ratio 0.49
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

    # tile_population_density carries TWO reviewer-editable handles:
    # locked_buckets AND locked_proxy. Both must be non-empty.
    tampered = copy.deepcopy(analysis)
    tampered["tile_population_density_proposal"]["locked_buckets"] = []
    with pytest.raises(SubDValidationError, match="locked_buckets"):
        validate_frequency_analysis(tampered)

    tampered = copy.deepcopy(analysis)
    tampered["tile_population_density_proposal"]["locked_proxy"] = None
    with pytest.raises(SubDValidationError, match="locked_proxy"):
        validate_frequency_analysis(tampered)

    # Empty candidate_proxies must be rejected: Layer 1 emits at least one
    # proxy on real data, and a missing list would silently break the
    # reviewer's proxy comparison at Gate 2.
    tampered = copy.deepcopy(analysis)
    tampered["tile_population_density_proposal"]["candidate_proxies"] = []
    with pytest.raises(SubDValidationError, match="candidate_proxies"):
        validate_frequency_analysis(tampered)

    # A proxy whose candidate_strategies is empty must also be rejected.
    tampered = copy.deepcopy(analysis)
    tampered["tile_population_density_proposal"]["candidate_proxies"][0][
        "candidate_strategies"
    ] = []
    with pytest.raises(SubDValidationError, match="candidate_strategies"):
        validate_frequency_analysis(tampered)

    # per_tile_evidence is required: select_layer3_subset depends on it and
    # silently returns [] if it's missing. Validation must catch that drop
    # as a contract violation rather than letting the silent empty subset
    # propagate to Gate 2.
    tampered = copy.deepcopy(analysis)
    del tampered["per_tile_evidence"]
    with pytest.raises(SubDValidationError, match="per_tile_evidence"):
        validate_frequency_analysis(tampered)


def _candidate_strategies_for_section(analysis: dict, section: str) -> list[list[dict]]:
    """Return list-of-strategy-lists for *section*.

    For flat sections (zoning, cell_density, road_skeleton) returns a
    single-element list. For tile_population_density (which has one
    candidate_strategies list per proxy under candidate_proxies[]) returns
    one entry per proxy.
    """
    if section == "tile_population_density_proposal":
        return [
            list(proxy["candidate_strategies"]) for proxy in analysis[section]["candidate_proxies"]
        ]
    return [list(analysis[section]["candidate_strategies"])]


def test_frequency_analysis_records_marginal_cost_monotonicity():
    analysis = build_frequency_analysis(_build_synthetic_inputs())
    for section in (
        "zoning_proposal",
        "cell_density_proposal",
        "tile_population_density_proposal",
        "road_skeleton_proposal",
    ):
        # tile_population_density has one strategies list per proxy; check
        # each independently. Other sections expose a single flat list.
        for cs in _candidate_strategies_for_section(analysis, section):
            assert isinstance(cs, list)
            assert len(cs) >= 2, f"{section} candidate_strategies too short: {cs}"
            # Entries are ordered from least-aggressive cut (most categories)
            # to most-aggressive cut. The first entry has marginal_cost==None
            # (no prior to compare to); the rest are non-decreasing floats on
            # synthetic data. Real bimodal data can violate monotonicity;
            # validate_frequency_analysis reports values without enforcing.
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
    out_path = tmp_path / "frequency_analysis.yaml"
    write_frequency_analysis(analysis, out_path)

    # File is written and round-trips.
    loaded = yaml.safe_load(out_path.read_text(encoding="utf-8"))

    # Top-level reviewer-facing keys. ``derivation_versions`` is the
    # per-namespace dict (4 entries) — never a bare global string.
    for key in (
        "analysis_version",
        "derivation_versions",
        "tile_count",
        "input_digests",
        "zoning_proposal",
        "cell_density_proposal",
        "tile_population_density_proposal",
        "road_skeleton_proposal",
        "zoning_orthogonality",
    ):
        assert key in loaded, f"missing reviewer-facing section: {key}"
    # The four derivation versions are present and stamped per namespace.
    assert set(loaded["derivation_versions"].keys()) == {
        "zoning",
        "cell_density",
        "tile_population_density",
        "road_skeleton",
    }

    # Flat-section sanity: locked_buckets + a single candidate_strategies
    # list at the section's top level. (tile_population_density is handled
    # separately below because it carries strategies *per proxy*.)
    for section in (
        "zoning_proposal",
        "cell_density_proposal",
        "road_skeleton_proposal",
    ):
        assert "locked_buckets" in loaded[section]
        assert "candidate_strategies" in loaded[section]
        assert isinstance(loaded[section]["locked_buckets"], list)
        assert len(loaded[section]["locked_buckets"]) >= 1
        for entry in loaded[section]["candidate_strategies"]:
            assert "strategy" in entry
            assert "categories" in entry
            assert "coverage" in entry
            assert "marginal_cost" in entry
            assert any(
                k in entry for k in ("kept_tokens", "bucket_boundaries", "bucket_lower_bounds")
            ), f"{section} candidate entry missing bucket definition: {entry}"

    # tile_population_density carries candidate_proxies[] (each with its own
    # candidate_strategies) plus locked_proxy + locked_buckets at the section
    # level. There is NO top-level candidate_strategies mirror — consumers
    # look up the locked proxy in candidate_proxies[].
    tpd = loaded["tile_population_density_proposal"]
    assert tpd.get("locked_proxy")
    assert "locked_buckets" in tpd and len(tpd["locked_buckets"]) >= 1
    assert "candidate_proxies" in tpd and len(tpd["candidate_proxies"]) >= 1
    assert "candidate_strategies" not in tpd, (
        "tile_population_density_proposal must not carry a top-level "
        "candidate_strategies mirror; strategies live under candidate_proxies[]"
    )
    for proxy_entry in tpd["candidate_proxies"]:
        assert proxy_entry.get("proxy_name")
        assert "candidate_strategies" in proxy_entry
        assert len(proxy_entry["candidate_strategies"]) >= 1

    # Bytes are identical on a second write of the same dict.
    other = tmp_path / "frequency_analysis_2.yaml"
    write_frequency_analysis(analysis, other)
    assert out_path.read_bytes() == other.read_bytes()

    # input_digests sorted by (tile_i, tile_j) for determinism.
    entries = loaded["input_digests"]
    assert entries == sorted(entries, key=lambda e: (e["tile_i"], e["tile_j"]))


def _build_inputs_for_subset_distinguishing() -> list[SubCTileInputs]:
    """Three tiles where evidence-based and fastest-only policies diverge.

    Tile (0, 0) is empty — would be picked first by a fastest-only / random
    policy but has no evidence to justify inclusion. Tile (1, 0) is mid.
    Tile (5, 5) is the rich tile carrying all five dimension signals.
    A rationale-driven selector with max_tiles=2 must include (5, 5) and
    must NOT pick (0, 0).
    """
    big_building = _wkb_square(8.0)  # 64 m^2 -> ratio 0.64 in 100 m^2 cell

    # Empty tile is set to coastal_inland_river=2 (riverside) on purpose:
    # a pure dimension-ranking selector would top "riverside_present" on
    # this tile (active_cell_count=0 doesn't disqualify under score-only
    # logic). The explicit eligibility predicate is what keeps it out of
    # the Layer-3 subset.
    empty_tile = _make_tile_inputs(
        tile_i=0,
        tile_j=0,
        active_cells=[],
        features_per_cell={},
        crossings_rows=[],
        coastal_inland_river=2,
    )
    mid_tile = _make_tile_inputs(
        tile_i=1,
        tile_j=0,
        active_cells=[(0, 0)],
        features_per_cell={(0, 0): [(0, "r_mid", b"")]},
        crossings_rows=[("r_mid", 0, 0, 0)],
        coastal_inland_river=0,  # inland
    )
    rich_tile = _make_tile_inputs(
        tile_i=5,
        tile_j=5,
        active_cells=[(0, 0), (1, 1), (2, 2)],
        features_per_cell={
            (0, 0): [
                (0, "r_a", b""),
                (0, "r_b", b""),
                (1, "b_a", big_building),
                (2, "p_a", b""),
            ],
            (1, 1): [(0, "r_c", b""), (1, "b_b", big_building)],
            (2, 2): [(3, "base_a", b"")],
        },
        crossings_rows=[
            ("r_a", 0, 0, 0),
            ("r_b", 0, 0, 1),
            ("r_c", 1, 1, 0),
        ],
        coastal_inland_river=1,  # coastal
    )
    return [empty_tile, mid_tile, rich_tile]


def test_subset_selection_is_deterministic_for_same_analysis():
    analysis_a = build_frequency_analysis(_build_synthetic_inputs())
    analysis_b = build_frequency_analysis(_build_synthetic_inputs())
    first = select_layer3_subset(analysis_a)
    second = select_layer3_subset(analysis_b)
    assert first == second
    # Repeated call on the same analysis dict also returns identical output.
    third = select_layer3_subset(analysis_a)
    assert first == third


def test_subset_selection_rejects_random_or_fastest_only_policy():
    inputs = _build_inputs_for_subset_distinguishing()
    analysis = build_frequency_analysis(inputs)
    per_tile = {(e["tile_i"], e["tile_j"]): e for e in analysis["per_tile_evidence"]}

    # Fixture invariants the test relies on:
    # - (0, 0) has zero active cells AND coastal_inland_river=2 (riverside),
    #   so a pure dimension-ranking policy would top "riverside_present" on
    #   this tile.
    # - (5, 5) carries every evidence dimension's positive signal.
    assert per_tile[(0, 0)]["active_cell_count"] == 0
    assert per_tile[(0, 0)]["coastal_inland_river"] == 2
    assert per_tile[(5, 5)]["active_cell_count"] >= 1

    # The explicit eligibility predicate is the layer of defense that rejects
    # (0, 0). Assert the predicate directly so a future refactor cannot move
    # the rejection back to score-based logic without this test failing.
    assert is_eligible_for_subset(per_tile[(0, 0)]) is False
    assert is_eligible_for_subset(per_tile[(5, 5)]) is True

    subset = select_layer3_subset(analysis, max_tiles=3)
    selected_keys = {(e["tile_i"], e["tile_j"]) for e in subset}
    # The rich tile (5, 5) MUST appear — a random or fastest-only policy
    # would never reach it before picking the empty tile (0, 0).
    assert (5, 5) in selected_keys
    # The empty tile (0, 0) MUST NOT appear: rejected by the eligibility
    # predicate, not by absent dimension favor (which it actually had).
    assert (0, 0) not in selected_keys, (
        "active_cell_count=0 tile must be rejected by eligibility predicate "
        "even when a meta dimension (riverside_present) would top it under "
        "score-only ranking"
    )


def test_subset_selection_records_rationale_per_tile():
    analysis = build_frequency_analysis(_build_inputs_for_subset_distinguishing())
    subset = select_layer3_subset(analysis, max_tiles=12)
    assert subset, "subset selection produced no tiles"
    # Every selected tile carries a non-empty rationale that names at least
    # one covered dimension from the documented set.
    allowed_dimension_keywords = {
        "zoning",
        "density",
        "road",
        "scope",
        "coastal",
        "inland",
        "riverside",
    }
    for entry in subset:
        assert "tile_i" in entry and "tile_j" in entry
        rationale = entry["rationale"]
        assert isinstance(rationale, str) and rationale.strip()
        # Rationale mentions one or more covered dimensions.
        hits = [w for w in allowed_dimension_keywords if w in rationale.lower()]
        assert hits, (
            f"rationale for tile ({entry['tile_i']},{entry['tile_j']}) "
            f"does not mention any documented dimension: {rationale!r}"
        )
    # Output is sorted by (tile_i, tile_j) for byte-determinism.
    keys = [(e["tile_i"], e["tile_j"]) for e in subset]
    assert keys == sorted(keys)


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


def test_proposal_artifacts_split_into_four_namespace_files_plus_index(tmp_path: Path):
    """The Gate 2 layout: 4 namespace files + 1 index file.

    Namespace files are content-pinned via sha256 references in the index;
    the reviewer-editable ``locked_buckets`` / ``locked_proxy`` live in the
    index, not in the namespace files. Task 8's promote-script flips
    ``status`` on the index file only.
    """
    analysis = build_frequency_analysis(_build_synthetic_inputs())
    layer3_subset = select_layer3_subset(analysis, max_tiles=3)

    index = write_proposal_artifacts(
        analysis, tmp_path, layer3_subset=layer3_subset, status="proposal"
    )

    # Five files on disk: 4 namespace + 1 index.
    expected_files = set(NAMESPACE_ARTIFACT_FILENAMES.values()) | {INDEX_ARTIFACT_FILENAME}
    actual_files = {p.name for p in tmp_path.iterdir() if p.is_file()}
    assert expected_files == actual_files

    # Index file shape.
    validate_proposal_index(index)
    assert index["status"] == "proposal"
    assert "selected_layer3_tiles" in index
    assert index["selected_layer3_tiles"] == layer3_subset

    # locked_buckets lives in the index keyed by namespace; locked_proxy
    # is set for tile_population_density.
    for namespace in (
        "zoning",
        "cell_density",
        "tile_population_density",
        "road_skeleton",
    ):
        assert namespace in index["locked_buckets"]
        assert len(index["locked_buckets"][namespace]) >= 1
    assert index["locked_proxy"]["tile_population_density"]

    # Each namespace_files entry has the actual file's sha256.
    for record in index["namespace_files"]:
        on_disk = (tmp_path / record["filename"]).read_bytes()
        assert record["sha256"] == compute_sha256(on_disk), (
            f"sha mismatch for {record['filename']!r}: index pin diverges from disk bytes"
        )

    # Namespace files MUST NOT contain locked_buckets / locked_proxy — those
    # reviewer-editable fields live exclusively in the index, otherwise an
    # edit there would invalidate the namespace digest without anyone noticing.
    for section_key, filename in NAMESPACE_ARTIFACT_FILENAMES.items():
        loaded = yaml.safe_load((tmp_path / filename).read_text(encoding="utf-8"))
        assert "locked_buckets" not in loaded
        assert "locked_proxy" not in loaded
        # Every namespace file carries its own derivation_version stamp +
        # the namespace label so an isolated file is self-describing.
        assert "derivation_version" in loaded
        assert "namespace" in loaded
        # Strategy storage shape varies by namespace:
        # - flat sections (zoning, cell_density, road_skeleton) carry a
        #   single top-level candidate_strategies list.
        # - tile_population_density carries candidate_proxies[] with
        #   strategies nested per proxy (no top-level mirror).
        if section_key == "tile_population_density_proposal":
            assert "candidate_strategies" not in loaded, (
                "tile_population_density namespace file must not carry a "
                "top-level candidate_strategies mirror"
            )
            assert "candidate_proxies" in loaded
            assert len(loaded["candidate_proxies"]) >= 1
            for proxy_entry in loaded["candidate_proxies"]:
                assert "candidate_strategies" in proxy_entry
                assert "proxy_name" in proxy_entry
        else:
            assert "candidate_strategies" in loaded

    # Determinism: a second call with the same analysis writes byte-identical
    # files. Use a separate dir to avoid mutating the first run's output.
    out_b = tmp_path / "run_b"
    write_proposal_artifacts(analysis, out_b, layer3_subset=layer3_subset, status="proposal")
    for filename in expected_files:
        assert (tmp_path / filename).read_bytes() == (out_b / filename).read_bytes(), (
            f"{filename} not byte-identical across runs"
        )

    # Locked status flip-by-string: write a locked variant; only the
    # ``status:`` line of the index differs. This is what Task 8's
    # promote-script will rely on for its byte-identity-modulo-status test.
    out_locked = tmp_path / "run_locked"
    write_proposal_artifacts(analysis, out_locked, layer3_subset=layer3_subset, status="locked")
    # Namespace files unchanged across the proposal/locked variants.
    for filename in NAMESPACE_ARTIFACT_FILENAMES.values():
        assert (tmp_path / filename).read_bytes() == (out_locked / filename).read_bytes()
    # Index bytes differ only on the status line.
    proposal_bytes = (tmp_path / INDEX_ARTIFACT_FILENAME).read_text(encoding="utf-8")
    locked_bytes = (out_locked / INDEX_ARTIFACT_FILENAME).read_text(encoding="utf-8")
    assert proposal_bytes.replace("status: proposal", "status: locked", 1) == locked_bytes

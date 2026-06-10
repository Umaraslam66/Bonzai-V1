"""Pure-logic tests for the Task-9 conditioning-discrimination gate (input (i)).

Mostly pure-logic: synthetic per-(city, stratum, metric) feature lists. Since
readiness Task 21, the IO function ``extract_features_by_city_stratum_metric`` IS
exercised locally too — against a synthetic monkeypatched-reader tile tree (the
coverage-counter + CRS-guard tests below); only the real-corpus run stays Leonardo.

The 6 TDD teeth (design note §"TDD teeth") plus unit tests for the pure stats
helpers (``noise_floor``, ``ks_pvalue``, ``benjamini_hochberg``).
"""

from __future__ import annotations

import dataclasses
import math
import random
from types import SimpleNamespace

import pytest
import yaml

import cfm.eval.conditioning_discrimination as CD
from cfm.eval.conditioning_discrimination import (
    benjamini_hochberg,
    conditioning_discrimination_verdict,
    ks_pvalue,
    noise_floor,
)
from cfm.eval.realism import FeatureMetric, ks_distance

_BUILDING = FeatureMetric.BUILDING_AREA.value
_ROAD = FeatureMetric.ROAD_LENGTH.value


def _gauss(rng: random.Random, *, mu: float, sigma: float, n: int) -> list[float]:
    return [rng.gauss(mu, sigma) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Pure stats helpers
# --------------------------------------------------------------------------- #


def test_noise_floor_matches_ks_critical_value_formula() -> None:
    # 1.36 * sqrt((n1+n2)/(n1*n2)) — the alpha=0.05 two-sample KS critical value.
    assert noise_floor(50, 50) == 1.36 * math.sqrt(100 / 2500)
    assert noise_floor(100, 200) == 1.36 * math.sqrt(300 / 20000)


def test_ks_pvalue_identical_samples_is_one() -> None:
    # D == 0 -> p == 1.0 (no evidence of difference).
    assert ks_pvalue(0.0, 50, 50) == 1.0


def test_ks_pvalue_disjoint_samples_is_near_zero() -> None:
    # D == 1 with large n -> p ~ 0 (overwhelming evidence of difference).
    p = ks_pvalue(1.0, 500, 500)
    assert p < 1e-6


def test_ks_pvalue_in_unit_interval() -> None:
    for d in (0.0, 0.1, 0.3, 0.5, 0.8, 1.0):
        p = ks_pvalue(d, 80, 120)
        assert 0.0 <= p <= 1.0


def test_benjamini_hochberg_hand_computed_example() -> None:
    # m=4, sorted p = [0.005, 0.01, 0.03, 0.5].
    # adj_(4)=0.5, adj_(3)=min(0.5, 0.03*4/3=0.04)=0.04,
    # adj_(2)=min(0.04, 0.01*4/2=0.02)=0.02, adj_(1)=min(0.02, 0.005*4/1=0.02)=0.02.
    # Input order [0.01, 0.5, 0.005, 0.03] -> [0.02, 0.5, 0.02, 0.04].
    out = benjamini_hochberg([0.01, 0.5, 0.005, 0.03])
    assert out == [0.02, 0.5, 0.02, 0.04]


def test_benjamini_hochberg_preserves_input_order_and_monotone() -> None:
    out = benjamini_hochberg([0.5, 0.005, 0.01, 0.03])
    assert out == [0.5, 0.02, 0.02, 0.04]


# --------------------------------------------------------------------------- #
# Verdict — the 6 TDD teeth
# --------------------------------------------------------------------------- #


def test_tooth1_pass_same_distribution_across_cities() -> None:
    """Same distribution, same stratum, across cities -> all D <= floor -> PASS."""
    rng = random.Random(7)
    n = 200
    features: dict[tuple[str, tuple, str], list[float]] = {}
    for stratum in ((1, 1, 0, 0), (2, 1, 1, 0)):
        for city in ("a", "b", "c"):
            features[(city, stratum, _BUILDING)] = _gauss(rng, mu=10.0, sigma=2.0, n=n)
    result = conditioning_discrimination_verdict(features, min_n=50, effect_size_floor=0.15)
    assert result.verdict == "PASS"
    assert result.n_qualifying_comparisons > 0
    assert all(not p.significant for p in result.pairs)


def test_tooth2_fail_genuine_difference_survives_bh() -> None:
    """One stratum, two cities from clearly different distributions -> FAIL."""
    rng = random.Random(7)
    n = 300
    stratum = (1, 1, 0, 0)
    features = {
        ("a", stratum, _BUILDING): _gauss(rng, mu=0.0, sigma=1.0, n=n),
        ("b", stratum, _BUILDING): _gauss(rng, mu=100.0, sigma=1.0, n=n),
    }
    result = conditioning_discrimination_verdict(features, min_n=50, effect_size_floor=0.15)
    assert result.verdict == "FAIL"
    assert result.per_metric_verdict[_BUILDING] == "FAIL"
    assert any(p.significant for p in result.pairs)


def test_tooth3_mc_guard_noise_tail_does_not_reopen_t5() -> None:
    """~40+ same-distribution pairs: BH suppresses chance raw exceedances -> PASS.

    Non-vacuous guard: at least one pair has raw ks > floor (a raw exceedance that
    WOULD have fired the un-corrected HALT), yet none is BH-significant -> PASS.
    """
    rng = random.Random(7)
    n = 60
    features: dict[tuple[str, tuple, str], list[float]] = {}
    # 14 strata x C(3,2)=3 pairs = 42 same-distribution comparisons.
    for s in range(14):
        stratum = (s, 0, 0, 0)
        for city in ("a", "b", "c"):
            features[(city, stratum, _BUILDING)] = _gauss(rng, mu=5.0, sigma=1.0, n=n)
    result = conditioning_discrimination_verdict(features, min_n=50, effect_size_floor=0.15)

    # The guard must hold: a noise-tail outlier does NOT reopen T5.
    assert result.verdict == "PASS"
    assert all(not p.significant for p in result.pairs)

    # Non-vacuity: prove a RAW exceedance existed (ks > floor) that BH suppressed.
    # If none exists, this test is vacuous and must fail.
    raw_exceedances = [p for p in result.pairs if p.ks > p.floor]
    assert raw_exceedances, "MC-guard test is vacuous: no raw ks>floor exceedance to suppress"
    # And confirm BH actually neutralised them (none is significant).
    assert all(not p.significant for p in raw_exceedances)


def test_tile_features_promotes_building_rings_to_area() -> None:
    """A building (closed-ring LineString by decoder contract) must become building_area,
    NOT road_length. RED-ON-DIVERGENCE: drop ``promote_building_rings`` and the closed
    ring stays LineString -> miscounted as a road -> building_area empty (the munich
    building_area=0 bug caught by the Leonardo sanity-extract 2026-06-10)."""
    from cfm.eval.conditioning_discrimination import _tile_features
    from cfm.eval.emergence import building_token_ids

    bid = min(building_token_ids())
    closed_ring = {"type": "LineString", "coordinates": [[0, 0], [1, 0], [1, 1], [0, 0]]}
    open_road = {"type": "LineString", "coordinates": [[0, 0], [2, 0]]}
    blocks = [[bid], [0]]  # block 0 carries a building token; block 1 does not
    out, n_bref_excluded = _tile_features(blocks, [closed_ring, open_road], [1, 1])

    areas = [v for m, v, _ in out if m == "building_area_m2"]
    lengths = [v for m, v, _ in out if m == "road_length_m"]
    assert len(areas) == 1 and areas[0] > 0, "building closed-ring not promoted to area"
    assert len(lengths) == 1 and lengths[0] > 0, "open road not classified as road_length"
    assert n_bref_excluded == 0  # no outbound-bref block in this fixture


def test_tooth4_thin_n_excluded_and_counted() -> None:
    """A (city, stratum, metric) below min_n is excluded, counted, in no pair."""
    rng = random.Random(7)
    stratum = (1, 1, 0, 0)
    features = {
        ("a", stratum, _BUILDING): _gauss(rng, mu=10.0, sigma=2.0, n=200),
        ("b", stratum, _BUILDING): _gauss(rng, mu=10.0, sigma=2.0, n=200),
        # thin: below min_n=50, excluded from the qualified set.
        ("c", stratum, _BUILDING): _gauss(rng, mu=10.0, sigma=2.0, n=10),
    }
    result = conditioning_discrimination_verdict(features, min_n=50, effect_size_floor=0.15)
    assert result.n_excluded_thin == 1
    # Only a<->b qualifies; c never appears in any pair.
    cities_in_pairs = {p.city_a for p in result.pairs} | {p.city_b for p in result.pairs}
    assert "c" not in cities_in_pairs
    assert result.n_qualifying_comparisons == 1


def test_tooth5_unsupported_when_all_strata_thin() -> None:
    """All cells thin -> verdict UNSUPPORTED, 0 qualifying comparisons (not PASS)."""
    rng = random.Random(7)
    stratum = (1, 1, 0, 0)
    features = {
        ("a", stratum, _BUILDING): _gauss(rng, mu=10.0, sigma=2.0, n=10),
        ("b", stratum, _BUILDING): _gauss(rng, mu=10.0, sigma=2.0, n=20),
    }
    result = conditioning_discrimination_verdict(features, min_n=50, effect_size_floor=0.15)
    assert result.verdict == "UNSUPPORTED"
    assert result.n_qualifying_comparisons == 0
    assert result.n_excluded_thin == 2


def test_tooth6_per_metric_road_fails_building_passes() -> None:
    """Identical building_area, clearly different road_length -> overall FAIL,
    road FAIL, building PASS."""
    rng = random.Random(7)
    n = 300
    stratum = (1, 1, 0, 0)
    features = {
        # building_area: same distribution -> PASS
        ("a", stratum, _BUILDING): _gauss(rng, mu=10.0, sigma=2.0, n=n),
        ("b", stratum, _BUILDING): _gauss(rng, mu=10.0, sigma=2.0, n=n),
        # road_length: clearly different -> FAIL
        ("a", stratum, _ROAD): _gauss(rng, mu=0.0, sigma=1.0, n=n),
        ("b", stratum, _ROAD): _gauss(rng, mu=100.0, sigma=1.0, n=n),
    }
    result = conditioning_discrimination_verdict(features, min_n=50, effect_size_floor=0.15)
    assert result.verdict == "FAIL"
    assert result.per_metric_verdict[_ROAD] == "FAIL"
    assert result.per_metric_verdict[_BUILDING] == "PASS"


def test_strata_too_few_cities_counted() -> None:
    """A stratum with only one qualified city contributes no pair and is counted."""
    rng = random.Random(7)
    features = {
        # stratum X: only one qualified city -> too few
        ("a", (9, 0, 0, 0), _BUILDING): _gauss(rng, mu=10.0, sigma=2.0, n=200),
        # stratum Y: two qualified cities -> a pair
        ("a", (1, 1, 0, 0), _BUILDING): _gauss(rng, mu=10.0, sigma=2.0, n=200),
        ("b", (1, 1, 0, 0), _BUILDING): _gauss(rng, mu=10.0, sigma=2.0, n=200),
    }
    result = conditioning_discrimination_verdict(features, min_n=50, effect_size_floor=0.15)
    assert result.n_strata_too_few_cities == 1
    assert result.n_qualifying_comparisons == 1


def test_n_by_city_stratum_metric_reports_every_n() -> None:
    """Every (city, stratum, metric) key's n is recorded, thin or not."""
    rng = random.Random(7)
    stratum = (1, 1, 0, 0)
    features = {
        ("a", stratum, _BUILDING): _gauss(rng, mu=10.0, sigma=2.0, n=200),
        ("b", stratum, _BUILDING): _gauss(rng, mu=10.0, sigma=2.0, n=7),
    }
    result = conditioning_discrimination_verdict(features, min_n=50, effect_size_floor=0.15)
    assert result.n_by_city_stratum_metric[("a", stratum, _BUILDING)] == 200
    assert result.n_by_city_stratum_metric[("b", stratum, _BUILDING)] == 7


def test_ks_distance_consistency_with_realism() -> None:
    """The verdict's ks matches realism.ks_distance for a known pair (one source)."""
    rng = random.Random(7)
    stratum = (1, 1, 0, 0)
    a = _gauss(rng, mu=0.0, sigma=1.0, n=120)
    b = _gauss(rng, mu=3.0, sigma=1.0, n=120)
    features = {
        ("a", stratum, _BUILDING): a,
        ("b", stratum, _BUILDING): b,
    }
    result = conditioning_discrimination_verdict(features, min_n=50, effect_size_floor=0.15)
    pair = result.pairs[0]
    assert pair.ks == ks_distance(a, b)


# --------------------------------------------------------------------------- #
# Gate-(i) extraction: tile-coverage counters + silent-shrinkage HALT (F3)
# and the extraction-site CRS regression guard (F1).
# --------------------------------------------------------------------------- #

#: A non-Singapore CRS label — distinct from tile_dirname's EPSG3414 default so the
#: CRS guard is RED if the signature default ever rides again.
_FIXTURE_EPSG = "EPSG25832"


def _stub_labels(tile_dir, *, tile_i, tile_j):  # signature mirrors the real reader
    return SimpleNamespace(
        morphology_stratum=SimpleNamespace(
            dominant_zoning_class=1,
            modal_road_skeleton_class=2,
        ),
        coastal_inland_river=0,
    )


def _setup_extraction_fixture(tmp_path, monkeypatch, *, city, n_tiles, missing):
    """Synthetic per-city held-out tree driven through the REAL extraction loop.

    Writes a holdout manifest with ``n_tiles`` tiles; tiles whose index is in
    ``missing`` get NO ``cells.parquet`` on disk (the silent-shrinkage path).
    Every per-tile reader past the existence check is stubbed so each read tile
    yields exactly one open-road feature.
    """
    manifest = {"regions": {city: {"tiles": [{"tile_i": i, "tile_j": 0} for i in range(n_tiles)]}}}
    mf = tmp_path / f"{city}_holdout.yaml"
    mf.write_text(yaml.safe_dump(manifest))

    sub_d = tmp_path / "sub_d" / city
    sub_f = tmp_path / "sub_f" / city
    for i in range(n_tiles):
        if i in missing:
            continue
        tile_dir = sub_f / f"tile={_FIXTURE_EPSG}_i{i}_j0"
        tile_dir.mkdir(parents=True)
        (tile_dir / "cells.parquet").touch()

    monkeypatch.setattr(CD, "holdout_manifest_for_region", lambda release, region: mf)
    monkeypatch.setattr(CD, "epsg_label_for_region", lambda region: _FIXTURE_EPSG)
    monkeypatch.setattr(CD, "sub_d_region_dir", lambda release, region: sub_d)
    monkeypatch.setattr(CD, "sub_f_region_dir", lambda release, region: sub_f)
    monkeypatch.setattr(CD, "read_tile_labels", _stub_labels)
    monkeypatch.setattr(CD, "_cell_density_by_cell", lambda tile_dir: {})
    monkeypatch.setattr(CD, "read_sub_f_cells", lambda path: [])
    monkeypatch.setattr(
        CD,
        "decode_region_blocks",
        lambda tokens, cdbc: (
            [[0]],
            [{"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 0.0]]}],
            [1],
        ),
    )


def test_extraction_reports_tile_coverage_counters(tmp_path, monkeypatch) -> None:
    """20-tile city, 1 missing cells.parquet (1/20 = 0.05 <= 0.1): NO halt; the
    result carries exact per-city n_tiles_expected/read/skipped counters and the
    read tiles still feed features."""
    _setup_extraction_fixture(tmp_path, monkeypatch, city="testcity", n_tiles=20, missing={3})
    out = CD.extract_features_by_city_stratum_metric("rel", ["testcity"])
    assert out.tile_coverage["testcity"] == CD.TileCoverage(
        n_tiles_expected=20, n_tiles_read=19, n_tiles_skipped=1
    )
    # One open-road feature per read tile — extraction still accumulates features.
    assert sum(len(v) for v in out.features.values()) == 19


def test_extraction_halts_above_silent_shrinkage_ceiling(tmp_path, monkeypatch) -> None:
    """3-tile city, 1 missing (1/3 > 0.1): extraction RAISES, naming the city, the
    counts, and the ceiling — a partial city can no longer quietly thin (F3)."""
    _setup_extraction_fixture(tmp_path, monkeypatch, city="testcity", n_tiles=3, missing={1})
    with pytest.raises(RuntimeError) as excinfo:
        CD.extract_features_by_city_stratum_metric("rel", ["testcity"])
    msg = str(excinfo.value)
    assert "testcity" in msg
    assert "1/3" in msg
    assert "0.1" in msg


def test_extraction_no_halt_at_exact_ceiling(tmp_path, monkeypatch) -> None:
    """10 tiles, 1 missing = exactly 0.1: strict > means NO raise (boundary)."""
    _setup_extraction_fixture(tmp_path, monkeypatch, city="testcity", n_tiles=10, missing={0})
    out = CD.extract_features_by_city_stratum_metric("rel", ["testcity"])
    assert out.tile_coverage["testcity"] == CD.TileCoverage(
        n_tiles_expected=10, n_tiles_read=9, n_tiles_skipped=1
    )


def test_extraction_zero_tile_city_is_loud(tmp_path, monkeypatch) -> None:
    """A city with zero tiles in its manifest is never a valid extraction target:
    its own loud error (not a division-by-zero, not a silent empty pass)."""
    _setup_extraction_fixture(tmp_path, monkeypatch, city="testcity", n_tiles=0, missing=set())
    with pytest.raises(ValueError, match="testcity"):
        CD.extract_features_by_city_stratum_metric("rel", ["testcity"])


def test_extraction_uses_region_crs_label(tmp_path, monkeypatch) -> None:
    """The extraction-site ``tile_dirname`` call (the 4th call site) must pass the
    REGION's CRS label. RED-ON-DIVERGENCE: reverting to ``tile_dirname(ti, tj)``
    (signature default rides) records EPSG3414 and fails. Mirrors
    tests/data/training/test_build_shards_multiregion.py::
    test_build_shards_in_memory_uses_region_crs_label."""
    _setup_extraction_fixture(tmp_path, monkeypatch, city="munichlike", n_tiles=1, missing=set())

    captured: list[str] = []
    real_tile_dirname = CD.tile_dirname

    def spy(tile_i: int, tile_j: int, *args, **kwargs):
        # EPSG3414 is the real signature's Singapore default; if the call site stops
        # passing the region label explicitly, the default is what gets recorded.
        label = args[0] if args else kwargs.get("epsg_label", "EPSG3414")
        captured.append(label)
        return real_tile_dirname(tile_i, tile_j, label)

    monkeypatch.setattr(CD, "tile_dirname", spy)

    out = CD.extract_features_by_city_stratum_metric("rel", ["munichlike"])
    assert captured == [_FIXTURE_EPSG], captured
    assert "EPSG3414" not in captured
    # The region-labelled dir resolved on disk: the tile was actually read.
    assert out.tile_coverage["munichlike"].n_tiles_read == 1


def test_result_tile_coverage_defaults_empty_and_threads_via_replace() -> None:
    """The verdict fn stays coverage-agnostic (default {}); the runner threads
    extraction coverage in via dataclasses.replace — pin that seam."""
    result = conditioning_discrimination_verdict({}, min_n=50, effect_size_floor=0.15)
    assert result.tile_coverage == {}
    cov = {"x": CD.TileCoverage(n_tiles_expected=2, n_tiles_read=2, n_tiles_skipped=0)}
    assert dataclasses.replace(result, tile_coverage=cov).tile_coverage == cov


# --------------------------------------------------------------------------- #
# Task 22 recalibration: δ effect-size floor + outbound-bref exclusion-by-identity
# --------------------------------------------------------------------------- #


def test_recalibrated_gate_CAN_pass_at_real_n() -> None:
    """Huge-n tiny-shift (KS≈0.03 << δ=0.15) is BH-significant but NOT a FAIL — the δ floor
    makes PASS reachable at real sample sizes (the structural-incapacity fix)."""
    rng = random.Random(7)
    n = 20000
    stratum = (1, 1, 0, 0)
    features = {
        ("a", stratum, _ROAD): _gauss(rng, mu=0.0, sigma=1.0, n=n),
        ("b", stratum, _ROAD): _gauss(rng, mu=0.1, sigma=1.0, n=n),
    }
    result = conditioning_discrimination_verdict(features, min_n=50, effect_size_floor=0.15)

    # Fixture regime self-check: the shift is real but tiny (0 < KS < δ), and at
    # huge n it IS BH-significant — exactly the structurally-incapable-of-PASS trap.
    pair = result.pairs[0]
    assert 0.0 < pair.ks < 0.15, f"fixture out of regime: ks={pair.ks}"
    assert pair.p_bh < result.alpha, f"fixture out of regime: p_bh={pair.p_bh}"

    # The δ's effect made visible: raw-BH fires, the effect-floored rule does not.
    assert result.n_significant_raw_bh > 0
    assert result.n_significant_effect == 0
    assert result.verdict == "PASS"
    assert all(not p.significant for p in result.pairs)


def test_recalibrated_gate_still_FAILS_on_large_effects() -> None:
    """KS≈0.4 at modest n → FAIL survives the recalibration (glasgow-vs-krakow regime)."""
    rng = random.Random(7)
    n = 150
    stratum = (1, 1, 0, 0)
    features = {
        ("a", stratum, _ROAD): _gauss(rng, mu=0.0, sigma=1.0, n=n),
        ("b", stratum, _ROAD): _gauss(rng, mu=1.0, sigma=1.0, n=n),
    }
    result = conditioning_discrimination_verdict(features, min_n=50, effect_size_floor=0.15)

    pair = result.pairs[0]
    assert pair.ks >= 0.15, f"fixture out of regime: ks={pair.ks}"
    assert result.verdict == "FAIL"
    assert result.n_significant_raw_bh > 0
    assert result.n_significant_effect > 0
    assert any(p.significant for p in result.pairs)


def test_bref_features_excluded_from_road_length_by_construction_identity_and_counted(
    tmp_path, monkeypatch
) -> None:
    """Outbound-bref road excluded + counted (n_bref_excluded per city); a zero-length
    geometry WITHOUT the bref identity is NOT excluded (regime-distinguishing twin —
    symptom-keyed exclusion would pass this; identity-keyed must)."""
    from cfm.data.sub_f.decoder import _is_bref_token
    from cfm.eval.conditioning_discrimination import _tile_features
    from cfm.eval.emergence import building_token_ids

    _BREF = 1500  # BP7 boundary-reference token band is 1500..1507 (sub-F decoder)
    # Authority anchor: the SAME predicate _has_outbound_bref ultimately uses
    # (cfm.data.sub_f.decoder._is_bref_token) must recognize the fixture token —
    # a band move becomes a loud fixture error, not a silent out-of-regime pass.
    assert _is_bref_token(_BREF), "fixture out of regime: _BREF not in decoder bref band"
    bid = min(building_token_ids())

    bref_road = {"type": "LineString", "coordinates": [[0, 0], [3, 0]]}
    zero_len_road = {"type": "LineString", "coordinates": [[0, 0], [0, 0]]}
    plain_road = {"type": "LineString", "coordinates": [[0, 0], [2, 0]]}
    building_ring = {"type": "LineString", "coordinates": [[0, 0], [1, 0], [1, 1], [0, 0]]}
    blocks = [
        [0, 5, _BREF, 0],  # body ends in bref -> outbound-bref road: EXCLUDED
        [0, 5, 0],  # zero-length but NO bref identity: KEPT (symptom != identity)
        [0, 5, 0],  # plain open road: KEPT
        [0, bid, _BREF, 0],  # building ring (promoted to Polygon): NEVER bref-excluded
    ]
    out, n_bref_excluded = _tile_features(
        blocks, [bref_road, zero_len_road, plain_road, building_ring], [1, 1, 1, 1]
    )

    assert n_bref_excluded == 1
    lengths = [v for m, v, _ in out if m == _ROAD]
    areas = [v for m, v, _ in out if m == _BUILDING]
    # The bref road's length (3.0) is gone; the zero-length non-bref twin survives.
    assert sorted(lengths) == [0.0, 2.0]
    assert len(areas) == 1 and areas[0] > 0

    # Per-city accounting: TileCoverage carries n_bref_excluded — excluded features
    # are COUNTED, never silently dropped (structural-exclusion discipline).
    _setup_extraction_fixture(tmp_path, monkeypatch, city="testcity", n_tiles=2, missing=set())
    # Each tile decodes to one outbound-bref road + one plain road.
    monkeypatch.setattr(
        CD,
        "decode_region_blocks",
        lambda tokens, cdbc: (
            [[0, 5, _BREF, 0], [0, 5, 0]],
            [
                {"type": "LineString", "coordinates": [[0.0, 0.0], [3.0, 0.0]]},
                {"type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 0.0]]},
            ],
            [1, 1],
        ),
    )
    out = CD.extract_features_by_city_stratum_metric("rel", ["testcity"])
    assert out.tile_coverage["testcity"].n_bref_excluded == 2  # 1 per tile x 2 tiles
    # Only the plain road per tile survives into the feature pool.
    assert sum(len(v) for v in out.features.values()) == 2

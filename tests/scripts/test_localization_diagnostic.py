"""Tests for scripts/run_localization_diagnostic.py (readiness-closure Task 23, F5; spec §4.2).

The localization diagnostic recomputes the recalibrated gate-(i) verdict under
one-layer-at-a-time stratum VARIANTS (V0 baseline / V1 un-collapse / V2_8+V2_16
un-quantize / V3 candidate sea dim) — the variant that kills the most
discrimination signal localizes where city character lives.

All tests are synthetic (2-city fixtures): the pure variant functions are hit
with in-memory per-cell records; the IO walk + ``main()`` are exercised against
tiny on-disk parquet fixtures written with the REAL sub-D/sub-C/sub-F writers
(path helpers monkeypatched at the module boundary — the established pattern in
tests/eval/test_conditioning_discrimination.py). No Leonardo, no real corpus.

Satisfiability screen (computed, not assumed): the kill-signal fixture uses two
fully separated value clouds (KS = 1.0 at n=60 per city ≥ min_n=50), so the
single V0 pair has p_raw ≈ 1e-27 << alpha=0.05 after a 1-pair BH no-op and
ks=1.0 ≥ δ=0.15 — V0 fires. Under V1 the per-cell zoning split leaves each
stratum with ONE qualified city → zero pairs → the signal is killed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

import cfm.eval.conditioning_discrimination as CD
from cfm.data.sub_c.epsilon import EPS_RATIO
from cfm.data.sub_c.io import CellAggregate
from cfm.data.sub_c.io import write_cells_parquet as write_sub_c_cells_parquet
from cfm.data.sub_d.enums import MetricNamespace, Scope, SlotKind
from cfm.data.sub_d.io import (
    DerivationEvidenceRow,
    MacroCoreRow,
    write_derivation_evidence_parquet,
    write_macro_core_parquet,
)
from cfm.data.sub_f.decoder import _is_bref_token
from cfm.data.sub_f.io import CellRow
from cfm.data.sub_f.io import write_cells_parquet as write_sub_f_cells_parquet
from cfm.eval.holdout.manifest import manifest_sha256
from cfm.eval.realism import FeatureMetric

_REPO = Path(__file__).resolve().parents[2]

_BUILDING = FeatureMetric.BUILDING_AREA.value
_ROAD = FeatureMetric.ROAD_LENGTH.value

#: A non-Singapore CRS label so the fixture never rides tile_dirname's default.
_EPSG = "EPSG25832"

#: A Case-A feature block (no bref) — verified to decode to a 2-distinct-vertex
#: LineString via the real decoder (same constant as tests/eval/holdout/test_roundtrip.py).
_SIMPLE_BLOCK = [509, 41, 300, 323, 363, 369, 1, 50, 510]

_BREF = 1500  # BP7 boundary-reference token band is 1500..1507 (sub-F decoder)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_localization_diagnostic", _REPO / "scripts" / "run_localization_diagnostic.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec: @dataclass resolves the module via sys.modules
    # (PEP 563 string annotations), which the bare importlib recipe skips.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


# --------------------------------------------------------------------------- #
# In-memory record helper (pure-function inputs)
# --------------------------------------------------------------------------- #


def _record(
    city: str,
    *,
    tile_zoning: int | None = 1,
    tile_skeleton: int | None = 2,
    tile_coastal: int | None = 0,
    cell_density: dict | None = None,
    cell_zoning: dict | None = None,
    cell_ratio: dict | None = None,
    cell_sea: dict | None = None,
    features: list | None = None,
    n_bref_excluded: int = 0,
):
    cell = (0, 0)
    return MOD.TileRecord(
        city=city,
        tile_zoning=tile_zoning,
        tile_skeleton=tile_skeleton,
        tile_coastal=tile_coastal,
        cell_density=cell_density if cell_density is not None else {cell: 1},
        cell_zoning=cell_zoning if cell_zoning is not None else {cell: 1},
        cell_ratio=cell_ratio if cell_ratio is not None else {cell: 0.5},
        cell_sea=cell_sea if cell_sea is not None else {cell: 0.0},
        features=features if features is not None else [(_ROAD, 5.0, cell)],
        n_bref_excluded=n_bref_excluded,
    )


# --------------------------------------------------------------------------- #
# V0: baseline must reproduce the gate-(i) stratum shape
# --------------------------------------------------------------------------- #


def test_v0_reproduces_gate_i_stratum_shape() -> None:
    """V0 stratum is EXACTLY the gate-(i) tuple
    (dominant_zoning, modal_skeleton, per-cell density bucket, coastal)."""
    rec = _record(
        "a",
        tile_zoning=3,
        tile_skeleton=2,
        tile_coastal=0,
        cell_density={(0, 0): 1},
        features=[(_ROAD, 5.0, (0, 0)), (_BUILDING, 9.0, (0, 0))],
    )
    feats = MOD.variant_features([rec], "V0")
    assert feats == {
        ("a", (3, 2, 1, 0), _ROAD): [5.0],
        ("a", (3, 2, 1, 0), _BUILDING): [9.0],
    }


# --------------------------------------------------------------------------- #
# V1: un-collapse — per-cell zoning + per-cell density REPLACE the tile dims
# --------------------------------------------------------------------------- #


def test_v1_uncollapses_to_per_cell_stratum() -> None:
    """V1 stratum is literally (per_cell_zoning, per_cell_density): length 2,
    tile-level dims gone (the sentinel tile_zoning=99 must not appear)."""
    rec = _record(
        "a",
        tile_zoning=99,
        tile_skeleton=98,
        tile_coastal=97,
        cell_density={(0, 0): 1},
        cell_zoning={(0, 0): 4},
        features=[(_ROAD, 5.0, (0, 0))],
    )
    feats = MOD.variant_features([rec], "V1")
    assert feats == {("a", (4, 1), _ROAD): [5.0]}
    for _city, stratum, _metric in feats:
        assert len(stratum) == 2
        assert 99 not in stratum and 98 not in stratum and 97 not in stratum


def test_v1_kills_signal_that_v0_sees() -> None:
    """THE localization tooth: two cities differ strongly WITHIN one V0 stratum but
    are homogeneous once V1 splits by per-cell zoning — V1's n_significant_effect
    must drop below V0's (here to zero: the per-cell split leaves no shared stratum)."""
    n = 60
    rec_a = _record(
        "a",
        cell_density={(0, 0): 0},
        cell_zoning={(0, 0): 1},
        features=[(_ROAD, 10.0 + i * 0.001, (0, 0)) for i in range(n)],
    )
    rec_b = _record(
        "b",
        cell_density={(0, 0): 0},
        cell_zoning={(0, 0): 2},
        features=[(_ROAD, 100.0 + i * 0.001, (0, 0)) for i in range(n)],
    )
    records = [rec_a, rec_b]

    out = MOD.diagnose(records, min_n=50, alpha=0.05, effect_size_floor=0.15)

    # Fixture regime self-check (gate-must-distinguish-regimes): V0 actually fires.
    v0 = out["V0"]["per_metric"][_ROAD]
    assert v0["n_pairs"] == 1
    assert v0["n_significant_effect"] == 1, "kill fixture out of regime: V0 never fired"
    assert v0["median_ks"] == 1.0

    # The un-collapse kills it: per-cell zoning partitions the cities apart.
    v1 = out["V1"]["per_metric"][_ROAD]
    assert v1["n_pairs"] == 0
    assert v1["n_significant_effect"] == 0
    assert v1["median_ks"] is None
    assert v1["n_significant_effect"] < v0["n_significant_effect"]


# --------------------------------------------------------------------------- #
# V2: un-quantize — equal-width raw-ratio buckets replace the 4-bucket slot
# --------------------------------------------------------------------------- #


def test_v2_ratio_bucket_edges() -> None:
    """Equal-width over [0, 1]; top edge inclusive; pathological >1 clips into the
    last bucket; the (0.50, 0.57) pair lands same-bucket at 8, different at 16."""
    assert MOD.ratio_bucket(0.0, 8) == 0
    assert MOD.ratio_bucket(1.0, 8) == 7  # top edge inclusive
    assert MOD.ratio_bucket(1.0, 16) == 15
    assert MOD.ratio_bucket(1.2, 8) == 7  # pathological ratio clipped into last
    # Same at 8 buckets...
    assert MOD.ratio_bucket(0.50, 8) == MOD.ratio_bucket(0.57, 8) == 4
    # ...different at 16 (the un-quantize resolution gain).
    assert MOD.ratio_bucket(0.50, 16) == 8
    assert MOD.ratio_bucket(0.57, 16) == 9


def test_v2_replaces_density_slot_with_raw_ratio_bucket() -> None:
    """V2_8/V2_16 strata keep the V0 tuple shape (length 4) with the density slot
    replaced by the N-bucket raw-ratio index — same stratum at 8, split at 16."""
    rec = _record(
        "a",
        tile_zoning=3,
        tile_skeleton=2,
        tile_coastal=0,
        cell_density={(0, 0): 1, (0, 1): 1},  # SAME 4-bucket density
        cell_ratio={(0, 0): 0.50, (0, 1): 0.57},
        cell_sea={(0, 0): 0.0, (0, 1): 0.0},
        cell_zoning={(0, 0): 1, (0, 1): 1},
        features=[(_ROAD, 5.0, (0, 0)), (_ROAD, 7.0, (0, 1))],
    )
    feats8 = MOD.variant_features([rec], "V2_8")
    assert feats8 == {("a", (3, 2, 4, 0), _ROAD): [5.0, 7.0]}
    feats16 = MOD.variant_features([rec], "V2_16")
    assert feats16 == {
        ("a", (3, 2, 8, 0), _ROAD): [5.0],
        ("a", (3, 2, 9, 0), _ROAD): [7.0],
    }


# --------------------------------------------------------------------------- #
# V3: candidate dim — sea bucket APPENDED to the V0 tuple
# --------------------------------------------------------------------------- #


def test_v3_sea_bucket_scheme() -> None:
    """{<=EPS_RATIO -> 0 (structural zero), (EPS, 0.5] -> 1, > 0.5 -> 2}."""
    assert MOD.sea_bucket(0.0) == 0
    assert MOD.sea_bucket(EPS_RATIO / 2) == 0  # structural-boundary EPS treatment
    assert MOD.sea_bucket(0.3) == 1
    assert MOD.sea_bucket(0.5) == 1  # chosen edge: strict, 0.5 inclusive in bucket 1
    assert MOD.sea_bucket(0.6) == 2


def test_v3_appends_sea_dim_to_v0_stratum() -> None:
    """V3 stratum = V0 tuple PLUS the sea bucket: length grows by exactly 1 and
    sea-0 vs sea-positive cells split."""
    rec = _record(
        "a",
        tile_zoning=3,
        tile_skeleton=2,
        tile_coastal=0,
        cell_density={(0, 0): 1, (0, 1): 1},
        cell_sea={(0, 0): 0.0, (0, 1): 0.6},
        cell_zoning={(0, 0): 1, (0, 1): 1},
        cell_ratio={(0, 0): 0.5, (0, 1): 0.5},
        features=[(_ROAD, 5.0, (0, 0)), (_ROAD, 7.0, (0, 1))],
    )
    feats_v0 = MOD.variant_features([rec], "V0")
    feats_v3 = MOD.variant_features([rec], "V3")
    assert feats_v3 == {
        ("a", (3, 2, 1, 0, 0), _ROAD): [5.0],
        ("a", (3, 2, 1, 0, 2), _ROAD): [7.0],
    }
    (v0_stratum,) = {s for _c, s, _m in feats_v0}
    for _c, stratum, _m in feats_v3:
        assert len(stratum) == len(v0_stratum) + 1
        assert stratum[:-1] == v0_stratum


# --------------------------------------------------------------------------- #
# Bref exclusion: SAME construction-identity rule, uniform feature pool
# --------------------------------------------------------------------------- #


def test_per_cell_classifier_applies_bref_exclusion_and_keeps_cell_key() -> None:
    """The per-cell twin of CD._tile_features: outbound-bref road excluded+counted,
    zero-length non-bref twin kept (identity, not symptom), building ring promoted
    to area — and every kept feature carries its cell key."""
    from cfm.eval.emergence import building_token_ids

    # Authority anchor: a band move becomes a loud fixture error, not a silent pass.
    assert _is_bref_token(_BREF), "fixture out of regime: _BREF not in decoder bref band"
    bid = min(building_token_ids())

    bref_road = {"type": "LineString", "coordinates": [[0, 0], [3, 0]]}
    zero_len_road = {"type": "LineString", "coordinates": [[0, 0], [0, 0]]}
    building_ring = {"type": "LineString", "coordinates": [[0, 0], [1, 0], [1, 1], [0, 0]]}

    # 41 is a verified NON-building token (the _SIMPLE_BLOCK road class); blocks must
    # be DISTINCT for the fake_decode lookup (bid == 5 collides with a bare 5).
    decoded = {
        (0, 0): ([0, 41, _BREF, 0], bref_road),  # body ends in bref -> EXCLUDED
        (0, 1): ([0, 41, 0], zero_len_road),  # zero-length, NO bref identity -> KEPT
        (0, 2): ([0, bid, 0], building_ring),  # building ring -> promoted, KEPT
    }

    def fake_split(token_sequence):
        return [token_sequence]

    def fake_decode(block):
        for _cell, (b, g) in decoded.items():
            if b == block:
                return g
        raise AssertionError(f"unexpected block {block}")

    import unittest.mock as mock

    tokens_by_cell = {cell: b for cell, (b, _g) in decoded.items()}
    density = {cell: 1 for cell in decoded}
    with (
        mock.patch.object(MOD, "split_cell_into_features", fake_split),
        mock.patch.object(MOD, "decode_feature", fake_decode),
    ):
        out, n_bref = MOD._tile_cell_features(tokens_by_cell, density)

    assert n_bref == 1
    assert (_ROAD, 0.0, (0, 1)) in out  # the symptom-twin survives
    assert all(cell != (0, 0) for _m, _v, cell in out)  # the bref road is gone
    areas = [(v, c) for m, v, c in out if m == _BUILDING]
    assert len(areas) == 1 and areas[0][0] > 0 and areas[0][1] == (0, 2)


def test_feature_pool_is_uniform_across_all_variants() -> None:
    """The variant comparison is confounded unless every variant sees the SAME
    feature pool: total feature count per city is identical across all 5 variants."""
    rec_a = _record("a", features=[(_ROAD, float(i), (0, 0)) for i in range(7)])
    rec_b = _record("b", features=[(_BUILDING, float(i), (0, 0)) for i in range(5)])
    totals = {}
    for variant in MOD.VARIANTS:
        feats = MOD.variant_features([rec_a, rec_b], variant)
        per_city: dict[str, int] = {}
        for (city, _s, _m), vals in feats.items():
            per_city[city] = per_city.get(city, 0) + len(vals)
        totals[variant] = per_city
    assert len(MOD.VARIANTS) == 5
    assert all(t == {"a": 7, "b": 5} for t in totals.values()), totals


# --------------------------------------------------------------------------- #
# Verdict-parameter threading (min_n / alpha / effect_size_floor)
# --------------------------------------------------------------------------- #


def test_min_n_alpha_floor_are_threaded_into_the_verdict() -> None:
    """The script must thread min_n/alpha/floor into the verdict call: each knob,
    moved alone, flips the kill fixture's V0 outcome."""
    n = 60
    rec_a = _record("a", features=[(_ROAD, 10.0 + i * 0.001, (0, 0)) for i in range(n)])
    rec_b = _record("b", features=[(_ROAD, 100.0 + i * 0.001, (0, 0)) for i in range(n)])
    records = [rec_a, rec_b]

    base = MOD.diagnose(records, min_n=50, alpha=0.05, effect_size_floor=0.15)
    assert base["V0"]["per_metric"][_ROAD]["n_significant_effect"] == 1

    # min_n above the fixture n -> both cells thin -> zero pairs.
    thin = MOD.diagnose(records, min_n=61, alpha=0.05, effect_size_floor=0.15)
    assert thin["V0"]["per_metric"][_ROAD]["n_pairs"] == 0

    # alpha = 0 -> nothing can be BH-significant.
    no_alpha = MOD.diagnose(records, min_n=50, alpha=0.0, effect_size_floor=0.15)
    assert no_alpha["V0"]["per_metric"][_ROAD]["n_significant_effect"] == 0

    # floor above KS=1.0 -> BH fires but the effect rule does not.
    no_floor = MOD.diagnose(records, min_n=50, alpha=0.05, effect_size_floor=1.01)
    assert no_floor["V0"]["per_metric"][_ROAD]["n_significant_effect"] == 0
    assert no_floor["V0"]["n_significant_raw_bh"] == 1


# --------------------------------------------------------------------------- #
# On-disk fixtures: IO walk parity with gate-(i), loud missing artifacts, main()
# --------------------------------------------------------------------------- #


def _write_city_tile(
    root: Path,
    city: str,
    *,
    n_blocks: int = 3,
    skip_evidence: bool = False,
    skip_sub_c: bool = False,
) -> None:
    """One real-writer tile (tile_i=0, tile_j=0) with two active cells (0,0)/(0,1)."""
    dirname = f"tile={_EPSG}_i0_j0"
    cells = [(0, 0), (0, 1)]
    zonings = {(0, 0): 1, (0, 1): 2}
    densities = {(0, 0): 0, (0, 1): 1}
    ratios = {(0, 0): 0.50, (0, 1): 0.57}
    seas = {(0, 0): 0.0, (0, 1): 0.6}

    sub_d = root / "sub_d" / city / dirname
    sub_d.mkdir(parents=True, exist_ok=True)
    macro_rows = [
        MacroCoreRow(
            slot_kind=SlotKind.CELL,
            slot_index=i * 8 + j,
            cell_i=i,
            cell_j=j,
            lower_cell_i=None,
            lower_cell_j=None,
            axis=None,
            scope=Scope.ACTIVE,
            zoning_class=zonings[(i, j)],
            cell_density_bucket=densities[(i, j)],
            road_skeleton_class=None,
        )
        for (i, j) in cells
    ]
    macro_rows.append(
        MacroCoreRow(
            slot_kind=SlotKind.INTERNAL_EDGE,
            slot_index=0,
            cell_i=None,
            cell_j=None,
            lower_cell_i=0,
            lower_cell_j=0,
            axis=0,
            scope=Scope.ACTIVE,
            zoning_class=None,
            cell_density_bucket=None,
            road_skeleton_class=2,
        )
    )
    write_macro_core_parquet(macro_rows, sub_d / "macro_core.parquet")
    (sub_d / "effective_conditioning.yaml").write_text(
        yaml.safe_dump({"conditioning": {"coastal_inland_river": 0}}), encoding="utf-8"
    )
    if not skip_evidence:
        evidence_rows = [
            DerivationEvidenceRow(
                slot_kind=SlotKind.CELL,
                slot_index=i * 8 + j,
                metric_namespace=MetricNamespace.CELL_DENSITY,
                metric_name="building_footprint_ratio",
                value=ratios[(i, j)],
                derivation_version="v1-test",
            )
            for (i, j) in cells
        ]
        write_derivation_evidence_parquet(evidence_rows, sub_d / "derivation_evidence.parquet")

    if not skip_sub_c:
        sub_c = root / "sub_c" / city / dirname
        sub_c.mkdir(parents=True, exist_ok=True)
        write_sub_c_cells_parquet(
            [
                CellAggregate(
                    cell_i=i,
                    cell_j=j,
                    water_fraction=seas[(i, j)],
                    sea_water_fraction=seas[(i, j)],
                    cell_area_admin_clipped_m2=1000.0,
                    kept_features_count=1,
                )
                for (i, j) in cells
            ],
            sub_c / "cells.parquet",
        )

    sub_f = root / "sub_f" / city / dirname
    sub_f.mkdir(parents=True, exist_ok=True)
    active = {(i, j): _SIMPLE_BLOCK * n_blocks for (i, j) in cells}
    rows = [
        CellRow(
            cell_i=i,
            cell_j=j,
            cell_slot_index=i * 8 + j,
            token_sequence=active.get((i, j), []),
            feature_count=n_blocks if (i, j) in active else 0,
            provenance_sha256="a" * 64,
        )
        for i in range(8)
        for j in range(8)
    ]
    write_sub_f_cells_parquet(sub_f / "cells.parquet", rows)


def _write_fixture(root: Path, cities: tuple[str, ...], **tile_kwargs) -> None:
    """Two-city on-disk fixture + a sha-stamped manifest with the lock marker."""
    for city in cities:
        _write_city_tile(root, city, **tile_kwargs)
    manifest = {
        "manifest_schema_version": "2.0",
        "regions": {city: {"tiles": [{"tile_i": 0, "tile_j": 0}]} for city in cities},
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    (root / "holdout_manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (root / "_EVAL_SET_LOCKED").touch()


def _patch_paths(monkeypatch, mod, root: Path) -> None:
    monkeypatch.setattr(
        mod, "holdout_manifest_for_region", lambda release, region: root / "holdout_manifest.yaml"
    )
    monkeypatch.setattr(mod, "epsg_label_for_region", lambda region: _EPSG)
    monkeypatch.setattr(mod, "sub_d_region_dir", lambda release, region: root / "sub_d" / region)
    monkeypatch.setattr(mod, "sub_f_region_dir", lambda release, region: root / "sub_f" / region)
    if hasattr(mod, "sub_c_region_dir"):
        monkeypatch.setattr(
            mod, "sub_c_region_dir", lambda release, region: root / "sub_c" / region
        )


_CITIES = ("alphaville", "bravotown")


def test_v0_features_identical_to_reference_extraction(tmp_path, monkeypatch) -> None:
    """External-source-of-truth gate: on the SAME on-disk fixture, the diagnostic's
    V0 features and coverage are IDENTICAL to gate-(i)'s
    extract_features_by_city_stratum_metric (real readers, real decoder)."""
    _write_fixture(tmp_path, _CITIES)
    _patch_paths(monkeypatch, MOD, tmp_path)
    _patch_paths(monkeypatch, CD, tmp_path)

    records, coverage = MOD.collect_tile_records("rel", list(_CITIES))
    v0 = MOD.variant_features(records, "V0")

    ref = CD.extract_features_by_city_stratum_metric("rel", list(_CITIES))
    assert v0 == ref.features
    assert coverage == ref.tile_coverage
    # Non-vacuity: the fixture actually produced features for both cities.
    assert sum(len(v) for v in v0.values()) == 12  # 2 cities x 2 cells x 3 blocks


def test_missing_derivation_evidence_is_loud(tmp_path, monkeypatch) -> None:
    """A tile that HAS sub-F cells but no derivation_evidence.parquet must raise
    (denominator integrity: variants must never silently see different tile sets)."""
    _write_fixture(tmp_path, _CITIES, skip_evidence=True)
    _patch_paths(monkeypatch, MOD, tmp_path)
    with pytest.raises(FileNotFoundError, match="derivation_evidence"):
        MOD.collect_tile_records("rel", list(_CITIES))


def test_missing_sub_c_cells_is_loud(tmp_path, monkeypatch) -> None:
    """Same denominator-integrity rule for the sub-C cells.parquet (V3's source)."""
    _write_fixture(tmp_path, _CITIES, skip_sub_c=True)
    _patch_paths(monkeypatch, MOD, tmp_path)
    with pytest.raises(FileNotFoundError, match=r"sub_c.*cells\.parquet"):
        MOD.collect_tile_records("rel", list(_CITIES))


def test_main_end_to_end_writes_full_variant_table(tmp_path, monkeypatch, capsys) -> None:
    """The real script main() against on-disk fixtures: exit 0, YAML lands with the
    methodology block (δ + bucket schemes), all 5 variants x both metrics, per-city
    feature totals, and the F3 coverage counters with n_bref_excluded."""
    _write_fixture(tmp_path, _CITIES)
    _patch_paths(monkeypatch, MOD, tmp_path)
    report = tmp_path / "report.yaml"

    rc = MOD.main(
        [
            "--release",
            "rel",
            "--cities",
            *_CITIES,
            "--min-n",
            "1",
            "--report-out",
            str(report),
        ]
    )
    assert rc == 0

    doc = yaml.safe_load(report.read_text(encoding="utf-8"))

    meth = doc["methodology"]
    assert meth["effect_size_floor"] == 0.15
    assert meth["min_n"] == 1
    assert meth["alpha"] == 0.05
    assert meth["variants"]["V2_8"]["density_bucket_scheme"]["n_buckets"] == 8
    assert meth["variants"]["V2_16"]["density_bucket_scheme"]["n_buckets"] == 16
    assert meth["variants"]["V2_8"]["density_bucket_scheme"]["scheme"] == "equal_width"
    assert "sea_bucket_scheme" in meth["variants"]["V3"]

    assert set(doc["variants"]) == {"V0", "V1", "V2_8", "V2_16", "V3"}
    for variant in doc["variants"].values():
        for metric in (_BUILDING, _ROAD):
            cell = variant["per_metric"][metric]
            assert set(cell) >= {"n_pairs", "n_significant_effect", "median_ks"}
        # The step-5 +/-20%-of-V0 denominator sanity check needs per-city totals.
        assert variant["n_features_by_city"] == {"alphaville": 6, "bravotown": 6}

    for city in _CITIES:
        cov = doc["tile_coverage"][city]
        assert cov["n_tiles_expected"] == 1
        assert cov["n_tiles_read"] == 1
        assert cov["n_tiles_skipped"] == 0
        assert cov["n_bref_excluded"] == 0

    # Human-readable variant table on stdout (the sanctioned runner-summary print).
    out = capsys.readouterr().out
    assert "V0" in out and "V2_16" in out and "V3" in out

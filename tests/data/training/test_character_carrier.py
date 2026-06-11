"""Task 24b data-side teeth — per-cell continuous character carrier (mini-spec §1/§3).

Covers (mini-spec §3 teeth 1-5, data half):
  1. round-trip + fail-closed: ``CellPayload.character_stats`` is REQUIRED (an
     old-schema construction without it fails loudly — the version-skew kill), and
     the stats ride shard -> flatten -> example EXACTLY (float equality);
  2. derivation parity vs the recon script's independent reader + ``statistics``
     calls on a SHARED on-disk fixture written by the REAL sub-C writers
     (external source of truth, never self-consistency);
  3. presence-flag discipline: a genuinely-zero-stat cell (one zero-length road:
     log10(0+1) == 0.0) and an absent-layer cell are distinguishable by the flag
     bit ALONE — collapsing the flags (force always-1) makes them identical
     (the mutation evidence the PI obligation names);
  4. ablation: "no_character" zeroes ONLY the continuous channel (all 10 prefix id
     positions bit-identical to full); "no_city" zeroes ONLY id slot 8 (stats
     untouched); the two compose;
  5. guards (spec §4.5 extension): constant stats across >=2 distinct regions ->
     loud; an all-absent region (every cell both flags 0) -> loud; wired at BOTH
     seams (build_multiregion_shards, CellDataModule.setup).
"""

from __future__ import annotations

import importlib.util
import math
import statistics
from pathlib import Path

import pytest
import yaml

from cfm.data.sub_c.io import FeatureRow, write_features_parquet
from cfm.data.training import build_shards as BS
from cfm.data.training import datamodule as DM
from cfm.data.training.shard_schema import CellPayload, TrainingShard
from cfm.eval.holdout.labels import MorphologyStratum, TileLabels

_REPO = Path(__file__).resolve().parents[3]
_RELEASE = "2026-04-15.0"
_REGION = "singapore"

# A healthy non-zero stats vector for synthetic payloads (flags present).
_STATS = (1.2, 0.4, 0.3, 1.1, 0.9, 1.0, 1.0)

_TILE_CONDITIONING = {
    "population_density_bucket": 0,
    "dominant_zoning_class": 0,
    "modal_road_skeleton_class": 1,
    "admin_region": None,
    "coastal_inland_river": 0,
    "sub_c_morphology_class": "Asian-megacity",
}


def _cell(ci, cj, n_tokens, density=1, character_stats=_STATS):
    return CellPayload(
        cell_i=ci,
        cell_j=cj,
        cell_slot_index=ci * 8 + cj,
        tokens=tuple(range(1, n_tokens + 1)),
        cell_density_bucket=density,
        boundary_contracts=(),
        character_stats=character_stats,
    )


def _shard(region, ti, tj, cells):
    return TrainingShard(
        region=region,
        tile_i=ti,
        tile_j=tj,
        tile_conditioning=dict(_TILE_CONDITIONING),
        macro_tokens=(),
        cells=tuple(cells),
        lineage=frozenset({(region, ti, tj)}),
    )


# --------------------------------------------------------------------------- #
# Shared on-disk fixture: a features.parquet written by the REAL sub-C writers.
# cell (0,0): 4 buildings + 3 roads; cell (0,1): one 1.0 m^2 building (the
# log-folds-to-zero screen); cell (0,2): one ZERO-LENGTH road (the flag-only
# regime pair); cell (1,0) exists only in sub-F (absent from features.parquet).
# --------------------------------------------------------------------------- #

_AREAS_00 = [10.0, 20.0, 40.0, 80.0]
_ROADS_00 = [5.0, 10.0, 20.0]

_GEOM_TYPE = {"Point": 0, "LineString": 1, "Polygon": 2}


def _row(ci, cj, fclass, geom, idx):
    b = geom.bounds
    return FeatureRow(
        cell_i=ci,
        cell_j=cj,
        feature_class=fclass,
        source_feature_id=f"f{idx:03d}",
        geometry=geom,
        geometry_type=_GEOM_TYPE[geom.geom_type],
        bbox_min_x=b[0],
        bbox_min_y=b[1],
        bbox_max_x=b[2],
        bbox_max_y=b[3],
        class_raw=None,
        subtype_raw=None,
        categories_primary=None,
        categories_alternate=None,
        sea_overlap_fraction=0.0,
    )


def _square(x, y, side):
    from shapely.geometry import Polygon

    return Polygon([(x, y), (x + side, y), (x + side, y + side), (x, y + side)])


def _write_fixture_features(path: Path) -> None:
    from shapely.geometry import LineString

    rows = []
    idx = 0
    for a in _AREAS_00:  # buildings: squares of area a
        rows.append(_row(0, 0, 1, _square(0.0, 0.0, math.sqrt(a)), idx))
        idx += 1
    for ln in _ROADS_00:  # roads: straight segments of length ln
        rows.append(_row(0, 0, 0, LineString([(0.0, 0.0), (ln, 0.0)]), idx))
        idx += 1
    rows.append(_row(0, 1, 1, _square(5.0, 5.0, 1.0), idx))  # one 1.0 m^2 building
    idx += 1
    rows.append(_row(0, 2, 0, LineString([(3.0, 3.0), (3.0, 3.0)]), idx))  # zero-length road
    path.parent.mkdir(parents=True, exist_ok=True)
    write_features_parquet(rows, path)


@pytest.fixture()
def fixture_features(tmp_path: Path) -> Path:
    p = tmp_path / _REGION / "tile=EPSG3414_i0_j0" / "features.parquet"
    _write_fixture_features(p)
    return p


def _load_recon():
    """The recon script (external source of truth for the derivation parity tooth)."""
    spec = importlib.util.spec_from_file_location(
        "recon_char", _REPO / "scripts" / "investigate_residual_character.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =========================================================================== #
# TOOTH 1 — schema fail-closed + exact round-trip
# =========================================================================== #


def test_cell_payload_refuses_construction_without_character_stats():
    """Version-skew kill: character_stats is REQUIRED (no default). An old-shard
    read that does not supply it must fail LOUDLY at construction, never default."""
    with pytest.raises(TypeError):
        CellPayload(
            cell_i=0,
            cell_j=0,
            cell_slot_index=0,
            tokens=(1, 2),
            cell_density_bucket=1,
            boundary_contracts=(),
        )


def test_character_stats_round_trip_shard_to_example_exact():
    """The stats ride CellPayload -> flatten -> CellExample EXACTLY (float ==),
    and the collated batch carries them as a float tensor [B, 7]."""
    import torch

    stats = (0.123456789012345, 0.4, 0.0, 1.75, 2.5, 1.0, 1.0)
    shard = _shard(_REGION, 0, 0, [_cell(0, 0, 10, character_stats=stats)])
    examples, _ = DM.flatten_shards_to_cells([shard])
    [ex] = examples
    assert ex.character_stats == stats  # exact, not approx
    batch = DM.collate_cells([DM._as_item(ex)])
    assert batch["char_stats"].shape == (1, 7)
    assert batch["char_stats"].dtype == torch.float32
    assert torch.allclose(batch["char_stats"][0], torch.tensor(stats, dtype=torch.float32))


def test_example_prefix_grows_by_the_placeholder_position():
    """Layout decision: prefix = [9 value-bearing ids | placeholder id], so
    prefix_len == 10 and the placeholder occupies ids[9] (its embedding is
    overwritten by the Linear projection model-side)."""
    shard = _shard(_REGION, 0, 0, [_cell(0, 0, 10)])
    [ex] = DM.flatten_shards_to_cells([shard])[0]
    assert ex.prefix_len == 10
    assert ex.prefix_ids[9] == DM.CHARACTER_PLACEHOLDER_ID


# =========================================================================== #
# TOOTH 2 — derivation parity vs the recon reader + statistics (external truth)
# =========================================================================== #


def test_derivation_parity_against_recon_reader_and_statistics(fixture_features):
    """Both sides on the SHARED on-disk fixture: build-shards derivation vs the
    recon script's _read_cell_building_areas + raw ``statistics`` calls. Equality
    is EXACT (same float ops). The recon side never routes through the build-side
    helper — external source of truth, not self-consistency."""
    recon = _load_recon()
    derived = BS.derive_character_stats(fixture_features)
    areas_by_cell = recon._read_cell_building_areas(fixture_features, {(0, 0), (0, 1), (0, 2)})
    # the reader sees exactly the fixture's building layers
    assert sorted(areas_by_cell[(0, 0)]) == pytest.approx(_AREAS_00)
    assert areas_by_cell[(0, 2)] == []  # roads-only cell: no buildings

    for cell in [(0, 0), (0, 1)]:
        areas = areas_by_cell[cell]
        got = derived[cell]
        med = statistics.median(areas)
        assert got[0] == math.log10(med)
        if len(areas) >= 2:
            qs4 = statistics.quantiles(areas, n=4)
            assert got[1] == math.log10(qs4[2] - qs4[0] + 1.0)
            p90 = statistics.quantiles(areas, n=10)[8]
            assert got[2] == math.log10(p90 / med)
        else:
            assert got[1] == 0.0 and got[2] == 0.0
        assert got[3] == math.log10(len(areas) + 1.0)
        assert got[5] == 1.0  # buildings present

    # road channel: the fixture's known lengths through raw statistics
    assert derived[(0, 0)][4] == math.log10(statistics.median(_ROADS_00) + 1.0)
    assert derived[(0, 0)][6] == 1.0
    assert derived[(0, 1)][4] == 0.0 and derived[(0, 1)][6] == 0.0  # no roads in (0,1)


# =========================================================================== #
# TOOTH 3 — zero-vs-absent regime pair (the carried PI obligation)
# =========================================================================== #


def test_zero_stat_cell_and_absent_layer_cell_differ_only_in_the_flag(fixture_features):
    """Satisfiability screened first: a single ZERO-LENGTH road log-folds channel 4
    to exactly 0.0 (log10(0+1)) — identical to the absent-roads value. The two
    7-vectors must DIFFER (flag bit), and collapsing the flags (mutation: force
    both flags to 1) makes them IDENTICAL — proof the distinguishability comes
    from the flag ALONE, exactly what the flag exists to carry."""
    zero_road = BS.character_stats_for_cell([], [0.0])
    absent = BS.character_stats_for_cell([], [])
    # screen: the chosen geometry really log-folds to 0.0 through THE transform
    assert zero_road[4] == 0.0 == absent[4]
    # the pair is distinguishable — by the roads flag bit ONLY
    assert zero_road != absent
    assert zero_road[:6] == absent[:6]
    assert zero_road[6] == 1.0 and absent[6] == 0.0
    # mutation evidence: collapsing the flags erases the distinction
    collapse = lambda v: (*v[:5], 1.0, 1.0)  # noqa: E731
    assert collapse(zero_road) == collapse(absent)

    # and through the REAL parquet fixture: cell (0,2) is the zero-length-road cell
    derived = BS.derive_character_stats(fixture_features)
    assert derived[(0, 2)] == zero_road


def test_one_square_meter_building_folds_channel_zero_to_zero():
    """The dispatch-named screen: one 1.0 m^2 building gives channel 0 == 0.0
    (log10(1) == 0) while the buildings flag still distinguishes it from absent."""
    one_sqm = BS.character_stats_for_cell([1.0], [])
    absent = BS.character_stats_for_cell([], [])
    assert one_sqm[0] == 0.0 == absent[0]
    assert one_sqm[5] == 1.0 and absent[5] == 0.0


# =========================================================================== #
# build_shards_in_memory wiring — derived stats land on CellPayload
# =========================================================================== #


def _fake_labels(tile_dir, *, tile_i, tile_j):
    return TileLabels(
        tile_i=tile_i,
        tile_j=tile_j,
        population_density_bucket=0,
        cell_density_buckets=(),
        morphology_stratum=MorphologyStratum(dominant_zoning_class=0, modal_road_skeleton_class=1),
        coastal_inland_river=0,
        admin_region=None,
        sub_c_morphology_class=None,
    )


def test_build_shards_in_memory_attaches_derived_stats(tmp_path, monkeypatch):
    """End-to-end wiring: the build derives per-cell stats from the tile's REAL
    sub-C features.parquet; a cell absent from features.parquet gets the absent
    vector (all 0.0 + both flags 0), never a KeyError."""
    fp = tmp_path / _REGION / "tile=EPSG3414_i0_j0" / "features.parquet"
    _write_fixture_features(fp)
    monkeypatch.setattr(BS, "sub_c_region_dir", lambda release, region: tmp_path / region)
    monkeypatch.setattr(BS, "epsg_label_for_region", lambda region: "EPSG3414")
    monkeypatch.setattr(BS, "read_tile_labels", _fake_labels)
    monkeypatch.setattr(BS, "_cell_density_by_cell", lambda tile_dir: {(0, 0): 1})
    monkeypatch.setattr(
        BS,
        "read_sub_f_cells",
        lambda p: {(0, 0): [1, 2, 3], (1, 0): [4, 5]},
    )
    [shard] = BS.build_shards_in_memory(_RELEASE, _REGION, tile_ids=[(0, 0)])
    by_cell = {(c.cell_i, c.cell_j): c for c in shard.cells}
    expected = BS.derive_character_stats(fp)
    assert by_cell[(0, 0)].character_stats == expected[(0, 0)]
    assert by_cell[(1, 0)].character_stats == BS.character_stats_for_cell([], [])


# =========================================================================== #
# TOOTH 4 — ablation: no_character zeroes ONLY the continuous channel; composes
# =========================================================================== #


def test_flatten_no_character_zeroes_only_the_continuous_channel():
    shard = _shard(_REGION, 0, 0, [_cell(0, 0, 10)])
    [full] = DM.flatten_shards_to_cells([shard])[0]
    [abl] = DM.flatten_shards_to_cells([shard], ablation="no_character")[0]
    # ALL 10 prefix id positions bit-identical (9 value ids + placeholder)
    assert abl.prefix_ids == full.prefix_ids
    assert abl.tokens == full.tokens
    # ONLY the continuous channel is zeroed — stats AND flags
    assert abl.character_stats == (0.0,) * 7
    assert full.character_stats == _STATS != abl.character_stats


def test_flatten_no_city_leaves_character_stats_untouched():
    """Composition (tooth 4): no_city zeroes ONLY id slot 8; the continuous
    channel rides untouched — the two ablations are orthogonal instruments."""
    shard = _shard(_REGION, 0, 0, [_cell(0, 0, 10)])
    [full] = DM.flatten_shards_to_cells([shard])[0]
    [no_city] = DM.flatten_shards_to_cells([shard], ablation="no_city")[0]
    assert no_city.character_stats == full.character_stats == _STATS
    assert no_city.prefix_ids[8] != full.prefix_ids[8]  # city slot ablated
    assert no_city.prefix_ids[:8] == full.prefix_ids[:8]
    assert no_city.prefix_ids[9] == full.prefix_ids[9]  # placeholder untouched


def _write_holdout(tmp):
    from cfm.eval.holdout.manifest import manifest_sha256

    holdout = {
        "manifest_schema_version": "2.0",
        "regions": {_REGION: {"tiles": [{"tile_i": 1, "tile_j": 7}]}},
    }
    holdout["manifest_sha256"] = manifest_sha256(holdout)
    p = tmp / "holdout.yaml"
    p.write_text(yaml.safe_dump(holdout))
    (tmp / "_EVAL_SET_LOCKED").touch()
    return p


def _write_training_manifest(tmp):
    p = tmp / "training_manifest.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "release": _RELEASE,
                "region": _REGION,
                "tiles": [{"tile_i": 2, "tile_j": 2, "lineage": [[_REGION, 2, 2]]}],
            }
        )
    )
    return p


def test_setup_no_character_zeroes_train_and_val_stats_but_not_ids(tmp_path, monkeypatch):
    """End-to-end through CellDataModule(conditioning_ablation="no_character"):
    every train AND val example (the generation-side conditioning source) carries
    the zero vector while its prefix ids stay bit-identical to the full twin.
    The §4.5 guard reads the SHARDS' raw stats, so an ablated run on a healthy
    region must still pass setup (the guard fires on data, not on the ablation)."""
    shards = [
        _shard(_REGION, 0, 0, [_cell(0, 0, 50), _cell(0, 1, 30)]),
        _shard(_REGION, 0, 1, [_cell(1, 1, 30, character_stats=(0.7,) * 5 + (1.0, 1.0))]),
    ]
    monkeypatch.setattr(DM, "build_shards_in_memory", lambda *a, **k: shards)
    clean = _write_training_manifest(tmp_path)
    hp = _write_holdout(tmp_path)
    dm_full = DM.CellDataModule(training_manifest=clean, holdout_manifest=hp, seed=7)
    dm_abl = DM.CellDataModule(
        training_manifest=clean,
        holdout_manifest=hp,
        seed=7,
        conditioning_ablation="no_character",
    )
    dm_full.setup("fit")
    dm_abl.setup("fit")  # guard must NOT fire on the ablation (it reads raw shard stats)
    assert dm_abl.val_cells, "generation-side source must be non-vacuous"
    for fe, ae in zip(
        dm_full.train_cells + dm_full.val_cells,
        dm_abl.train_cells + dm_abl.val_cells,
        strict=True,
    ):
        assert fe.key == ae.key
        assert ae.character_stats == (0.0,) * 7
        assert fe.character_stats != ae.character_stats
        assert fe.prefix_ids == ae.prefix_ids  # id positions bit-identical
        assert fe.tokens == ae.tokens


# =========================================================================== #
# TOOTH 5 — §4.5 guards: constant-across-regions + all-absent region
# =========================================================================== #


def test_guard_character_stats_constant_across_two_regions_fires():
    v = (1.0, 0.5, 0.2, 1.1, 0.9, 1.0, 1.0)
    with pytest.raises(BS.CharacterStatsError, match="constant"):
        BS.guard_character_stats({"almere": [v, v], "welwyn": [v]})


def test_guard_character_stats_constant_within_one_region_is_healthy():
    v = (1.0, 0.5, 0.2, 1.1, 0.9, 1.0, 1.0)
    w = (2.0, 0.5, 0.2, 1.1, 0.9, 1.0, 1.0)
    BS.guard_character_stats({"almere": [v, v], "welwyn": [w]})  # must not raise
    BS.guard_character_stats({"almere": [v, w]})  # varied single region: healthy
    BS.guard_character_stats({"almere": []})  # vacuous, never a false positive


def test_guard_character_stats_all_absent_region_fires():
    absent = BS.character_stats_for_cell([], [])
    with pytest.raises(BS.CharacterStatsError, match="almere"):
        BS.guard_character_stats({"almere": [absent, absent]})


def test_guard_all_absent_fires_even_when_some_continuous_channels_are_nonzero():
    """The all-absent regime keys on the FLAGS (both 0), not on the stats being
    zero — a vector with nonzero stats but zero flags is still 'absent layer'
    by the carrier's own convention and must fire."""
    weird = (1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0)  # flags 0 despite nonzero stats
    with pytest.raises(BS.CharacterStatsError, match="welwyn"):
        BS.guard_character_stats({"welwyn": [weird]})


def test_build_multiregion_shards_runs_the_character_guard(monkeypatch):
    """Wire-in proof at the build seam: per-city builds whose stats are one
    constant vector across BOTH cities must raise (city_identity itself is
    healthy here, so the firing guard is the character one)."""
    v = (1.0, 0.5, 0.2, 1.1, 0.9, 1.0, 1.0)

    def fake_city_shards(release, city):
        return [_shard(city, 0, 0, [_cell(0, 0, 10, character_stats=v)])]

    monkeypatch.setattr(BS, "build_train_city_shards", fake_city_shards)
    with pytest.raises(BS.CharacterStatsError, match="constant"):
        BS.build_multiregion_shards(_RELEASE, ["almere", "welwyn"])


def test_setup_runs_the_character_guard_on_the_union(tmp_path, monkeypatch):
    """Wire-in proof at the datamodule seam: two per-city manifests whose built
    shards carry the SAME constant stats vector -> setup halts loud BEFORE any
    split/loader exists."""
    v = (1.0, 0.5, 0.2, 1.1, 0.9, 1.0, 1.0)
    mf = {}
    for city, (ti, tj) in {"almere": (2, 2), "welwyn": (3, 3)}.items():
        p = tmp_path / f"{city}.yaml"
        p.write_text(
            yaml.safe_dump(
                {
                    "release": _RELEASE,
                    "region": city,
                    "tiles": [{"tile_i": ti, "tile_j": tj, "lineage": [[city, ti, tj]]}],
                }
            )
        )
        mf[city] = p
    monkeypatch.setattr(
        DM,
        "build_shards_in_memory",
        lambda release, region, *, tile_ids=None: [
            _shard(region, tile_ids[0][0], tile_ids[0][1], [_cell(0, 0, 10, character_stats=v)])
        ],
    )
    dm = DM.CellDataModule(
        training_manifests=[mf["almere"], mf["welwyn"]],
        holdout_manifest=_write_holdout(tmp_path),
        seed=7,
    )
    with pytest.raises(BS.CharacterStatsError, match="constant"):
        dm.setup("fit")

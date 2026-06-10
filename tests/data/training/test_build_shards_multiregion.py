"""Phase C Task 7 — multi-region train build (the teeth).

NET-NEW build over the legacy single-region (Singapore) pipeline. Four teeth, all
on SYNTHETIC multi-CRS fixtures (the EU corpus is Leonardo-only; the real shard
build is Task 8):

  1. (held-out-excluded-IN-THE-BUILD) ``train_cities`` over a synthetic G4 roll-up
     (validated + unvalidated) and a synthetic holdout (the 4 held-out cities) keeps
     ONLY the validated, non-held-out cities. This is the PRIMARY exclusion (the 4
     held-out + every unvalidated city are absent from the build, not merely
     leak-audited downstream).

  2. (Task-7 boundary — the I1 bite) building a synthetic TRAIN city (not held-out,
     not singapore) via the driver SUCCEEDS with an empty tile-level holdout, NOT a
     ``ValueError``. RED-ON-DIVERGENCE: the NAIVE path
     (``compute_training_tile_ids(release, train_city)``) RAISES ``ValueError`` (the
     ``_holdout_ids`` train-city raise), while the driver's all-validated-tiles path
     does NOT. That contrast IS the proof the boundary is handled at the loop.

  3. (datamodule union) two synthetic per-city manifests union into one example set
     whose provenance/region spans BOTH cities (not just one).

  4. (CRS-agnostic) a synthetic city tagged a non-z32 CRS (EPSG:25833) loads through
     the union; the loaded examples are pure token ints with NO CRS attribute/field
     anywhere on the example (CRS is baked out at encode time).

The driver/union NEVER call ``_holdout_ids`` for a train city, and NEVER modify
``_holdout_ids`` / ``holdout_manifest_for_region`` (the Task-1 fail-closed guarantee
stays intact — its own raise on the naive path is asserted here as the teeth).
"""

from __future__ import annotations

import pytest

from cfm.data.training import build_shards as BS
from cfm.data.training import datamodule as DM
from cfm.data.training.shard_schema import CellPayload, TrainingShard

_RELEASE = "2026-04-15.0"


# --------------------------------------------------------------------------- #
# Synthetic fixtures (no real sub-D/sub-F data; multi-CRS by tagging)
# --------------------------------------------------------------------------- #
def _synthetic_g4_rollup() -> dict:
    """A roll-up mixing validated / unvalidated cities, including the 4 held-out
    (which ARE present in the real roll-up with ``validated: true`` — see the G4
    audit). ``train_cities`` must drop the held-out AND the unvalidated."""
    return {
        "per_city": [
            {"name": "prague", "crs": "EPSG:25833", "validated": True},
            {"name": "barcelona", "crs": "EPSG:25831", "validated": True},
            {"name": "munich", "crs": "EPSG:25832", "validated": True},  # HELD-OUT
            {"name": "glasgow", "crs": "EPSG:25830", "validated": True},  # HELD-OUT
            {"name": "krakow", "crs": "EPSG:25834", "validated": True},  # HELD-OUT
            {"name": "eisenhuttenstadt", "crs": "EPSG:25833", "validated": True},  # HELD-OUT
            {"name": "half_baked", "crs": "EPSG:25832", "validated": False},  # UNVALIDATED
            {"name": "broken_city", "crs": "EPSG:25831", "validated": False},  # UNVALIDATED
        ]
    }


def _synthetic_multiregion_holdout() -> dict:
    """The schema-2.0 multiregion holdout: ``held_out_cities`` is the selector
    ``train_cities`` reads to exclude whole cities by construction."""
    return {
        "manifest_schema_version": "2.0",
        "held_out_cities": ["eisenhuttenstadt", "glasgow", "krakow", "munich"],
        "regions": {
            c: {"tiles": [{"tile_i": 0, "tile_j": 0}]}
            for c in ("eisenhuttenstadt", "glasgow", "krakow", "munich")
        },
    }


def _cell(ci: int, cj: int, n_tokens: int, density: int = 1) -> CellPayload:
    return CellPayload(
        cell_i=ci,
        cell_j=cj,
        cell_slot_index=ci * 8 + cj,
        tokens=tuple(range(1, n_tokens + 1)),  # pure ints, 1-based (non-empty, non-PAD)
        cell_density_bucket=density,
        boundary_contracts=(),
    )


def _shard(region: str, ti: int, tj: int, cells: list[CellPayload]) -> TrainingShard:
    return TrainingShard(
        region=region,
        tile_i=ti,
        tile_j=tj,
        tile_conditioning={"admin_region": None},  # division (None for EU), never the city
        macro_tokens=(),
        cells=tuple(cells),
        lineage=frozenset({(region, ti, tj)}),
    )


# =========================================================================== #
# TOOTH 1 — held-out (and unvalidated) excluded IN THE BUILD
# =========================================================================== #
def test_train_cities_excludes_heldout_and_unvalidated_in_the_build():
    cities = BS.train_cities(
        _RELEASE,
        g4_rollup=_synthetic_g4_rollup(),
        holdout_manifest=_synthetic_multiregion_holdout(),
    )
    # validated, non-held-out -> kept
    assert set(cities) == {"prague", "barcelona"}
    # the 4 held-out -> absent (primary exclusion, in the build)
    for held in ("eisenhuttenstadt", "glasgow", "krakow", "munich"):
        assert held not in cities
    # the unvalidated -> absent
    for unval in ("half_baked", "broken_city"):
        assert unval not in cities


def test_train_cities_accepts_paths_too(tmp_path):
    """Injectable as dict OR path (so Task 8 can point at the on-disk roll-up /
    holdout without a code change)."""
    import yaml

    rollup_p = tmp_path / "g4.yaml"
    holdout_p = tmp_path / "holdout.yaml"
    rollup_p.write_text(yaml.safe_dump(_synthetic_g4_rollup()))
    holdout_p.write_text(yaml.safe_dump(_synthetic_multiregion_holdout()))
    cities = BS.train_cities(_RELEASE, g4_rollup=rollup_p, holdout_manifest=holdout_p)
    assert set(cities) == {"prague", "barcelona"}


# =========================================================================== #
# TOOTH 2 — the I1 bite (red-on-divergence)
# =========================================================================== #
def test_naive_path_raises_valueerror_for_train_city():
    """RED-ON-DIVERGENCE (half 1): the NAIVE path — compute_training_tile_ids on a
    TRAIN city — routes through _holdout_ids -> holdout_manifest_for_region, which
    RAISES ValueError for a city that is neither singapore nor a held-out city. This
    is the I1 boundary biting exactly as predicted (and proves _holdout_ids is the
    untouched fail-closed backstop)."""
    with pytest.raises(ValueError, match="unknown region 'prague'"):
        BS.compute_training_tile_ids(_RELEASE, "prague")


def test_driver_all_validated_tiles_path_does_not_raise(monkeypatch):
    """RED-ON-DIVERGENCE (half 2): the DRIVER's all-validated-tiles path for the SAME
    train city does NOT raise — it bypasses compute_training_tile_ids/_holdout_ids by
    passing tile_ids=<all validated ids for the city> straight to
    build_shards_in_memory. The empty tile-level holdout is correct: whole-city
    exclusion already removed the held-out cities, so a train city keeps ALL its
    validated tiles.

    We stub _validated_inventory (the all-validated-tiles source) and the per-region
    shard build so the test stays data-free; the assertion is that the driver reaches
    build_shards_in_memory WITH tile_ids and NEVER calls _holdout_ids."""
    # Stub the inventory: prague has 3 validated tiles.
    monkeypatch.setattr(
        BS,
        "_validated_inventory",
        lambda release, region: [
            {"tile_i": 0, "tile_j": 0, "provenance_sha256": "a"},
            {"tile_i": 0, "tile_j": 1, "provenance_sha256": "b"},
            {"tile_i": 1, "tile_j": 0, "provenance_sha256": "c"},
        ],
    )
    # Stub the per-region build so no real sub-D/sub-F data is needed; it MUST be
    # called WITH explicit tile_ids (the bypass) or the test fails.
    seen: dict[str, object] = {}

    def fake_in_memory(release, region, *, tile_ids=None):
        seen["region"] = region
        seen["tile_ids"] = tile_ids
        assert tile_ids is not None, "driver must pass explicit tile_ids (the bypass)"
        return [_shard(region, ti, tj, [_cell(0, 0, 5)]) for (ti, tj) in tile_ids]

    monkeypatch.setattr(BS, "build_shards_in_memory", fake_in_memory)

    # Guard: _holdout_ids must NEVER be called for a train city (would raise).
    def boom(*a, **k):  # pragma: no cover - asserts non-invocation
        raise AssertionError("_holdout_ids must NOT be called for a train city")

    monkeypatch.setattr(BS, "_holdout_ids", boom)

    shards = BS.build_train_city_shards(_RELEASE, "prague")  # must NOT raise
    assert seen["region"] == "prague"
    assert seen["tile_ids"] == [(0, 0), (0, 1), (1, 0)]  # all validated tiles, sorted
    assert len(shards) == 3
    assert all(s.region == "prague" for s in shards)


# =========================================================================== #
# TOOTH 2b — the I1-SAFE WRITER (build_train_city_manifest) — Task-8 persistence
# =========================================================================== #
def test_build_train_city_manifest_writes_all_validated_tiles(tmp_path, monkeypatch):
    """The I1-safe WRITER builds a train city from ALL validated tiles (no holdout
    subtraction) and writes a schema-1.0 per-city manifest. This is the persistence the
    Task-8 driver needs — build_train_city_shards is in-memory only."""
    monkeypatch.setattr(
        BS,
        "_validated_inventory",
        lambda release, region: [
            {"tile_i": 0, "tile_j": 0, "provenance_sha256": "a"},
            {"tile_i": 0, "tile_j": 1, "provenance_sha256": "b"},
            {"tile_i": 1, "tile_j": 0, "provenance_sha256": "c"},
        ],
    )
    monkeypatch.setattr(
        BS,
        "build_shards_in_memory",
        lambda release, region, *, tile_ids=None: [
            _shard(region, ti, tj, [_cell(0, 0, 5)]) for (ti, tj) in (tile_ids or [])
        ],
    )
    shards = BS.build_train_city_manifest(_RELEASE, "prague", out_dir=tmp_path)
    assert len(shards) == 3
    import yaml

    m = yaml.safe_load((tmp_path / "training_manifest.yaml").read_text())
    assert m["region"] == "prague"
    assert m["manifest_schema_version"] == "1.0"
    assert m["n_training_tiles"] == 3  # ALL validated tiles — no tile-level holdout


def test_i1_writer_does_not_raise_unlike_single_region_writer(tmp_path, monkeypatch):
    """RED-ON-DIVERGENCE: the single-region writer ``build_training_shards`` RAISES the
    I1 ValueError for a train city (it routes through _holdout_ids); the I1-safe writer
    ``build_train_city_manifest`` does NOT. This contrast is the proof Task 8 must call
    the I1-safe writer — the exact bug small-before-big caught on Leonardo (2026-06-10)."""
    # _validated_inventory stubbed so build_training_shards gets PAST the prov step and
    # reaches build_shards_in_memory(no tile_ids) -> compute_training_tile_ids ->
    # _holdout_ids -> the I1 raise (build_shards_in_memory NOT yet stubbed).
    monkeypatch.setattr(
        BS,
        "_validated_inventory",
        lambda release, region: [{"tile_i": 0, "tile_j": 0, "provenance_sha256": "a"}],
    )
    with pytest.raises(ValueError, match="unknown region 'prague'"):
        BS.build_training_shards(_RELEASE, "prague", out_dir=tmp_path)
    # Now stub the in-memory build; the I1-safe writer does NOT raise and writes a manifest.
    monkeypatch.setattr(
        BS,
        "build_shards_in_memory",
        lambda release, region, *, tile_ids=None: [
            _shard(region, ti, tj, [_cell(0, 0, 5)]) for (ti, tj) in (tile_ids or [])
        ],
    )
    shards = BS.build_train_city_manifest(_RELEASE, "prague", out_dir=tmp_path)
    assert len(shards) == 1
    assert (tmp_path / "training_manifest.yaml").exists()


# =========================================================================== #
# TOOTH — multi-region CRS: per-tile dir names use the REGION's CRS label
# =========================================================================== #
def test_build_shards_in_memory_uses_region_crs_label(monkeypatch):
    """build_shards_in_memory must construct per-tile dir names with the REGION's CRS
    label (e.g. EPSG25832), NOT the Singapore EPSG3414 default. RED-ON-DIVERGENCE:
    reverting to ``tile_dirname(ti, tj)`` (defaulted) makes the path carry EPSG3414 and
    fails the EPSG25832 assertion. Locks the multi-region read fix caught by the Task-8
    small-before-big build on Leonardo (2026-06-10)."""
    monkeypatch.setattr(
        BS,
        "_validated_inventory",
        lambda release, region: [{"tile_i": 5, "tile_j": 9, "provenance_sha256": "x"}],
    )
    monkeypatch.setattr(BS, "epsg_label_for_region", lambda region: "EPSG25832")

    captured: dict[str, str] = {}

    class _Stop(Exception):
        pass

    def fake_read_tile_labels(tile_dir, *, tile_i, tile_j):
        captured["dir"] = str(tile_dir)
        raise _Stop  # short-circuit before the (absent) real parquet read

    monkeypatch.setattr(BS, "read_tile_labels", fake_read_tile_labels)

    with pytest.raises(_Stop):
        BS.build_shards_in_memory(_RELEASE, "munich", tile_ids=[(5, 9)])

    assert "tile=EPSG25832_i5_j9" in captured["dir"], captured["dir"]
    assert "EPSG3414" not in captured["dir"]


# =========================================================================== #
# TOOTH 3 — datamodule union spans BOTH regions
# =========================================================================== #
def _write_city_manifest(tmp, region, tiles):
    import yaml

    manifest = {
        "manifest_schema_version": "1.0",
        "release": _RELEASE,
        "region": region,
        "n_training_tiles": len(tiles),
        "tiles": [
            {"tile_i": ti, "tile_j": tj, "lineage": [[region, ti, tj]]} for (ti, tj) in tiles
        ],
    }
    p = tmp / f"{region}_training_manifest.yaml"
    p.write_text(yaml.safe_dump(manifest))
    return p


def _write_union_holdout(tmp):
    """A schema-2.0 holdout whose held-out cities are disjoint from the train cities
    under test — so the union audit passes (train tiles never match holdout tiles)."""
    import yaml

    holdout = {
        "manifest_schema_version": "2.0",
        "held_out_cities": ["munich"],
        "regions": {"munich": {"tiles": [{"tile_i": 9, "tile_j": 9}]}},
    }
    p = tmp / "union_holdout.yaml"
    p.write_text(yaml.safe_dump(holdout))
    return p


def test_datamodule_union_spans_both_regions(tmp_path, monkeypatch):
    """Two per-city manifests union into one example set spanning BOTH regions."""
    prague_mf = _write_city_manifest(tmp_path, "prague", [(0, 0), (0, 1)])
    barca_mf = _write_city_manifest(tmp_path, "barcelona", [(0, 0)])
    holdout = _write_union_holdout(tmp_path)

    # Stub the per-region build so the union is data-free. Each region's shards carry
    # that region as provenance (region field), so a union that spans both regions
    # yields examples from both.
    def fake_in_memory(release, region, *, tile_ids=None):
        return [_shard(region, ti, tj, [_cell(0, 0, 6)]) for (ti, tj) in (tile_ids or [])]

    monkeypatch.setattr(DM, "build_shards_in_memory", fake_in_memory)

    dm = DM.CellDataModule(
        training_manifests=[prague_mf, barca_mf],
        holdout_manifest=holdout,
        seed=7,
        val_fraction=0.0,  # keep all examples in train so the span check is exhaustive
    )
    dm.setup("fit")
    regions = {ex.region for ex in dm.train_cells + dm.val_cells}
    assert regions == {"prague", "barcelona"}, f"union must span both regions, got {regions}"


def test_single_region_path_still_works(tmp_path, monkeypatch):
    """The legacy single-region constructor (one training_manifest) must keep working
    unchanged — Singapore path is locked."""
    import yaml

    sg_mf = tmp_path / "sg_training_manifest.yaml"
    sg_mf.write_text(
        yaml.safe_dump(
            {
                "manifest_schema_version": "1.0",
                "release": _RELEASE,
                "region": "singapore",
                "tiles": [{"tile_i": 2, "tile_j": 2, "lineage": [["singapore", 2, 2]]}],
            }
        )
    )
    holdout = tmp_path / "holdout.yaml"
    holdout.write_text(
        yaml.safe_dump(
            {
                "manifest_schema_version": "2.0",
                "regions": {"singapore": {"tiles": [{"tile_i": 1, "tile_j": 7}]}},
            }
        )
    )
    monkeypatch.setattr(
        DM,
        "build_shards_in_memory",
        lambda *a, **k: [_shard("singapore", 2, 2, [_cell(0, 0, 5)])],
    )
    dm = DM.CellDataModule(training_manifest=sg_mf, holdout_manifest=holdout, seed=7)
    dm.setup("fit")
    assert len(dm.train_cells) + len(dm.val_cells) > 0
    assert {ex.region for ex in dm.train_cells + dm.val_cells} == {"singapore"}


# =========================================================================== #
# TOOTH 4 — CRS-agnostic (no CRS on examples)
# =========================================================================== #
def test_union_examples_are_crs_agnostic_pure_token_ints(tmp_path, monkeypatch):
    """A city tagged a non-z32 CRS (EPSG:25833) loads through the union; the loaded
    examples are pure token ints with NO CRS attribute/field anywhere (CRS is baked
    out at encode time — the token layer is CRS-agnostic)."""
    city_mf = _write_city_manifest(tmp_path, "prague", [(0, 0)])  # prague is EPSG:25833
    holdout = _write_union_holdout(tmp_path)

    def fake_in_memory(release, region, *, tile_ids=None):
        return [_shard(region, ti, tj, [_cell(0, 0, 7)]) for (ti, tj) in (tile_ids or [])]

    monkeypatch.setattr(DM, "build_shards_in_memory", fake_in_memory)

    dm = DM.CellDataModule(
        training_manifests=[city_mf], holdout_manifest=holdout, seed=7, val_fraction=0.0
    )
    dm.setup("fit")
    examples = dm.train_cells + dm.val_cells
    assert examples, "non-vacuous"
    for ex in examples:
        # tokens are pure ints
        assert all(isinstance(t, int) for t in ex.tokens)
        assert all(isinstance(i, int) for i in ex.ids)
        # NO CRS attribute/field anywhere on the example
        fields = set(vars(ex)) if hasattr(ex, "__dict__") else set(ex.__dataclass_fields__)
        for name in fields:
            assert "crs" not in name.lower() and "epsg" not in name.lower(), (
                f"CRS leaked onto example field {name!r}"
            )
        for attr in dir(ex):
            assert "crs" not in attr.lower() and "epsg" not in attr.lower(), (
                f"CRS leaked onto example attr {attr!r}"
            )


# =========================================================================== #
# CELL-KEY COLLISION-FREE PROPERTY — the region-keyed key disambiguates the union
# =========================================================================== #
def test_cell_key_is_collision_free_across_regions_for_same_coords():
    """The 5-tuple key exists for ONE reason: two cities can share a
    ``(tile_i, tile_j)`` (and a cell within it), and the union must NOT collapse
    them into one key. Two ``CellExample``s identical in every coordinate but
    differing in ``region`` must therefore have DISTINCT keys.

    Reverting ``CellExample.key`` to the old 4-tuple (dropping ``region``) makes
    these two keys COLLIDE (``a.key == b.key``) — the exact bug the widening
    prevents — so this test goes RED on that revert."""
    shared = dict(
        tile_i=3,
        tile_j=4,
        cell_i=1,
        cell_j=2,
        prefix_ids=(1, 2, 3),
        tokens=(4, 5, 6),
        cell_density_bucket=1,
    )
    a = DM.CellExample(region="prague", **shared)
    b = DM.CellExample(region="barcelona", **shared)
    # Identical coordinates (the collision surface), only region differs.
    assert (a.tile_i, a.tile_j, a.cell_i, a.cell_j) == (b.tile_i, b.tile_j, b.cell_i, b.cell_j)
    assert a.region != b.region
    # The collision-free property: region-keyed keys are DISTINCT.
    assert a.key != b.key

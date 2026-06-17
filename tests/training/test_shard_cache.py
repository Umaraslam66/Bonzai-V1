"""W3: the shard-derivation cache (design: reports/2026-06-12-shard-cache-design-memo.md).

CHECKPOINT 1 — the build instrument: per-city parquet serialization of the
FULL locked TrainingShard format (provisioned fields and None-representability
included, never just the slice's subset), byte-determinism, the source-file
enumeration the cache key pins (the per-city manifest sha does NOT transitively
pin sub-C/sub-F bytes — the cache records its own per-file path/size/sha256),
and the no-marker-without-end-state-verification build: parquets are written,
RE-READ and compared, a seeded sample is RE-DERIVED FROM SOURCE and compared,
and only then is the cache manifest sealed (locked_yaml grammar, the fourth
instance) and ``_SHARD_CACHE_VALID`` written.

CHECKPOINT 2 (read/verify path) tests live below the marker comment once the
build instrument is reviewed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cfm.data.training.shard_schema import CellPayload, TrainingShard

_REPO_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

_TILE_CONDITIONING = {
    "population_density_bucket": 0,
    "dominant_zoning_class": 0,
    "modal_road_skeleton_class": 1,
    "admin_region": None,
    "coastal_inland_river": 0,
    "sub_c_morphology_class": "Asian-megacity",
}


def _cell(ci, cj, tokens, density, contracts=(), stats=None):
    return CellPayload(
        cell_i=ci,
        cell_j=cj,
        cell_slot_index=ci * 8 + cj,
        tokens=tuple(tokens),
        cell_density_bucket=density,
        boundary_contracts=tuple(contracts),
        character_stats=stats if stats is not None else (1.5, 0.25, 0.0, 1.0, 0.875, 1.0, 0.0),
    )


def _full_format_shards() -> list[TrainingShard]:
    """Deliberately exercises the FULL locked format: non-empty macro_tokens and
    boundary_contracts (provisioned for bake-off candidates 1/2), a None
    cell_density_bucket, a None lineage (G-F4: must survive as a genuine None),
    and exact float character_stats (incl. a clip-boundary -3.0)."""
    return [
        TrainingShard(
            region="cityx",
            tile_i=0,
            tile_j=0,
            tile_conditioning=dict(_TILE_CONDITIONING),
            macro_tokens=(7, 8, 9),
            cells=(
                _cell(0, 0, [1, 2, 3], 2, contracts=(11, 12)),
                _cell(0, 1, [4], None, stats=(-3.0, 0.0, 0.0, 0.30102999566398, 0.0, 1.0, 0.0)),
            ),
            lineage=frozenset({("cityx", 0, 0)}),
        ),
        TrainingShard(
            region="cityx",
            tile_i=0,
            tile_j=1,
            tile_conditioning={**_TILE_CONDITIONING, "dominant_zoning_class": 3},
            macro_tokens=(),
            cells=(_cell(2, 2, [5, 6], 1),),
            lineage=None,
        ),
    ]


# --- serialization fidelity ------------------------------------------------------------


def test_city_cache_round_trips_the_full_locked_format(tmp_path: Path):
    from cfm.data.training.shard_cache import read_city_cache, write_city_cache

    shards = _full_format_shards()
    write_city_cache(shards, tmp_path)
    assert read_city_cache(tmp_path) == shards


def test_city_cache_bytes_are_deterministic(tmp_path: Path):
    from cfm.data.training.shard_cache import write_city_cache

    d1, d2 = tmp_path / "one", tmp_path / "two"
    write_city_cache(_full_format_shards(), d1)
    write_city_cache(_full_format_shards(), d2)
    for name in ("cells.parquet", "tiles.parquet"):
        assert (d1 / name).read_bytes() == (d2 / name).read_bytes(), name


def test_city_cache_records_match_the_written_bytes(tmp_path: Path):
    from cfm.data.determinism import compute_sha256
    from cfm.data.training.shard_cache import write_city_cache

    records = write_city_cache(_full_format_shards(), tmp_path)
    assert set(records) == {"cells.parquet", "tiles.parquet"}
    for name, rec in records.items():
        raw = (tmp_path / name).read_bytes()
        assert rec["size"] == len(raw)
        assert rec["sha256"] == compute_sha256(raw)


# --- the source-file enumeration the key pins -------------------------------------------


def test_source_files_for_city_enumerates_all_four_per_tile(tmp_path: Path, monkeypatch):
    import cfm.data.training.shard_cache as sc

    # fake the three subsystem layouts the real walk reads (one tile)
    monkeypatch.setattr(sc, "epsg_label_for_region", lambda region: "EPSG9999")
    for sub, fname in (
        ("sub_d", "macro_core.parquet"),
        ("sub_d", "effective_conditioning.yaml"),
        ("sub_f", "cells.parquet"),
        ("sub_c", "features.parquet"),
    ):
        p = tmp_path / sub / "tile=EPSG9999_i3_j4" / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"content-of-{fname}")
    monkeypatch.setattr(sc, "sub_d_region_dir", lambda release, region: tmp_path / "sub_d")
    monkeypatch.setattr(sc, "sub_f_region_dir", lambda release, region: tmp_path / "sub_f")
    monkeypatch.setattr(sc, "sub_c_region_dir", lambda release, region: tmp_path / "sub_c")

    records = sc.source_file_records("r", "cityx", [(3, 4)], data_root=tmp_path)
    names = sorted(Path(r["path"]).name for r in records)
    assert names == sorted(
        ["macro_core.parquet", "effective_conditioning.yaml", "cells.parquet", "features.parquet"]
    )
    for r in records:
        full = tmp_path / r["path"]
        assert r["size"] == full.stat().st_size
        assert len(r["sha256"]) == 64


def test_source_file_records_refuse_a_missing_source(tmp_path: Path, monkeypatch):
    import cfm.data.training.shard_cache as sc

    monkeypatch.setattr(sc, "epsg_label_for_region", lambda region: "EPSG9999")
    monkeypatch.setattr(sc, "sub_d_region_dir", lambda release, region: tmp_path / "sub_d")
    monkeypatch.setattr(sc, "sub_f_region_dir", lambda release, region: tmp_path / "sub_f")
    monkeypatch.setattr(sc, "sub_c_region_dir", lambda release, region: tmp_path / "sub_c")
    with pytest.raises(sc.ShardCacheBuildError, match=r"macro_core\.parquet"):
        sc.source_file_records("r", "cityx", [(3, 4)], data_root=tmp_path)


# --- the build: no marker without end-state verification --------------------------------


def _fake_builder(shards_by_city):
    calls = []

    def fake(release, region, *, tile_ids=None):
        calls.append((release, region, tuple(tile_ids or ())))
        return list(shards_by_city[region])

    return fake, calls


def _patch_layout(monkeypatch, sc, tmp_path):
    """Minimal fake source layout: source_file_records returns a stable list."""
    monkeypatch.setattr(
        sc,
        "source_file_records",
        lambda release, city, tile_ids, data_root=None: [
            {"path": f"{city}/src.bin", "size": 3, "sha256": "ab" * 32}
        ],
    )


def test_build_seals_manifest_and_marker_only_after_verification(tmp_path: Path, monkeypatch):
    import cfm.data.training.shard_cache as sc

    fake, calls = _fake_builder({"cityx": _full_format_shards()})
    monkeypatch.setattr(sc, "build_shards_in_memory", fake)
    _patch_layout(monkeypatch, sc, tmp_path)

    root = tmp_path / "cache"
    sc.build_shard_cache(
        release="r",
        cities=["cityx"],
        tile_ids_by_city={"cityx": [(0, 0), (0, 1)]},
        cache_root=root,
        manifest_sha_by_city={"cityx": "cd" * 32},
        sample_cells_per_city=1,
    )
    assert (root / "r" / sc.SHARD_CACHE_LOCK_NAME).exists()
    data = yaml.safe_load((root / "r" / sc.SHARD_CACHE_MANIFEST_NAME).read_text())
    assert data["cache_schema_version"] == sc.SHARD_CACHE_SCHEMA_VERSION
    assert data["derivation_version"] == sc.SHARD_CACHE_DERIVATION_VERSION
    assert data["key"]["per_city"]["cityx"]["training_manifest_sha256"] == "cd" * 32
    # the build re-derived for the sample verification (>= 2 builder calls)
    assert len(calls) >= 2


def test_build_refuses_marker_when_reread_diverges(tmp_path: Path, monkeypatch):
    import cfm.data.training.shard_cache as sc

    fake, _ = _fake_builder({"cityx": _full_format_shards()})
    monkeypatch.setattr(sc, "build_shards_in_memory", fake)
    _patch_layout(monkeypatch, sc, tmp_path)
    # poison the RE-READ: the write succeeded but reading back yields different content
    broken = _full_format_shards()[:1]
    monkeypatch.setattr(sc, "read_city_cache", lambda city_dir: broken)

    root = tmp_path / "cache"
    with pytest.raises(sc.ShardCacheBuildError, match="re-read"):
        sc.build_shard_cache(
            release="r",
            cities=["cityx"],
            tile_ids_by_city={"cityx": [(0, 0), (0, 1)]},
            cache_root=root,
            manifest_sha_by_city={"cityx": "cd" * 32},
        )
    assert not (root / "r" / sc.SHARD_CACHE_LOCK_NAME).exists()
    assert not (root / "r" / sc.SHARD_CACHE_MANIFEST_NAME).exists()


def test_build_refuses_marker_when_sample_rederivation_diverges(tmp_path: Path, monkeypatch):
    import cfm.data.training.shard_cache as sc

    shards = _full_format_shards()
    calls = {"n": 0}

    def divergent_builder(release, region, *, tile_ids=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return list(shards)  # the full build
        # the verification re-derive returns DIFFERENT content (a wrong-derivation
        # regime the marker must never bless)
        mutated = shards[0]
        bad = TrainingShard(
            region=mutated.region,
            tile_i=mutated.tile_i,
            tile_j=mutated.tile_j,
            tile_conditioning=mutated.tile_conditioning,
            macro_tokens=mutated.macro_tokens,
            cells=tuple(
                CellPayload(
                    cell_i=c.cell_i,
                    cell_j=c.cell_j,
                    cell_slot_index=c.cell_slot_index,
                    tokens=c.tokens,
                    cell_density_bucket=c.cell_density_bucket,
                    boundary_contracts=c.boundary_contracts,
                    character_stats=tuple(x + 1.0 for x in c.character_stats),
                )
                for c in mutated.cells
            ),
            lineage=mutated.lineage,
        )
        return [bad, *shards[1:]]

    monkeypatch.setattr(sc, "build_shards_in_memory", divergent_builder)
    _patch_layout(monkeypatch, sc, tmp_path)

    root = tmp_path / "cache"
    with pytest.raises(sc.ShardCacheBuildError, match="re-deriv"):
        sc.build_shard_cache(
            release="r",
            cities=["cityx"],
            tile_ids_by_city={"cityx": [(0, 0), (0, 1)]},
            cache_root=root,
            manifest_sha_by_city={"cityx": "cd" * 32},
            sample_cells_per_city=1,
        )
    assert not (root / "r" / sc.SHARD_CACHE_LOCK_NAME).exists()


# --- the golden pin: derivation/schema version travels with the bytes -------------------


def test_golden_fixture_pins_versions_to_the_cache_bytes(tmp_path: Path):
    """Lock-and-guards travel together: changing the serialization OR the
    derivation-relevant format without bumping the version constants goes RED
    here (and bumping the constants without re-goldening goes red too)."""
    from cfm.data.determinism import compute_sha256
    from cfm.data.training.shard_cache import (
        SHARD_CACHE_DERIVATION_VERSION,
        SHARD_CACHE_SCHEMA_VERSION,
        write_city_cache,
    )

    write_city_cache(_full_format_shards(), tmp_path)
    golden = {
        "schema": SHARD_CACHE_SCHEMA_VERSION,
        "derivation": SHARD_CACHE_DERIVATION_VERSION,
        "cells_sha256": compute_sha256((tmp_path / "cells.parquet").read_bytes()),
        "tiles_sha256": compute_sha256((tmp_path / "tiles.parquet").read_bytes()),
    }
    assert golden == {
        "schema": "1.0",
        "derivation": "1",
        "cells_sha256": "626c28684049c50df12f38e4ef798caef3f6d9aba1350fd1a269940a4d4c3c50",
        "tiles_sha256": "34dd8b8967e934f72a59bbb78a6ae0f0b0774679232a18dae7d717d7c05f05bc",
    }


# ======================================================================================
# CHECKPOINT 2 — the read/verify path (load_verified_shard_cache)
# ======================================================================================


def _real_source(tmp_path: Path, content: bytes = b"abc") -> dict:
    """One REAL on-disk source file so tier-(b) verification verifies something."""
    from cfm.data.determinism import compute_sha256

    f = tmp_path / "data" / "cityx" / "src.bin"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(content)
    return {"path": "cityx/src.bin", "size": len(content), "sha256": compute_sha256(content)}


def _built_cache(tmp_path: Path, monkeypatch, *, source_content: bytes = b"abc") -> Path:
    import cfm.data.training.shard_cache as sc

    rec = _real_source(tmp_path, source_content)
    monkeypatch.setattr(
        sc, "source_file_records", lambda release, city, tile_ids, data_root=None: [rec]
    )
    fake, _ = _fake_builder({"cityx": _full_format_shards()})
    monkeypatch.setattr(sc, "build_shards_in_memory", fake)
    root = tmp_path / "cache"
    sc.build_shard_cache(
        release="r",
        cities=["cityx"],
        tile_ids_by_city={"cityx": [(0, 0), (0, 1)]},
        cache_root=root,
        manifest_sha_by_city={"cityx": "cd" * 32},
        sample_cells_per_city=1,
        data_root=tmp_path / "data",
    )
    return root


def _load(tmp_path: Path, root: Path, **overrides):
    from cfm.data.training.shard_cache import load_verified_shard_cache

    kwargs = dict(
        release="r",
        cities=["cityx"],
        training_manifest_sha_by_city={"cityx": "cd" * 32},
        data_root=tmp_path / "data",
    )
    kwargs.update(overrides)
    return load_verified_shard_cache(root, **kwargs)


def test_load_verified_returns_the_cached_shards(tmp_path: Path, monkeypatch):
    root = _built_cache(tmp_path, monkeypatch)
    assert _load(tmp_path, root) == {"cityx": _full_format_shards()}


def test_load_refuses_missing_marker(tmp_path: Path, monkeypatch):
    from cfm.data.training.shard_cache import SHARD_CACHE_LOCK_NAME, ShardCacheStale

    root = _built_cache(tmp_path, monkeypatch)
    (root / "r" / SHARD_CACHE_LOCK_NAME).unlink()
    with pytest.raises(ShardCacheStale, match="_SHARD_CACHE_VALID"):
        _load(tmp_path, root)


def test_load_refuses_tampered_manifest(tmp_path: Path, monkeypatch):
    from cfm.data.training.shard_cache import SHARD_CACHE_MANIFEST_NAME, ShardCacheStale

    root = _built_cache(tmp_path, monkeypatch)
    m = root / "r" / SHARD_CACHE_MANIFEST_NAME
    m.write_text(m.read_text().replace("cityx", "cityz"), encoding="utf-8")
    with pytest.raises(ShardCacheStale, match="sha mismatch"):
        _load(tmp_path, root)


def test_load_refuses_derivation_version_skew(tmp_path: Path, monkeypatch):
    import cfm.data.training.shard_cache as sc

    root = _built_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(sc, "SHARD_CACHE_DERIVATION_VERSION", "999")
    with pytest.raises(sc.ShardCacheStale, match="derivation_version"):
        _load(tmp_path, root)


def test_load_refuses_conditioning_scheme_skew(tmp_path: Path, monkeypatch):
    import cfm.data.training.shard_cache as sc

    root = _built_cache(tmp_path, monkeypatch)
    monkeypatch.setattr(sc, "SHARD_CACHE_CONDITIONING_SCHEME", "value-char-v2")
    with pytest.raises(sc.ShardCacheStale, match="conditioning_scheme"):
        _load(tmp_path, root)


def test_load_refuses_city_set_mismatch(tmp_path: Path, monkeypatch):
    from cfm.data.training.shard_cache import ShardCacheStale

    root = _built_cache(tmp_path, monkeypatch)
    with pytest.raises(ShardCacheStale, match="cities"):
        _load(
            tmp_path,
            root,
            cities=["cityx", "cityy"],
            training_manifest_sha_by_city={"cityx": "cd" * 32, "cityy": "ee" * 32},
        )


def test_load_refuses_training_manifest_sha_change(tmp_path: Path, monkeypatch):
    from cfm.data.training.shard_cache import ShardCacheStale

    root = _built_cache(tmp_path, monkeypatch)
    with pytest.raises(ShardCacheStale, match="training_manifest"):
        _load(tmp_path, root, training_manifest_sha_by_city={"cityx": "ee" * 32})


def test_load_refuses_missing_source_file(tmp_path: Path, monkeypatch):
    from cfm.data.training.shard_cache import ShardCacheStale

    root = _built_cache(tmp_path, monkeypatch)
    (tmp_path / "data" / "cityx" / "src.bin").unlink()
    with pytest.raises(ShardCacheStale, match="source file"):
        _load(tmp_path, root)


def test_load_refuses_changed_small_source_file(tmp_path: Path, monkeypatch):
    # tier (b'): the seeded sample re-hashes the file (a single-file fixture is
    # always sampled), catching a same-size content flip
    from cfm.data.training.shard_cache import ShardCacheStale

    root = _built_cache(tmp_path, monkeypatch)
    (tmp_path / "data" / "cityx" / "src.bin").write_bytes(b"zzz")  # same size, new bytes
    with pytest.raises(ShardCacheStale, match="source file"):
        _load(tmp_path, root)


def test_load_refuses_changed_big_source_file_via_sample(tmp_path: Path, monkeypatch):
    # tier (b'): size-checked ALL + seeded-sample re-hashed; with one file the
    # sample always covers it, so a same-size content flip must be caught
    from cfm.data.training.shard_cache import ShardCacheStale

    big = b"a" * 70_000
    root = _built_cache(tmp_path, monkeypatch, source_content=big)
    (tmp_path / "data" / "cityx" / "src.bin").write_bytes(b"b" + big[1:])
    with pytest.raises(ShardCacheStale, match="source file"):
        _load(tmp_path, root)


def test_load_refuses_tampered_cache_parquet(tmp_path: Path, monkeypatch):
    from cfm.data.training.shard_cache import ShardCacheStale

    root = _built_cache(tmp_path, monkeypatch)
    cells = root / "r" / "cityx" / "cells.parquet"
    cells.write_bytes(cells.read_bytes() + b"x")
    with pytest.raises(ShardCacheStale, match="cache file"):
        _load(tmp_path, root)


def test_load_refuses_release_mismatch_in_payload(tmp_path: Path, monkeypatch):
    import shutil

    from cfm.data.training.shard_cache import ShardCacheStale

    root = _built_cache(tmp_path, monkeypatch)
    shutil.move(str(root / "r"), str(root / "r2"))  # dir renamed; payload still says "r"
    with pytest.raises(ShardCacheStale, match="release"):
        _load(tmp_path, root, release="r2")


# --- the CLI thread (scored runs reach the cache through config, one source) -----------


def test_shard_cache_cli_flag_threads_to_config():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_scaffold_shard_cache", _REPO_SCRIPTS / "train_scaffold.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # --region supplied (now REQUIRED); this test's subject is the --shard-cache flag
    args = mod._build_parser().parse_args(
        ["--region", "singapore", "--shard-cache", "/x/cache", "--devices", "1"]
    )
    cfg = mod.build_config_from_args(args)
    assert cfg.shard_cache == "/x/cache"
    assert mod._build_parser().parse_args([]).shard_cache is None

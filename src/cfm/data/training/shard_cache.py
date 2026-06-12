"""The shard-derivation cache (W3; design: reports/2026-06-12-shard-cache-design-memo.md).

Every training-job start pays the ~40-min ``build_shards_in_memory`` features
walk on ALL DDP ranks (~2.7 GPU-h of idle A100s per start). This module caches
that layer — and ONLY that layer: the cache is deliberately upstream of every
per-run knob, so the holdout audit, the budget drop, prefix/seed/ablation, the
train/val split, and both wiring guards all stay read-time and the cache is
invariant to the 13,312 budget lock.

Layout (under ``$WORK``, never ``$SCRATCH``):

    <cache_root>/<release>/<city>/cells.parquet   one row per cell (full payload)
    <cache_root>/<release>/<city>/tiles.parquet   one row per tile (full shard head)
    <cache_root>/<release>/cache_manifest.yaml    the sealed union manifest
    <cache_root>/<release>/_SHARD_CACHE_VALID     marker — ONLY after verification

The manifest is the FOURTH locked-YAML instance (W2's ``locked_yaml`` grammar:
``cache_sha256`` over the canonical YAML excluding itself, sealed write-once).
Its key covers every input that determines shard content — release, sorted city
list, per-city training-manifest sha, per-tile (path, size, sha256) of EVERY
source file the walk reads (the manifest sha does NOT transitively pin
sub-C/sub-F bytes, so the cache records its own), the derivation version, the
conditioning scheme, and the cache format version. A stale cache must HALT
naming the differing component, never silently feed a wrong-derivation run.

NO MARKER WITHOUT END-STATE VERIFICATION: the build writes the parquets,
RE-READS them and compares against the in-memory derivation, RE-DERIVES a
seeded sample of cells per city FROM SOURCE and compares exactly — only then is
the manifest sealed and ``_SHARD_CACHE_VALID`` written.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from cfm.data.determinism import compute_sha256
from cfm.data.io import canonicalize_yaml
from cfm.data.locked_yaml import stamp_and_seal
from cfm.data.training.build_shards import build_shards_in_memory
from cfm.data.training.conditioning import CHARACTER_STAT_CHANNELS
from cfm.data.training.paths import (
    epsg_label_for_region,
    sub_c_region_dir,
    sub_d_region_dir,
    sub_f_region_dir,
    tile_dirname,
)
from cfm.data.training.shard_schema import CellPayload, TrainingShard

logger = logging.getLogger(__name__)

#: Cache FORMAT version (the parquet/manifest serialization itself).
SHARD_CACHE_SCHEMA_VERSION: str = "1.0"

#: DERIVATION version: bump whenever shard-derivation CODE changes meaning —
#: character_stats_for_cell (channels, _CHAR_LOG10_CLIP, presence flags),
#: _tile_conditioning_dict's field mapping, _cell_density_by_cell, or the
#: CellPayload/TrainingShard schema. The golden-fixture test pins this constant
#: TO the cache bytes, so a silent un-bumped derivation change goes red.
SHARD_CACHE_DERIVATION_VERSION: str = "1"

#: The conditioning scheme the builder produces (Task 24b carrier). Recorded in
#: the key: a cache built under one scheme must never feed a run expecting another.
SHARD_CACHE_CONDITIONING_SCHEME: str = "value-char-v1"

SHARD_CACHE_MANIFEST_NAME: str = "cache_manifest.yaml"
SHARD_CACHE_LOCK_NAME: str = "_SHARD_CACHE_VALID"

#: The four files the per-tile walk reads (the cache key pins each by content).
_TILE_SOURCE_FILES: tuple[tuple[str, str], ...] = (
    ("sub_d", "macro_core.parquet"),
    ("sub_d", "effective_conditioning.yaml"),
    ("sub_f", "cells.parquet"),
    ("sub_c", "features.parquet"),
)

#: Seed for the sample re-derivation pick (fixed: the verification is part of
#: the reproducible build record, never wall-clock-random).
_SAMPLE_SEED: int = 7


def default_cache_root() -> Path:
    """The default cache location: data/processed/training_cache (gitignored,
    on $WORK on Leonardo — never $SCRATCH, whose cleanup would silently
    re-introduce the 40-min walk mid-campaign)."""
    from cfm.data.training.paths import _data_processed

    return _data_processed() / "training_cache"


def default_cache_data_root() -> Path:
    """Cache-key paths are stored RELATIVE to the repo root, so the sealed
    manifest is portable across machines (Mac dev / Leonardo $WORK).
    Private import SANCTIONED: _repo_root is the one-source path anchor."""
    from cfm.eval.holdout.paths import _repo_root

    return _repo_root()


class ShardCacheBuildError(RuntimeError):
    """The cache build failed verification — no manifest, no marker (a false
    _SHARD_CACHE_VALID would poison every later training run)."""


class ShardCacheStale(RuntimeError):
    """A cache component diverged from the live inputs (named in the message);
    fail-closed — rebuild deliberately via scripts/build_shard_cache.py."""


# --------------------------------------------------------------------------- #
# Serialization: the FULL locked TrainingShard format
# --------------------------------------------------------------------------- #

_CELLS_SCHEMA = pa.schema(
    [
        ("tile_i", pa.int64()),
        ("tile_j", pa.int64()),
        ("cell_i", pa.int64()),
        ("cell_j", pa.int64()),
        ("cell_slot_index", pa.int64()),
        ("tokens", pa.list_(pa.int64())),
        ("cell_density_bucket", pa.int64()),  # nullable: None survives as null
        ("boundary_contracts", pa.list_(pa.int64())),
        ("character_stats", pa.list_(pa.float64(), CHARACTER_STAT_CHANNELS)),
    ]
)

_TILES_SCHEMA = pa.schema(
    [
        ("region", pa.string()),
        ("tile_i", pa.int64()),
        ("tile_j", pa.int64()),
        # canonical-YAML string columns: schema-stable under conditioning-field
        # growth, exact round-trip for the int/str/None values they carry
        ("tile_conditioning_yaml", pa.string()),
        ("macro_tokens", pa.list_(pa.int64())),
        ("lineage_yaml", pa.string()),  # nullable: lineage=None survives as null
    ]
)


def write_city_cache(shards: list[TrainingShard], city_dir: Path) -> dict[str, dict]:
    """Serialize one city's shards (cells.parquet + tiles.parquet, byte-
    deterministic) and return ``{filename: {size, sha256}}`` for the manifest."""
    city_dir.mkdir(parents=True, exist_ok=True)
    tiles = {
        "region": [s.region for s in shards],
        "tile_i": [s.tile_i for s in shards],
        "tile_j": [s.tile_j for s in shards],
        "tile_conditioning_yaml": [canonicalize_yaml(s.tile_conditioning) for s in shards],
        "macro_tokens": [list(s.macro_tokens) for s in shards],
        "lineage_yaml": [
            canonicalize_yaml({"lineage": sorted([list(r) for r in s.lineage])})
            if s.lineage is not None
            else None
            for s in shards
        ],
    }
    cells: dict[str, list] = {name: [] for name in _CELLS_SCHEMA.names}
    for s in shards:
        for c in s.cells:
            cells["tile_i"].append(s.tile_i)
            cells["tile_j"].append(s.tile_j)
            cells["cell_i"].append(c.cell_i)
            cells["cell_j"].append(c.cell_j)
            cells["cell_slot_index"].append(c.cell_slot_index)
            cells["tokens"].append(list(c.tokens))
            cells["cell_density_bucket"].append(c.cell_density_bucket)
            cells["boundary_contracts"].append(list(c.boundary_contracts))
            cells["character_stats"].append(list(c.character_stats))
    records: dict[str, dict] = {}
    for name, table in (
        ("tiles.parquet", pa.table(tiles, schema=_TILES_SCHEMA)),
        ("cells.parquet", pa.table(cells, schema=_CELLS_SCHEMA)),
    ):
        path = city_dir / name
        pq.write_table(table, path)
        raw = path.read_bytes()
        records[name] = {"size": len(raw), "sha256": compute_sha256(raw)}
    return records


def read_city_cache(city_dir: Path) -> list[TrainingShard]:
    """Pure deserialization (NO verification — ``load_verified_shard_cache`` is
    the only sanctioned consumer path; this is its internal/build-time read)."""
    return _parse_city_cache(
        (city_dir / "tiles.parquet").read_bytes(), (city_dir / "cells.parquet").read_bytes()
    )


def _parse_city_cache(tiles_bytes: bytes, cells_bytes: bytes) -> list[TrainingShard]:
    """Parse the two cache tables from BYTES (the verified read hashes and parses
    the same buffer — one IO pass, nothing unverified reaches deserialization)."""
    tiles = pq.ParquetFile(pa.BufferReader(tiles_bytes)).read()
    cells = pq.ParquetFile(pa.BufferReader(cells_bytes)).read()
    cells_by_tile: dict[tuple[int, int], list[CellPayload]] = {}
    c = {n: cells.column(n).to_pylist() for n in cells.column_names}
    for i in range(cells.num_rows):
        payload = CellPayload(
            cell_i=int(c["cell_i"][i]),
            cell_j=int(c["cell_j"][i]),
            cell_slot_index=int(c["cell_slot_index"][i]),
            tokens=tuple(int(t) for t in c["tokens"][i]),
            cell_density_bucket=(
                None if c["cell_density_bucket"][i] is None else int(c["cell_density_bucket"][i])
            ),
            boundary_contracts=tuple(int(b) for b in c["boundary_contracts"][i]),
            character_stats=tuple(float(x) for x in c["character_stats"][i]),
        )
        cells_by_tile.setdefault((int(c["tile_i"][i]), int(c["tile_j"][i])), []).append(payload)
    t = {n: tiles.column(n).to_pylist() for n in tiles.column_names}
    shards: list[TrainingShard] = []
    for i in range(tiles.num_rows):
        ti, tj = int(t["tile_i"][i]), int(t["tile_j"][i])
        lineage_yaml = t["lineage_yaml"][i]
        shards.append(
            TrainingShard(
                region=t["region"][i],
                tile_i=ti,
                tile_j=tj,
                tile_conditioning=yaml.safe_load(t["tile_conditioning_yaml"][i]),
                macro_tokens=tuple(int(m) for m in t["macro_tokens"][i]),
                cells=tuple(cells_by_tile.get((ti, tj), [])),
                lineage=(
                    frozenset(tuple(r) for r in yaml.safe_load(lineage_yaml)["lineage"])
                    if lineage_yaml is not None
                    else None
                ),
            )
        )
    return shards


# --------------------------------------------------------------------------- #
# The cache key: every input that determines shard content
# --------------------------------------------------------------------------- #


def source_file_records(
    release: str,
    city: str,
    tile_ids: list[tuple[int, int]],
    *,
    data_root: Path | None = None,
) -> list[dict]:
    """(path, size, sha256) for EVERY file the per-tile walk reads — the
    manifest sha does not transitively pin sub-C/sub-F bytes, so the cache key
    records its own. Refuses a missing source loudly (a cache key over absent
    files would seal a walk that cannot have happened)."""
    roots = {
        "sub_d": sub_d_region_dir(release, city),
        "sub_f": sub_f_region_dir(release, city),
        "sub_c": sub_c_region_dir(release, city),
    }
    epsg = epsg_label_for_region(city)
    records: list[dict] = []
    for ti, tj in sorted(tile_ids):
        dirname = tile_dirname(ti, tj, epsg)
        for sub, fname in _TILE_SOURCE_FILES:
            f = roots[sub] / dirname / fname
            if not f.exists():
                raise ShardCacheBuildError(
                    f"source file missing for {city} tile ({ti},{tj}): {f} — refusing "
                    f"to key a cache over an absent input."
                )
            raw = f.read_bytes()
            path_str = str(f.relative_to(data_root)) if data_root is not None else str(f)
            records.append({"path": path_str, "size": len(raw), "sha256": compute_sha256(raw)})
    return records


# --------------------------------------------------------------------------- #
# The build: write -> re-read -> sample re-derive -> seal
# --------------------------------------------------------------------------- #


def build_shard_cache(
    release: str,
    cities: list[str],
    *,
    tile_ids_by_city: dict[str, list[tuple[int, int]]],
    cache_root: Path,
    manifest_sha_by_city: dict[str, str],
    sample_cells_per_city: int = 8,
    data_root: Path | None = None,
) -> Path:
    """Build, verify, and seal the union cache. Returns the sealed manifest path.

    Verification BEFORE any marker: (1) each city's parquets are re-read and
    compared exactly against the in-memory derivation; (2) a seeded sample of
    cells per city is re-derived FROM SOURCE (a second ``build_shards_in_memory``
    over the sampled tiles) and compared exactly against the cached bytes'
    content. Any divergence: ShardCacheBuildError, no manifest, no marker."""
    release_dir = cache_root / release
    per_city: dict[str, dict] = {}
    for city in sorted(cities):
        tile_ids = sorted(tile_ids_by_city[city])
        shards = build_shards_in_memory(release, city, tile_ids=tile_ids)
        records = write_city_cache(shards, release_dir / city)
        # (1) re-read: the bytes on disk must reconstruct EXACTLY what was derived
        reread = read_city_cache(release_dir / city)
        if reread != shards:
            raise ShardCacheBuildError(
                f"cache re-read mismatch for {city}: the serialized cache does not "
                f"reconstruct the derived shards; refusing to seal (no marker)."
            )
        # (2) seeded sample re-derivation FROM SOURCE
        _verify_sample_rederivation(
            release, city, shards, sample_cells_per_city=sample_cells_per_city
        )
        per_city[city] = {
            "training_manifest_sha256": manifest_sha_by_city[city],
            "n_tiles": len(shards),
            "n_cells": sum(len(s.cells) for s in shards),
            "source_files": source_file_records(release, city, tile_ids, data_root=data_root),
            "cache_files": records,
        }
    payload = {
        "cache_schema_version": SHARD_CACHE_SCHEMA_VERSION,
        "derivation_version": SHARD_CACHE_DERIVATION_VERSION,
        "conditioning_scheme": SHARD_CACHE_CONDITIONING_SCHEME,
        "release": release,
        "cities": sorted(cities),
        "key": {"per_city": per_city},
    }
    manifest_path = release_dir / SHARD_CACHE_MANIFEST_NAME
    stamp_and_seal(
        payload, manifest_path, sha_field="cache_sha256", lock_name=SHARD_CACHE_LOCK_NAME
    )
    logger.info("shard cache sealed: %s (%d cities)", manifest_path, len(cities))
    return manifest_path


def _verify_sample_rederivation(
    release: str,
    city: str,
    shards: list[TrainingShard],
    *,
    sample_cells_per_city: int,
) -> None:
    """Re-derive a seeded sample of tiles from SOURCE and compare their cells
    exactly against the just-built shards (the end-state check the marker
    depends on; a wrong-derivation cache must never get sealed)."""
    if sample_cells_per_city <= 0 or not shards:
        return
    all_cells = [(s.tile_i, s.tile_j) for s in shards for _ in s.cells]
    if not all_cells:
        return
    rng = random.Random(_SAMPLE_SEED)
    picked = rng.sample(all_cells, min(sample_cells_per_city, len(all_cells)))
    sample_tiles = sorted(set(picked))
    rederived = {
        (s.tile_i, s.tile_j): s
        for s in build_shards_in_memory(release, city, tile_ids=sample_tiles)
    }
    by_tile = {(s.tile_i, s.tile_j): s for s in shards}
    for tile in sample_tiles:
        if tile not in rederived or rederived[tile] != by_tile[tile]:
            raise ShardCacheBuildError(
                f"sample re-derivation mismatch for {city} tile {tile}: the cached "
                f"derivation does not match a fresh from-source re-derivation; "
                f"refusing to seal (no marker)."
            )


# --------------------------------------------------------------------------- #
# The verified read: the ONLY sanctioned consumer path
# --------------------------------------------------------------------------- #

#: Informational small-file threshold (reporting only; the read tier does not
#: branch on it — see the tier-(b') note on _verify_source_files).
SMALL_SOURCE_FILE_BYTES: int = 65_536

#: Source files FULLY re-hashed per city at read (seeded sample over ALL files).
SAMPLE_SOURCE_FILES_PER_CITY: int = 8


def load_verified_shard_cache(
    cache_root: Path,
    *,
    release: str,
    cities: list[str],
    training_manifest_sha_by_city: dict[str, str],
    data_root: Path | None = None,
) -> dict[str, list[TrainingShard]]:
    """Verified cache read -> ``{city: shards}`` — materializes the WHOLE union
    (>25 GB in Python objects for the 38-city corpus; OOMed twice on serial
    nodes, jobs 46065304/46068302). Fine for tests/small sets; the datamodule
    and any union-scale consumer MUST use ``iter_verified_shard_cache`` and
    release each city after consuming it (the walk path's memory profile)."""
    return dict(
        iter_verified_shard_cache(
            cache_root,
            release=release,
            cities=cities,
            training_manifest_sha_by_city=training_manifest_sha_by_city,
            data_root=data_root,
        )
    )


def iter_verified_shard_cache(
    cache_root: Path,
    *,
    release: str,
    cities: list[str],
    training_manifest_sha_by_city: dict[str, str],
    data_root: Path | None = None,
) -> Iterator[tuple[str, list[TrainingShard]]]:
    """Verified STREAMING cache read: yields ``(city, shards)`` in sorted-city
    order, one city materialized at a time (the consumer releases each before
    the next — matching the walk path's peak-memory profile).

    Verification is identical to the dict form: the sealed manifest + every
    GLOBAL component (derivation/scheme/release/cities) verify EAGERLY before
    the first yield; per-city components (training-manifest sha, tier-(b')
    source files, full cache-parquet integrity) verify as each city is read.
    ANY mismatch raises ``ShardCacheStale`` NAMING the differing component —
    fail-closed, NO silent fallback to the walk (a fallback would mask exactly
    the staleness this exists to catch). Rebuild deliberately via
    scripts/build_shard_cache.py."""
    from cfm.data.locked_yaml import verify_sealed_yaml

    release_dir = Path(cache_root) / release
    payload = verify_sealed_yaml(
        release_dir / SHARD_CACHE_MANIFEST_NAME,
        sha_field="cache_sha256",
        lock_name=SHARD_CACHE_LOCK_NAME,
        schema_field="cache_schema_version",
        schema_version=SHARD_CACHE_SCHEMA_VERSION,
        required_key="key",
        error=ShardCacheStale,
    )
    if payload["derivation_version"] != SHARD_CACHE_DERIVATION_VERSION:
        raise ShardCacheStale(
            f"shard cache STALE (component: derivation_version): cache was built at "
            f"derivation_version={payload['derivation_version']!r} but this code is "
            f"{SHARD_CACHE_DERIVATION_VERSION!r} — the derivation changed; rebuild."
        )
    if payload["conditioning_scheme"] != SHARD_CACHE_CONDITIONING_SCHEME:
        raise ShardCacheStale(
            f"shard cache STALE (component: conditioning_scheme): cache carries "
            f"{payload['conditioning_scheme']!r}, this code expects "
            f"{SHARD_CACHE_CONDITIONING_SCHEME!r}; rebuild."
        )
    if payload["release"] != release:
        raise ShardCacheStale(
            f"shard cache STALE (component: release): cache manifest says "
            f"{payload['release']!r} but the run requests {release!r}."
        )
    if payload["cities"] != sorted(cities):
        raise ShardCacheStale(
            f"shard cache STALE (component: cities): cache covers {payload['cities']} "
            f"but the run requests {sorted(cities)}; rebuild for the requested union."
        )
    for city in sorted(cities):
        entry = payload["key"]["per_city"][city]
        live_sha = training_manifest_sha_by_city[city]
        if entry["training_manifest_sha256"] != live_sha:
            raise ShardCacheStale(
                f"shard cache STALE (component: training_manifest, city {city}): cache "
                f"keyed on manifest sha {entry['training_manifest_sha256'][:12]}… but the "
                f"live manifest hashes {live_sha[:12]}…; the tile inventory moved; rebuild."
            )
        _verify_source_files(city, entry["source_files"], data_root=data_root)
        yield city, _read_verified_city(release_dir / city, entry["cache_files"])


def _verify_source_files(city: str, records: list[dict], *, data_root: Path | None) -> None:
    """Tier (b'): stat (existence + size) on ALL records + a seeded sample of
    SAMPLE_SOURCE_FILES_PER_CITY files FULLY re-hashed.

    TIER MEASURED, NOT ASSUMED (2026-06-12, real 38-city union on Leonardo):
    tier (a) full re-hash = 26.8 min (jobs 46050613) — per-file open latency
    (~18 ms on Lustre) dominates, not bytes; the original tier (b) "fully
    re-hash all small files" would open 71,355 of 88,076 files (~20 min) and
    was REJECTED by the same measurement. The stat walk costs 59 s for all
    88,076 files (job 46055141) — tier (b') totals ~1 min per job start.

    Residual, stated honestly: a same-size single-file content edit is caught
    only when sampled. Every realistic staleness event (a regen under the
    uniform-defect rule) moves many files, sizes, and the per-city manifests —
    all caught by the all-files stat + manifest shas. The deliberate-rebuild
    discipline is the primary defense; the sample is the tripwire."""
    paths: list[tuple[Path, dict]] = []
    for rec in records:
        f = (Path(data_root) / rec["path"]) if data_root is not None else Path(rec["path"])
        if not f.exists():
            raise ShardCacheStale(
                f"shard cache STALE (component: source file, city {city}): {f} is "
                f"missing — a keyed input vanished; rebuild (or restore the source)."
            )
        size = f.stat().st_size
        if size != rec["size"]:
            raise ShardCacheStale(
                f"shard cache STALE (component: source file, city {city}): {f} is "
                f"{size} bytes, cache keyed {rec['size']}; the input changed; rebuild."
            )
        paths.append((f, rec))
    if paths:
        rng = random.Random(_SAMPLE_SEED)
        for f, rec in rng.sample(paths, min(SAMPLE_SOURCE_FILES_PER_CITY, len(paths))):
            if compute_sha256(f.read_bytes()) != rec["sha256"]:
                raise ShardCacheStale(
                    f"shard cache STALE (component: source file, city {city}): {f} "
                    f"content changed since the cache was keyed (sampled re-hash); rebuild."
                )


def _read_verified_city(city_dir: Path, cache_records: dict[str, dict]) -> list[TrainingShard]:
    """Full-integrity city read: ONE IO pass — bytes are read, hashed against the
    sealed manifest, and parsed from the same buffer (a tampered/corrupt cache
    parquet can never reach deserialization unverified)."""
    raw: dict[str, bytes] = {}
    for name, rec in cache_records.items():
        f = city_dir / name
        if not f.exists():
            raise ShardCacheStale(
                f"shard cache STALE (component: cache file): {f} is missing; rebuild."
            )
        data = f.read_bytes()
        if len(data) != rec["size"] or compute_sha256(data) != rec["sha256"]:
            raise ShardCacheStale(
                f"shard cache STALE (component: cache file): {f} does not match the "
                f"sealed manifest (size/sha); the cache bytes were modified; rebuild."
            )
        raw[name] = data
    return _parse_city_cache(raw["tiles.parquet"], raw["cells.parquet"])

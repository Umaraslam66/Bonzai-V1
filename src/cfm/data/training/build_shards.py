"""build_training_shards — materialize per-tile training shards (spec §6 trigger 1, §10.1).

Training set = the sub-D-validated tiles MINUS the frozen 132 holdout tiles, BY
TILE ID from the frozen manifest (single source — no "recompute which tiles are
holdout" path). Each shard's lineage is STAMPED from the tile's recorded
provenance (read, never synthesized from a path) so a missing lineage reaches the
holdout audit as a genuine None (G-F4 fail-closed).

The tile inventory is the sub-D manifest's ``tiles[]`` (authoritative one source,
matching ``cfm.eval.holdout.pipeline._load_inventory``); labels come from the
sub-D tile dir, tokens from the sub-F tile dir (``read_sub_f_cells``).

DECISION (slice v1): ``macro_tokens`` and per-cell ``boundary_contracts`` are
provisioned-but-empty here — the locked FORMAT carries them (shard_schema), but
the cell-unit slice does not read them. Populating them is bake-off-prep work:
macro_tokens from the sub-D macro plan tokenization (candidate 1), boundary
contracts from sub-E ``boundary_contract.parquet`` (candidate 2). Provisioning the
fields (not the values) is what §10.1 requires; under-provisioning the format is
the fatal write-once direction, and the format is not under-provisioned.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from cfm.data.io import canonicalize_yaml
from cfm.data.sub_d.enums import SlotKind
from cfm.data.sub_d.io import read_macro_core_parquet
from cfm.data.sub_g.readers import read_sub_f_cells
from cfm.data.training.atomic_io import atomic_write_text
from cfm.data.training.paths import (
    epsg_label_for_region,
    holdout_manifest_for_region,
    sub_d_region_dir,
    sub_f_region_dir,
    tile_dirname,
    training_manifest_path,
    training_region_dir,
)
from cfm.data.training.shard_schema import CellPayload, TrainingShard
from cfm.eval.holdout.labels import TileLabels, read_tile_labels

# THE one source for the Task-8 G4 corpus-DoD roll-up path (repo-relative).
# Every consumer (train_scaffold._g4_rollup_path, build_multiregion_train_shards
# _DEFAULT_G4, the bakeoff_run.sbatch union preamble) derives from THIS constant —
# a drifted copy would let the preamble verify a different city set than training
# consumes, and the guard would pass while guarding nothing.
DEFAULT_G4_ROLLUP: str = "reports/2026-06-05-phase-2-g4-corpus-dod.yaml"


def _validated_inventory(release: str, region: str) -> list[dict]:
    """The validated-tile inventory from the sub-D manifest (authoritative, one
    source — same read as cfm.eval.holdout.pipeline._load_inventory)."""
    md = yaml.safe_load(
        (sub_d_region_dir(release, region) / "manifest.yaml").read_text(encoding="utf-8")
    )
    return md["tiles"]


def _holdout_ids(release: str, region: str) -> set[tuple[int, int]]:
    """SINGLE SOURCE: the frozen holdout manifest, by tile ID. No re-derivation.

    REGION-AWARE (obligation (a), delta-spec §3 CORRECTION): ``region`` selects the
    manifest — ``singapore`` -> SG manifest (schema 1.0); the 4 EU held-out cities ->
    multiregion manifest (schema 2.0); unknown region -> raise. Both manifests share the
    ``regions[<region>]["tiles"][{tile_i, tile_j}]`` nesting, so the read is identical."""
    m = yaml.safe_load(holdout_manifest_for_region(release, region).read_text(encoding="utf-8"))
    return {(int(t["tile_i"]), int(t["tile_j"])) for t in m["regions"][region]["tiles"]}


def compute_training_tile_ids(release: str, region: str) -> list[tuple[int, int]]:
    """validated minus holdout, by ID. Sorted for deterministic build order.

    SINGLE-REGION path (Singapore + the held-out cities). For a multi-region TRAIN
    city this RAISES via ``_holdout_ids`` (the city is neither singapore nor a held-out
    city) — that is the I1 fail-closed boundary. The multi-region driver
    (``build_train_city_shards``) DELIBERATELY does not call this for a train city; it
    builds with ALL validated tiles instead (whole-city exclusion already removed the
    held-out cities, so a train city has no tile-level holdout)."""
    holdout = _holdout_ids(release, region)
    ids = [
        (int(e["tile_i"]), int(e["tile_j"]))
        for e in _validated_inventory(release, region)
        if (int(e["tile_i"]), int(e["tile_j"])) not in holdout
    ]
    return sorted(ids)


def _load_or_pass(obj: dict | Path | str) -> dict:
    """Accept a parsed mapping OR a path to a YAML file (injectable for tests)."""
    if isinstance(obj, dict):
        return obj
    return yaml.safe_load(Path(obj).read_text(encoding="utf-8"))


def train_cities(
    release: str,
    *,
    g4_rollup: dict | Path | str,
    holdout_manifest: dict | Path | str,
) -> list[str]:
    """The multi-region TRAIN-city names: validated cities MINUS the held-out cities.

    Parse the G4 roll-up's ``per_city`` list, keep ``validated: true`` names, and
    EXCLUDE the held-out cities read from ``holdout_manifest["held_out_cities"]``.

    The 4 held-out cities are present in the roll-up with ``validated: true`` (they
    passed validation; they are simply reserved for evaluation), so this exclusion is
    ACTIVE — the held-out set is removed here, BY CONSTRUCTION, before any shard is
    built. Unvalidated cities are dropped too (they are not corpus-eligible).

    ``g4_rollup`` / ``holdout_manifest`` accept a parsed dict OR a path so the real
    on-disk artifacts (Task 8 / Leonardo) and synthetic fixtures share one code path.
    Result is sorted for deterministic build order. The ``release`` arg is carried for
    call-site symmetry with the rest of the builder API (the roll-up is already
    release-scoped by the caller)."""
    rollup = _load_or_pass(g4_rollup)
    held = set(_load_or_pass(holdout_manifest).get("held_out_cities", []))
    names = [c["name"] for c in rollup["per_city"] if c.get("validated") and c["name"] not in held]
    return sorted(names)


def verify_union_manifests(
    release: str,
    *,
    g4_rollup: dict | Path | str,
    holdout_manifest: dict | Path | str,
) -> list[str]:
    """Fail-loud gate for the eu-train-union consumers: resolve the train cities and
    verify EVERY per-city training manifest exists on disk; return the sorted cities.

    The bakeoff_run.sbatch preamble calls this before burning node-hours — the SAME
    resolution (``train_cities`` over the SAME roll-up constant) the training process
    consumes, so the guard cannot drift from what it guards. RAISES (never ``assert``,
    which vanishes under ``python -O``):

      - ``ValueError`` naming ``held_out_cities`` if the holdout manifest lacks the key
        (fail-closed: ``train_cities`` itself uses ``.get(..., [])`` and would silently
        exclude NOTHING — the same strict-read contract as the scaffold's
        ``_union_datamodule``);
      - ``ValueError`` listing ALL missing per-city manifests if any are absent (this
        path NEVER rebuilds; manifests are built only by the Task-8 driver).
    """
    holdout = _load_or_pass(holdout_manifest)
    if "held_out_cities" not in holdout:
        raise ValueError(
            "holdout manifest has no 'held_out_cities' key; refusing to verify a "
            "training union with an empty exclusion set"
        )
    cities = train_cities(release, g4_rollup=g4_rollup, holdout_manifest=holdout)
    missing = [c for c in cities if not training_manifest_path(release, c).exists()]
    if missing:
        raise ValueError(
            "missing per-city training manifests (run the Task-8 build "
            "(scripts/build_multiregion_train_shards.py) first; this path never "
            "rebuilds): " + ", ".join(missing)
        )
    return cities  # train_cities already returns sorted


def build_train_city_shards(release: str, city: str) -> list[TrainingShard]:
    """Build one TRAIN city's shards from ALL its validated tiles — the I1-boundary
    bypass (handled AT THE LOOP, never by touching ``_holdout_ids``).

    A train city has NO tile-level holdout: whole-city exclusion (``train_cities``)
    already removed the 4 held-out cities, so every validated tile of a train city is
    a training tile. Building with ``tile_ids=<all validated ids>`` routes straight to
    ``build_shards_in_memory`` and DELIBERATELY skips ``compute_training_tile_ids`` /
    ``_holdout_ids`` — which would otherwise RAISE ``ValueError`` for a train city
    (the Task-1 fail-closed guarantee, left intact as the backstop)."""
    tile_ids = sorted(
        (int(e["tile_i"]), int(e["tile_j"])) for e in _validated_inventory(release, city)
    )
    return build_shards_in_memory(release, city, tile_ids=tile_ids)


def build_train_city_manifest(
    release: str, city: str, *, out_dir: Path | None = None
) -> list[TrainingShard]:
    """I1-SAFE writing build for a TRAIN city: build from ALL validated tiles AND write
    the byte-deterministic per-city ``training_manifest.yaml``.

    The persistence sibling of ``build_train_city_shards`` (which is in-memory only) and
    the train-city counterpart of ``build_training_shards``. Unlike the latter — which
    routes through ``compute_training_tile_ids`` -> ``_holdout_ids`` and therefore RAISES
    ``ValueError`` for a train city (the I1 fail-closed boundary: a train city is neither
    ``singapore`` nor a held-out city) — this builds via ``build_train_city_shards`` (all
    validated tiles, no tile-level holdout, since whole-city exclusion already removed the
    held-out cities). Task 8's multi-region driver calls THIS per train city; the locked
    single-region Singapore/held-out writer ``build_training_shards`` stays untouched."""
    out = out_dir or training_region_dir(release, city)
    out.mkdir(parents=True, exist_ok=True)
    prov_by_id = {
        (int(e["tile_i"]), int(e["tile_j"])): e["provenance_sha256"]
        for e in _validated_inventory(release, city)
    }
    shards = build_train_city_shards(release, city)
    _write_training_manifest(out, release, city, shards, prov_by_id)
    return shards


def build_multiregion_shards(release: str, cities: list[str]) -> list[TrainingShard]:
    """Build the UNION of shards across all train cities (per-city all-validated-tiles
    build, concatenated). Cities are built in sorted order for deterministic output.

    Runs the Task-24a city-identity guard over the union: the per-city carried
    ``TrainingShard.region`` values feed the ``city_identity`` conditioning field,
    so an all-None city or one value shared across distinct cities is a wiring bug
    that must halt HERE, before any prefix is built."""
    shards: list[TrainingShard] = []
    carried_by_city: dict[str, list[str | None]] = {}
    for city in sorted(cities):
        city_shards = build_train_city_shards(release, city)
        carried_by_city[city] = [s.region for s in city_shards]
        shards.extend(city_shards)
    guard_city_identity(carried_by_city)
    return shards


class CityIdentityError(RuntimeError):
    """The city_identity conditioning wiring is broken (Task 24a guard).

    Two regimes, both wiring bugs that would otherwise train silently wrong:
      * ALL-NONE: a city's shards all carry ``region=None`` -> the city_identity
        field constant-buckets to 0 for the whole city (city-blind while claiming
        identity conditioning);
      * CONSTANT-ACROSS-CITIES: the same city value carried by >=2 DISTINCT
        requested cities (the #22 constant-column class) -> the field cannot
        discriminate the cities it exists to separate.
    """


def guard_city_identity(carried_by_source: dict[str, list[str | None]]) -> None:
    """Task-24a wiring guard over ``{requested_city: [carried city_identity values]}``.

    The carried values are the per-shard ``TrainingShard.region`` (the city name the
    conditioning's ``city_identity`` field will encode). Empty value lists (no shards
    built for a source) are vacuous — never an all-None false positive. The same
    source key appearing with its own value repeatedly is healthy; only a value
    shared across DISTINCT source keys fires the constant-across-cities regime.
    """
    sources_by_value: dict[str, set[str]] = {}
    for source, values in carried_by_source.items():
        if values and all(v is None for v in values):
            raise CityIdentityError(
                f"region {source!r}: city_identity is all-None across its "
                f"{len(values)} shards — the conditioning city field would silently "
                f"constant-bucket to 0 for the whole city (wiring bug; Task 24a)."
            )
        for v in values:
            if v is not None:
                sources_by_value.setdefault(v, set()).add(source)
    for value, sources in sources_by_value.items():
        if len(sources) >= 2:
            raise CityIdentityError(
                f"city_identity value {value!r} is carried by {len(sources)} distinct "
                f"regions ({sorted(sources)}) — constant-across-cities is a wiring bug "
                f"(the field cannot discriminate the cities it exists to separate; "
                f"Task 24a / the #22 constant-column class)."
            )


def _cell_density_by_cell(sub_d_tile_dir: Path) -> dict[tuple[int, int], int]:
    """(cell_i, cell_j) -> cell_density_bucket (same derivation as
    cfm.eval.holdout.pipeline._cell_density_by_cell)."""
    rows = read_macro_core_parquet(sub_d_tile_dir / "macro_core.parquet")
    return {
        (int(r.cell_i), int(r.cell_j)): int(r.cell_density_bucket)
        for r in rows
        if r.slot_kind == SlotKind.CELL and r.cell_density_bucket is not None
    }


def _tile_conditioning_dict(labels: TileLabels) -> dict:
    """Tile-level conditioning (the locked schema's tile fields). Per-cell
    cell_density lives on CellPayload; the run seed is applied at training time."""
    return {
        "population_density_bucket": labels.population_density_bucket,
        "dominant_zoning_class": labels.morphology_stratum.dominant_zoning_class,
        "modal_road_skeleton_class": labels.morphology_stratum.modal_road_skeleton_class,
        # "admin_region", NOT "region": the city name lives on TrainingShard.region;
        # this is the admin DIVISION (None for every EU tile). Sharing the key would
        # let conditioning wiring silently grab the wrong value (F6 trap).
        "admin_region": labels.admin_region,
        "coastal_inland_river": labels.coastal_inland_river,
        "sub_c_morphology_class": labels.sub_c_morphology_class,
    }


def build_shards_in_memory(
    release: str,
    region: str,
    *,
    tile_ids: list[tuple[int, int]] | None = None,
) -> list[TrainingShard]:
    """Build the in-memory shards (full tile structure) WITHOUT writing anything.

    The DataModule reads cell tokens from here (the persisted manifest carries only
    lineage/provenance, not tokens). ``tile_ids`` restricts the build to a subset
    (sorted) for fast tests; default is the full validated-minus-holdout set."""
    sub_d_dir = sub_d_region_dir(release, region)
    sub_f_dir = sub_f_region_dir(release, region)
    ids = sorted(tile_ids) if tile_ids is not None else compute_training_tile_ids(release, region)

    # Per-tile dir names embed the REGION's CRS label (e.g. EPSG25834), NOT the Singapore
    # EPSG3414 default — derive it once per region. Without this, an EU region's tiles
    # resolve to a Singapore-named dir that does not exist (multi-region read bug, caught
    # by the Task-8 small-before-big build on Leonardo 2026-06-10).
    epsg_label = epsg_label_for_region(region)

    shards: list[TrainingShard] = []
    for ti, tj in ids:  # sorted -> deterministic
        dirname = tile_dirname(ti, tj, epsg_label)
        labels = read_tile_labels(sub_d_dir / dirname, tile_i=ti, tile_j=tj)
        density = _cell_density_by_cell(sub_d_dir / dirname)
        tokens_by_cell = read_sub_f_cells(sub_f_dir / dirname / "cells.parquet")
        cells = tuple(
            CellPayload(
                cell_i=ci,
                cell_j=cj,
                cell_slot_index=ci * 8 + cj,
                tokens=tuple(toks),
                cell_density_bucket=density.get((ci, cj)),
                boundary_contracts=(),  # provisioned-empty (slice unread; see docstring)
            )
            for (ci, cj), toks in sorted(tokens_by_cell.items())
        )
        shards.append(
            TrainingShard(
                region=region,
                tile_i=ti,
                tile_j=tj,
                tile_conditioning=_tile_conditioning_dict(labels),
                macro_tokens=(),  # provisioned-empty (slice does not read; see module docstring)
                cells=cells,
                lineage=frozenset({(region, ti, tj)}),  # STAMPED from provenance, points at self
            )
        )
    return shards


def build_training_shards(
    release: str, region: str, *, out_dir: Path | None = None
) -> list[TrainingShard]:
    """Build the in-memory shards (full tile structure) and write a
    byte-deterministic training_manifest.yaml carrying per-tile stamped lineage."""
    out = out_dir or training_region_dir(release, region)
    out.mkdir(parents=True, exist_ok=True)
    prov_by_id = {
        (int(e["tile_i"]), int(e["tile_j"])): e["provenance_sha256"]
        for e in _validated_inventory(release, region)
    }
    shards = build_shards_in_memory(release, region)
    _write_training_manifest(out, release, region, shards, prov_by_id)
    return shards


def _write_training_manifest(
    out: Path,
    release: str,
    region: str,
    shards: list[TrainingShard],
    prov_by_id: dict[tuple[int, int], str],
) -> None:
    """Byte-deterministic manifest: the lineage-bearing artifact the DataModule
    reads (so the holdout audit reads STAMPED lineage, never synthesized)."""
    manifest = {
        "manifest_schema_version": "1.0",
        "release": release,
        "region": region,
        "n_training_tiles": len(shards),
        "tiles": [
            {
                "tile_i": s.tile_i,
                "tile_j": s.tile_j,
                "provenance_sha256": prov_by_id[(s.tile_i, s.tile_j)],
                "lineage": sorted([list(ref) for ref in s.lineage]),
            }
            for s in shards  # already sorted by (ti, tj)
        ],
    }
    path = out / training_manifest_path(release, region).name
    atomic_write_text(path, canonicalize_yaml(manifest))  # crash-safe (F17)

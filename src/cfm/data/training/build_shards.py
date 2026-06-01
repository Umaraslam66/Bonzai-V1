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
from cfm.data.training.paths import (
    holdout_manifest_path,
    sub_d_region_dir,
    sub_f_region_dir,
    tile_dirname,
    training_manifest_path,
    training_region_dir,
)
from cfm.data.training.shard_schema import CellPayload, TrainingShard
from cfm.eval.holdout.labels import TileLabels, read_tile_labels


def _validated_inventory(release: str, region: str) -> list[dict]:
    """The validated-tile inventory from the sub-D manifest (authoritative, one
    source — same read as cfm.eval.holdout.pipeline._load_inventory)."""
    md = yaml.safe_load(
        (sub_d_region_dir(release, region) / "manifest.yaml").read_text(encoding="utf-8")
    )
    return md["tiles"]


def _holdout_ids(release: str, region: str) -> set[tuple[int, int]]:
    """SINGLE SOURCE: the frozen holdout manifest, by tile ID. No re-derivation."""
    m = yaml.safe_load(holdout_manifest_path(release).read_text(encoding="utf-8"))
    return {(int(t["tile_i"]), int(t["tile_j"])) for t in m["regions"][region]["tiles"]}


def compute_training_tile_ids(release: str, region: str) -> list[tuple[int, int]]:
    """validated minus holdout, by ID. Sorted for deterministic build order."""
    holdout = _holdout_ids(release, region)
    ids = [
        (int(e["tile_i"]), int(e["tile_j"]))
        for e in _validated_inventory(release, region)
        if (int(e["tile_i"]), int(e["tile_j"])) not in holdout
    ]
    return sorted(ids)


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
        "region": labels.admin_region,
        "coastal_inland_river": labels.coastal_inland_river,
        "sub_c_morphology_class": labels.sub_c_morphology_class,
    }


def build_training_shards(
    release: str, region: str, *, out_dir: Path | None = None
) -> list[TrainingShard]:
    """Build the in-memory shards (full tile structure) and write a
    byte-deterministic training_manifest.yaml carrying per-tile stamped lineage."""
    out = out_dir or training_region_dir(release, region)
    out.mkdir(parents=True, exist_ok=True)
    sub_d_dir = sub_d_region_dir(release, region)
    sub_f_dir = sub_f_region_dir(release, region)
    prov_by_id = {
        (int(e["tile_i"]), int(e["tile_j"])): e["provenance_sha256"]
        for e in _validated_inventory(release, region)
    }

    shards: list[TrainingShard] = []
    for ti, tj in compute_training_tile_ids(release, region):  # sorted -> deterministic
        dirname = tile_dirname(ti, tj)
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
    path.write_text(canonicalize_yaml(manifest), encoding="utf-8")

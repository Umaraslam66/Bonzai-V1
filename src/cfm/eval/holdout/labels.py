"""Per-tile conditioning-label aggregation (spec §E), one-source over sub-C/sub-D.

This module READS already-derived labels; it never re-derives a density,
zoning, or skeleton determination (planning protocol v2 Gate 6 + §3).

NAMING COLLISION (verified 2026-06-01): the SCORED "morphology" dimension is
sub-D's road_skeleton_class + zoning_class (io.py:51-53), which vary across
Singapore. sub-C's field literally named ``morphology_class`` is the CONSTANT
string "Asian-megacity" (sub_c/conditioning.py) -> UNSCORED v1. We never call
the sub-D stratum "morphology"; it is ``morphology_stratum``. The sub-C constant
is carried verbatim and listed in ``UNSCORED_V1_DIMENSIONS``.

DENSITY is an AGGREGATE (verified evidence.py:308-337): tile_population_density
= p75 of the same per-cell building_footprint_ratio that cell_density_bucket
buckets. So ``population_density_bucket`` (tile) is the held-out-unit + §9
conditioning label, and ``cell_density_buckets`` (per-cell) is D's stratification
key - a tile mean would mask intra-tile spread.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from cfm.data.sub_d.enums import SlotKind
from cfm.data.sub_d.io import read_macro_core_parquet
from cfm.data.sub_d.macro_vocab import load_macro_vocab

#: v1 conditioning dimensions with no real Singapore variation - UNSCORED-stated,
#: never read as a met bar (spec §E + §5 unscored-not-passing).
UNSCORED_V1_DIMENSIONS: frozenset[str] = frozenset(
    {"region", "morphology_class", "coastal_inland_river"}
)


@dataclass(frozen=True)
class MorphologyStratum:
    """The SCORED morphology stratum = sub-D skeleton + zoning (io.py:51-53).

    Deliberately NOT named 'morphology' - that word is sub-C's constant field.
    """

    dominant_zoning_class: int | None
    modal_road_skeleton_class: int | None


@dataclass(frozen=True)
class TileLabels:
    tile_i: int
    tile_j: int
    population_density_bucket: int | None  # tile-level (conditioning yaml); held-out unit
    cell_density_buckets: tuple[int, ...]  # per active CELL; D's stratification key
    morphology_stratum: MorphologyStratum  # sub-D skeleton + zoning (SCORED)
    coastal_inland_river: int | None  # sub-C enum; UNSCORED (near-constant)
    admin_region: str | None  # sub-C; UNSCORED
    sub_c_morphology_class: str | None  # the constant; recorded, UNSCORED


def valid_cell_density_bucket_ids(vocab_path: Path) -> set[int]:
    """Ground-truth cell_density token_ids straight from the locked vocab."""
    vocab = load_macro_vocab(vocab_path)
    return {int(b["token_id"]) for b in vocab["locked_buckets"]["cell_density"]}


def read_tile_labels(tile_dir: Path, *, tile_i: int, tile_j: int) -> TileLabels:
    """Aggregate one tile's conditioning labels from sub-D artifacts on disk."""
    rows = read_macro_core_parquet(tile_dir / "macro_core.parquet")

    cell_density = tuple(
        int(r.cell_density_bucket)
        for r in rows
        if r.slot_kind == SlotKind.CELL and r.cell_density_bucket is not None
    )
    zoning = [
        int(r.zoning_class)
        for r in rows
        if r.slot_kind == SlotKind.CELL and r.zoning_class is not None
    ]
    skeleton = [
        int(r.road_skeleton_class)
        for r in rows
        if r.slot_kind == SlotKind.INTERNAL_EDGE and r.road_skeleton_class is not None
    ]
    morphology = MorphologyStratum(
        dominant_zoning_class=Counter(zoning).most_common(1)[0][0] if zoning else None,
        modal_road_skeleton_class=(Counter(skeleton).most_common(1)[0][0] if skeleton else None),
    )

    ec = yaml.safe_load((tile_dir / "effective_conditioning.yaml").read_text(encoding="utf-8"))
    cond = ec.get("conditioning", {})
    pdb = cond.get("population_density_bucket")

    return TileLabels(
        tile_i=int(tile_i),
        tile_j=int(tile_j),
        population_density_bucket=int(pdb) if pdb is not None else None,
        cell_density_buckets=cell_density,
        morphology_stratum=morphology,
        coastal_inland_river=cond.get("coastal_inland_river"),
        admin_region=cond.get("admin_region"),
        sub_c_morphology_class=cond.get("morphology_class"),
    )

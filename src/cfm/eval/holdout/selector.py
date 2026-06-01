"""Fresh per-stratum quota selector for the held-out set (spec §F selection).

This is NOT sub-D's select_layer3_subset (a tile-granularity, one-tile-per-
dimension reviewer-diversity picker, frequency_analysis.py:884). The eval needs
a per-stratum QUOTA sampler sized to power statistical floors - a different tool.
sub-D's #11 selector stays untouched on its reviewer-diversity path.

It CONSUMES sub-D-derived TileLabels (one source); it never re-derives a density
or morphology determination (no import of sub-D derivation modules; the
import-surface test enforces this).

#11's failure class was a SILENT under-pick of the sparse side (a sign error that
made a guard skip a dimension and report success anyway). The regime-distinguishing
guard here is the opposite: a tile-stratum that cannot fill its quota, or a
cell-density stratum whose selected cells fall below its floor, is SURFACED as
UNDERPOWERED (G's degradation policy, spec §G) - never silently omitted.

A tile stratum key is (population_density_bucket, (dominant_zoning_class,
modal_road_skeleton_class)). Cell-density floors are checked on the UNION of the
selected tiles' cells (D's stratification key - density is an aggregate).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from cfm.eval.holdout.labels import TileLabels

TileKey = tuple[int, int]
TileStratum = tuple[int, tuple[int | None, int | None]]


@dataclass(frozen=True)
class Shortfall:
    available: int
    floor: int


@dataclass
class SelectionResult:
    selected: list[TileKey]
    per_tile_stratum_counts: dict[TileStratum, int]
    underpowered_tile_strata: dict[TileStratum, Shortfall] = field(default_factory=dict)
    underpowered_cell_density_strata: dict[int, Shortfall] = field(default_factory=dict)


def _tile_stratum(tl: TileLabels) -> TileStratum:
    return (
        int(tl.population_density_bucket) if tl.population_density_bucket is not None else -1,
        (
            tl.morphology_stratum.dominant_zoning_class,
            tl.morphology_stratum.modal_road_skeleton_class,
        ),
    )


def select_holdout_tiles(
    tile_labels: list[TileLabels],
    quotas: dict[TileStratum, int],
    cell_density_floor: dict[int, int],
) -> SelectionResult:
    """Pick tiles to fill per-stratum quotas; surface every shortfall."""
    by_stratum: dict[TileStratum, list[TileLabels]] = defaultdict(list)
    for tl in tile_labels:
        by_stratum[_tile_stratum(tl)].append(tl)

    selected: list[TileKey] = []
    counts: dict[TileStratum, int] = {}
    underpowered_tiles: dict[TileStratum, Shortfall] = {}

    for stratum, quota in sorted(quotas.items()):
        pool = sorted(by_stratum.get(stratum, []), key=lambda tl: (tl.tile_i, tl.tile_j))
        take = pool[:quota]
        counts[stratum] = len(take)
        selected.extend((tl.tile_i, tl.tile_j) for tl in take)
        if len(take) < quota:
            underpowered_tiles[stratum] = Shortfall(available=len(take), floor=quota)

    # Cell-density coverage on the UNION of selected tiles' cells.
    selected_set = set(selected)
    cell_counts: dict[int, int] = defaultdict(int)
    for tl in tile_labels:
        if (tl.tile_i, tl.tile_j) in selected_set:
            for b in tl.cell_density_buckets:
                cell_counts[b] += 1
    underpowered_cells: dict[int, Shortfall] = {}
    for bucket, floor in sorted(cell_density_floor.items()):
        have = cell_counts.get(bucket, 0)
        if have < floor:
            underpowered_cells[bucket] = Shortfall(available=have, floor=floor)

    return SelectionResult(
        selected=sorted(selected),
        per_tile_stratum_counts=counts,
        underpowered_tile_strata=underpowered_tiles,
        underpowered_cell_density_strata=underpowered_cells,
    )

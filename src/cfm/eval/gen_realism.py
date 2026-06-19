"""Generated-side realism feature extraction (Phase-2 bake-off Task 11/[B]).

The decision layer (``bakeoff_decision.decide`` -> ``lane_s_excess``) scores per
``(metric, stratum)`` where ``stratum`` is the floor's frozen 4-tuple
``(zoning, road_skeleton, cell_density_bucket, coastal_inland_river)``. The real side is
keyed by ``extract_features_by_city_stratum_metric``; this module is its GENERATED-side
counterpart: it turns a trained backbone's decoded cells into ``gen_by_city`` GenFeatures
in the IDENTICAL grammar, so Lane-S can match gen against real per stratum.

GRAMMAR FIDELITY (the whole point — a mismatch silently makes Lane-S vacuous, every stratum
skipped-thin): the 4-tuple is derived from ``read_tile_labels`` — the SAME source the floor
used (sub-D ``io.py:51-53``) — NOT from the shard's ``tile_conditioning`` (which carries the
same dims but is a second encoding path that could drift). And feature classification reuses
``conditioning_discrimination._tile_features`` (ring promotion + outbound-bref exclusion) so
gen and real are classified by ONE rule. A generated cell inherits ONE density (the density
it was conditioned on), applied to every feature it emitted.
"""

from __future__ import annotations

from dataclasses import dataclass

# Private import SANCTIONED (the run_conditioning_floor._validated_inventory precedent):
# _tile_features is the ONE feature-classification rule (promote rings, building_area vs
# road_length, outbound-bref exclusion) the floor's real side uses. Reusing it means gen and
# real can never be classified by two drifting copies.
from cfm.eval.conditioning_discrimination import _tile_features
from cfm.eval.holdout.labels import read_tile_labels
from cfm.eval.holdout.paths import (
    epsg_label_for_region,
    sub_d_region_dir,
    tile_dirname,
)

#: {(metric, stratum) -> samples}; stratum is the floor's 4-tuple. Mirrors
#: ``bakeoff_decision.GenFeatures`` (kept structural to avoid a torch-pulling import).
GenFeatures = dict[tuple[str, tuple], list[float]]


@dataclass(frozen=True)
class DecodedCell:
    """One generated, decoded cell, with the conditioning context needed to stratum-key it.

    ``blocks``/``geoms`` are aligned (the kept decode results). ``cell_density_bucket`` is the
    density the cell was CONDITIONED on (the matched-conditioning real example's bucket); every
    feature the cell emitted is scored in that density."""

    city: str
    tile_i: int
    tile_j: int
    cell_density_bucket: int | None
    blocks: list[list[int]]
    geoms: list[dict]


def gen_features_by_city(cells: list[DecodedCell], *, release: str) -> dict[str, GenFeatures]:
    """Decoded generated cells -> ``{city: GenFeatures}`` keyed by the floor's 4-tuple grammar.

    For each cell, ``(zoning, road_skeleton, coastal)`` come from ``read_tile_labels`` of the
    cell's tile (cached per tile) and ``density`` is the cell's conditioned bucket; each feature
    is binned into ``(metric, (zoning, road_skeleton, density, coastal))`` — byte-for-byte the
    key shape ``extract_features_by_city_stratum_metric`` froze into the locked floor.
    """
    label_cache: dict[tuple[str, int, int], tuple] = {}
    by_city: dict[str, GenFeatures] = {}
    for cell in cells:
        ck = (cell.city, cell.tile_i, cell.tile_j)
        if ck not in label_cache:
            epsg = epsg_label_for_region(cell.city)
            tdir = sub_d_region_dir(release, cell.city) / tile_dirname(
                cell.tile_i, cell.tile_j, epsg
            )
            labels = read_tile_labels(tdir, tile_i=cell.tile_i, tile_j=cell.tile_j)
            label_cache[ck] = (
                labels.morphology_stratum.dominant_zoning_class,
                labels.morphology_stratum.modal_road_skeleton_class,
                labels.coastal_inland_river,
            )
        zoning, skeleton, coastal = label_cache[ck]
        density = cell.cell_density_bucket
        # One density per cell -> every feature in this cell gets it (the conditioned stratum).
        feats, _n_bref = _tile_features(cell.blocks, cell.geoms, [density] * len(cell.blocks))
        gf = by_city.setdefault(cell.city, {})
        for metric, value, dens in feats:
            gf.setdefault((metric, (zoning, skeleton, dens, coastal)), []).append(value)
    return by_city

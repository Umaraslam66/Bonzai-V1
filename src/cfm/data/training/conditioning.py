"""Trigger-2 one-source conditioning + the append-only conditioning id-block.

The model's conditioning and the eval's conditioning-compliance scoring read the
SAME derivation (``derive_tile_conditioning`` re-exports the factored core of
``cfm.eval.holdout.labels.read_tile_labels``). The identity is structural: a test
asserts ``derive_tile_conditioning is labels._derive_tile_conditioning``, which
FAILS the moment someone forks the derivation — not merely "equal values today".

Tier line (spec §5): the VALUES this produces are tier-1 (the locked conditioning
schema); the model-side ENCODING of those values into embedding inputs (Task 7) is
tier-2, OUTSIDE the trigger-2 compared surface.
"""

from __future__ import annotations

from cfm.data.sub_f.vocab import vocab_tag_to_id
from cfm.eval.holdout.labels import TileLabels, _derive_tile_conditioning

#: Trigger-2 single source. The model conditioning and the eval both resolve here.
derive_tile_conditioning = _derive_tile_conditioning

#: Conditioning id-block: appended STRICTLY above the sealed sub-F vocab so the
#: sub-F vocab lock is untouched (never reindexed).
CONDITIONING_ID_BASE: int = max(vocab_tag_to_id().values()) + 1

#: Append-only, recorded ordering (one source, read by both the shard build and
#: the model). A future conditioning dimension APPENDS at the end — never reindex.
_CONDITIONING_FIELDS: tuple[str, ...] = (
    "population_density_bucket",  # scored (tile p75 aggregate)
    "zoning_class",  # scored (morphology stratum)
    "road_skeleton_class",  # scored (morphology stratum)
    "cell_density_bucket",  # scored (per-cell scalar; trigger-2 granularity)
    "region",  # unscored-recorded (forward-compat: second region)
    "coastal_inland_river",  # unscored-recorded
    "sub_c_morphology_class",  # unscored-recorded constant
    "seed",  # PRD §8 deterministic seed
)


def conditioning_field_to_id() -> dict[str, int]:
    """The one-source field->id mapping (read by both shard build and model)."""
    return {f: CONDITIONING_ID_BASE + i for i, f in enumerate(_CONDITIONING_FIELDS)}


def conditioning_prefix_ids(
    labels: TileLabels, *, cell_density_bucket: int | None, seed: int
) -> list[object]:
    """The positional conditioning VALUES (tier-1) for one cell example.

    Returned in ``_CONDITIONING_FIELDS`` order. These are the tier-1 values; the
    model-side int-encoding into embedding inputs is Task 7 (tier-2). ``None`` in
    a slot means the label was absent for this tile.
    """
    return [
        labels.population_density_bucket,
        labels.morphology_stratum.dominant_zoning_class,
        labels.morphology_stratum.modal_road_skeleton_class,
        cell_density_bucket,
        labels.admin_region,
        labels.coastal_inland_river,
        labels.sub_c_morphology_class,
        seed,
    ]

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

import hashlib

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
    VALUE-BEARING model-side encoding into embedding ids is ``build_value_bearing_prefix``.
    ``None`` in a slot means the label was absent for this tile.
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


#: Number of conditioning POSITIONS in the prefix (one per field; append-only).
CONDITIONING_PREFIX_LEN: int = len(_CONDITIONING_FIELDS)

#: VALUE-BEARING layout. Each field reserves a block of ``_VALUE_STRIDE`` embedding ids
#: above the sealed sub-F vocab; a field's value maps to one id inside its block.
#: DECISION (bake-off Task 6, tier-1 conditioning schema, identical across all runs):
#:  * STRIDE=64 generously covers every current field's bucket cardinality (densities,
#:    zoning/skeleton enums, the two ~constant string fields) with headroom; revisit only
#:    if a field exceeds 64 buckets (it would APPEND a wider block, never reindex).
#:  * bucket 0 is reserved for None/absent so a missing label never collides with value 0.
#:  * the ``seed`` field is NOT value-embedded (a generation-sampling control, not a
#:    learnable category) -> it maps to its block's constant bucket 0, preserving the
#:    8-position layout without teaching the model per-seed embeddings.
#:  * string fields hash via SHA-256 (NOT builtin hash(), which is PYTHONHASHSEED-salted
#:    and would differ across cold processes -- determinism across runs is mandatory).
_VALUE_STRIDE: int = 64
CONDITIONING_VALUE_BASE: int = CONDITIONING_ID_BASE


def conditioning_id_span() -> int:
    """Embedding ids the value-bearing conditioning occupies ABOVE the sub-F vocab.

    The model's embedding table must span ``n_subf_vocab + conditioning_id_span()``
    (wired in Task 7). Distinct from ``CONDITIONING_PREFIX_LEN`` (the position count).
    """
    return CONDITIONING_PREFIX_LEN * _VALUE_STRIDE


def _value_bucket(value: object) -> int:
    """Map a tier-1 conditioning value to a bucket in ``[0, _VALUE_STRIDE)``.

    0 is reserved for None/absent. Ints fold by modulo; strings hash deterministically
    (SHA-256, never builtin ``hash()``). Both reserve bucket 0 for absence.
    """
    if value is None:
        return 0
    if isinstance(value, str):
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % (_VALUE_STRIDE - 1) + 1
    return int(value) % (_VALUE_STRIDE - 1) + 1


def build_value_bearing_prefix(
    *,
    population_density_bucket: int | None,
    zoning_class: int | None,
    road_skeleton_class: int | None,
    cell_density_bucket: int | None,
    region: str | None,
    coastal_inland_river: int | None,
    sub_c_morphology_class: str | None,
    seed: int,
) -> list[int]:
    """The VALUE-BEARING conditioning prefix: 8 embedding ids encoding the real values.

    Field order is ``_CONDITIONING_FIELDS`` (append-only). Each id lands in its field's
    reserved block above the sub-F vocab, so distinct conditioning -> distinct prefix.
    ``seed`` is intentionally constant-bucketed (not value-embedded; see _VALUE_STRIDE note).
    """
    # Field-order values; seed -> None so it constant-buckets to 0 (not value-embedded).
    values: list[object] = [
        population_density_bucket,
        zoning_class,
        road_skeleton_class,
        cell_density_bucket,
        region,
        coastal_inland_river,
        sub_c_morphology_class,
        None,  # seed slot: sampling control, not a learnable embedding
    ]
    return [
        CONDITIONING_VALUE_BASE + i * _VALUE_STRIDE + _value_bucket(v) for i, v in enumerate(values)
    ]

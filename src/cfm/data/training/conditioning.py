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
from functools import cache
from pathlib import Path

import yaml

from cfm.data.determinism import compute_sha256
from cfm.data.io import canonicalize_yaml
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
    "city_identity",  # Task 24a (spec §8): registry-encoded city name; Lane-S ablates it
)


# ----- Task 24a: sha-locked append-only city-identity registry (spec §8 Lane S/D) -----
#
# WHY a registry and not _value_bucket: the generic string hash (sha256 % 63 + 1)
# COLLIDES over the 49 known cities (11 colliding groups / 13 pairs, incl.
# madrid=rome, berlin=warsaw, manchester=tilburg=toledo) — identity must be
# injective, so city_identity_bucket = registry_index + 1 (bucket 0 stays reserved
# for None/ablation). The registry is SORTED ONCE at the 2026-06-11 freeze, then
# APPEND-ONLY FOREVER: future cities append at the end; never re-sort, never remove
# (ids must never move). 49 < 63 fits the 64-stride block with headroom; if ever
# exceeded, a wider block APPENDS (the existing stride rule), never reindexes.
#
# Lock grammar mirrors the Task-20 holdout-manifest discipline: ``registry_sha256``
# over the canonical YAML EXCLUDING itself + a ``_CITY_REGISTRY_LOCKED`` marker
# beside the file. The reader REFUSES on sha mismatch / missing sha / missing
# marker / malformed YAML / schema-version skew / stride overflow / unknown city —
# fail-loud, never silent bucket-0.

_REPO_ROOT = Path(__file__).resolve().parents[4]
CITY_REGISTRY_PATH: Path = _REPO_ROOT / "configs" / "training" / "city_identity_registry.yaml"
CITY_REGISTRY_LOCK_NAME: str = "_CITY_REGISTRY_LOCKED"
CITY_REGISTRY_SCHEMA_VERSION: str = "1.0"


class CityRegistryError(RuntimeError):
    """The city-identity registry failed verification, or a city name is unknown."""


def city_registry_sha256(data: dict) -> str:
    """SHA over the canonical registry EXCLUDING the registry_sha256 field itself
    (the same ``*_sha256`` exclusion grammar as the holdout manifest freeze)."""
    payload = {k: v for k, v in data.items() if k != "registry_sha256"}
    return compute_sha256(canonicalize_yaml(payload).encode("utf-8"))


def freeze_city_registry(cities: list[str], path: Path) -> None:
    """Stamp the registry sha, write ONCE, and seal with the lock marker beside it.

    Refuses to overwrite: the registry is append-only — appending a city means
    deliberately re-freezing (delete + re-write with the old prefix INTACT)."""
    if path.exists():
        raise FileExistsError(
            f"city-identity registry already locked at {path}; it is append-only — "
            f"re-freeze deliberately with the existing order intact, never overwrite."
        )
    data: dict = {
        "registry_schema_version": CITY_REGISTRY_SCHEMA_VERSION,
        "cities": list(cities),
    }
    data["registry_sha256"] = city_registry_sha256(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonicalize_yaml(data), encoding="utf-8")
    (path.parent / CITY_REGISTRY_LOCK_NAME).touch()


def _read_verified_registry(path: Path) -> tuple[str, ...]:
    """Verified read: lock marker beside the file, stored sha == recomputed sha."""
    marker = path.parent / CITY_REGISTRY_LOCK_NAME
    if not marker.exists():
        raise CityRegistryError(
            f"no {CITY_REGISTRY_LOCK_NAME} marker beside the city-identity registry "
            f"(expected {marker}); refusing to read an unsealed registry."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "cities" not in data:
        raise CityRegistryError(
            f"malformed registry at {path}: expected a YAML mapping with a 'cities' "
            f"key (got {type(data).__name__}); refusing (fail-closed)."
        )
    stored = data.get("registry_sha256")
    if stored is None:
        raise CityRegistryError(
            f"city-identity registry {path} carries NO registry_sha256 field — an "
            f"unstamped registry is unverifiable; refusing (fail-closed)."
        )
    recomputed = city_registry_sha256(data)
    if stored != recomputed:
        raise CityRegistryError(
            f"city-identity registry sha mismatch at {path}: stored "
            f"registry_sha256={stored!r} but recomputed {recomputed!r} — the content "
            f"was edited after the freeze; refusing (ids must never move)."
        )
    version = data.get("registry_schema_version")
    if version != CITY_REGISTRY_SCHEMA_VERSION:
        raise CityRegistryError(
            f"city-identity registry {path} declares "
            f"registry_schema_version={version!r} but this reader requires "
            f"{CITY_REGISTRY_SCHEMA_VERSION!r}; refusing a version-skewed registry."
        )
    cities = tuple(data["cities"])
    if len(set(cities)) != len(cities):
        raise CityRegistryError(f"city-identity registry {path} has duplicate entries.")
    if len(cities) >= _VALUE_STRIDE:
        raise CityRegistryError(
            f"city-identity registry {path} has {len(cities)} entries — bucket "
            f"index+1 would overflow the {_VALUE_STRIDE}-stride block; a wider block "
            f"must APPEND (never reindex) before more cities can register."
        )
    return cities


@cache
def _default_registry() -> tuple[str, ...]:
    return _read_verified_registry(CITY_REGISTRY_PATH)


def load_city_registry(path: Path | None = None) -> tuple[str, ...]:
    """The verified city list (lazy + cached for the committed default path)."""
    if path is None:
        return _default_registry()
    return _read_verified_registry(Path(path))


def city_identity_bucket(city: str | None) -> int:
    """``registry_index + 1`` for a known city; 0 for None (absent/ablated).

    Unknown city names REFUSE loud — silently encoding bucket 0 would train a
    city-blind model while claiming identity conditioning."""
    if city is None:
        return 0
    registry = load_city_registry()
    try:
        return registry.index(city) + 1
    except ValueError:
        raise CityRegistryError(
            f"city {city!r} is not in the append-only city-identity registry "
            f"({CITY_REGISTRY_PATH}); refusing to encode it (never silent bucket-0). "
            f"A new city must be APPENDED to the registry deliberately."
        ) from None


#: The conditioning ablation switch (spec §8): Lane S scores with the city-identity
#: block ABLATED ("no_city"); Lane D runs it live ("full"). "no_character" (Task 24b)
#: ablates the CONTINUOUS character carrier: the 9 id positions stay identical to
#: "full" here, and the datamodule zeroes the stats vector + presence flags data-side
#: BEFORE the model's projection (the all-zeros input is the learned-nothing signal,
#: the bucket-0 analog).
_ABLATION_MODES: tuple[str, ...] = ("full", "no_city", "no_character")


def conditioning_field_to_id() -> dict[str, int]:
    """The one-source field->id mapping (read by both shard build and model)."""
    return {f: CONDITIONING_ID_BASE + i for i, f in enumerate(_CONDITIONING_FIELDS)}


def conditioning_prefix_ids(
    labels: TileLabels, *, cell_density_bucket: int | None, seed: int, city_identity: str | None
) -> list[object]:
    """The positional conditioning VALUES (tier-1) for one cell example.

    Returned in ``_CONDITIONING_FIELDS`` order. These are the tier-1 values; the
    VALUE-BEARING model-side encoding into embedding ids is ``build_value_bearing_prefix``.
    ``None`` in a slot means the label was absent for this tile. ``city_identity``
    is the shard's CITY name (``TrainingShard.region``) — never the admin division.
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
        city_identity,
    ]


#: Number of conditioning ID POSITIONS in the prefix (one per field; append-only).
CONDITIONING_PREFIX_LEN: int = len(_CONDITIONING_FIELDS)

# ----- Task 24b: the continuous character-carrier prefix position (mini-spec §2) -----
#
# AXIS SEPARATION (the recurring n_cond conflation trap): these two constants live on
# the POSITION axis and the CHANNEL axis respectively — NEITHER is an embedding row.
# The carrier adds ONE position (the 10th, after the 9 id positions) whose input
# embedding is Linear(CHARACTER_STAT_CHANNELS -> d_model) of the cell's
# CellPayload.character_stats; the token-id span (conditioning_id_span() == 576 rows)
# is UNCHANGED — no new vocabulary ids.

#: Continuous prefix POSITIONS appended after the id positions (position axis).
CHARACTER_PREFIX_POSITIONS: int = 1

#: Width of the per-cell character_stats vector (channel axis; mini-spec §1's 7
#: channels: building median/IQR/p90-p50/count, road median length, two presence flags).
CHARACTER_STAT_CHANNELS: int = 7

#: VALUE-BEARING layout. Each field reserves a block of ``_VALUE_STRIDE`` embedding ids
#: above the sealed sub-F vocab; a field's value maps to one id inside its block.
#: DECISION (bake-off Task 6, tier-1 conditioning schema, identical across all runs):
#:  * STRIDE=64 generously covers every current field's bucket cardinality (densities,
#:    zoning/skeleton enums, the two ~constant string fields) with headroom; revisit only
#:    if a field exceeds 64 buckets (it would APPEND a wider block, never reindex).
#:  * bucket 0 is reserved for None/absent so a missing label never collides with value 0.
#:  * the ``seed`` field is NOT value-embedded (a generation-sampling control, not a
#:    learnable category) -> it maps to its block's constant bucket 0, preserving the
#:    9-position layout without teaching the model per-seed embeddings.
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
    city_identity: str | None,
    ablation: str = "full",
) -> list[int]:
    """The VALUE-BEARING conditioning prefix: 9 embedding ids encoding the real values.

    Field order is ``_CONDITIONING_FIELDS`` (append-only). Each id lands in its field's
    reserved block above the sub-F vocab, so distinct conditioning -> distinct prefix.
    ``seed`` is intentionally constant-bucketed (not value-embedded; see _VALUE_STRIDE note).

    ``city_identity`` (Task 24a) is the shard's CITY name, encoded via the sha-locked
    append-only registry (``city_identity_bucket``: index+1; injective over the 49 —
    NEVER ``_value_bucket``, which collides madrid=rome among 11 groups). None -> 0.

    ``ablation`` (spec §8): "full" = everything live; "no_city" forces ONLY the
    city_identity slot to bucket 0 (Lane S's instrument); "no_character" (Task 24b)
    leaves ALL 9 id positions identical to "full" — it ablates the CONTINUOUS
    character position instead, zeroed data-side where the stats are threaded
    (datamodule.flatten_shards_to_cells), never here.
    """
    if ablation not in _ABLATION_MODES:
        raise ValueError(f"unknown conditioning ablation {ablation!r}; expected {_ABLATION_MODES}")
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
    prefix = [
        CONDITIONING_VALUE_BASE + i * _VALUE_STRIDE + _value_bucket(v) for i, v in enumerate(values)
    ]
    # city_identity (slot 8): registry-encoded, never hashed; ablation forces bucket 0.
    city_bucket = 0 if ablation == "no_city" else city_identity_bucket(city_identity)
    prefix.append(CONDITIONING_VALUE_BASE + len(values) * _VALUE_STRIDE + city_bucket)
    return prefix

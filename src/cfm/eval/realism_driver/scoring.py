"""Realism-eval scoring helpers: torch-free decode + real-features serialization.

Task 5 creates this module with (a) ``decode_tokens_to_cell`` — the SAME
split-into-features + decode-or-None chain ``scripts.train_scaffold.score_cell``
uses, but WITHOUT a model (already-real tokens, no generation), and (b) the
``real-features.yaml`` record serialization keyed to the schema
``scripts/run_bakeoff_decision.py`` loads (``{metric, stratum, samples}`` per
city). Task 6 extends this module with the scored-lane driver.

TORCH DISCIPLINE (verified 2026-07-20; single authority per Task-5 review): the
decode primitives live in torch-free modules — ``split_cell_into_features`` in
``cfm.data.sub_g``'s ``seam_decodability`` and ``try_decode_block`` in
``cfm.data.sub_f.decoder`` (its home since the review fix; ``cfm.inference.generate``
re-exports it unchanged for its torch-side importers). Importing this module never
pulls torch, matching the Task-1/Task-2 import discipline of the realism_driver
package.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

# BOTH torch-free (checked 2026-07-20): importing this module must not pull torch.
from cfm.data.sub_f.decoder import try_decode_block
from cfm.data.sub_g.seam_decodability import split_cell_into_features

# All torch-free (verified 2026-07-20): read_held_out_cities lives in bakeoff_decision
# (imports only city_aggregate/conditioning_floor/ladder — no torch); city_aggregate,
# conditioning_floor and gen_realism are the locked scored-lane primitives this module
# orchestrates. Importing this module must STILL not pull torch (Task-5 discipline).
from cfm.eval.bakeoff_decision import read_held_out_cities
from cfm.eval.city_aggregate import (
    BindingVerdict,
    NoDecisiveWinner,
    PerCityKS,
    binding_city_verdict,
)
from cfm.eval.conditioning_floor import LaneSResult, VerifiedFloorArtifact
from cfm.eval.gen_realism import DecodedCell

if TYPE_CHECKING:
    from cfm.eval.realism_driver.driver import GenCellRecord

logger = logging.getLogger(__name__)


class MemorizationHalt(RuntimeError):
    """A checkpoint FAILED the Lane-M memorization discriminator during the scored
    realism eval — a hard halt mirroring ``bakeoff_decision.MemorizationRefusal``
    (``bakeoff_decision.py``): a regurgitator passes realism by construction, so NO
    scoring (coverage / excess / verdict) may run past this. The CLI writes
    ``memorization.yaml`` and raises this BEFORE any ``lane_s_excess`` call."""


#: {(metric, stratum) -> samples}; stratum is the floor's 4-tuple. Structural
#: mirror of ``gen_realism.GenFeatures`` / ``bakeoff_decision.GenFeatures`` (kept
#: local so importing this module never drags a torch-pulling module in).
GenFeatures = dict[tuple[str, tuple], list[float]]


# --------------------------------------------------------------------------- #
# Decode (real tokens -> aligned kept blocks/geoms), torch-free
# --------------------------------------------------------------------------- #


def decode_tokens_to_cell(tokens: Sequence[int]) -> tuple[list[list[int]], list[dict]]:
    """Split a real cell's token body into feature blocks and decode each, keeping
    only the blocks that decode — returning ALIGNED ``(blocks, geoms)`` exactly as
    ``score_cell`` does (``[b for b, d in decoded if d is not None]`` /
    ``[d for ... if d is not None]``), so a decoded real cell feeds
    ``gen_realism.DecodedCell`` in the identical shape a generated cell does.

    No model, no torch: the tokens are already-real (no generation step). A block
    that fails to decode is skipped (measured rate), never raised."""
    kept_blocks: list[list[int]] = []
    kept_geoms: list[dict] = []
    for block in split_cell_into_features(list(tokens)):
        geom = try_decode_block(block)
        if geom is not None:
            kept_blocks.append(block)
            kept_geoms.append(geom)
    return kept_blocks, kept_geoms


# --------------------------------------------------------------------------- #
# real-features.yaml serialization (run_bakeoff_decision record schema)
# --------------------------------------------------------------------------- #

#: Spec tag stamped into the artifact meta (lineage; bump on layout change).
REAL_FEATURES_SPEC = "realism-eval-real-features-v1"


def _stratum_sort_key(stratum: tuple) -> tuple[str, ...]:
    return tuple(str(x) for x in stratum)


def features_to_records(features: GenFeatures) -> list[dict]:
    """``{(metric, stratum): samples}`` -> ``[{metric, stratum, samples}, ...]`` — the
    schema ``run_bakeoff_decision._features_from_records`` reads. Sorted by
    (metric, str(stratum)) so the artifact is insertion-order independent
    (PYTHONHASHSEED-proof), mirroring the floor artifact's ordering discipline."""
    return [
        {
            "metric": metric,
            "stratum": list(stratum),
            "samples": [float(x) for x in samples],
        }
        for (metric, stratum), samples in sorted(
            features.items(), key=lambda kv: (kv[0][0], _stratum_sort_key(kv[0][1]))
        )
    ]


def features_from_records(records: list[dict]) -> GenFeatures:
    """Inverse of :func:`features_to_records` — byte-for-byte the same shape as
    ``run_bakeoff_decision._features_from_records`` (the downstream reader), kept
    here so a roundtrip can be asserted without importing the scripts layer."""
    return {
        (rec["metric"], tuple(rec["stratum"])): [float(x) for x in rec["samples"]]
        for rec in records
    }


def city_features_to_records(by_city: Mapping[str, GenFeatures]) -> dict[str, list[dict]]:
    """``{city: GenFeatures}`` -> ``{city: [record, ...]}`` (cities sorted)."""
    return {city: features_to_records(by_city[city]) for city in sorted(by_city)}


def build_real_features_payload(
    *,
    meta: dict,
    real_by_city: Mapping[str, GenFeatures] | None = None,
    real_train_by_city: Mapping[str, GenFeatures] | None = None,
) -> dict:
    """Assemble the ``real-features.yaml`` payload.

    A half is included ONLY when computed: a held-out-only (or train-only) partial
    OMITS the other key ENTIRELY, so ``run_bakeoff_decision._load_real``'s STRICT
    read refuses it (missing key) rather than silently accepting an empty set —
    only a full run (both halves) yields the canonical artifact the memorization
    check accepts."""
    payload: dict[str, Any] = {"spec": REAL_FEATURES_SPEC, "meta": dict(meta)}
    if real_by_city is not None:
        payload["real_by_city"] = city_features_to_records(real_by_city)
    if real_train_by_city is not None:
        payload["real_train_by_city"] = city_features_to_records(real_train_by_city)
    return payload


def write_real_features(path: str | Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` as write-once YAML (atomic tmp + rename).

    Refuses to overwrite an existing file (``FileExistsError``): re-extracting real
    features means deleting the old artifact deliberately (the eval-set / gen-artifact
    write-once discipline), so a stale extraction is never silently clobbered."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"real-features artifact already exists at {path}; it is write-once — "
            "delete deliberately only to re-extract."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    logger.info("wrote real-features artifact -> %s", path)


def load_real_features(path: str | Path) -> dict:
    """Read a real-features YAML back into its raw mapping (for end-state verify)."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Task 6 — scored-lane driver: decode -> city-set guard -> seed aggregation
# --------------------------------------------------------------------------- #


def decoded_cells_from_artifact(
    meta: Mapping[str, Any],
    records: Sequence[GenCellRecord],
    *,
    release: str,
    verify_tokens: bool = False,
) -> list[DecodedCell]:
    """Map Task-2 ``GenCellRecord``s to ``gen_realism.DecodedCell`` in manifest order.

    REUSES each record's ALIGNED ``(blocks, geoms)`` — the kept decode results the driver
    already produced GPU-side — so scoring consumes the exact geometry that was generated.
    ``cell_key`` -> ``(city, tile_i, tile_j)`` (the cell_i/cell_j identity dims are dropped;
    the tile is what keys the floor's ``(zoning, skeleton, coastal)`` labels), and
    ``density_bucket`` -> ``cell_density_bucket`` (the conditioned stratum dim).

    ``verify_tokens`` (a dry-run determinism assert): re-decode ``record.tokens`` on the CPU
    via the torch-free :func:`decode_tokens_to_cell` and require the result to EQUAL the
    stored ``(blocks, geoms)`` — catches a gen/scoring decode drift before it silently
    changes features. ``release`` is cross-checked against ``meta['release']`` when present
    (a lineage warning — the tile labels ``gen_features_by_city`` reads are release-scoped)."""
    meta_release = meta.get("release")
    if meta_release is not None and meta_release != release:
        logger.warning(
            "decoded_cells_from_artifact: release=%r but artifact meta.release=%r — a "
            "lineage mismatch (features key on THIS release's tile labels).",
            release,
            meta_release,
        )
    out: list[DecodedCell] = []
    for rec in records:
        city, tile_i, tile_j, _cell_i, _cell_j = rec.cell_key
        blocks = [list(b) for b in rec.blocks]
        geoms = list(rec.geoms)
        if verify_tokens:
            re_blocks, re_geoms = decode_tokens_to_cell(rec.tokens)
            if re_blocks != blocks or re_geoms != geoms:
                raise ValueError(
                    f"decoded_cells_from_artifact: decode determinism drift for cell "
                    f"{rec.cell_key} — re-decoded (blocks, geoms) != the artifact's stored "
                    "aligned decode; the gen-side and score-side decoders disagree."
                )
        out.append(
            DecodedCell(
                city=city,
                tile_i=int(tile_i),
                tile_j=int(tile_j),
                cell_density_bucket=rec.density_bucket,
                blocks=blocks,
                geoms=geoms,
            )
        )
    return out


def assert_city_sets(
    manifest: Mapping[str, Any],
    artifact: VerifiedFloorArtifact,
    real_by_city: Mapping[str, Any],
    gen_by_city_per_ckpt: Mapping[tuple[str, int], Mapping[str, Any]],
) -> frozenset[str]:
    """STRICT held-out set-equality across the manifest, the floor artifact, the real
    features, and EVERY checkpoint's gen — ``bakeoff_decision.decide()``'s Tooth-2,
    replicated explicitly here because the crown path in this module does NOT call
    ``decide()``.

    Reads the manifest's ``held_out_cities`` via the STRICT
    ``bakeoff_decision.read_held_out_cities`` (never a ``.get(..., [])`` that would make
    completeness vacuously pass on zero cities). Any mismatch raises ``ValueError`` naming
    the offender — a silently-shrunk or padded city set changes the worst-case max domain.
    Returns the single agreed held-out set."""
    held = read_held_out_cities(manifest)
    artifact_held = frozenset(artifact.payload["held_out_cities"])
    if artifact_held != held:
        raise ValueError(
            "assert_city_sets: floor artifact held_out_cities "
            f"{sorted(artifact_held)} != manifest {sorted(held)} — floors frozen for a "
            "DIFFERENT held-out set are lineage skew; refusing."
        )
    real_set = frozenset(real_by_city)
    if real_set != held:
        raise ValueError(
            "assert_city_sets: real_by_city "
            f"(missing: {sorted(held - real_set)}, extra: {sorted(real_set - held)}) != the "
            f"held-out set {sorted(held)}; refusing a shrunk/padded reference domain."
        )
    for (backbone, seed), gen in sorted(gen_by_city_per_ckpt.items()):
        gen_set = frozenset(gen)
        if gen_set != held:
            raise ValueError(
                f"assert_city_sets: gen for ({backbone}, seed {seed}) "
                f"(missing: {sorted(held - gen_set)}, extra: {sorted(gen_set - held)}) != the "
                f"held-out set {sorted(held)} — every checkpoint must generate for every "
                "held-out city."
            )
    return held


def n_reference_by_city(
    artifact: VerifiedFloorArtifact,
    real_by_city: Mapping[str, GenFeatures],
) -> dict[str, int]:
    """Per held-out city: total REAL feature count over the city's FLOORED strata ONLY —
    the #21 power-gate reference population, EXACTLY ``bakeoff_decision.decide()``'s rule
    (the floored-strata sum, never the all-strata sum).

    THE HONEST ``PerCityKS.n_features`` (Task-6 constraint): ``n_features`` feeds
    ``feature_resolution.single_region_floor_gap`` (the ``C/sqrt(n)`` resolution floor), so
    it must be the backbone-independent reference the floor was measured against. Unfloored
    strata are excluded — counting them would inflate n and SHRINK the resolution floor into
    a silently more permissive gate (Task-26 spec review #3)."""
    floored_by_city: dict[str, set[tuple[str, tuple]]] = {}
    for rec in artifact.payload["floors"]:
        floored_by_city.setdefault(rec["city"], set()).add((rec["metric"], tuple(rec["stratum"])))
    out: dict[str, int] = {}
    for city, feats in real_by_city.items():
        floored = floored_by_city.get(city, set())
        out[city] = sum(len(samples) for key, samples in feats.items() if key in floored)
    return out


def seed_aggregated_per_backbone(
    lane_s_by_ckpt: Mapping[tuple[str, int], Mapping[str, LaneSResult]],
    *,
    n_reference_by_city: Mapping[str, int],
) -> dict[str, list[PerCityKS]]:
    """Per (backbone, city): mean of that backbone's per-seed ``median_excess`` = ``ks``;
    std-error of those per-seed values over seeds = ``seed_sem`` (sample stdev / sqrt(n)).
    Returns the ``{backbone: [PerCityKS, ...]}`` map ``binding_city_verdict`` consumes.

    Requires >= 2 backbones (a single entry never auto-wins — ``pick_winner``'s rule), a
    consistent seed COUNT across backbones (the seed-noise floor compares like with like),
    and >= 2 seeds per backbone (review fix I-1: a single seed would set ``seed_sem=0`` and
    silently drop the locked seed-noise floor — refused, never warned past). Every
    (backbone, seed) must score exactly ``n_reference_by_city``'s cities."""
    keys = list(lane_s_by_ckpt)
    if not keys:
        raise ValueError("seed aggregation: no (backbone, seed) results supplied")
    backbones = sorted({bb for bb, _seed in keys})
    if len(backbones) < 2:
        raise ValueError(
            f"seed aggregation: got {len(backbones)} backbone(s) ({backbones}); a bake-off "
            "verdict needs >= 2 backbones — a single entry never auto-wins."
        )
    cities = sorted(n_reference_by_city)
    if not cities:
        raise ValueError("seed aggregation: n_reference_by_city is empty")

    seeds_by_backbone = {bb: sorted(seed for b, seed in keys if b == bb) for bb in backbones}
    seed_counts = {bb: len(s) for bb, s in seeds_by_backbone.items()}
    if len(set(seed_counts.values())) != 1:
        raise ValueError(
            f"seed aggregation: backbones ran DIFFERENT seed counts {seed_counts} — the "
            "seed-noise floor compares like with like; refusing an unbalanced sweep."
        )
    n_seeds = next(iter(seed_counts.values()))
    if n_seeds < 2:
        # REVIEW FIX I-1 (2026-07-20): RAISE, never warn-and-proceed. A single seed makes
        # seed_sem=0, which silently re-opens the exact seed-noise-floor hole this module
        # closes (a partial run would crown with the locked floor dropped). No override
        # flag by design: the locked run shape is 3 seeds (GROUND_TRUTH §4); a lost
        # checkpoint goes back to the PI, not to a flag.
        raise ValueError(
            f"seed aggregation: only {n_seeds} seed per backbone — seed_sem would be 0.0, "
            "silently DROPPING the locked seed-noise floor (GROUND_TRUTH §4: 3 seeds). "
            "Refusing; a single-seed crown is exactly the hole this crown path closes."
        )

    per_backbone: dict[str, list[PerCityKS]] = {}
    for bb in backbones:
        per_city_vals: dict[str, list[float]] = {c: [] for c in cities}
        for seed in seeds_by_backbone[bb]:
            ckpt = lane_s_by_ckpt[(bb, seed)]
            if set(ckpt) != set(cities):
                raise ValueError(
                    f"seed aggregation: ({bb}, seed {seed}) scored cities "
                    f"(missing: {sorted(set(cities) - set(ckpt))}, "
                    f"extra: {sorted(set(ckpt) - set(cities))}) != {cities}; refusing."
                )
            for c in cities:
                per_city_vals[c].append(float(ckpt[c].median_excess))
        per_backbone[bb] = [
            PerCityKS(
                city=c,
                ks=statistics.fmean(per_city_vals[c]),
                n_features=int(n_reference_by_city[c]),
                seed_sem=(
                    statistics.stdev(per_city_vals[c]) / math.sqrt(len(per_city_vals[c]))
                    if len(per_city_vals[c]) >= 2
                    else 0.0
                ),
            )
            for c in cities
        ]
    return per_backbone


def aggregate_seed_verdict(
    lane_s_by_ckpt: Mapping[tuple[str, int], Mapping[str, LaneSResult]],
    *,
    n_reference_by_city: Mapping[str, int],
) -> BindingVerdict | NoDecisiveWinner:
    """The ONLY crown path (orchestrator decision 2026-07-20): the locked two-floor rule
    with its seed-noise input POPULATED.

    Builds per-backbone ``PerCityKS`` via :func:`seed_aggregated_per_backbone` (mean
    per-seed ``median_excess`` = ``ks``; per-seed std-error = ``seed_sem``) and hands the
    map to ``city_aggregate.binding_city_verdict`` — which demotes any city whose
    winner-vs-runner-up gap fails to clear ``max(C/sqrt(n) resolution floor, seed-noise
    floor)`` and returns ``NoDecisiveWinner`` when no city is decisive.

    Deliberately NOT ``bakeoff_decision.decide()``: decide() builds ``PerCityKS`` with
    ``seed_sem=0`` (its scalar is a single fixed-scale excess), which would silently DROP the
    locked seed-noise floor of GROUND_TRUTH §4. Here the 3-seed spread IS the input, so the
    seed floor actually binds."""
    per_backbone = seed_aggregated_per_backbone(
        lane_s_by_ckpt, n_reference_by_city=n_reference_by_city
    )
    return binding_city_verdict(per_backbone)

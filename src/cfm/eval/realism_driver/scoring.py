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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

# BOTH torch-free (checked 2026-07-20): importing this module must not pull torch.
from cfm.data.sub_f.decoder import try_decode_block
from cfm.data.sub_g.seam_decodability import split_cell_into_features

# All torch-free (verified 2026-07-20): read_held_out_cities lives in bakeoff_decision
# (imports only city_aggregate/conditioning_floor/ladder — no torch); city_aggregate,
# conditioning_floor, gen_realism, lane_s_sampler and conditioning_discrimination are the
# locked scored-lane primitives this module orchestrates. Importing this module must STILL not
# pull torch (Task-5 discipline; lane_s_sampler / _tile_features re-checked 2026-07-20).
from cfm.eval.bakeoff_decision import read_held_out_cities
from cfm.eval.city_aggregate import (
    BindingVerdict,
    NoDecisiveWinner,
    PerCityKS,
    binding_city_verdict,
)
from cfm.eval.conditioning_discrimination import _tile_features
from cfm.eval.conditioning_floor import LaneSResult, VerifiedFloorArtifact, lane_s_excess
from cfm.eval.gen_realism import DecodedCell, gen_features_by_city
from cfm.eval.lane_s_sampler import CoverageReport, verify_gen_coverage

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


# --------------------------------------------------------------------------- #
# Task 7a — LOCAL scoring dry-run (no GPU, no checkpoint, no verdict)
# --------------------------------------------------------------------------- #


def single_stratum_gen_features(
    cells: Sequence[DecodedCell],
    *,
    stratum: tuple,
    release: str | None = None,
) -> dict[str, GenFeatures]:
    """LOCAL dry-run ONLY: assign EVERY decoded cell's features to ONE synthetic 4-tuple
    ``stratum`` (NO disk read).

    The heldout-cache stand-in (``data/_diag/heldout_cache.json``) carries no tile identity —
    only ``{region, body_tokens}`` — and the held-out EU tile labels are not on the local disk,
    so the real ``gen_realism.gen_features_by_city`` (which reads
    ``(zoning, road_skeleton, coastal)`` off disk per tile) cannot key the floor's 4-tuple
    grammar locally. This reuses the IDENTICAL feature classification
    (``conditioning_discrimination._tile_features``: ring promotion, building-area vs
    road-length, outbound-bref exclusion) so decode->feature is faithful, and stamps the
    caller's single synthetic stratum in place of the disk labels. NEVER used on Leonardo (real
    tiles -> the real ``gen_features_by_city``). ``release`` is accepted for signature parity
    with ``gen_features_by_city`` and is unused."""
    key_stratum = tuple(stratum)
    by_city: dict[str, GenFeatures] = {}
    for cell in cells:
        density = cell.cell_density_bucket
        feats, _n_bref = _tile_features(cell.blocks, cell.geoms, [density] * len(cell.blocks))
        gf = by_city.setdefault(cell.city, {})
        for metric, value, _dens in feats:
            gf.setdefault((metric, key_stratum), []).append(float(value))
    return by_city


def synthesize_dry_run_manifest(gen_by_city: Mapping[str, GenFeatures], *, min_n: int) -> dict:
    """A synthetic, sealed-SHAPED Lane-S manifest built FROM the observed gen features so
    ``verify_gen_coverage`` can run without the real sealed manifest (whose strata key real EU
    tile labels unavailable locally).

    Every observed ``(city, stratum)`` becomes a NON-ceiling stratum owing its observed
    metrics, ``target_features = max(1, min_n)`` so a dry run passes coverage on whatever it
    generated. This is a wiring check, NEVER the sealed manifest and NEVER a real coverage
    gate — the scored run always loads the sealed manifest instead."""
    strata: list[dict] = []
    for city in sorted(gen_by_city):
        by_stratum: dict[tuple, list[str]] = {}
        for metric, stratum in gen_by_city[city]:
            by_stratum.setdefault(stratum, []).append(metric)
        for stratum in sorted(by_stratum, key=_stratum_sort_key):
            metrics = sorted(set(by_stratum[stratum]))
            strata.append(
                {
                    "city": city,
                    "stratum": list(stratum),
                    "owed_metrics": metrics,
                    "binding_metric": metrics[0],
                    "ceiling_bound": False,
                }
            )
    return {
        "held_out_cities": sorted(gen_by_city),
        "methodology": {"target_features": max(1, int(min_n))},
        "strata": strata,
    }


@dataclass(frozen=True)
class DryRunReport:
    """Result of the Task-7a LOCAL scoring dry-run over ONE artifact.

    Deliberately carries NO ``BindingVerdict`` / ``NoDecisiveWinner`` field: a dry run is
    structurally incapable of a crown — it stops before any seed aggregation and never invokes
    ``aggregate_seed_verdict`` / ``binding_city_verdict``. It surfaces the decode + gen-feature
    + coverage + Lane-S wiring so a human can confirm the chain runs, plus the determinism
    ``verify_tokens`` outcome."""

    n_cells: int
    n_self_terminated: int
    verify_tokens: bool
    cities: list[str]
    gen_stratum_keys: dict[str, list[tuple[str, tuple]]]
    coverage: CoverageReport
    lane_s_by_city: dict[str, LaneSResult]
    scored_cities: list[str]


def dry_run_score(
    *,
    meta: Mapping[str, Any],
    records: Sequence[GenCellRecord],
    real_by_city: Mapping[str, GenFeatures],
    verified: VerifiedFloorArtifact,
    release: str,
    gen_features_fn: Callable[..., Mapping[str, GenFeatures]] = gen_features_by_city,
    manifest: dict | None = None,
    min_n: int | None = None,
    verify_tokens: bool = True,
) -> DryRunReport:
    """Task-7a LOCAL scoring dry-run: ``decode -> gen features -> verify_gen_coverage ->
    lane_s_excess`` for ONE artifact, then STOP.

    It NEVER calls ``aggregate_seed_verdict`` / ``binding_city_verdict`` and writes nothing — a
    dry run cannot emit a decision.yaml or any crown language. ``verify_tokens`` (default ON)
    re-decodes each record's tokens on the CPU and asserts they reproduce the stored aligned
    ``(blocks, geoms)`` bit-identically (the determinism check the 7b PASS criteria require);
    a drift raises ``ValueError`` in :func:`decoded_cells_from_artifact`.

    ``gen_features_fn`` is the real ``gen_features_by_city`` on Leonardo (real tile labels) or
    :func:`single_stratum_gen_features` locally (heldout-cache stand-in, no tile identity). When
    ``manifest`` is ``None`` a synthetic 1-per-observed-stratum manifest is built so coverage can
    run without the sealed manifest. A city whose Lane-S is vacuous (thin / mismatched strata) is
    WARNED and reported, never crowned."""
    decoded = decoded_cells_from_artifact(
        meta, records, release=release, verify_tokens=verify_tokens
    )
    gen_by_city = dict(gen_features_fn(decoded, release=release))
    cov_min_n = min_n if min_n is not None else 1
    mf = (
        manifest
        if manifest is not None
        else synthesize_dry_run_manifest(gen_by_city, min_n=cov_min_n)
    )
    coverage = verify_gen_coverage(gen_by_city, mf, min_n=min_n)

    lane_s: dict[str, LaneSResult] = {}
    for city in sorted(gen_by_city):
        if city not in real_by_city:
            logger.warning(
                "dry-run: no real features for city %r — Lane-S skipped (reported, not crowned)",
                city,
            )
            continue
        try:
            lane_s[city] = lane_s_excess(
                gen_by_city[city], real_by_city[city], verified, city=city, min_n=min_n
            )
        except ValueError as exc:
            # A vacuous Lane-S locally (thin slice / mismatched synthetic strata) is EXPECTED
            # for a dry run — report it, never fail the wiring check (and never crown).
            logger.warning(
                "dry-run: Lane-S vacuous for city %r (%s) — reported, not crowned", city, exc
            )

    n_self_terminated = sum(1 for r in records if r.self_terminated)
    return DryRunReport(
        n_cells=len(records),
        n_self_terminated=n_self_terminated,
        verify_tokens=verify_tokens,
        cities=sorted(gen_by_city),
        gen_stratum_keys={
            city: sorted(gen_by_city[city], key=lambda kv: (kv[0], _stratum_sort_key(kv[1])))
            for city in sorted(gen_by_city)
        },
        coverage=coverage,
        lane_s_by_city=lane_s,
        scored_cities=sorted(lane_s),
    )

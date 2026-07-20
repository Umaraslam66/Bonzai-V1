"""Checkpoint-independent real-feature extraction CLI (realism-eval Task 5).

Emits ONE ``real-features.yaml`` — the memorization-check's required real side —
with TWO halves:

  * ``real_by_city``       — the 4 held-out cities, features computed from the
    sealed Lane-S manifest cells' REAL body tokens
    (``ConditionedCell.real_body_tokens`` -> decode -> ``gen_features_by_city``).
  * ``real_train_by_city`` — EXACTLY the ``train_cities`` frozen in the floor
    artifact (read FROM the artifact via ``load_verified_floor``, NEVER hardcoded:
    ``memorization_check`` refuses a train-city-set mismatch), features from each
    train city's tiles via ``build_shards_in_memory`` + ``flatten_shards_to_cells``
    -> decode ``CellExample.tokens`` -> the SAME ``gen_features_by_city`` classifier.

CHECKPOINT-INDEPENDENT (the whole point — no model, no GPU): every field
``gen_features_by_city`` reads is ablation-independent — the decoded geometry comes
from a cell's real ``tokens``, the density from ``cell_density_bucket``, the
(zoning, skeleton, coastal) from the tile labels on disk. The conditioning
``ablation`` only shapes the PREFIX/char-stats, which this extraction never touches.
So the real features are the SAME for every bake-off checkpoint; extract them once.

DECISION (Task-5, simplest design): the canonical artifact is produced by ONE full
run (both halves). ``--held-out-only`` / ``--train-only`` / ``--cities`` produce
PARTIAL smoke/sizing artifacts (a partial omits the uncomputed half's key, so the
downstream STRICT read refuses it; ``--cities`` narrows the train set so its keys no
longer equal the artifact's ``train_cities``). Those partials let ops size each half's
CPU cost before the full run; they are NOT canonical and ``memorization_check`` will
refuse them. No ``--merge`` mode — merging two write-once partials adds surface for no
benefit, since the full run is a single CPU job.

TORCH DISCIPLINE: ``build_arg_parser``, ``select_cities``, ``train_cities_from_floor``
and ``heldout_decoded_cells`` are torch-free (unit-tested without a GPU). Every
torch-pulling import (``build_conditioned_cells`` -> datamodule, ``build_shards_in_memory``,
``flatten_shards_to_cells``) is LAZY, inside the heavy extractors called only from
``main`` — so importing this module never pulls torch (mirrors ``realism_eval_gen.py``).

Run (ops; NOT this session — CPU, no GPU, run on a Leonardo serial/data node):
    python scripts/realism_eval_real_features.py \\
        --floor-artifact <conditioning-floor.yaml> \\
        --manifest <sealed_lane_s_manifest.json> \\
        --out reports/realism_eval/real-features.yaml

Sizing a single half first (partial, non-canonical):
    python scripts/realism_eval_real_features.py ... --train-only --cities <one_city>
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from cfm.eval.conditioning_floor import load_verified_floor
from cfm.eval.gen_realism import DecodedCell, gen_features_by_city
from cfm.eval.realism_driver import scoring
from cfm.eval.realism_driver.conditioning import ConditionedCell

logger = logging.getLogger(__name__)

#: Overture release the sealed manifest's tiles + the train cities' shards belong to.
DEFAULT_RELEASE = "2026-04-15.0"

#: Flatten seed (constant-bucketed in the prefix — inert for the body ``tokens`` this
#: extraction reads; kept a flag for parity with the conditioning join).
DEFAULT_SEED = 0

#: Ablation threaded to the flattener. DECISION: real-feature extraction reads only
#: ablation-INDEPENDENT fields (tokens -> geoms, density bucket, tile labels); the
#: ablation shapes only the prefix/char-stats, never used here. "full" is a safe
#: constant. Revisit only if a future feature keys on the prefix.
DEFAULT_ABLATION = "full"

#: rank-0 stdout sentinel — printed ONLY after the artifact is re-read and its city
#: sets + non-empty samples are verified (no marker without end-state verify).
SENTINEL = "REALISM_EVAL_REAL_FEATURES_DONE"


def build_arg_parser() -> argparse.ArgumentParser:
    """The CLI. Torch-free so it is unit-testable without a GPU."""
    ap = argparse.ArgumentParser(
        description="Checkpoint-independent real-feature extraction (realism-eval Task 5)."
    )
    ap.add_argument(
        "--floor-artifact",
        required=True,
        help="frozen conditioning-floor YAML; its train_cities set is the train half",
    )
    ap.add_argument("--manifest", required=True, help="sealed Lane-S sampler manifest JSON")
    ap.add_argument("--out", required=True, help="output real-features YAML (write-once)")
    ap.add_argument("--release", default=DEFAULT_RELEASE, help="Overture release for the join")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="flatten seed (inert in body)")
    ap.add_argument(
        "--ablation",
        default=DEFAULT_ABLATION,
        help="flatten ablation; real features are ablation-independent (see module doc)",
    )
    ap.add_argument(
        "--cities",
        default=None,
        help="smoke/sizing: comma-separated subset of cities to extract (NON-canonical)",
    )
    half = ap.add_mutually_exclusive_group()
    half.add_argument(
        "--held-out-only",
        action="store_true",
        help="sizing: extract ONLY the held-out half (partial artifact, non-canonical)",
    )
    half.add_argument(
        "--train-only",
        action="store_true",
        help="sizing: extract ONLY the training half (partial artifact, non-canonical)",
    )
    return ap


def select_cities(requested: str | None, universe: Sequence[str]) -> list[str]:
    """Resolve a ``--cities`` request against the ``universe`` of extractable cities.

    ``None`` -> the full universe (canonical). A comma-separated request -> the
    intersection returned in UNIVERSE order (deterministic, not request order). A
    requested city absent from the universe is a hard ``SystemExit`` — never a
    silent drop that would shrink the artifact behind the operator's back."""
    if requested is None:
        return list(universe)
    wanted = [c.strip() for c in requested.split(",") if c.strip()]
    universe_set = set(universe)
    unknown = [c for c in wanted if c not in universe_set]
    if unknown:
        raise SystemExit(
            f"--cities requests {unknown} not in the extractable set {sorted(universe_set)}; "
            "refusing to silently drop unknown cities."
        )
    return [c for c in universe if c in set(wanted)]


def train_cities_from_floor(floor_path: str | Path) -> list[str]:
    """The training-city set read FROM the verified floor artifact (NEVER hardcoded).

    ``memorization_check`` refuses a train-city-set mismatch, so the artifact's own
    frozen ``train_cities`` are the single source of truth for the train half. The
    load is a verified read (sha/lock checked) — a tampered floor refuses here."""
    verified = load_verified_floor(Path(floor_path))
    return list(verified.payload["train_cities"])


def heldout_decoded_cells(cells: Sequence[ConditionedCell]) -> list[DecodedCell]:
    """Map each manifest ``ConditionedCell`` to a ``DecodedCell`` by decoding its REAL
    body tokens (``real_body_tokens``) through the shared torch-free decode chain.

    The cell key carries ``(city, tile_i, tile_j, cell_i, cell_j)``; the DecodedCell
    keeps ``city``/``tile_i``/``tile_j`` (for the tile-label lookup) and the density it
    was conditioned on (``density_bucket``), so ``gen_features_by_city`` stratum-keys the
    real features in the IDENTICAL grammar the generated side uses."""
    out: list[DecodedCell] = []
    for cell in cells:
        city, tile_i, tile_j, _ci, _cj = cell.cell_key
        blocks, geoms = scoring.decode_tokens_to_cell(cell.real_body_tokens)
        out.append(
            DecodedCell(
                city=city,
                tile_i=tile_i,
                tile_j=tile_j,
                cell_density_bucket=cell.density_bucket,
                blocks=blocks,
                geoms=geoms,
            )
        )
    return out


def extract_heldout(
    *, manifest_path: str | Path, release: str, ablation: str, seed: int, cities: str | None
) -> dict[str, scoring.GenFeatures]:
    """Held-out real features: verified manifest -> matched conditioning join ->
    decode ``real_body_tokens`` -> ``gen_features_by_city``. Torch-pulling imports
    (``build_conditioned_cells`` -> datamodule) are LAZY here."""
    from cfm.eval.realism_driver.conditioning import (  # lazy: pulls torch via datamodule
        build_conditioned_cells,
        load_verified_manifest_or_raise,
    )

    manifest = load_verified_manifest_or_raise(Path(manifest_path))
    conditioned = build_conditioned_cells(
        manifest, release=release, ablation=ablation, conditioning_seed=seed
    )
    decoded = heldout_decoded_cells(conditioned)
    if cities is not None:
        universe = sorted({d.city for d in decoded})
        keep = set(select_cities(cities, universe))
        decoded = [d for d in decoded if d.city in keep]
    by_city = gen_features_by_city(decoded, release=release)
    logger.info("held-out real: %d cells -> %d cities", len(decoded), len(by_city))
    return by_city


def extract_train(
    *, release: str, ablation: str, seed: int, train_cities: Sequence[str]
) -> dict[str, scoring.GenFeatures]:
    """Training real features: per city, ``build_shards_in_memory`` (all its tiles) ->
    ``flatten_shards_to_cells`` -> decode ``CellExample.tokens`` -> ``gen_features_by_city``.
    Torch-pulling imports are LAZY here. This is the heaviest CPU step (29 cities x tiles)."""
    from cfm.data.training.build_shards import build_shards_in_memory  # lazy
    from cfm.data.training.datamodule import flatten_shards_to_cells  # lazy: pulls torch

    by_city: dict[str, scoring.GenFeatures] = {}
    for city in train_cities:
        shards = build_shards_in_memory(release, city)  # tile_ids=None -> full city
        examples, dropped = flatten_shards_to_cells(shards, seed=seed, ablation=ablation)
        decoded: list[DecodedCell] = []
        for ex in examples:
            blocks, geoms = scoring.decode_tokens_to_cell(ex.tokens)
            decoded.append(
                DecodedCell(
                    city=ex.region,
                    tile_i=ex.tile_i,
                    tile_j=ex.tile_j,
                    cell_density_bucket=ex.cell_density_bucket,
                    blocks=blocks,
                    geoms=geoms,
                )
            )
        city_features = gen_features_by_city(decoded, release=release)
        # One region per build -> gen_features_by_city yields at most this one city.
        by_city.update(city_features)
        logger.info(
            "train real: %s — %d cells (dropped: %s) -> %d feature-keys",
            city,
            len(decoded),
            dropped,
            len(city_features.get(city, {})),
        )
    return by_city


def _verify_end_state(
    out: str | Path,
    *,
    expect_heldout: set[str] | None,
    expect_train: set[str] | None,
) -> None:
    """Re-read the just-written artifact and prove the requested halves are present,
    each expected city carries at least one non-empty-sample record, and (canonical)
    the train keys EXACTLY equal the artifact's train_cities. A false DONE poisons the
    downstream decision, so the sentinel is earned by disk state, never by control flow."""
    data = scoring.load_real_features(out)

    def _check(key: str, expect: set[str]) -> None:
        if key not in data:
            raise SystemExit(f"end-state verify FAILED: {key!r} missing from {out}.")
        got = set(data[key])
        if got != expect:
            raise SystemExit(
                f"end-state verify FAILED: {key} cities {sorted(got)} != expected "
                f"{sorted(expect)} (city-set mismatch)."
            )
        for city, records in data[key].items():
            if not records or not any(rec["samples"] for rec in records):
                raise SystemExit(
                    f"end-state verify FAILED: {key}[{city!r}] has no non-empty samples."
                )

    if expect_heldout is not None:
        _check("real_by_city", expect_heldout)
    if expect_train is not None:
        _check("real_train_by_city", expect_train)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_arg_parser().parse_args(argv)

    do_heldout = not args.train_only
    do_train = not args.held_out_only
    canonical = do_heldout and do_train and args.cities is None

    train_cities_full = train_cities_from_floor(args.floor_artifact)

    real_by_city = None
    real_train_by_city = None
    expect_heldout: set[str] | None = None
    expect_train: set[str] | None = None

    if do_heldout:
        real_by_city = extract_heldout(
            manifest_path=args.manifest,
            release=args.release,
            ablation=args.ablation,
            seed=args.seed,
            cities=args.cities,
        )
        expect_heldout = set(real_by_city)

    if do_train:
        train_cities = select_cities(args.cities, train_cities_full)
        real_train_by_city = extract_train(
            release=args.release,
            ablation=args.ablation,
            seed=args.seed,
            train_cities=train_cities,
        )
        expect_train = set(train_cities)

    meta = {
        "spec": scoring.REAL_FEATURES_SPEC,
        "release": args.release,
        "ablation": args.ablation,
        "seed": args.seed,
        "manifest_path": str(args.manifest),
        "floor_artifact_path": str(args.floor_artifact),
        "floor_train_cities": train_cities_full,
        "held_out_only": args.held_out_only,
        "train_only": args.train_only,
        "cities_filter": args.cities,
        "canonical": canonical,
    }
    payload = scoring.build_real_features_payload(
        meta=meta, real_by_city=real_by_city, real_train_by_city=real_train_by_city
    )
    scoring.write_real_features(args.out, payload)
    _verify_end_state(args.out, expect_heldout=expect_heldout, expect_train=expect_train)

    if canonical and expect_train != set(train_cities_full):
        # Defensive: a canonical run must carry the full floor train-city set.
        raise SystemExit(
            "end-state verify FAILED: canonical run's train cities "
            f"{sorted(expect_train or set())} != floor train_cities {sorted(train_cities_full)}."
        )
    logger.info(
        "real-features written (canonical=%s): held-out=%s train=%s -> %s",
        canonical,
        sorted(expect_heldout) if expect_heldout else None,
        sorted(expect_train) if expect_train else None,
        args.out,
    )
    # Deliberate stdout sentinel — emitted ONLY after the artifact was re-read + verified.
    print(SENTINEL, flush=True)


if __name__ == "__main__":
    main()

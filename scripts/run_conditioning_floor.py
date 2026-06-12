#!/usr/bin/env python3
"""scripts/run_conditioning_floor.py — conditioning-floor artifact runner
(Task 25 step 1; spec §8; the real run is separately GATED on Leonardo CPU).

Thin DRIVER over ``cfm.eval.conditioning_floor``:

  1. Resolve the held-out cities: a VERIFIED, STRICT read of ``held_out_cities``
     from the multiregion holdout manifest — the holdout guard's F9 verifier
     (sha + ``_EVAL_SET_LOCKED``) runs first, then the strict key read (never
     ``.get`` — correction #12) — unless ``--held-out-cities`` overrides.
  2. Optionally (``--include-train-cities``) resolve the training cities via
     ``verify_union_manifests`` (the Gate-2 / G4-roll-up source the codebase
     already uses) so the gated Leonardo run can stage held-out-only first.
  3. Extract per-(city, full-stratum, metric) REAL feature scalars
     (``extract_features_by_city_stratum_metric``: bref exclusion + the F3
     tile-coverage halts are built in).
  4. Build the TWO BH families (PI call 2026-06-11) inside the payload
     builder: family 1 (D-D held-out pairwise — the builder RESTRICTS the
     features to held-out cities, so family 1 is the stage-1 computation and
     bit-identity determinism anchor even with ``--include-train-cities``) and
     family 2 (D-T cross, its OWN BH, built only when training cities are
     present; no T-T pair ever). The integrity halts (FloorCollapseError
     < 0.049 / FloorExplosionError > 0.5, PI knob 4; UNSUPPORTED zero-pairs is
     loud) run on FAMILY 1 BEFORE any artifact byte is written.
  5. Derive per-held-out-city two-variant floors (STRICT min, PI knob 1:
     ``floor_heldout`` over family 1; ``floor_all`` over family 1 u family 2 —
     the bar Lane S scores) + discriminating strata (CROSS family, real data
     only) + the family-1 δ ladder, and FREEZE the sha-stamped write-once
     artifact (``_CONDITIONING_FLOOR_LOCKED`` beside).
  6. Re-load the artifact VERIFIED (end-state verification, never a marker
     written on control flow) and print a summary.

Exit 0 on success; every halt RAISES (no exit-code masking).

    uv run python scripts/run_conditioning_floor.py --release 2026-04-15.0
    # default --out: reports/conditioning_floor/<release>/conditioning-floor.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# iCloud-safe sys.path inject — mirrors scripts/run_gate_i_conditioning_discrimination.py
# (parents[1] = repo root; underscore-prefixed .pth files are hidden in the
# iCloud-synced .venv so the editable install can't be relied on).
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import yaml  # noqa: E402

# _validated_inventory private import SANCTIONED (stage-2 integration fix, Slurm
# job 45835276; the has_outbound_bref / _verify_manifest_integrity precedent):
# it is the ONE source of a training city's tile inventory — the sub-D manifest
# tiles[] the training data itself walks (build_train_city_shards; a train city
# has no tile-level holdout, see compute_training_tile_ids). Reusing it means the
# floor's training-city tiles can never drift from what training consumes;
# holdout_manifest_for_region rightly RAISES for a train city (fail-closed).
from cfm.data.training.build_shards import (  # noqa: E402
    DEFAULT_G4_ROLLUP,
    _validated_inventory,
    verify_union_manifests,
)

# Private import SANCTIONED (Task-25 quality review #2; the has_outbound_bref
# precedent — that one since promoted public at W7): _verify_manifest_integrity is the ONE F9 authority for "this
# holdout manifest is sealed and untampered" (sha + _EVAL_SET_LOCKED) — reusing
# it here means the D-vs-T boundary the floor artifact freezes is defined by
# the same verifier training uses, never a second hand-rolled copy.
from cfm.data.training.holdout_guard import _verify_manifest_integrity  # noqa: E402
from cfm.eval.conditioning_discrimination import (  # noqa: E402
    extract_features_by_city_stratum_metric,
)
from cfm.eval.conditioning_floor import (  # noqa: E402
    build_floor_artifact_payload,
    freeze_floor_artifact,
    load_verified_floor,
)
from cfm.eval.holdout.paths import multiregion_holdout_manifest_path  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_RELEASE = "2026-04-15.0"
# DEDICATED per-release directory (Task-25 quality review #4): the freeze's
# _CONDITIONING_FLOOR_LOCKED marker is per-DIRECTORY, so the artifact must own
# its directory — in a busy reports/ root a stale marker from an earlier freeze
# would seal a later hand-dropped artifact (rationale in freeze_floor_artifact).
_DEFAULT_OUT_TEMPLATE = "reports/conditioning_floor/{release}/conditioning-floor.yaml"
# One-sourced (correction #12): the G4 roll-up path is build_shards' constant,
# never a hand-copied literal that could drift from what training consumes.
_DEFAULT_G4_ROLLUP = DEFAULT_G4_ROLLUP


def _held_out_cities_from_manifest(manifest_path: Path) -> list[str]:
    """VERIFIED, STRICT read of ``held_out_cities``.

    VERIFIED (Task-25 quality review #2): the holdout guard's F9 verifier runs
    FIRST — recomputed sha == stored ``manifest_sha256`` AND ``_EVAL_SET_LOCKED``
    beside the file — so a tampered/unsealed manifest can never define the
    D-vs-T boundary the floor artifact freezes (raises HoldoutLeakError).

    STRICT (correction #12): a manifest without the key is refused loudly —
    ``.get(..., [])`` would silently scope zero cities and freeze a floor
    artifact that floors nothing."""
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            f"holdout manifest {manifest_path} is not a YAML mapping "
            f"(got {type(data).__name__}); refusing to derive a conditioning floor."
        )
    _verify_manifest_integrity(data, manifest_path)
    if "held_out_cities" not in data:
        raise ValueError(
            f"holdout manifest {manifest_path} has no 'held_out_cities' key; "
            "refusing to derive a conditioning floor from an empty held-out set."
        )
    return list(data["held_out_cities"])


def _print_summary(payload: dict, out_path: Path) -> None:
    print("=" * 72)
    print("conditioning-floor artifact (Task 25; spec §8)")
    print("=" * 72)
    print(f"  release                  : {payload['release']}")
    print(f"  held-out cities          : {payload['held_out_cities']}")
    print(f"  train cities             : {len(payload['train_cities'])}")
    meth = payload["methodology"]
    print(f"  min_n / alpha / delta    : {meth['min_n']} / {meth['alpha']} / {meth['delta']}")
    print(f"  family-1 (D-D) pairs     : {len(payload['pairs'])}")
    print(f"  family-1 median KS       : {payload['pair_table_median_ks']:.4f}")
    print(f"  family-2 (D-T) pairs     : {len(payload['cross_pairs'])}")
    cross_median = payload["cross_median_ks"]
    cross_median_txt = "n/a (no cross pairs)" if cross_median is None else f"{cross_median:.4f}"
    print(f"  cross median KS          : {cross_median_txt}")
    print(f"  thin-n cells excluded    : {payload['n_excluded_thin']}")
    print(f"  strata <2 cities         : {payload['n_strata_too_few_cities']}")
    print(f"  floors (city,metric,stratum) rows: {len(payload['floors'])}")
    n_tightened = sum(1 for rec in payload["floors"] if rec["floor_all"] < rec["floor_heldout"])
    print(f"  floor_all < floor_heldout: {n_tightened} rows (training cities tightened the bar)")
    ladder = {row["delta"]: row["n_pairs"] for row in payload["delta_ladder"]}
    print(f"  delta ladder             : {ladder}")
    for city, cov in sorted(payload["tile_coverage"].items()):
        print(
            f"  tiles {city:<18}: expected={cov['n_tiles_expected']} "
            f"read={cov['n_tiles_read']} skipped={cov['n_tiles_skipped']} "
            f"bref_excluded={cov['n_bref_excluded']}"
        )
    print(f"  artifact                 : {out_path}")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Freeze the conditioning-floor artifact (spec §8 Lane S/M inputs)"
    )
    parser.add_argument("--release", default=_DEFAULT_RELEASE)
    parser.add_argument(
        "--held-out-cities",
        nargs="+",
        default=None,
        help="override; default is a STRICT held_out_cities read from the manifest",
    )
    parser.add_argument(
        "--holdout-manifest",
        default=None,
        help="path to the multiregion holdout manifest (default: the release's)",
    )
    parser.add_argument(
        "--include-train-cities",
        action="store_true",
        help="also extract the training cities (verify_union_manifests source); "
        "OPTIONAL so the gated Leonardo run can stage held-out-only first",
    )
    parser.add_argument("--g4-rollup", default=_DEFAULT_G4_ROLLUP)
    parser.add_argument("--min-n", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.15)
    parser.add_argument(
        "--out",
        default=None,
        help="artifact path (default: reports/conditioning_floor/<release>/"
        "conditioning-floor.yaml — a DEDICATED dir so the per-directory "
        "_CONDITIONING_FLOOR_LOCKED marker can never be a stale seal from an "
        "unrelated reports/ artifact)",
    )
    args = parser.parse_args(argv)

    manifest_path = (
        Path(args.holdout_manifest)
        if args.holdout_manifest is not None
        else multiregion_holdout_manifest_path(args.release)
    )
    held_out = (
        list(args.held_out_cities)
        if args.held_out_cities is not None
        else _held_out_cities_from_manifest(manifest_path)
    )

    train: list[str] = []
    if args.include_train_cities:
        train = verify_union_manifests(
            args.release, g4_rollup=args.g4_rollup, holdout_manifest=manifest_path
        )
        logger.info("union verifier resolved %d training cities", len(train))

    held_out_set = set(held_out)
    train_only = [c for c in train if c not in held_out_set]
    cities = held_out + train_only
    # TRAINING cities have no tile-level holdout (their holdout-manifest read
    # would rightly raise): hand the extractor their sub-D validated inventories
    # as an override; held-out cities stay OFF the map (frozen holdout-manifest
    # path, byte-identical to a held-out-only run when the map is empty -> None).
    tiles_by_city = {c: _validated_inventory(args.release, c) for c in train_only}
    logger.info("extracting features for %d cities, release=%s", len(cities), args.release)
    extraction = extract_features_by_city_stratum_metric(
        args.release, cities, tiles_by_city=tiles_by_city or None
    )
    logger.info("extracted %d (city, stratum, metric) cells", len(extraction.features))

    # The builder constructs BOTH BH families itself: family 1 from the
    # held-out-restricted features (so it is the stage-1 computation even with
    # --include-train-cities), family 2 (D-T cross) only when training cities
    # are present. Integrity halts (collapse/explosion/UNSUPPORTED) fire on
    # family 1 BEFORE any artifact byte exists; they RAISE — never exit-coded.
    payload = build_floor_artifact_payload(
        extraction.features,
        release=args.release,
        held_out_cities=held_out,
        train_cities=train,
        min_n=args.min_n,
        alpha=args.alpha,
        delta=args.delta,
        tile_coverage=extraction.tile_coverage,
    )

    out_arg = (
        args.out if args.out is not None else _DEFAULT_OUT_TEMPLATE.format(release=args.release)
    )
    out_path = Path(out_arg) if Path(out_arg).is_absolute() else _REPO / out_arg
    freeze_floor_artifact(payload, out_path)
    # End-state verification: the artifact must load VERIFIED (sha + marker +
    # version) before this runner reports success — no marker-on-control-flow.
    verified = load_verified_floor(out_path)
    logger.info("floor artifact verified: sha=%s", verified.payload["floor_sha256"])

    _print_summary(payload, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

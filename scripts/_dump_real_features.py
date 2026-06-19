#!/usr/bin/env python3
"""SCRATCH (uncommitted): dump real_by_city + real_train_by_city for the Phase-2 bake-off
decision layer, in run_bakeoff_decision.py's --real-features shape, keyed in the EXACT
4-tuple stratum grammar the locked floor froze.

Reuses extract_features_by_city_stratum_metric — the SAME extractor that built the locked
conditioning-floor artifact (floor_sha256 95abb88...). Produces an artifact for review and
VERIFIES it; does NOT wire, commit, or lock.

Teeth:
  1. FLOOR REPRODUCTION (from the WRITTEN file, the way decide consumes it): reload the dump,
     reconstruct the (city, stratum, metric) feature dict, rebuild floors via
     build_floor_artifact_payload, and confirm floor_all reproduces the locked artifact's
     per-(city,metric,stratum) values EXACTLY. If not -> the dump does not match what decide
     consumes; HALT signalled via FLOOR_REPRODUCTION_OK=False.
  2. min_n=50 COVERAGE: how many (city,metric,stratum) cells survive >=50, which fall below
     (thin -> excluded-and-reported, never silently dropped), with held-out detail.
  3. BYTE-DETERMINISM: samples sorted (KS is order-independent, so floors are unaffected);
     re-serialization of the same payload is confirmed byte-identical in-process.

City lists are sourced FROM THE LOCKED ARTIFACT (held_out_cities / train_cities) so the dump's
city sets match EXACTLY what decide/memorization_check enforce.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import yaml  # noqa: E402

from cfm.data.training.build_shards import _validated_inventory  # noqa: E402
from cfm.eval.conditioning_discrimination import (  # noqa: E402
    extract_features_by_city_stratum_metric,
)
from cfm.eval.conditioning_floor import (  # noqa: E402
    build_floor_artifact_payload,
    load_verified_floor,
)

logger = logging.getLogger(__name__)
_DEFAULT_RELEASE = "2026-04-15.0"


def _records_for(features: dict, cities: list[str]) -> dict[str, list[dict]]:
    """{(city,stratum,metric)->samples} -> {city: [{metric, stratum, samples}]} for `cities`,
    re-keyed (stratum,metric)->(metric,stratum) into run_bakeoff_decision's --real-features
    record shape. Samples sorted (KS is order-independent -> floors unaffected, bytes stable);
    records sorted by (metric, stratum) -> deterministic file."""
    by_city: dict[str, list[dict]] = {c: [] for c in cities}
    cset = set(cities)
    for (city, stratum, metric), samples in features.items():
        if city in cset:
            by_city[city].append(
                {
                    "metric": metric,
                    "stratum": list(stratum),
                    "samples": sorted(float(x) for x in samples),
                }
            )
    for c in by_city:
        by_city[c].sort(key=lambda r: (r["metric"], tuple(str(x) for x in r["stratum"])))
    return by_city


def _features_from_dump(by_city: dict[str, list[dict]]) -> dict:
    """Reconstruct the floor-builder feature dict {(city, stratum, metric): samples} from the
    dump records — mirrors run_bakeoff_decision._features_from_records, then re-keys to the
    (city, stratum, metric) order build_floor_artifact_payload expects."""
    out: dict[tuple, list[float]] = {}
    for city, records in by_city.items():
        for rec in records:
            out[(city, tuple(rec["stratum"]), rec["metric"])] = [float(x) for x in rec["samples"]]
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Dump + verify real_by_city / real_train_by_city")
    ap.add_argument("--release", default=_DEFAULT_RELEASE)
    ap.add_argument(
        "--out",
        default=str(_REPO / "reports/phase-2-bakeoff" / "real-features-2026-04-15.0.yaml"),
    )
    args = ap.parse_args()
    release = args.release

    locked_path = _REPO / "reports/conditioning_floor" / release / "conditioning-floor.yaml"
    locked = load_verified_floor(locked_path)
    held_out = list(locked.payload["held_out_cities"])
    train = list(locked.payload["train_cities"])
    logger.info(
        "locked floor sha=%s held_out=%s n_train=%d",
        locked.payload["floor_sha256"], held_out, len(train),
    )

    held_set = set(held_out)
    train_only = [c for c in train if c not in held_set]
    cities = held_out + train_only
    # train cities have no tile-level holdout -> hand the extractor their sub-D validated
    # inventory (same as run_conditioning_floor); held-out cities use the holdout-manifest path.
    tiles_by_city = {c: _validated_inventory(release, c) for c in train_only}
    logger.info(
        "extracting %d cities (%d held-out + %d train)", len(cities), len(held_out), len(train_only)
    )
    extraction = extract_features_by_city_stratum_metric(
        release, cities, tiles_by_city=tiles_by_city or None
    )
    logger.info("extracted %d (city,stratum,metric) cells", len(extraction.features))

    # ----- DUMP (run_bakeoff_decision --real-features shape) -----
    payload = {
        "real_by_city": _records_for(extraction.features, held_out),
        "real_train_by_city": _records_for(extraction.features, train),
    }
    text = yaml.safe_dump(payload, sort_keys=True, default_flow_style=False)
    text2 = yaml.safe_dump(payload, sort_keys=True, default_flow_style=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    sha = hashlib.sha256(text.encode()).hexdigest()
    logger.info(
        "DUMP written: %s  bytes=%d  sha256=%s  reserialize_byte_identical=%s",
        out, len(text.encode()), sha, text == text2,
    )

    # ----- VERIFY 1: floor reproduction FROM THE WRITTEN FILE -----
    reloaded = yaml.safe_load(out.read_text(encoding="utf-8"))
    feats = _features_from_dump(reloaded["real_by_city"])
    feats.update(_features_from_dump(reloaded["real_train_by_city"]))
    rebuilt = build_floor_artifact_payload(
        feats, release=release, held_out_cities=held_out, train_cities=train,
        min_n=50, alpha=0.05, delta=0.15, tile_coverage=extraction.tile_coverage,
    )

    def fmap(floors: list[dict]) -> dict:
        return {(r["city"], r["metric"], tuple(r["stratum"])): r["floor_all"] for r in floors}

    fm_new, fm_lock = fmap(rebuilt["floors"]), fmap(locked.payload["floors"])
    kn, kl = set(fm_new), set(fm_lock)
    mism = [(k, fm_new[k], fm_lock[k]) for k in (kn & kl) if abs(fm_new[k] - fm_lock[k]) > 1e-12]
    exact = sum(1 for k in (kn & kl) if fm_new[k] == fm_lock[k])
    logger.info(
        "FLOOR REPRO: rebuilt_rows=%d locked_rows=%d shared=%d only_rebuilt=%d "
        "only_locked=%d exact_match=%d mismatch=%d",
        len(kn), len(kl), len(kn & kl), len(kn - kl), len(kl - kn), exact, len(mism),
    )
    for k, a, b in mism[:8]:
        logger.info("  MISMATCH %s rebuilt=%r locked=%r", k, a, b)
    for k in sorted(kl - kn)[:8]:
        logger.info("  ONLY_LOCKED %s", k)
    for k in sorted(kn - kl)[:8]:
        logger.info("  ONLY_REBUILT %s", k)
    repro_ok = (kn == kl) and not mism
    logger.info("FLOOR_REPRODUCTION_OK=%s", repro_ok)

    # ----- VERIFY 2: min_n=50 coverage -----
    surv: Counter = Counter()
    thin: Counter = Counter()
    thin_detail = []
    for (city, stratum, metric), samples in extraction.features.items():
        if len(samples) >= 50:
            surv[city] += 1
        else:
            thin[city] += 1
            thin_detail.append((city, metric, tuple(stratum), len(samples)))
    logger.info(
        "COVERAGE (min_n=50): survive_cells=%d thin_cells=%d",
        sum(surv.values()), sum(thin.values()),
    )
    logger.info("--- per-city (city: survive / thin) ---")
    for c in cities:
        tag = "HELDOUT" if c in held_set else "train"
        logger.info("  [%s] %-22s survive=%4d thin=%4d", tag, c, surv[c], thin[c])
    logger.info("--- HELD-OUT thin (city, metric, stratum, n<50) — excluded-and-reported ---")
    for d in sorted(t for t in thin_detail if t[0] in held_set):
        logger.info("    %s", d)

    sys.stdout.flush()
    sys.stderr.flush()
    print("DUMP_VERIFY_DONE")
    sys.stdout.flush()
    os._exit(0)  # dodge the GPU-less torch interpreter-teardown hang (memory lesson)


if __name__ == "__main__":
    main()

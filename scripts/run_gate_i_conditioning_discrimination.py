#!/usr/bin/env python3
"""scripts/run_gate_i_conditioning_discrimination.py — Task-9 gate input (i).

Thin DRIVER over the LOCKED verdict API
(``cfm.eval.conditioning_discrimination``). It:

  1. Extracts per-(city, full-stratum, metric) real feature scalars from the held-out
     EU tiles (``extract_features_by_city_stratum_metric``).
  2. Computes the conditioning-discrimination verdict (BH multiple-comparison guard;
     ``conditioning_discrimination_verdict``).
  3. Writes a canonical-YAML report with every n reported beside its denominator:
     the verdict, per-metric verdict, the full per-(city, stratum, metric) n map, and
     the per-pair list (metric, stratum, cities, n, ks, floor, raw/BH p, significance).

PASS ⇒ the worst-case bar is valid; FAIL ⇒ T5 reopens; UNSUPPORTED ⇒ the held-out
set can't support the test at full granularity (report, do NOT coarsen).

Runs on Leonardo CPU against the real corpus BEFORE any GPU pilot (model-independent).
There is no corpus locally; the verdict logic is unit-tested in
``tests/eval/test_conditioning_discrimination.py``.

    uv run python scripts/run_gate_i_conditioning_discrimination.py \
        --release 2026-04-15.0 \
        --cities eisenhuttenstadt glasgow krakow munich
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# iCloud-safe sys.path inject — mirrors scripts/build_multiregion_train_shards.py
# (parents[1] = repo root; underscore-prefixed .pth files are hidden in the
# iCloud-synced .venv so the editable install can't be relied on).
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.io import canonicalize_yaml  # noqa: E402
from cfm.eval.conditioning_discrimination import (  # noqa: E402
    DEFAULT_CITIES,
    ConditioningDiscriminationResult,
    conditioning_discrimination_verdict,
    extract_features_by_city_stratum_metric,
)

logger = logging.getLogger(__name__)

_DEFAULT_RELEASE = "2026-04-15.0"
_DEFAULT_REPORT_OUT = "reports/2026-06-10-gate-i-conditioning-discrimination-result.yaml"


def _result_to_report_dict(result: ConditioningDiscriminationResult) -> dict:
    """Flatten the result into YAML-safe primitives (tuple keys -> explicit lists)."""
    n_map = [
        {
            "city": city,
            "stratum": list(stratum),
            "metric": metric,
            "n": n,
        }
        for (city, stratum, metric), n in sorted(
            result.n_by_city_stratum_metric.items(),
            key=lambda kv: (kv[0][2], [str(x) for x in kv[0][1]], kv[0][0]),
        )
    ]
    pairs = [
        {
            "metric": p.metric,
            "stratum": list(p.stratum),
            "city_a": p.city_a,
            "city_b": p.city_b,
            "n_a": p.n_a,
            "n_b": p.n_b,
            "ks": float(p.ks),
            "floor": float(p.floor),
            "p_raw": float(p.p_raw),
            "p_bh": float(p.p_bh),
            "significant": bool(p.significant),
        }
        for p in result.pairs
    ]
    return {
        "verdict": result.verdict,
        "per_metric_verdict": dict(result.per_metric_verdict),
        "min_n": result.min_n,
        "alpha": result.alpha,
        "n_qualifying_comparisons": result.n_qualifying_comparisons,
        "n_excluded_thin": result.n_excluded_thin,
        "n_strata_too_few_cities": result.n_strata_too_few_cities,
        "n_by_city_stratum_metric": n_map,
        "pairs": pairs,
    }


def _print_summary(result: ConditioningDiscriminationResult, report_path: Path) -> None:
    sig = [p for p in result.pairs if p.significant]
    print("=" * 72)
    print("Gate input (i): conditioning-discrimination")
    print("=" * 72)
    print(f"  VERDICT                  : {result.verdict}")
    print(f"  per-metric verdict       : {result.per_metric_verdict}")
    print(f"  min_n / alpha            : {result.min_n} / {result.alpha}")
    print(f"  qualifying comparisons   : {result.n_qualifying_comparisons}")
    print(f"  thin-n cells excluded    : {result.n_excluded_thin}")
    print(f"  strata <2 cities         : {result.n_strata_too_few_cities}")
    print(f"  BH-significant pairs     : {len(sig)}")
    for p in sig:
        print(
            f"    FAIL  metric={p.metric} stratum={p.stratum} "
            f"{p.city_a}-{p.city_b} ks={p.ks:.4f} floor={p.floor:.4f} "
            f"p_raw={p.p_raw:.4g} p_bh={p.p_bh:.4g}"
        )
    print(f"  report                   : {report_path}")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Task-9 gate input (i): conditioning-discrimination"
    )
    parser.add_argument("--release", default=_DEFAULT_RELEASE)
    parser.add_argument("--cities", nargs="+", default=list(DEFAULT_CITIES))
    parser.add_argument("--min-n", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--report-out", default=_DEFAULT_REPORT_OUT)
    args = parser.parse_args(argv)

    logger.info("extracting features for cities=%s release=%s", args.cities, args.release)
    features = extract_features_by_city_stratum_metric(args.release, args.cities)
    logger.info("extracted %d (city, stratum, metric) cells", len(features))

    result = conditioning_discrimination_verdict(features, min_n=args.min_n, alpha=args.alpha)

    report_path = (
        (_REPO / args.report_out)
        if not Path(args.report_out).is_absolute()
        else Path(args.report_out)
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(canonicalize_yaml(_result_to_report_dict(result)), encoding="utf-8")

    _print_summary(result, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

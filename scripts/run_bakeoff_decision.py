"""Bake-off decision runner CLI (readiness-closure Task 26; Task-12 surface).

Thin orchestration over ``cfm.eval.bakeoff_decision.decide``: load the
per-(backbone, scale) eval-results YAML + the real-feature YAML, hand decide()
the floor-artifact PATH (so the sha/lock refusal happens inside the decision
layer, never against a pre-parsed payload), the holdout manifest and the
persisted Rule-2 basis record, then persist the YAML-safe decision record.

Input grammars (feature records share the floor artifact's stratum grammar):

eval-results YAML::

    backbones:
      - backbone: transformer-ar
        structural_check_ok: true
        scales:
          - scale_params: 30000000
            gen_by_city:
              krakow:
                - {metric: building_area_m2, stratum: [R, S1, 1, inland],
                   samples: [12.5, ...]}

real-features YAML::

    real_by_city:        {<held-out city>: [<feature record>, ...], ...}
    real_train_by_city:  {<training city>: [<feature record>, ...], ...}

Any refusal (tampered floor, incomplete cities, basis mismatch, memorizer)
propagates as its named exception — the CLI never converts a refusal into a
quiet nonzero, and no decision file is written on refusal.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# iCloud-Drive-safe path inject (same pattern as scripts/smoke.py): rely on
# the repo's src/ directly rather than the editable-install .pth files.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfm.eval.bakeoff_decision import (  # after the path inject
    BackboneEval,
    GenFeatures,
    decide,
    decision_record,
)

logger = logging.getLogger(__name__)


def _features_from_records(records: list[dict]) -> GenFeatures:
    """[{metric, stratum, samples}] -> {(metric, tuple(stratum)): samples} —
    the same (metric, stratum) grammar the floor artifact freezes."""
    return {
        (rec["metric"], tuple(rec["stratum"])): [float(x) for x in rec["samples"]]
        for rec in records
    }


def _city_features(by_city: dict[str, list[dict]]) -> dict[str, GenFeatures]:
    return {city: _features_from_records(records) for city, records in by_city.items()}


def _load_evals(path: Path) -> list[BackboneEval]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "backbones" not in data:
        raise ValueError(f"eval-results {path}: expected a YAML mapping with a 'backbones' key")
    return [
        BackboneEval(
            backbone=entry["backbone"],
            structural_check_ok=bool(entry["structural_check_ok"]),
            gen_by_city_by_scale={
                int(scale["scale_params"]): _city_features(scale["gen_by_city"])
                for scale in entry["scales"]
            },
        )
        for entry in data["backbones"]
    ]


def _load_real(path: Path) -> tuple[dict[str, GenFeatures], dict[str, GenFeatures]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("real_by_city", "real_train_by_city"):
        if not isinstance(data, dict) or key not in data:
            raise ValueError(f"real-features {path}: missing the '{key}' key (STRICT read)")
    return _city_features(data["real_by_city"]), _city_features(data["real_train_by_city"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase-2 bake-off decision runner")
    parser.add_argument("--eval-results", required=True, help="per-(backbone, scale) YAML")
    parser.add_argument("--real-features", required=True, help="real held-out + training YAML")
    parser.add_argument(
        "--floor-artifact",
        required=True,
        help="frozen conditioning-floor YAML (sha/lock verified INSIDE decide)",
    )
    parser.add_argument("--holdout-manifest", required=True, help="held_out_cities source")
    parser.add_argument("--persisted-basis", required=True, help="Rule-2 basis record YAML")
    parser.add_argument("--out", required=True, help="decision record YAML to write")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = _build_parser().parse_args(argv)
    real_by_city, real_train_by_city = _load_real(Path(args.real_features))
    decision = decide(
        _load_evals(Path(args.eval_results)),
        real_by_city,
        real_train_by_city,
        artifact=args.floor_artifact,  # the PATH: refusal lives in the decision layer
        holdout_manifest=args.holdout_manifest,
        persisted_basis=args.persisted_basis,
    )
    record = decision_record(decision)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(record, sort_keys=True), encoding="utf-8")
    logger.info(
        "decision: winner=%s (binding city %s, gap %.4f > floor %.4f; demoted %s) -> %s",
        record["winner"],
        record["binding_city"],
        record["binding_gap"],
        record["binding_city_floor"],
        record["demoted_cities"],
        out,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""scripts/build_multiregion_train_shards.py — Task 8 multi-region train-shard build.

Thin DRIVER over the LOCKED build API (``cfm.data.training.build_shards``). It:

  1. Resolves the TRAIN cities via ``train_cities`` (validated MINUS held-out — the
     whole-city exclusion that removes the 4 EU held-out cities by construction).
  2. STRUCTURAL GUARD: asserts no held-out city leaked into the build list.
  3. PRE-SNAPSHOT: records, per held-out city, whether a training manifest exists
     BEFORE the run (the baseline for the negative end-state check).
  4. Builds each train city's per-region ``training_manifest.yaml`` via
     ``build_training_shards`` (THIS is the per-city persistence call).
  5. VERIFIED END-STATE — RE-READS every manifest from disk (never trusts the build
     return value): each train-city manifest exists, has ``region == city``,
     ``manifest_schema_version == "1.0"``, and a ``n_training_tiles`` int; the count of
     train-city manifests on disk equals ``len(cities)``; and NO held-out manifest
     APPEARED relative to the pre-snapshot.

The core ``build_all_train_cities`` is import-testable (no argparse); the CLI is a thin
wrapper. Run on Leonardo against the real corpus (the controller submits the sbatch);
there is no corpus locally, so the unit tests drive the core function with synthetic
fixtures and tmp_path path-helper monkeypatches.

    uv run python scripts/build_multiregion_train_shards.py \
        --release 2026-04-15.0 \
        --g4 reports/2026-06-05-phase-2-g4-corpus-dod.yaml \
        --holdout-manifest data/processed/eval_set/2026-04-15.0/multiregion/holdout_manifest.yaml
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import subprocess
import sys
from pathlib import Path

import yaml

# iCloud-safe sys.path inject — mirrors scripts/eval/build_multiregion_manifest.py
# (parents[1] = repo root; underscore-prefixed .pth files are hidden in the
# iCloud-synced .venv so the editable install can't be relied on).
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.io import canonicalize_yaml  # noqa: E402
from cfm.data.training.build_shards import (  # noqa: E402
    DEFAULT_G4_ROLLUP,
    _load_or_pass,
    build_train_city_manifest,
    train_cities,
)
from cfm.data.training.paths import (  # noqa: E402
    training_manifest_path,
    training_region_dir,  # noqa: F401  (re-exported so the runner's namespace carries it; tests monkeypatch R.training_region_dir to redirect the per-city build under tmp_path)
)

_LOG = logging.getLogger("build_multiregion_train_shards")

_DEFAULT_RELEASE = "2026-04-15.0"
_DEFAULT_G4 = DEFAULT_G4_ROLLUP  # one-source constant; CLI default behavior unchanged
_DEFAULT_HOLDOUT = "data/processed/eval_set/2026-04-15.0/multiregion/holdout_manifest.yaml"
_DEFAULT_REPORT = "reports/2026-06-10-task8-multiregion-train-shards-build.yaml"


def build_all_train_cities(
    release: str,
    *,
    g4_rollup: dict | Path | str,
    holdout_manifest: dict | Path | str,
    report_out: Path | None = None,
) -> dict:
    """Build every TRAIN city's per-region training manifest, then verify end-state.

    Returns a DETERMINISTIC summary dict (no timestamps/volatile fields) so the unit
    tests can assert byte-identity across runs. The CLI may additionally stamp the git
    commit + a generated_at into the written report file.
    """
    # 1. Resolve the train cities (validated minus held-out), sorted/deterministic.
    cities = train_cities(release, g4_rollup=g4_rollup, holdout_manifest=holdout_manifest)

    # 2. Held-out selector (accept dict or path, same helper the build API uses).
    holdout = _load_or_pass(holdout_manifest)
    held = set(holdout["held_out_cities"])

    # 3. STRUCTURAL GUARD — no held-out city may be in the build list.
    leaked = held & set(cities)
    assert held.isdisjoint(set(cities)), (
        f"held-out city leaked into the train-build list: {sorted(leaked)}"
    )

    # 4. PRE-SNAPSHOT — held-out manifest existence BEFORE building (negative baseline).
    held_pre = {h: training_manifest_path(release, h).exists() for h in sorted(held)}

    # 5. Build each train city's per-region manifest (the persistence call).
    #    I1-SAFE writer: build_train_city_manifest uses the all-validated-tiles path; the
    #    single-region build_training_shards RAISES for a train city (the I1 boundary).
    _LOG.info("building %d train cities: %s", len(cities), cities)
    for city in cities:
        _LOG.info("building city %s", city)
        build_train_city_manifest(release, city)

    # 6. VERIFIED END-STATE — RE-READ from disk; never trust the build return value.
    per_city: dict[str, int] = {}
    for city in cities:
        p = training_manifest_path(release, city)
        assert p.exists(), f"train-city manifest missing on disk: {city} ({p})"
        m = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert m["region"] == city, f"manifest region mismatch for {city}: got {m['region']!r}"
        assert m["manifest_schema_version"] == "1.0", (
            f"unexpected manifest schema for {city}: {m['manifest_schema_version']!r}"
        )
        n = m["n_training_tiles"]
        assert isinstance(n, int), f"n_training_tiles not an int for {city}: {n!r}"
        per_city[city] = n

    # Count of train-city manifests on disk equals the build list length.
    assert len(per_city) == len(cities), (
        f"expected {len(cities)} train-city manifests, verified {len(per_city)}"
    )

    # NEGATIVE: no held-out manifest APPEARED relative to the pre-snapshot.
    for h in sorted(held):
        now = training_manifest_path(release, h).exists()
        appeared = now and not held_pre[h]
        assert not appeared, (
            f"a held-out manifest was CREATED by the run (must never happen): {h} "
            f"({training_manifest_path(release, h)})"
        )

    total = sum(per_city.values())
    summary: dict = {
        "release": release,
        "n_train_cities": len(cities),
        "total_training_tiles": total,
        "held_out_cities": sorted(held),
        "per_city": {c: per_city[c] for c in sorted(per_city)},
        "train_cities": cities,
    }

    # 7. Optionally persist the summary as canonical YAML (deterministic).
    if report_out is not None:
        report_out = Path(report_out)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(canonicalize_yaml(summary), encoding="utf-8")

    return summary


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_REPO)).decode().strip()
        )
    except Exception:  # pragma: no cover - best-effort stamp, never fatal
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default=_DEFAULT_RELEASE)
    parser.add_argument("--g4", default=_DEFAULT_G4, type=Path)
    parser.add_argument("--holdout-manifest", default=_DEFAULT_HOLDOUT, type=Path)
    parser.add_argument("--report-out", default=_DEFAULT_REPORT, type=Path)
    args = parser.parse_args(argv)

    summary = build_all_train_cities(
        args.release,
        g4_rollup=args.g4,
        holdout_manifest=args.holdout_manifest,
        report_out=args.report_out,
    )

    # CLI-only volatile stamps written ON TOP of the deterministic core summary.
    stamped = dict(summary)
    stamped["git_commit"] = _git_commit()
    stamped["generated_at"] = _dt.datetime.now(_dt.UTC).isoformat()
    Path(args.report_out).write_text(canonicalize_yaml(stamped), encoding="utf-8")

    print("=== multi-region train-shard build ===")
    print(f"release            : {summary['release']}")
    print(f"n_train_cities     : {summary['n_train_cities']}")
    print(f"total_training_tiles: {summary['total_training_tiles']}")
    print(f"held_out (excluded): {', '.join(summary['held_out_cities'])}")
    print(f"report             : {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

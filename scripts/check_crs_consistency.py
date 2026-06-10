#!/usr/bin/env python3
"""scripts/check_crs_consistency.py — Task 12 three-way CRS consistency check (F2).

For each region, verifies that THREE mandatory sources agree on the CRS:

  1. region config ``configs/data/regions/<region>.yaml`` -> ``projected_crs``
  2. sub-D manifest ``data/processed/sub_d/<release>/<region>/manifest.yaml``
     -> ``region_crs``
  3. on-disk tile-dir labels ``tile=EPSG<code>_i<i>_j<j>`` under the region's
     sub-D dir (ALL labels must match the config-derived label)

plus a FOURTH, optional leg: the holdout manifest's per-region ``crs`` where
present (multiregion manifest first, then the SG manifest; a region absent
from both, or present without a ``crs`` key, SKIPS this leg — recorded as
``holdout_crs: null``, never a failure).

METERS GUARD: the CRS must be projected/metric. Pin = label matches
``^EPSG\\d+$`` AND ``projected_crs`` is in the allowlist derived from the
region configs (the 8 values they declare). Geographic codes (4326-class) are
checked against an explicit reject-set FIRST so the diff names them as such.

Exit 0 iff every checked region is PASS; nonzero otherwise, with named
per-city diffs on the log. ``--report`` writes a YAML with per-city verdicts.

    uv run python scripts/check_crs_consistency.py \\
        --regions singapore --report /tmp/crs_singapore.yaml
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Callable
from pathlib import Path

import yaml

# iCloud-safe sys.path inject — mirrors scripts/build_multiregion_train_shards.py
# (parents[1] = repo root; underscore-prefixed .pth files are hidden in the
# iCloud-synced .venv so the editable install can't be relied on).
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.io import canonicalize_yaml  # noqa: E402
from cfm.eval.holdout.paths import (  # noqa: E402
    DEFAULT_RELEASE,
    holdout_manifest_path,
    multiregion_holdout_manifest_path,
    sub_d_region_dir,
)

_LOG = logging.getLogger("check_crs_consistency")

#: Projected/metric CRS allowlist — exactly the 8 distinct ``projected_crs``
#: values declared across configs/data/regions/*.yaml (verified 2026-06-10).
#: A guard test below the configs is the lock; extend ONLY when a new region
#: config introduces a new projected CRS.
PROJECTED_CRS_ALLOWLIST: frozenset[str] = frozenset(
    {
        "EPSG:3414",  # Singapore SVY21
        "EPSG:25829",  # ETRS89 / UTM 29N
        "EPSG:25830",
        "EPSG:25831",
        "EPSG:25832",
        "EPSG:25833",
        "EPSG:25834",
        "EPSG:25835",
    }
)

#: Explicit geographic (degrees, NOT meters) reject-set, checked BEFORE the
#: allowlist so the diff message names "geographic CRS" specifically.
GEOGRAPHIC_CRS_REJECT: frozenset[str] = frozenset(
    {"EPSG:4326", "EPSG:4258", "EPSG:4269", "EPSG:4979"}
)

_LABEL_RE = re.compile(r"^EPSG\d+$")
_TILE_DIR_RE = re.compile(r"^tile=(EPSG\d+)_i-?\d+_j-?\d+$")

_DEFAULT_CONFIG_ROOT = _REPO / "configs" / "data" / "regions"


def _holdout_crs_for_region(
    region: str, sg_holdout: dict | None, mr_holdout: dict | None
) -> str | None:
    """Resolve the holdout-manifest ``crs`` leg for ``region``, or None to skip.

    Rule (locked): check the multiregion manifest's ``regions[region]['crs']``
    if the region appears there; else the SG manifest's entry (``crs`` absent
    -> skip). A region in neither manifest skips the leg entirely.
    """
    for manifest in (mr_holdout, sg_holdout):
        if manifest is None:
            continue
        entry = manifest.get("regions", {}).get(region)
        if entry is not None:
            return entry.get("crs")
    return None


def check_region(
    release: str,
    region: str,
    *,
    config_root: Path,
    sub_d_root_fn: Callable[[str, str], Path],
    sg_holdout: dict | None,
    mr_holdout: dict | None,
) -> dict:
    """Check one region; returns its verdict dict (never raises on data defects).

    The config / sub-D-manifest / tile-dir legs are MANDATORY (missing -> FAIL);
    the holdout leg is optional (absent -> skipped, ``holdout_crs: null``).
    """
    diffs: list[str] = []
    meters_diffs: list[str] = []

    # Leg 1 (MANDATORY): region config projected_crs.
    config_crs: str | None = None
    config_path = Path(config_root) / f"{region}.yaml"
    if not config_path.exists():
        diffs.append(f"region config missing: {config_path}")
    else:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config_crs = cfg.get("projected_crs")
        if config_crs is None:
            diffs.append(f"region config has no projected_crs key: {config_path}")

    # Leg 2 (MANDATORY): sub-D manifest region_crs.
    sub_d_crs: str | None = None
    region_dir = sub_d_root_fn(release, region)
    manifest_path = region_dir / "manifest.yaml"
    if not manifest_path.exists():
        diffs.append(f"sub-D manifest missing: {manifest_path}")
    else:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        sub_d_crs = manifest.get("region_crs")
        if sub_d_crs is None:
            diffs.append(f"sub-D manifest has no region_crs key: {manifest_path}")

    # Leg 3 (MANDATORY): on-disk tile-dir labels.
    tile_dir_labels: list[str] = []
    if region_dir.is_dir():
        labels = set()
        for child in region_dir.iterdir():
            if not child.is_dir():
                continue
            m = _TILE_DIR_RE.match(child.name)
            if m:
                labels.add(m.group(1))
        tile_dir_labels = sorted(labels)
    if not tile_dir_labels:
        diffs.append(f"no tile=EPSG..._i_j dirs found under {region_dir}")

    # Leg 4 (OPTIONAL): holdout manifest per-region crs, where present.
    holdout_crs = _holdout_crs_for_region(region, sg_holdout, mr_holdout)

    # Meters guard (on the config CRS + the dir labels).
    expected_label: str | None = None
    if config_crs is not None:
        expected_label = config_crs.replace(":", "")
        if config_crs in GEOGRAPHIC_CRS_REJECT:
            meters_diffs.append(
                f"geographic CRS (degrees, not meters): projected_crs={config_crs!r}"
            )
        elif config_crs not in PROJECTED_CRS_ALLOWLIST:
            meters_diffs.append(
                f"projected_crs {config_crs!r} not in the region-config allowlist "
                f"{sorted(PROJECTED_CRS_ALLOWLIST)}"
            )
        if not _LABEL_RE.match(expected_label):
            meters_diffs.append(
                f"config-derived label {expected_label!r} does not match ^EPSG\\d+$"
            )

    # Cross-source comparisons (only where both sides were readable).
    if config_crs is not None and sub_d_crs is not None and sub_d_crs != config_crs:
        diffs.append(
            f"sub-D manifest region_crs={sub_d_crs!r} != config projected_crs={config_crs!r}"
        )
    if expected_label is not None and tile_dir_labels:
        wrong = [lbl for lbl in tile_dir_labels if lbl != expected_label]
        if wrong:
            diffs.append(f"tile-dir label(s) {wrong} != config-derived label {expected_label!r}")
        if len(tile_dir_labels) > 1:
            diffs.append(f"multiple distinct tile-dir labels in one region: {tile_dir_labels}")
    if config_crs is not None and holdout_crs is not None and holdout_crs != config_crs:
        diffs.append(f"holdout manifest crs={holdout_crs!r} != config projected_crs={config_crs!r}")

    projected_ok = config_crs is not None and not meters_diffs
    consistent = not diffs
    all_diffs = diffs + meters_diffs
    return {
        "region": region,
        "config_crs": config_crs,
        "sub_d_crs": sub_d_crs,
        "tile_dir_labels": tile_dir_labels,
        "holdout_crs": holdout_crs,
        "projected_ok": projected_ok,
        "consistent": consistent,
        "verdict": "PASS" if (consistent and projected_ok) else "FAIL",
        "diffs": all_diffs,
    }


def run_check(
    release: str,
    regions: list[str],
    *,
    config_root: Path,
    sub_d_root_fn: Callable[[str, str], Path],
    sg_holdout: dict | None,
    mr_holdout: dict | None,
    report: Path | None = None,
) -> tuple[int, dict]:
    """Check every region; return (exit_code, summary). Exit 0 iff all PASS."""
    per_region: dict[str, dict] = {}
    for region in regions:
        verdict = check_region(
            release,
            region,
            config_root=config_root,
            sub_d_root_fn=sub_d_root_fn,
            sg_holdout=sg_holdout,
            mr_holdout=mr_holdout,
        )
        per_region[region] = verdict
        if verdict["verdict"] == "FAIL":
            for diff in verdict["diffs"]:
                _LOG.error("%s: %s", region, diff)
        else:
            _LOG.info("%s: PASS", region)

    n_pass = sum(1 for v in per_region.values() if v["verdict"] == "PASS")
    summary = {
        "release": release,
        "n_regions": len(per_region),
        "n_pass": n_pass,
        "n_fail": len(per_region) - n_pass,
        "per_region": {r: per_region[r] for r in sorted(per_region)},
    }

    if report is not None:
        report = Path(report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(canonicalize_yaml(summary), encoding="utf-8")

    return (0 if summary["n_fail"] == 0 else 1), summary


def _load_yaml_if_exists(path: Path) -> dict | None:
    """Tolerant holdout-manifest read: absent file -> None (leg skipped)."""
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _discover_regions(release: str) -> list[str]:
    """Default region list: sub-D region dirs present on disk for the release."""
    sub_d_release_dir = sub_d_region_dir(release, "_x").parent
    if not sub_d_release_dir.is_dir():
        return []
    return sorted(d.name for d in sub_d_release_dir.iterdir() if d.is_dir())


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help="regions to check (default: every sub-D region dir for the release)",
    )
    parser.add_argument("--report", default=None, type=Path, help="write YAML report here")
    args = parser.parse_args(argv)

    regions = args.regions if args.regions is not None else _discover_regions(args.release)
    if not regions:
        _LOG.error("no regions to check (none given, none discovered under sub-D)")
        return 1

    rc, summary = run_check(
        args.release,
        regions,
        config_root=_DEFAULT_CONFIG_ROOT,
        sub_d_root_fn=sub_d_region_dir,
        sg_holdout=_load_yaml_if_exists(holdout_manifest_path(args.release)),
        mr_holdout=_load_yaml_if_exists(multiregion_holdout_manifest_path(args.release)),
        report=args.report,
    )

    print("=== three-way CRS consistency check ===")
    print(f"release  : {summary['release']}")
    print(f"n_regions: {summary['n_regions']}")
    print(f"n_pass   : {summary['n_pass']}")
    print(f"n_fail   : {summary['n_fail']}")
    for region, v in summary["per_region"].items():
        print(f"  {region}: {v['verdict']}")
        for diff in v["diffs"]:
            print(f"    DIFF: {diff}")
    if args.report is not None:
        print(f"report   : {args.report}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

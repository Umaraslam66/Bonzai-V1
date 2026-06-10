#!/usr/bin/env python3
"""scripts/measure_emergence_floor.py — derive a region's emergence floor (Task 13, F13/F15).

Computes ``density = holdout_polygons_per_active_cell(release, region)`` on the
region's REAL holdout, derives ``floor = frac * density``, and writes/replaces the
region's entry in the provenance-bearing artifact ``configs/eval/emergence_floors.yaml``
(deterministic ``canonicalize_yaml`` serialization; ``derived_at`` = git sha).

The density measurement needs the real sub-F tile data (Leonardo ``$WORK``); tests
inject a synthetic ``density_fn``. Rewrites drop hand-written comments by design:
the writer emits a fixed header + canonical YAML, so the file stays byte-deterministic.

NOTE: ``floor`` in the artifact is the authoritative gate value; writer-emitted entries
satisfy floor == frac * holdout_density exactly, but the singapore SEED hand-carries the
historical 1.96 (vs 1.9625 exact) for diagnostic comparability — re-deriving singapore
writes 1.9625 and deliberately fails the seeded-value resolution test
(tests/training/test_emergence_floor_resolution.py::
test_resolves_seeded_singapore_floor_from_real_yaml); that fail-loud is intended.

    uv run python scripts/measure_emergence_floor.py \\
        --release 2026-04-15.0 --region krakow
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import yaml

# iCloud-safe sys.path inject — mirrors scripts/check_crs_consistency.py
# (parents[1] = repo root; underscore-prefixed .pth files are hidden in the
# iCloud-synced .venv so the editable install can't be relied on).
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.io import canonicalize_yaml  # noqa: E402
from cfm.eval.emergence import (  # noqa: E402
    EMERGENCE_FRAC_OF_HOLDOUT_DENSITY,
    emergence_floor_polygons_per_cell,
)
from cfm.eval.geometry import holdout_polygons_per_active_cell  # noqa: E402

_LOG = logging.getLogger("measure_emergence_floor")

#: Every per-region entry must carry the full schema (resolution + writer both validate).
REQUIRED_ENTRY_KEYS: frozenset[str] = frozenset(
    {"floor", "holdout_density", "frac", "derived_at", "derivation_regime"}
)

#: The denominator convention the density measurement runs under — recorded in every
#: entry so a future regime change (e.g. truncated cells, active-only denominator)
#: is visible in provenance instead of silently shifting the floor's meaning.
DERIVATION_REGIME: dict[str, str] = {
    "cell_length": "full",
    "denominator": "all_nonempty_cells",
}

DEFAULT_FLOORS_PATH = _REPO / "configs" / "eval" / "emergence_floors.yaml"

_HEADER = (
    "# Per-region emergence floors (readiness-closure Task 13, F13/F15).\n"
    "# `floor` is the AUTHORITATIVE gate value; writer-emitted entries satisfy\n"
    "# floor == frac * holdout_density (polygons per active cell, REAL holdout)\n"
    "# exactly by construction. Hand-seeded entries may hand-carry a historical\n"
    "# floor for diagnostic comparability (see their derived_at).\n"
    "# Written by scripts/measure_emergence_floor.py (derived_at = git sha);\n"
    "# resolved by scripts/train_scaffold.py run_short via cfg.region (fail-closed).\n"
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover - mirrors train_scaffold._git_commit
        return "unknown"


def _validate_entries(regions: dict, path: Path) -> None:
    """Refuse to (re)serialize an artifact with a schema-broken entry."""
    for region, entry in regions.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"emergence floor entry for {region!r} in {path} is not a mapping: {entry!r}"
            )
        missing = sorted(REQUIRED_ENTRY_KEYS - entry.keys())
        if missing:
            raise ValueError(
                f"emergence floor entry for {region!r} in {path} is missing required "
                f"keys {missing}; refusing to write a schema-broken artifact"
            )


def measure_and_write(
    release: str,
    region: str,
    *,
    floors_path: Path,
    frac: float = EMERGENCE_FRAC_OF_HOLDOUT_DENSITY,
    density_fn=holdout_polygons_per_active_cell,
    git_sha: str | None = None,
) -> dict:
    """Measure ``region``'s holdout density, derive the floor, persist the entry.

    Loads (or initializes) the floors YAML, writes/replaces the region's entry with
    the full schema, schema-validates EVERY entry, and writes back deterministically.
    Returns the new entry.
    """
    floors_path = Path(floors_path)
    density = float(density_fn(release=release, region=region))
    entry = {
        "floor": emergence_floor_polygons_per_cell(holdout_polys_per_cell=density, frac=frac),
        "holdout_density": density,
        "frac": frac,
        "derived_at": git_sha if git_sha is not None else _git_sha(),
        "derivation_regime": dict(DERIVATION_REGIME),
    }

    if floors_path.exists():
        data = yaml.safe_load(floors_path.read_text(encoding="utf-8")) or {}
    else:
        data = {"schema_version": "1.0", "regions": {}}
    regions = data.setdefault("regions", {})
    regions[region] = entry
    _validate_entries(regions, floors_path)

    floors_path.parent.mkdir(parents=True, exist_ok=True)
    floors_path.write_text(_HEADER + canonicalize_yaml(data), encoding="utf-8")
    _LOG.info(
        "%s: floor=%.4f (frac=%s x holdout_density=%.4f) -> %s",
        region,
        entry["floor"],
        frac,
        density,
        floors_path,
    )
    return entry


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--floors-path", type=Path, default=DEFAULT_FLOORS_PATH)
    parser.add_argument(
        "--frac",
        type=float,
        default=EMERGENCE_FRAC_OF_HOLDOUT_DENSITY,
        help="floor = frac * holdout density (recorded PI choice; default 0.25)",
    )
    args = parser.parse_args(argv)

    # density_fn looked up via the MODULE global (not the def-time default) so tests
    # can monkeypatch the measurement without real sub-F tile data.
    entry = measure_and_write(
        args.release,
        args.region,
        floors_path=args.floors_path,
        frac=args.frac,
        density_fn=holdout_polygons_per_active_cell,
    )
    print(f"{args.region}: floor={entry['floor']} derived_at={entry['derived_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

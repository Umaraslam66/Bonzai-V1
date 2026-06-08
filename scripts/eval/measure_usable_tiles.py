"""scripts/eval/measure_usable_tiles.py — read-only usable-tile census (gate b).

For each city, glob its sub-D region dir for ``tile=*`` directories, read each
tile's ``macro_core.parquet``, and count (a) total tiles that have a macro_core
and (b) tiles that are 'usable' for coherence power (>= 3 road-carrying interior
edges, per ``cfm.eval.usable_tiles.tile_is_usable`` / the shared
``interior_road_graph`` builder). Writes a YAML summary.

This is a measurement harness only: it never writes into the corpus. Run on
Leonardo against the real sub-D corpus (the controller runs Step 5):

    uv run python scripts/eval/measure_usable_tiles.py \
        --release 2026-04-15.0 \
        --cities glasgow,almere,... \
        --out reports/usable-n.yaml

Output YAML:

    release: "2026-04-15.0"
    cities:
      glasgow: {n_tiles: <int>, n_usable_tiles: <int>}
      ...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# iCloud-safe sys.path inject — mirrors scripts/sub_g/derive_phase1_region.py
# (parents[2] = repo root; underscore-prefixed .pth files are hidden in the
# iCloud-synced .venv so the editable install can't be relied on).
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.sub_d.io import read_macro_core_parquet  # noqa: E402
from cfm.eval.holdout.paths import sub_d_region_dir  # noqa: E402
from cfm.eval.usable_tiles import tile_is_usable  # noqa: E402

logger = logging.getLogger("measure_usable_tiles")

#: Per-tile sub-D core artifact filename (spec §11.2 / sub_d/pipeline.py).
MACRO_CORE_FILENAME = "macro_core.parquet"


def measure_city(release: str, city: str) -> dict[str, int]:
    """Count total + usable tiles for one city's sub-D region dir.

    A 'tile' is any ``tile=*`` dir that contains a ``macro_core.parquet``.
    'usable' is decided by the shared ``tile_is_usable`` predicate so it stays
    identical to what the coherence metric scores.
    """
    region_dir = sub_d_region_dir(release, city)
    n_tiles = 0
    n_usable = 0
    if not region_dir.is_dir():
        logger.warning("city %s: region dir missing (%s); 0 tiles", city, region_dir)
        return {"n_tiles": 0, "n_usable_tiles": 0}

    for tile_dir in sorted(region_dir.glob("tile=*")):
        macro_core = tile_dir / MACRO_CORE_FILENAME
        if not macro_core.exists():
            logger.debug(
                "city %s: %s has no %s; skipping", city, tile_dir.name, MACRO_CORE_FILENAME
            )
            continue
        rows = read_macro_core_parquet(macro_core)
        n_tiles += 1
        if tile_is_usable(rows):
            n_usable += 1

    logger.info("city %s: %d tiles, %d usable", city, n_tiles, n_usable)
    return {"n_tiles": n_tiles, "n_usable_tiles": n_usable}


def measure(release: str, cities: list[str]) -> dict[str, object]:
    """Build the full census payload for all cities."""
    return {
        "release": release,
        "cities": {city: measure_city(release, city) for city in cities},
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, help="sub-D release id, e.g. 2026-04-15.0")
    parser.add_argument(
        "--cities",
        required=True,
        help="comma-separated city/region names (sub-D region dir names)",
    )
    parser.add_argument("--out", required=True, type=Path, help="output YAML path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    payload = measure(args.release, cities)

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

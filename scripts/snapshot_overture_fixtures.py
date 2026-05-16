"""Snapshot Overture fixtures used by tests.

Two modes:

  --mode bootstrap  Write synthetic, schema-matching parquets to
                    tests/fixtures/overture_mini/. No S3, no network.
                    Used to initialise the fixtures during sub-project A,
                    and to regenerate them on re-pin in offline situations.

  --mode s3         Fetch a tiny real bbox (currently a 0.01 deg x 0.01 deg
                    square near central Singapore) and write the resulting
                    rows to tests/fixtures/overture_mini/. Requires
                    network + the pinned Overture release. (Implemented in
                    Task 12; not available in this commit.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "overture_mini"

# Five synthetic rows per theme. Geometry bytes are placeholder values (we
# don't decode them in fast tests; the slow opt-in test uses real geometry).
_FAKE_WKB = b"\x01\x01\x00\x00\x00" + b"\x00" * 16  # WKB POINT placeholder

_SYNTHETIC: dict[str, dict] = {
    "buildings": {
        "id": ["b1", "b2", "b3", "b4", "b5"],
        "geometry": [_FAKE_WKB] * 5,
        "class": ["residential", "commercial", "residential", "industrial", "residential"],
        "height": [10.0, 25.0, 8.0, 15.0, 12.0],
        "num_floors": [3, 7, 2, 4, 4],
    },
    "places": {
        "id": ["p1", "p2", "p3", "p4", "p5"],
        "geometry": [_FAKE_WKB] * 5,
        "categories": [
            '{"primary": "restaurant"}',
            '{"primary": "school"}',
            '{"primary": "retail"}',
            '{"primary": "park_amenity"}',
            '{"primary": "transit_stop"}',
        ],
    },
    "transportation": {
        "id": ["t1", "t2", "t3", "t4", "t5"],
        "geometry": [_FAKE_WKB] * 5,
        "class": ["motorway", "primary", "residential", "service", "secondary"],
        "subtype": ["road", "road", "road", "road", "road"],
    },
    "base": {
        "id": ["a1", "a2", "a3", "a4", "a5"],
        "geometry": [_FAKE_WKB] * 5,
        "subtype": ["land", "water", "land", "water", "land_cover"],
    },
    "divisions": {
        "id": ["d1"],
        "geometry": [_FAKE_WKB],
        "country": ["SG"],
        "subtype": ["country"],
    },
}

# Explicit pyarrow types so casts are deterministic across pyarrow versions.
_TYPES: dict[str, dict[str, pa.DataType]] = {
    "buildings": {
        "id": pa.string(),
        "geometry": pa.binary(),
        "class": pa.string(),
        "height": pa.float64(),
        "num_floors": pa.int32(),
    },
    "places": {
        "id": pa.string(),
        "geometry": pa.binary(),
        "categories": pa.string(),
    },
    "transportation": {
        "id": pa.string(),
        "geometry": pa.binary(),
        "class": pa.string(),
        "subtype": pa.string(),
    },
    "base": {
        "id": pa.string(),
        "geometry": pa.binary(),
        "subtype": pa.string(),
    },
    "divisions": {
        "id": pa.string(),
        "geometry": pa.binary(),
        "country": pa.string(),
        "subtype": pa.string(),
    },
}


def _build_table(theme: str) -> pa.Table:
    cols = _SYNTHETIC[theme]
    types = _TYPES[theme]
    arrays = {name: pa.array(values, type=types[name]) for name, values in cols.items()}
    return pa.table(arrays)


def bootstrap() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for theme in _SYNTHETIC:
        out = FIXTURES_DIR / f"{theme}.parquet"
        table = _build_table(theme)
        pq.write_table(table, out)
        print(f"wrote {out.relative_to(REPO_ROOT)} ({table.num_rows} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("bootstrap", "s3"),
        default="bootstrap",
        help="bootstrap = synthetic, no network; s3 = real fetch (Task 12)",
    )
    args = parser.parse_args()
    if args.mode == "bootstrap":
        bootstrap()
        return 0
    s3_snapshot()
    return 0


def s3_snapshot() -> None:
    """Fetch a tiny real bbox in central Singapore from the pinned release."""

    import yaml

    from cfm.data.overture.backend import S3DuckDBBackend
    from cfm.data.overture.region import BboxScope

    # Read the pin so the snapshot tracks whatever is currently pinned.
    with (REPO_ROOT / "configs" / "data" / "overture_release.yaml").open() as f:
        release = yaml.safe_load(f)["release"]

    # 0.01 deg x 0.01 deg around 1.295N 103.855E (Bukit Timah/Bishan area).
    bbox = BboxScope.from_tuple((103.85, 1.29, 103.86, 1.30))
    backend = S3DuckDBBackend()
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for theme in _SYNTHETIC:  # reuse the theme list
        print(f"fetching theme={theme} release={release} bbox={bbox.as_tuple()} ...", flush=True)
        table = backend.read_theme(theme=theme, bbox=bbox, release=release)
        out = FIXTURES_DIR / f"{theme}.parquet"
        pq.write_table(table, out)
        print(f"  wrote {out.relative_to(REPO_ROOT)} ({table.num_rows} rows)")


if __name__ == "__main__":
    sys.exit(main())

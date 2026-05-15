from __future__ import annotations

from pathlib import Path
from typing import Protocol

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from cfm.data.overture.errors import OvertureUnreachable
from cfm.data.overture.region import BboxScope, SizeEstimate

# Phase-1 simplification: one primary type per theme.
# Overture's S3 layout is partitioned by both theme and type:
#   s3://.../release/{release}/theme={theme}/type={type}/*
# Multi-type fetches (e.g., base.land + base.water) are deferred to a later phase.
# See docs.overturemaps.org/getting-data/ for the full type catalogue.
THEME_TO_TYPE: dict[str, str] = {
    "buildings": "building",
    "places": "place",
    "transportation": "segment",  # roads/paths/etc.; connectors deferred
    "base": "water",  # needed for sea masking; land/land_cover deferred
    "divisions": "division_area",  # polygon geometry; division/boundary types deferred
}


class OvertureBackend(Protocol):
    """Reads Overture theme parquet for a bounding box at a given release.

    Phase 1 backends apply only the bounding-box filter. The region's
    admin polygon is NOT used here; downstream consumers apply it for
    precise clipping. See docs/data/handoffs.md.
    """

    def read_theme(
        self,
        *,
        theme: str,
        bbox: BboxScope,
        release: str,
    ) -> pa.Table: ...

    def estimate_size(
        self,
        *,
        theme: str,
        bbox: BboxScope,
        release: str,
    ) -> SizeEstimate: ...


class LocalFixtureBackend:
    """Reads from tests/fixtures/overture_mini/. Ignores bbox and release.

    Used by the fast test suite. The fixtures are committed parquets generated
    by scripts/snapshot_overture_fixtures.py.
    """

    def __init__(self, fixtures_dir: Path) -> None:
        self._fixtures_dir = Path(fixtures_dir)

    def read_theme(
        self,
        *,
        theme: str,
        bbox: BboxScope,
        release: str,
    ) -> pa.Table:
        path = self._fixtures_dir / f"{theme}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"no fixture parquet for theme={theme!r} at {path}")
        return pq.read_table(path)

    def estimate_size(
        self,
        *,
        theme: str,
        bbox: BboxScope,
        release: str,
    ) -> SizeEstimate:
        path = self._fixtures_dir / f"{theme}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"no fixture parquet for theme={theme!r} at {path}")
        meta = pq.read_metadata(path)
        return SizeEstimate(rows=meta.num_rows, bytes=path.stat().st_size)


class S3DuckDBBackend:
    """Reads Overture themes from public S3 via DuckDB + httpfs extensions.

    The Overture S3 bucket (s3://overturemaps-us-west-2/) is public-read; no
    credentials required. Phase 1 filters by bounding box only; precise
    admin-polygon clipping is the responsibility of downstream consumers
    (see docs/data/handoffs.md).
    """

    DEFAULT_BUCKET = "overturemaps-us-west-2"

    def __init__(self, bucket: str | None = None) -> None:
        self.bucket = bucket or self.DEFAULT_BUCKET

    def build_s3_url(self, *, theme: str, release: str) -> str:
        try:
            type_ = THEME_TO_TYPE[theme]
        except KeyError as e:
            known = ", ".join(sorted(THEME_TO_TYPE))
            raise ValueError(f"unknown Overture theme {theme!r}; known themes: {known}") from e
        return f"s3://{self.bucket}/release/{release}/theme={theme}/type={type_}/*"

    def build_query(
        self,
        *,
        theme: str,
        bbox: BboxScope,
        release: str,
    ) -> str:
        url = self.build_s3_url(theme=theme, release=release)
        return f"""
            SELECT *
            FROM read_parquet('{url}', filename=false, hive_partitioning=1)
            WHERE bbox.xmin <= {bbox.max_lon}
              AND bbox.xmax >= {bbox.min_lon}
              AND bbox.ymin <= {bbox.max_lat}
              AND bbox.ymax >= {bbox.min_lat}
        """.strip()

    def build_count_query(
        self,
        *,
        theme: str,
        bbox: BboxScope,
        release: str,
    ) -> str:
        url = self.build_s3_url(theme=theme, release=release)
        return f"""
            SELECT COUNT(*) AS n
            FROM read_parquet('{url}', filename=false, hive_partitioning=1)
            WHERE bbox.xmin <= {bbox.max_lon}
              AND bbox.xmax >= {bbox.min_lon}
              AND bbox.ymin <= {bbox.max_lat}
              AND bbox.ymax >= {bbox.min_lat}
        """.strip()

    def read_theme(
        self,
        *,
        theme: str,
        bbox: BboxScope,
        release: str,
    ) -> pa.Table:
        try:
            con = self._open()
            return con.execute(self.build_query(theme=theme, bbox=bbox, release=release)).arrow()
        except duckdb.IOException as e:  # type: ignore[attr-defined]
            raise OvertureUnreachable(f"reading theme={theme!r}: {e}") from e

    def estimate_size(
        self,
        *,
        theme: str,
        bbox: BboxScope,
        release: str,
    ) -> SizeEstimate:
        try:
            con = self._open()
            (rows,) = con.execute(
                self.build_count_query(theme=theme, bbox=bbox, release=release)
            ).fetchone()
        except duckdb.IOException as e:  # type: ignore[attr-defined]
            raise OvertureUnreachable(f"estimating theme={theme!r}: {e}") from e
        # Rough byte estimate: 200 bytes per row is a Phase-1 guess; refined when
        # actual cache writes report real sizes.
        return SizeEstimate(rows=int(rows), bytes=int(rows) * 200)

    @staticmethod
    def _open() -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("SET s3_region='us-west-2';")
        return con

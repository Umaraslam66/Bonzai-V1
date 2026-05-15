from __future__ import annotations

import pytest

from cfm.data.overture.backend import S3DuckDBBackend
from cfm.data.overture.region import BboxScope


def test_s3_backend_default_bucket() -> None:
    backend = S3DuckDBBackend()
    assert backend.bucket == "overturemaps-us-west-2"


def test_s3_backend_custom_bucket() -> None:
    backend = S3DuckDBBackend(bucket="my-mirror")
    assert backend.bucket == "my-mirror"


def test_build_s3_url_for_release_and_theme() -> None:
    backend = S3DuckDBBackend()
    url = backend.build_s3_url(theme="buildings", release="2026-04-15.0")
    assert url == "s3://overturemaps-us-west-2/release/2026-04-15.0/theme=buildings/*"


def test_build_query_contains_bbox_clauses_and_not_polygon(singapore_bbox: BboxScope) -> None:
    backend = S3DuckDBBackend()
    sql = backend.build_query(theme="buildings", bbox=singapore_bbox, release="2026-04-15.0")
    # Theme path
    assert "release/2026-04-15.0/theme=buildings/" in sql
    # Bbox-only filter (Overture parquet has bbox.xmin/xmax/ymin/ymax)
    assert "bbox.xmin" in sql
    assert "bbox.ymax" in sql
    # No polygon refinement in Phase 1 — handoff contract is bbox-only at fetch time.
    assert "ST_Intersects" not in sql and "st_intersects" not in sql.lower()


def test_build_count_query_returns_count_star(singapore_bbox: BboxScope) -> None:
    backend = S3DuckDBBackend()
    sql = backend.build_count_query(theme="buildings", bbox=singapore_bbox, release="2026-04-15.0")
    assert sql.strip().lower().startswith("select count(*)")
    assert "release/2026-04-15.0/theme=buildings/" in sql


@pytest.mark.slow
def test_real_s3_smoke_buildings() -> None:
    """Sanity smoke that S3 is reachable. Excluded from default suite."""
    backend = S3DuckDBBackend()
    tiny = BboxScope.from_tuple((103.85, 1.29, 103.86, 1.30))
    est = backend.estimate_size(theme="buildings", bbox=tiny, release="2026-04-15.0")
    assert est.rows >= 0

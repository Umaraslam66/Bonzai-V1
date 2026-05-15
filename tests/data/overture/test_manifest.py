from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from cfm.data.overture.manifest import (
    CacheManifest,
    ThemeEntry,
    sha256_of_file,
)


def test_sha256_of_file_known_value(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    # sha256("hello") == 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
    assert sha256_of_file(p) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_manifest_round_trip_yaml(tmp_path: Path) -> None:
    m = CacheManifest(
        schema_version=1,
        release="2026-04-15.0",
        release_date="2026-04-15",
        release_subversion=0,
        overture_schema_version="v1.16.0",
        region="singapore",
        admin_polygon_source="overture://divisions:country:SG",
        bbox=(103.6, 1.16, 104.05, 1.48),
        backend="S3DuckDBBackend",
        fetched_at=datetime(2026, 5, 16, 14, 32, 11, tzinfo=UTC),
        themes={
            "buildings": ThemeEntry(
                s3_url="s3://overturemaps-us-west-2/release/2026-04-15.0/theme=buildings/",
                rows=1000,
                bytes=50_000,
                sha256="abc123",
                parquet_filename="buildings.parquet",
            ),
        },
    )
    path = tmp_path / "manifest.yaml"
    m.to_yaml(path)
    loaded = CacheManifest.from_yaml(path)
    assert loaded == m


def test_manifest_rejects_wrong_schema_version(tmp_path: Path) -> None:
    bad = tmp_path / "manifest.yaml"
    bad.write_text("schema_version: 99\nrelease: x\n")
    with pytest.raises(ValueError, match="schema_version"):
        CacheManifest.from_yaml(bad)


def test_theme_entry_fields() -> None:
    e = ThemeEntry(
        s3_url="s3://x/",
        rows=10,
        bytes=100,
        sha256="abc",
        parquet_filename="x.parquet",
    )
    assert e.rows == 10
    assert e.parquet_filename == "x.parquet"


def test_manifest_fetched_at_serialises_to_iso_z(tmp_path: Path) -> None:
    m = CacheManifest(
        schema_version=1,
        release="2026-04-15.0",
        release_date="2026-04-15",
        release_subversion=0,
        overture_schema_version="v1.16.0",
        region="singapore",
        admin_polygon_source="overture://divisions:country:SG",
        bbox=(103.6, 1.16, 104.05, 1.48),
        backend="S3DuckDBBackend",
        fetched_at=datetime(2026, 5, 16, 14, 32, 11, tzinfo=UTC),
        themes={},
    )
    path = tmp_path / "manifest.yaml"
    m.to_yaml(path)
    text = path.read_text()
    assert "2026-05-16T14:32:11Z" in text

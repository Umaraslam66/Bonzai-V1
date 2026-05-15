from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

CURRENT_SCHEMA_VERSION = 1


def sha256_of_file(path: Path) -> str:
    """Hex-encoded SHA-256 of the file at `path`."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ThemeEntry:
    """One entry under the `themes` mapping in a cache manifest."""

    s3_url: str
    rows: int
    bytes: int
    sha256: str
    parquet_filename: str

    def to_dict(self) -> dict:
        return {
            "s3_url": self.s3_url,
            "rows": self.rows,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "parquet_filename": self.parquet_filename,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ThemeEntry:
        return cls(
            s3_url=data["s3_url"],
            rows=int(data["rows"]),
            bytes=int(data["bytes"]),
            sha256=data["sha256"],
            parquet_filename=data["parquet_filename"],
        )


@dataclass(frozen=True)
class CacheManifest:
    """Per-region cache manifest, written to manifest.yaml on every fetch.

    Matches the format in spec §7 of
    docs/superpowers/specs/2026-05-16-phase-1-sub-A-overture-loader-design.md.
    """

    schema_version: int
    release: str
    release_date: str
    release_subversion: int
    overture_schema_version: str
    region: str
    admin_polygon_source: str
    bbox: tuple[float, float, float, float]
    backend: str
    fetched_at: datetime
    themes: dict[str, ThemeEntry] = field(default_factory=dict)

    def to_yaml(self, path: Path) -> None:
        data = {
            "schema_version": self.schema_version,
            "release": self.release,
            "release_date": self.release_date,
            "release_subversion": self.release_subversion,
            "overture_schema_version": self.overture_schema_version,
            "region": self.region,
            "scope": {
                "admin_polygon_source": self.admin_polygon_source,
                "bbox": list(self.bbox),
            },
            "backend": self.backend,
            "fetched_at": _format_iso_z(self.fetched_at),
            "themes": {name: entry.to_dict() for name, entry in self.themes.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: Path) -> CacheManifest:
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        version = int(data.get("schema_version", 0))
        if version != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"manifest at {path} has schema_version={version}, "
                f"expected {CURRENT_SCHEMA_VERSION}"
            )
        scope = data.get("scope", {})
        bbox_list = scope.get("bbox", [0.0, 0.0, 0.0, 0.0])
        return cls(
            schema_version=version,
            release=data["release"],
            release_date=data["release_date"],
            release_subversion=int(data["release_subversion"]),
            overture_schema_version=data["overture_schema_version"],
            region=data["region"],
            admin_polygon_source=scope["admin_polygon_source"],
            bbox=(bbox_list[0], bbox_list[1], bbox_list[2], bbox_list[3]),
            backend=data["backend"],
            fetched_at=_parse_iso_z(data["fetched_at"]),
            themes={
                name: ThemeEntry.from_dict(entry)
                for name, entry in (data.get("themes") or {}).items()
            },
        )


def _format_iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso_z(s: str) -> datetime:
    # Accept both "Z" suffix and explicit +00:00.
    cleaned = s.replace("Z", "+00:00")
    return datetime.fromisoformat(cleaned).astimezone(UTC)

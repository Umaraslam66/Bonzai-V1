from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from shapely.geometry import box

from cfm.data.overture.backend import (
    OvertureBackend,
    S3DuckDBBackend,
)
from cfm.data.overture.errors import (
    CacheCorrupt,
    OversizedFetch,
    RegionNotFound,
    ReleaseNotConfigured,
)
from cfm.data.overture.manifest import (
    CURRENT_SCHEMA_VERSION,
    CacheManifest,
    ThemeEntry,
    sha256_of_file,
)
from cfm.data.overture.region import (
    BboxScope,
    Region,
    RegionGeometry,
    SizeEstimate,
)

logger = logging.getLogger(__name__)

THEMES_TO_LOAD: tuple[str, ...] = (
    "divisions",  # fetched first; admin polygon comes from here
    "buildings",
    "places",
    "transportation",
    "base",
)

OVERSIZED_THRESHOLD_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GB


def load_region(
    name: str,
    *,
    backend: OvertureBackend | None = None,
    refresh: bool = False,
    confirm: bool = False,
    repo_root: Path | None = None,
) -> Region:
    """Load Overture themes for region `name`, caching on disk.

    Phase 1 contract: themes are bbox-filtered only. The admin polygon is
    surfaced on Region.geometry for downstream consumers to apply. See
    docs/data/handoffs.md and the spec at
    docs/superpowers/specs/2026-05-16-phase-1-sub-A-overture-loader-design.md.
    """
    root = Path(repo_root) if repo_root is not None else _find_repo_root()
    release = _load_release_pin(root)
    region_cfg = _load_region_config(root, name)
    bbox = _build_bbox_scope(region_cfg)
    geometry = _build_region_geometry(region_cfg)
    backend = backend or S3DuckDBBackend()

    cache_dir = root / "data" / "cache" / "overture" / release["release"] / name
    manifest_path = cache_dir / "manifest.yaml"

    if not refresh and manifest_path.exists():
        existing = CacheManifest.from_yaml(manifest_path)
        if existing.release == release["release"]:
            _verify_cache_or_raise(cache_dir, existing)
            return _region_from_cache(name, bbox, geometry, cache_dir, existing)
        logger.info(
            "[overture] cached release %s differs from pin %s; re-fetching",
            existing.release,
            release["release"],
        )

    _check_total_size(backend, bbox, release["release"], confirm=confirm)

    # Remember the prior manifest's fetched_at so a back-to-back `refresh=True`
    # call produces a strictly newer timestamp. The on-disk format has
    # second resolution; a same-second refresh would otherwise compare equal.
    prior_fetched_at: datetime | None = None
    if manifest_path.exists():
        try:
            prior_fetched_at = CacheManifest.from_yaml(manifest_path).fetched_at
        except Exception:
            # Prior manifest may be unreadable (mismatched schema, partial write);
            # treat as no prior and proceed to a fresh fetch.
            prior_fetched_at = None

    cache_dir.mkdir(parents=True, exist_ok=True)
    themes: dict = {}
    theme_entries: dict[str, ThemeEntry] = {}
    for theme in THEMES_TO_LOAD:
        table = backend.read_theme(theme=theme, bbox=bbox, release=release["release"])
        out_path = cache_dir / f"{theme}.parquet"
        pq.write_table(table, out_path)
        sha = sha256_of_file(out_path)
        themes[theme] = table
        theme_entries[theme] = ThemeEntry(
            s3_url=_s3_url(backend, theme, release["release"]),
            rows=table.num_rows,
            bytes=out_path.stat().st_size,
            sha256=sha,
            parquet_filename=f"{theme}.parquet",
        )

    now = datetime.now(UTC).replace(microsecond=0)
    if prior_fetched_at is not None and now <= prior_fetched_at:
        now = prior_fetched_at + timedelta(seconds=1)
    manifest = CacheManifest(
        schema_version=CURRENT_SCHEMA_VERSION,
        release=release["release"],
        release_date=release["release_date"],
        release_subversion=int(release["release_subversion"]),
        overture_schema_version=release["overture_schema_version"],
        region=name,
        admin_polygon_source=geometry.source,
        bbox=bbox.as_tuple(),
        backend=type(backend).__name__,
        fetched_at=now,
        themes=theme_entries,
    )
    manifest.to_yaml(manifest_path)
    return Region(
        name=name,
        release=release["release"],
        fetch_bbox=bbox,
        geometry=geometry,
        themes=themes,
        manifest_path=manifest_path,
    )


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("could not locate repo root from cfm.data.overture.loader")


def _load_release_pin(root: Path) -> dict:
    path = root / "configs" / "data" / "overture_release.yaml"
    if not path.exists():
        raise ReleaseNotConfigured(f"missing {path}")
    data = yaml.safe_load(path.read_text())
    for key in ("release", "overture_schema_version", "release_date", "release_subversion"):
        if key not in data:
            raise ReleaseNotConfigured(f"{path} missing key {key!r}")
    return data


def _load_region_config(root: Path, name: str) -> dict:
    path = root / "configs" / "data" / "regions" / f"{name}.yaml"
    if not path.exists():
        raise RegionNotFound(f"no region config at {path}")
    return yaml.safe_load(path.read_text())


def _build_bbox_scope(region_cfg: dict) -> BboxScope:
    """The fetch-time spatial filter. Phase 1: read straight from the region config."""
    return BboxScope.from_tuple(tuple(region_cfg["fallback_bbox"]))


def _build_region_geometry(region_cfg: dict) -> RegionGeometry:
    """The handoff-record geometry. Phase 1 placeholder: a Polygon equal to
    the bbox. C-stage (or a future implementation upgrade) replaces this
    with the precise polygon from the divisions theme. See
    docs/data/handoffs.md.
    """
    bbox = tuple(region_cfg["fallback_bbox"])
    polygon = box(bbox[0], bbox[1], bbox[2], bbox[3])
    admin = region_cfg["admin"]
    source = f"{admin['source']}:{admin['level']}:{admin['country_code']}"
    return RegionGeometry(admin_polygon=polygon, source=source)


def _verify_cache_or_raise(cache_dir: Path, manifest: CacheManifest) -> None:
    for entry in manifest.themes.values():
        parquet_path = cache_dir / entry.parquet_filename
        if not parquet_path.exists():
            raise CacheCorrupt(f"manifest lists {parquet_path} but file is missing")
        actual = sha256_of_file(parquet_path)
        if actual != entry.sha256:
            raise CacheCorrupt(
                f"sha256 mismatch for {parquet_path}: manifest={entry.sha256!r} actual={actual!r}"
            )


def _region_from_cache(
    name: str,
    bbox: BboxScope,
    geometry: RegionGeometry,
    cache_dir: Path,
    manifest: CacheManifest,
) -> Region:
    themes: dict = {}
    for theme, entry in manifest.themes.items():
        themes[theme] = pq.read_table(cache_dir / entry.parquet_filename)
    return Region(
        name=name,
        release=manifest.release,
        fetch_bbox=bbox,
        geometry=geometry,
        themes=themes,
        manifest_path=cache_dir / "manifest.yaml",
    )


def _s3_url(backend: OvertureBackend, theme: str, release: str) -> str:
    if isinstance(backend, S3DuckDBBackend):
        return backend.build_s3_url(theme=theme, release=release).rstrip("*")
    return f"local-fixture://{theme}"


def _check_total_size(
    backend: OvertureBackend,
    bbox: BboxScope,
    release: str,
    *,
    confirm: bool,
) -> None:
    estimates: dict[str, SizeEstimate] = {}
    total = 0
    for theme in THEMES_TO_LOAD:
        est = backend.estimate_size(theme=theme, bbox=bbox, release=release)
        estimates[theme] = est
        total += est.bytes
        logger.info(
            "[overture] estimated fetch: theme=%-15s rows~%d size~%d bytes",
            theme,
            est.rows,
            est.bytes,
        )
    logger.info("[overture] estimated total: %d themes, ~%d bytes", len(estimates), total)
    if total > OVERSIZED_THRESHOLD_BYTES and not confirm:
        raise OversizedFetch(
            f"estimated total {total} bytes exceeds {OVERSIZED_THRESHOLD_BYTES} threshold; "
            "pass confirm=True if intended"
        )

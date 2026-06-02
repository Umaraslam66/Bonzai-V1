from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class BboxScope:
    """Bounding box used by Overture backends as the fetch-time spatial filter.

    This is the only spatial filter applied at fetch time in Phase 1. The
    admin polygon (see RegionGeometry) is NOT used by backends; downstream
    consumers (e.g. C-stage tile extraction) must apply it themselves for
    precise clipping. See docs/data/handoffs.md.
    """

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @classmethod
    def from_tuple(cls, t: tuple[float, float, float, float]) -> BboxScope:
        return cls(min_lon=t[0], min_lat=t[1], max_lon=t[2], max_lat=t[3])

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)


@dataclass(frozen=True)
class RegionGeometry:
    """Precise geometric description of a region, plus where it came from.

    Phase 1 uses this as a downstream-handoff record: it lives on Region for
    C-stage and later consumers, but is NOT used by backends as a fetch-time
    filter. See docs/data/handoffs.md.
    """

    admin_polygon: BaseGeometry
    source: str  # e.g. "overture://divisions:country:SG"


@dataclass(frozen=True)
class SizeEstimate:
    """Cheap pre-fetch estimate: row count + approximate byte size."""

    rows: int
    bytes: int


@dataclass(frozen=True)
class Region:
    """A fully-loaded Overture region for one release.

    HANDOFF CONTRACT — read before consuming this object:

    The `themes` parquet tables are filtered ONLY by `fetch_bbox` at load
    time. The `geometry.admin_polygon` is a HANDOFF record describing the
    region's precise shape; it is NOT applied to the themes at load time.

    Downstream consumers (e.g. C-stage tile extraction in sub-project C)
    MUST apply `admin_polygon` to clip themes before training data leaves
    A's contract. Failing to do so means open sea — which falls inside
    `fetch_bbox` but outside `admin_polygon` — silently enters the
    training set. See docs/data/handoffs.md.
    """

    name: str
    release: str
    fetch_bbox: BboxScope
    geometry: RegionGeometry
    themes: dict[str, pa.Table]
    manifest_path: Path
    #: Projected (metric, conformal) CRS for sub-C reprojection, e.g. "EPSG:3414"
    #: (Singapore SVY21) or "EPSG:25833" (a UTM33N city). A downstream-handoff
    #: record like ``geometry`` — sub-A does not use it; sub-C reprojects with it.
    projected_crs: str

    @property
    def admin_polygon(self) -> BaseGeometry:
        return self.geometry.admin_polygon

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.fetch_bbox.as_tuple()

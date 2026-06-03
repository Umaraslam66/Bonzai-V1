"""City selection (spec §4): hard filters (single-UTM-zone, defer cross-border-
admin) and region-config emission. The NAMED list is PI-ratified (Task G1).

The single-UTM filter rejects on the FULL bbox, not the centroid: a city whose
centroid sits in one zone but whose area spills into the next would pass a
centroid classification and then reproject with distortion in the spillover.
``utm_epsg_for_centroid`` is used ONLY to pick the CRS string after the bbox has
been confirmed to fit one zone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from cfm.data.sub_c.coords import utm_epsg_for_centroid


def _utm_zone(lon: float) -> int:
    """UTM zone for a longitude (half-open lower bound, like the tile grid)."""
    return int((lon + 180.0) / 6.0) + 1


def single_utm_zone_ok(bbox: tuple[float, float, float, float]) -> tuple[bool, str | None]:
    """``(ok, projected_crs)``. ``ok`` iff the FULL bbox lies in one UTM zone,
    northern hemisphere, inside the ETRS89 European range. Returns the centroid
    CRS when ok, else ``(False, None)``. Rejection is by bbox extent — NOT by
    centroid — so a zone-straddling city is rejected even if its centroid is
    cleanly inside one zone."""
    min_lon, min_lat, max_lon, max_lat = bbox
    if min_lat < 0 or max_lat < 0:
        return (False, None)  # southern hemisphere — outside ETRS89-North policy
    if _utm_zone(min_lon) != _utm_zone(max_lon):
        return (False, None)  # straddles a UTM zone boundary
    centroid_lon = (min_lon + max_lon) / 2.0
    centroid_lat = (min_lat + max_lat) / 2.0
    try:
        crs = utm_epsg_for_centroid(centroid_lon, centroid_lat)
    except ValueError:
        return (False, None)  # outside the European zone range
    return (True, crs)


@dataclass(frozen=True)
class CityCandidate:
    name: str
    country_code: str
    admin_level: str  # country | region | locality (Overture divisions level)
    bbox: tuple[float, float, float, float]  # [min_lon, min_lat, max_lon, max_lat]
    morphology: str  # medieval-organic | planned-grid | modernist-sprawl | mixed
    density: str  # dense-core | moderate | sparse
    projected_crs: str  # must equal single_utm_zone_ok(bbox)'s CRS


def write_region_config(candidate: CityCandidate, configs_dir: Path) -> Path:
    """Emit ``configs/data/regions/<name>.yaml`` mirroring berlin.yaml."""
    out = configs_dir / f"{candidate.name}.yaml"
    payload = {
        "name": candidate.name,
        "admin": {
            "source": "overture://divisions",
            "country_code": candidate.country_code,
            "level": candidate.admin_level,
        },
        "fallback_bbox": list(candidate.bbox),
        "crs": "EPSG:4326",
        "projected_crs": candidate.projected_crs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False))
    return out


def load_canary_manifest(path: Path) -> list[dict]:
    """Load the PI-ratified canary axis labels (the machine-readable home for the
    morphology/density/geography labels that region configs do NOT carry).

    Returns the per-city dicts (name, country_code, projected_crs, morphology,
    density, geography) the Phase-G roll-up consumes to build CityRecord axis
    labels for the coverage gate — so a cold session never re-guesses them."""
    data = yaml.safe_load(Path(path).read_text())
    return data["cities"]

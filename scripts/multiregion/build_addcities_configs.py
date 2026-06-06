#!/usr/bin/env python
"""Add-cities (Path B token+NL-coverage lever) -> single-UTM-zone gate -> region
configs + addcities_v1 manifest.

The shipped corpus = 29 validated + these add-cities, MINUS the 6 excluded
(quantum-inflation: 3 corpus-normal-but-edge-tripped recoverable post-deadline, 2
degraded-source rotterdam/warsaw, 1 elevated amsterdam) and MINUS paris/lyon/madrid
(zero-coverage-at-extreme-cost). These 7 are moderate/sprawl (high tok/tile, ~30-45k,
NOT dense cores); eindhoven closes the only open coverage axis (NL). Boxes generous
(known_issues #15); the FULL bbox runs through single_utm_zone_ok (a straddler drops
+ is reported so a dropped cell never silently vacates an axis label).

Writes configs/data/regions/<city>.yaml (passers) + configs/multiregion/addcities_v1.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.multiregion import selection  # noqa: E402

# generous half-widths by tier (deg), mirroring build_batch2_configs
_TIERS = {"big": (0.40, 0.30), "mid": (0.30, 0.24), "small": (0.16, 0.13)}

# (name, country, center_lon, center_lat, morphology, density, tier)
CANDIDATES: list[tuple] = [
    ("eindhoven", "NL", 5.48, 51.44, "modernist-sprawl", "moderate", "mid"),  # closes NL
    ("tilburg", "NL", 5.08, 51.56, "modernist-sprawl", "moderate", "mid"),
    ("wolfsburg", "DE", 10.79, 52.42, "modernist-sprawl", "moderate", "mid"),
    ("telford", "GB", -2.44, 52.68, "modernist-sprawl", "moderate", "mid"),
    ("szczecin", "PL", 14.55, 53.43, "mixed", "moderate", "mid"),
    ("linz", "AT", 14.29, 48.31, "planned-grid", "moderate", "mid"),
    ("debrecen", "HU", 21.63, 47.53, "planned-grid", "moderate", "mid"),
]


def _bbox(lon: float, lat: float, tier: str) -> tuple[float, float, float, float]:
    hw_lon, hw_lat = _TIERS[tier]
    return (
        round(lon - hw_lon, 4),
        round(lat - hw_lat, 4),
        round(lon + hw_lon, 4),
        round(lat + hw_lat, 4),
    )


def main() -> int:
    configs_dir = _REPO / "configs" / "data" / "regions"
    passers: list[dict] = []
    drops: list[tuple[str, str]] = []
    for name, cc, lon, lat, morph, density, tier in CANDIDATES:
        bbox = _bbox(lon, lat, tier)
        ok, crs = selection.single_utm_zone_ok(bbox)
        if not ok:
            drops.append((name, f"bbox {bbox} straddles a UTM zone / out of ETRS89-N range"))
            continue
        cand = selection.CityCandidate(
            name=name,
            country_code=cc,
            admin_level="locality",
            bbox=bbox,
            morphology=morph,
            density=density,
            projected_crs=crs,
        )
        selection.write_region_config(cand, configs_dir)
        passers.append(
            {
                "name": name,
                "country_code": cc,
                "projected_crs": crs,
                "morphology": morph,
                "density": density,
                "geography": cc,
            }
        )

    manifest = {
        "version": "v1",
        "note": "Add-cities (Path B token+NL-coverage lever, 2026-06-06). Shipped corpus "
        "= canary_v1 + batch2_v1 (minus the 6 inflation-excluded + paris/lyon/madrid) "
        "+ these. eindhoven closes NL. The G4 roll-up reads all three manifests.",
        "cities": passers,
    }
    man_path = _REPO / "configs" / "multiregion" / "addcities_v1.yaml"
    man_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))

    print(f"=== PASSERS ({len(passers)}) ===")
    for c in passers:
        axis = f"{c['morphology']}/{c['density']}"
        print(f"  {c['name']:<12} {c['geography']:<3} {c['projected_crs']:<11} {axis}")
    print(f"\n=== DROPS ({len(drops)}) ===")
    for name, reason in drops:
        print(f"  {name}: {reason}")
    if not drops:
        print("  (none — all add-cities fit one UTM zone)")
    print(f"\nwrote {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

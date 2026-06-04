#!/usr/bin/env python
"""Batch-2 candidate → single-UTM-zone gate → region configs + batch2_v1 manifest.

Diversity-sized (PI 2026-06-04): fill the morphology x density x geography matrix
to original breadth; counts/tokens are NOT the sizing basis (G3). Boxes are drawn
GENEROUSLY (over-include rather than clip a dense core; known_issues #15). Every
candidate's FULL generous bbox is run through ``selection.single_utm_zone_ok`` —
generous boxes straddle zone boundaries more easily, so the gate runs on the real
box. Straddlers are dropped + reported; coverage (morphology/density/zone) is
checked so a drop never silently vacates an axis label.

Run from repo root. Writes configs/data/regions/<city>.yaml (passers) +
configs/multiregion/batch2_v1.yaml. Idempotent. Read-only on everything else.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.multiregion import selection  # noqa: E402

# (name, country, center_lon, center_lat, morphology, density, size_tier)
# size_tier sets the generous half-widths below. Coordinates are city centres.
_BIG = (0.40, 0.30)  # big metro
_MID = (0.30, 0.24)  # city
_SMALL = (0.16, 0.13)  # town
_TIERS = {"big": _BIG, "mid": _MID, "small": _SMALL}

CANDIDATES: list[tuple] = [
    # --- medieval-organic ---
    ("vienna", "AT", 16.37, 48.21, "medieval-organic", "dense-core", "mid"),
    ("lyon", "FR", 4.84, 45.76, "medieval-organic", "dense-core", "mid"),
    ("bologna", "IT", 11.34, 44.49, "medieval-organic", "moderate", "mid"),
    ("tallinn", "EE", 24.75, 59.44, "medieval-organic", "moderate", "mid"),
    ("krakow", "PL", 19.94, 50.06, "medieval-organic", "moderate", "mid"),
    ("edinburgh", "GB", -3.19, 55.95, "medieval-organic", "moderate", "mid"),
    ("bruges", "BE", 3.22, 51.21, "medieval-organic", "sparse", "small"),
    ("toledo", "ES", -4.03, 39.86, "medieval-organic", "sparse", "small"),
    ("ljubljana", "SI", 14.51, 46.05, "medieval-organic", "moderate", "mid"),
    # --- planned-grid ---
    ("turin", "IT", 7.69, 45.07, "planned-grid", "dense-core", "mid"),
    ("valencia", "ES", -0.38, 39.47, "planned-grid", "dense-core", "mid"),
    ("mannheim", "DE", 8.47, 49.49, "planned-grid", "moderate", "mid"),
    ("glasgow", "GB", -4.25, 55.86, "planned-grid", "moderate", "mid"),
    ("lodz", "PL", 19.46, 51.76, "planned-grid", "moderate", "mid"),
    ("helsinki", "FI", 24.94, 60.17, "planned-grid", "dense-core", "mid"),
    ("a_coruna", "ES", -8.41, 43.36, "planned-grid", "moderate", "mid"),
    ("karlsruhe", "DE", 8.40, 49.01, "planned-grid", "moderate", "mid"),
    # --- modernist-sprawl ---
    ("almere", "NL", 5.22, 52.35, "modernist-sprawl", "moderate", "mid"),
    ("rotterdam", "NL", 4.48, 51.92, "modernist-sprawl", "dense-core", "mid"),
    ("le_havre", "FR", 0.11, 49.49, "modernist-sprawl", "moderate", "mid"),
    ("cergy", "FR", 2.04, 49.04, "modernist-sprawl", "moderate", "mid"),
    ("tychy", "PL", 18.99, 50.13, "modernist-sprawl", "moderate", "mid"),
    ("vallingby", "SE", 17.87, 59.36, "modernist-sprawl", "moderate", "mid"),
    ("espoo", "FI", 24.66, 60.21, "modernist-sprawl", "sparse", "small"),
    ("welwyn", "GB", -0.20, 51.80, "modernist-sprawl", "sparse", "small"),
    # --- mixed ---
    # (berlin EXCLUDED: a pilot cache with a different bbox exists on Leonardo and
    #  load_region keys cache on region name, not bbox -> would cache-hit the stale
    #  pilot extent. mixed/dense-core is covered by 8 other cities.)
    ("paris", "FR", 2.35, 48.86, "mixed", "dense-core", "big"),
    ("madrid", "ES", -3.70, 40.42, "mixed", "dense-core", "big"),
    ("rome", "IT", 12.50, 41.90, "mixed", "dense-core", "big"),
    ("amsterdam", "NL", 4.90, 52.37, "mixed", "dense-core", "big"),
    ("hamburg", "DE", 10.00, 53.55, "mixed", "dense-core", "big"),
    ("warsaw", "PL", 21.01, 52.23, "mixed", "dense-core", "big"),
    ("budapest", "HU", 19.04, 47.50, "mixed", "dense-core", "big"),
    ("lisbon", "PT", -9.14, 38.72, "mixed", "moderate", "mid"),
    ("copenhagen", "DK", 12.57, 55.68, "mixed", "dense-core", "big"),
    ("manchester", "GB", -2.24, 53.48, "mixed", "moderate", "mid"),
    ("malmo", "SE", 13.00, 55.60, "mixed", "moderate", "mid"),
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
    drops: list[tuple[str, str, str]] = []  # (name, cell, reason)
    for name, cc, lon, lat, morph, dens, tier in CANDIDATES:
        bbox = _bbox(lon, lat, tier)
        ok, crs = selection.single_utm_zone_ok(bbox)
        zone = selection._utm_zone((bbox[0] + bbox[2]) / 2.0)
        cell = f"{morph}/{dens}/z{zone}"
        if not ok:
            drops.append((name, cell, f"bbox {bbox} straddles a UTM zone / out of range"))
            continue
        cand = selection.CityCandidate(
            name=name,
            country_code=cc,
            admin_level="locality",
            bbox=bbox,
            morphology=morph,
            density=dens,
            projected_crs=crs,
        )
        selection.write_region_config(cand, configs_dir)
        passers.append(
            {
                "name": name,
                "country_code": cc,
                "projected_crs": crs,
                "morphology": morph,
                "density": dens,
                "geography": cc,
                "_zone": zone,
                "_bbox": list(bbox),
            }
        )

    # write batch2_v1 manifest (axis labels; mirrors canary_v1.yaml shape)
    manifest = {
        "version": "v1",
        "ratified_utc": "2026-06-04",
        "note": "Batch-2 NEW cities (diversity fill). Canary 5 are in canary_v1.yaml; "
        "the G4 roll-up reads BOTH for the full coverage matrix. Labels are "
        "pre-data hypotheses (PI-ratified). bboxes generous (known_issues #15).",
        "cities": [{k: v for k, v in c.items() if not k.startswith("_")} for c in passers],
    }
    man_path = _REPO / "configs" / "multiregion" / "batch2_v1.yaml"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))

    # ---- report ----
    print(f"=== PASSERS ({len(passers)}) ===")
    for c in sorted(passers, key=lambda x: (x["morphology"], x["density"], x["_zone"])):
        print(
            f"  {c['name']:<12} {c['geography']:<3} z{c['_zone']:<2} {c['projected_crs']:<11} "
            f"{c['morphology']:<16} {c['density']:<10} bbox={c['_bbox']}"
        )
    print(f"\n=== DROPS ({len(drops)}) ===")
    for name, cell, reason in drops:
        print(f"  DROP {name:<12} cell={cell:<28} {reason}")

    # ---- coverage (does any drop vacate a morphology / density / zone?) ----
    def labels(field):
        return {c[field] for c in passers}

    print("\n=== COVERAGE (passers + canary axes) ===")
    canary = {
        "morph": {"medieval-organic", "planned-grid", "modernist-sprawl", "mixed"},
        "dens": {"dense-core", "moderate", "sparse"},
        "zone": {30, 31, 32, 33, 34},
    }
    p_morph, p_dens, p_zone = labels("morphology"), labels("density"), {c["_zone"] for c in passers}
    print(f"  morphology covered: {sorted(p_morph | canary['morph'])}")
    print(f"  density covered:    {sorted(p_dens | canary['dens'])}")
    print(f"  zones covered:      {sorted(p_zone | canary['zone'])}")
    # per (morphology,density) count across passers
    from collections import Counter

    cells = Counter((c["morphology"], c["density"]) for c in passers)
    print("  (morphology,density) cell counts among passers:")
    for k, v in sorted(cells.items()):
        print(f"     {k[0]:<16} {k[1]:<10} {v}")
    print(f"\nwrote {man_path} + {len(passers)} region configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

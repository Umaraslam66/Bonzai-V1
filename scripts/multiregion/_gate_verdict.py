#!/usr/bin/env python3
"""Gate helper: parse the sub-G verdict for one city and print the four-checkpoint
summary (decoded_vertex == 0 for #1; no decodable_to_valid_geojson group for #4)."""

from __future__ import annotations

import pathlib
import sys

import yaml

RELEASE = "2026-04-15.0"


def main() -> int:
    city = sys.argv[1]
    base = pathlib.Path(f"data/processed/sub_g/{RELEASE}/{city}")
    validated = (base / "_PHASE1_VALIDATED").exists()
    print(f"  _PHASE1_VALIDATED present: {validated}")

    qp = base / "quarantine_report.yaml"
    groups = []
    if qp.exists():
        groups = yaml.safe_load(qp.read_text()).get("groups") or []
        print(f"  quarantine groups: {len(groups)}")
        for g in groups:
            print(
                f"    - {g.get('invariant_name')}: instance_count={g.get('instance_count')} "
                f"tiles={g.get('tile_ids')}"
            )
    else:
        print("  quarantine_report.yaml: ABSENT (clean)")

    bl = base / "_PHASE1_ACCURACY_BASELINE.yaml"
    if bl.exists():
        b = yaml.safe_load(bl.read_text())
        print(f"  structural_bound_breaches (decoded_vertex): {b.get('structural_bound_breaches')}")
        print(
            f"  position_full_p99_9: {b.get('position_full_p99_9')}  "
            f"position_core_p99_9: {b.get('position_core_p99_9')}"
        )
        print(
            "  ogc_bref_collapse_excluded (by-construction, NOT a defect): "
            f"{b.get('ogc_bref_collapse_excluded_from_gate')}"
        )

    dv = sum(
        g.get("instance_count", 0)
        for g in groups
        if g.get("invariant_name") == "decoded_vertex_within_cell_bound"
    )
    ogc = sum(
        g.get("instance_count", 0)
        for g in groups
        if g.get("invariant_name") == "decodable_to_valid_geojson"
    )
    print("  --- CHECKPOINT SUMMARY ---")
    print(f"  #1 #19 clears:        decoded_vertex_within_cell_bound count = {dv}  (PASS if 0)")
    print(f"  #4 valid geometry:    decodable_to_valid_geojson count = {ogc}  (PASS if 0)")
    clean = len(groups) == 0 and validated
    print(f"     overall sub_g clean (groups==0 + _PHASE1_VALIDATED): {clean}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

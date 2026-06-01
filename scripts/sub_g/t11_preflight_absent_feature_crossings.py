"""sub-G T11 cycle-2 pre-flight: count sub-C crossings referencing a feature
absent from the same tile's features.parquet.

Gates the full-494 sub-E regen after the §5.1 non-road-exclusion fix
(commit 99f9e43). That fix excludes any crossing whose source_feature_id is
not a road feature in the tile. A crossing referencing a feature ABSENT from
the tile entirely is excluded too, but that is an unexpected data-integrity
regime, NOT a benign non-road skip — so the pipeline log.warnings on it, and
this probe quantifies it region-wide BEFORE re-deriving 494 tiles.

Expected: 0. Nonzero -> escalate (do not proceed to regen).

Read-only. Run:
    uv run python scripts/sub_g/t11_preflight_absent_feature_crossings.py
"""

from __future__ import annotations

from pathlib import Path

from cfm.data.sub_e.io import read_sub_c_crossings, read_sub_c_features

REPO_ROOT = Path(__file__).resolve().parents[2]
SUB_C = REPO_ROOT / "data" / "processed" / "sub_c" / "2026-04-15.0" / "singapore"


def main() -> int:
    tile_dirs = sorted(SUB_C.glob("tile=EPSG3414_*"))
    if not tile_dirs:
        print(f"NO_TILES at {SUB_C}")
        return 2

    total_crossings = 0
    total_absent = 0
    tiles_with_absent = 0
    examples: list[str] = []

    for td in tile_dirs:
        cpath = td / "crossings.parquet"
        fpath = td / "features.parquet"
        if not cpath.exists():
            continue  # tile with no crossings -> no absent-feature crossings
        crossings = read_sub_c_crossings(cpath)
        features = read_sub_c_features(fpath) if fpath.exists() else []
        feature_ids = {f.source_feature_id for f in features}
        absent = [c for c in crossings if c.source_feature_id not in feature_ids]
        total_crossings += len(crossings)
        total_absent += len(absent)
        if absent:
            tiles_with_absent += 1
            if len(examples) < 5:
                a = absent[0]
                examples.append(
                    f"{td.name}: feature_id={a.source_feature_id} on edge "
                    f"({a.lower_cell_i},{a.lower_cell_j},{a.axis})"
                )

    print(f"TILES={len(tile_dirs)}")
    print(f"TOTAL_CROSSINGS={total_crossings}")
    print(f"ABSENT_FEATURE_CROSSINGS={total_absent}")
    print(f"TILES_WITH_ABSENT={tiles_with_absent}")
    for e in examples:
        print(f"  example: {e}")
    print("PREFLIGHT_RESULT=" + ("CLEAN" if total_absent == 0 else "DIRTY_ESCALATE"))
    print("END_PREFLIGHT")
    return 0 if total_absent == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

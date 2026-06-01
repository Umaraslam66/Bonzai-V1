#!/usr/bin/env python3
"""scripts/sub_g/t11_step0.py — sub-G T11 Step-0 read-only scan (Singapore).

Step-0a: assert sub-C-minus-sub-D gap tiles are all empty (else sub-D bug -> exit 1).
Step-0b: select densest-200 non-empty tiles; report density distribution + the
         #200 cutoff; write the subset tile list (the 'measurement run' config).

READ-ONLY — no derive is fired. Feature counts use parquet footer metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.sub_g.subset import (  # noqa: E402
    assert_gap_empty,
    density_distribution,
    feature_count,
    select_densest,
    tile_names,
)

_RELEASE = "2026-04-15.0"
_REGION = "singapore"
_N = 200


def main() -> int:
    sub_c = _REPO / "data" / "processed" / "sub_c" / _RELEASE / _REGION
    sub_d = _REPO / "data" / "processed" / "sub_d" / _RELEASE / _REGION
    print(f"sub-C tiles: {len(tile_names(sub_c))}  |  sub-D tiles: {len(tile_names(sub_d))}")

    # Step-0a -- sub-C-minus-sub-D gap assertion.
    gap = assert_gap_empty(sub_c, sub_d)
    print(
        f"\n[Step-0a] gap (sub-C-minus-sub-D) = {gap.gap_tile_count} tiles; "
        f"max feature count across gap = {gap.max_feature_count}; PASS={gap.passed}"
    )
    if not gap.passed:
        print(f"  HALT — sub-D skipped non-empty tiles: {gap.offending[:10]}")
        return 1

    # Step-0b — densest-200 subset + density distribution.
    all_counts = {t: feature_count(sub_c, t) for t in tile_names(sub_c)}
    non_empty = {t: c for t, c in all_counts.items() if c > 0}
    subset = select_densest(all_counts, _N)
    subset_counts = [all_counts[t] for t in subset]
    cutoff = subset_counts[-1] if subset_counts else 0

    print(f"\n[Step-0b] non-empty tiles: {len(non_empty)}")
    print(f"  population (all non-empty) dist: {density_distribution(list(non_empty.values()))}")
    print(f"  densest-{_N} subset dist:        {density_distribution(subset_counts)}")
    print(f"  CUTOFF (#{_N} densest feature count): {cutoff}")

    out = _REPO / "reports" / "sub_g_t11_measurement_subset.txt"
    out.write_text("\n".join(subset) + "\n")
    print(f"\n  subset tile list ({len(subset)}) -> {out.relative_to(_REPO)}")
    print(f"  first 3: {subset[:3]}")
    print(f"  last 3:  {subset[-3:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

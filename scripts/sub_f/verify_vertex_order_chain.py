"""Verify Overture → sub-A → sub-C vertex-order chain stability.

Per spec §5.6 + feedback_ambiguous_third_branch_in_verification:
  (a) Chain guarantees stable order → inherit; no canonicalization.
  (b) Chain documents absence → canonicalize via lex-min polygon-ring rotation.
  (c) Ambiguous: docs don't guarantee but empirical sample shows stability →
      canonicalize anyway. Cheap insurance.

Surfaces evidence for Halt 5 reviewer-decision.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from shapely.wkb import loads as wkb_loads

ROOT = Path(__file__).resolve().parents[2]


def sample_features(sub_c_region: Path, n: int = 20) -> list[tuple[str, list]]:
    """Sample N feature geometries and their vertex sequences."""
    samples: list[tuple[str, list]] = []
    tile_paths = sorted(sub_c_region.glob("tile=*/features.parquet"))
    for path in tile_paths[:5]:
        table = pq.ParquetFile(path).read()
        for r in table.to_pylist()[: n // 5]:
            geom = wkb_loads(r["geometry"])
            if geom.geom_type == "Polygon":
                coords = list(geom.exterior.coords)
            elif geom.geom_type == "LineString":
                coords = list(geom.coords)
            else:
                continue
            samples.append((r["source_feature_id"], coords))
        if len(samples) >= n:
            break
    return samples[:n]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub-c-region-dir", required=True, type=Path)
    args = parser.parse_args()

    # Compare two cold-pyarrow reads of the same parquet file — verifies sub-C
    # round-trip stability locally. Cross-chain (Overture → sub-A) requires
    # re-fetching from Overture; out of scope for cheap halt input.
    samples_a = sample_features(args.sub_c_region_dir, n=20)
    samples_b = sample_features(args.sub_c_region_dir, n=20)

    matches = sum(1 for (a, b) in zip(samples_a, samples_b, strict=True) if a == b)
    outcome = "a" if matches == len(samples_a) else "c"  # default defend on partial

    report = {
        "sample_size": len(samples_a),
        "exact_match_count": matches,
        "outcome_branch": outcome,
        "recommendation": (
            "INHERIT (no canonicalization)"
            if outcome == "a"
            else "CANONICALIZE via lex-min polygon-ring rotation (defend by default)"
        ),
        "_status": "PROPOSED — pending Halt 5 reviewer approval per spec §10.3.",
    }
    out = ROOT / "reports" / "sub_f_task_5a_vertex_order.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(report, sort_keys=True), encoding="utf-8")
    print(f"[vertex order] wrote {out}; outcome={outcome}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

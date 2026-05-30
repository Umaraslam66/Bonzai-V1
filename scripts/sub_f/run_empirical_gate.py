#!/usr/bin/env python3
"""scripts/sub_f/run_empirical_gate.py — sub-F per-type RETENTION gate (T14).

The empirical complement to T13 (which gates round-trip + BP7 on real Singapore).
T14 verifies the ONE locked surface T13 does not touch: per-type retention against
the Halt-4 floors.

Retention per type = 1 - (features of that type dropped by alpha-truncation at the
padded budget) / (total features of that type). Computed from sub-C features +
the chunked encoder cost + the §7.2 formula stage-4 (sub-E ABSENT) via the
already-pinned `compute_alpha_drop_report` — so it runs against the PRESENT
sub-C Singapore cache; it does NOT need sub-E.

EVERY locked value is READ from the operative YAML block, never hardcoded:
  - floors  ← sequence_length_analysis.yaml  lock.retention_floors_per_type[*].floor
  - budget  ← sequence_length_analysis.yaml  lock.elbow_budget_padded_tokens
Read `lock.*` (operative), NOT the top-level budget_surface / retention_by_quantile
sections (PRE-CHUNKING historical, 5792/5888 — see the YAML's
_prechunking_analysis_note) and NOT retention_defaults_per_spec_7_5 (the §7.5
audit-trail defaults the master-plan snippet wrongly hardcoded).

PRECISION: floors are stored at 4dp (e.g. roads 0.9936 = round(0.99356, 4)). A
naive `measured >= floor` would falsely fail roads (0.99356 < 0.9936). The gate
allows a half-ulp tolerance (_FLOOR_TOLERANCE = 5e-5) so the lock-derivation
value passes while real drift (> half a 4dp ulp) is still caught.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from scripts.sub_f.compute_alpha_drop_report import compute_alpha_drop_report  # noqa: E402

_SEQ_LEN_YAML = _REPO / "configs" / "sub_f" / "sequence_length_analysis.yaml"

# Floors are stored at 4dp; tolerance = half the last-digit ulp so the
# lock-derivation retention passes while genuine drift is still caught.
_FLOOR_TOLERANCE = 5e-5


def _lock_block() -> dict:
    """The OPERATIVE lock block (NOT the pre-chunking top-level analysis sections)."""
    return yaml.safe_load(_SEQ_LEN_YAML.read_text(encoding="utf-8"))["lock"]


def load_locked_floors() -> dict[int, dict]:
    """{feature_class_int -> {"role": str, "floor": float}} from the operative lock."""
    raw = _lock_block()["retention_floors_per_type"]
    # Skip the `_measurement_basis` prose key — only the numeric fc entries.
    return {
        int(fc): {"role": d["role"], "floor": float(d["floor"])}
        for fc, d in raw.items()
        if str(fc).isdigit()
    }


def load_locked_budget_padded() -> int:
    """The operative padded budget (the alpha-cut basis for the retention floors)."""
    return int(_lock_block()["elbow_budget_padded_tokens"])


def evaluate_retention(
    sub_c_region_dir: Path,
    *,
    floors: dict[int, float] | None = None,
    budget_padded: int | None = None,
) -> dict:
    """Measure per-type retention on a sub-C region and compare to the floors.

    ``floors`` / ``budget_padded`` default to the operative lock; tests may
    override (e.g. to leg-neuter a single floor for rule isolation).
    """
    locked = load_locked_floors()
    if floors is None:
        floors = {fc: d["floor"] for fc, d in locked.items()}
    if budget_padded is None:
        budget_padded = load_locked_budget_padded()

    # Padded-budget alpha cut (budget_raw == budget_padded) — the floors' measurement
    # basis (sequence_length_analysis.yaml retention_floors_per_type._measurement_basis).
    report = compute_alpha_drop_report(sub_c_region_dir, budget_padded, budget_padded)
    by_type = report[
        "drop_set_by_type"
    ]  # {fc: {n_features_dropped, n_features_total, fraction_of_type_dropped_pct}}

    per_type: dict[int, dict] = {}
    all_pass = True
    for fc, floor in sorted(floors.items()):
        stats = by_type.get(fc)
        if stats is None:
            retention = 1.0  # type absent from this region → vacuously retained
            n_total = 0
        else:
            retention = 1.0 - stats["fraction_of_type_dropped_pct"] / 100.0
            n_total = int(stats["n_features_total"])
        passed = retention >= floor - _FLOOR_TOLERANCE
        per_type[fc] = {
            "role": locked.get(fc, {}).get("role", f"fc_{fc}"),
            "measured_retention": retention,
            "locked_floor": floor,
            "n_features_total": n_total,
            "pass": passed,
        }
        all_pass = all_pass and passed

    return {
        "budget_padded": budget_padded,
        "stage_4_provenance": report.get("stage_4_provenance"),
        "per_type": per_type,
        "all_pass": all_pass,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sub-F per-type retention gate vs the Halt-4 locked floors (reads sub-C)."
    )
    parser.add_argument("--sub-c-region-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="optional path to write the summary YAML")
    args = parser.parse_args(argv)

    result = evaluate_retention(args.sub_c_region_dir)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(yaml.safe_dump(result, sort_keys=True), encoding="utf-8")
        print(f"[empirical gate] wrote {args.out}")

    for _fc, d in sorted(result["per_type"].items()):
        status = "PASS" if d["pass"] else "FAIL"
        print(
            f"[empirical gate] {d['role']:<12} retention={d['measured_retention']:.4f} "
            f"floor={d['locked_floor']:.4f}  {status}"
        )
    print(f"[empirical gate] all_pass={result['all_pass']}")
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

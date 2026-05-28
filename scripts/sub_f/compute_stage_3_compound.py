"""Compose stage-3 compound from Task 3a joint x Task 2 encoder lock.

Per spec §7.2 stage-3 formula:
  Case A: 3 + N_anchor + 2(V-1)
  Case B: 4 + N_anchor + 2(V-2)
  Case C: 4 + N_anchor + 2(V-1)
  Case D: 5 + N_anchor + 2(V-2)

At Task 3b: stage-4 (cross-cell overhead) is NOT yet added. Output is
per-observation length distribution sans cross-cell overhead. Task 3c adds
stage-4 and aggregates to per-cell.

Pre-3c assumption: all features are Case A (uncrossed). Cross-cell
classification happens at Task 3c when the §7.2 stage-4 overhead is applied.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

import yaml

ROOT = Path(__file__).resolve().parents[2]


def case_a_tokens(v: int, n_anchor: int) -> int:
    """Stage-3 Case A: 3 + N_anchor + 2*(V-1), V >= 1.

    V = 0 should not arise in practice (every sub-C feature has at least
    one coordinate); guard with 2 ( <feature> + <semantic_tag> ).
    """
    return 3 + n_anchor + 2 * (v - 1) if v >= 1 else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    primitives = yaml.safe_load(
        (ROOT / "configs" / "sub_f" / "encoding_primitives.yaml").read_text(encoding="utf-8")
    )
    joint = yaml.safe_load(
        (ROOT / "configs" / "sub_f" / "stage_1_2_joint.yaml").read_text(encoding="utf-8")
    )

    # Locked values live under `lock_metadata.approved_lock_values` (per Halt 2 lock).
    # Fall back to `proposed_lock` then to top-level for older shapes.
    lock = (
        primitives.get("lock_metadata", {}).get("approved_lock_values")
        or primitives.get("proposed_lock")
        or primitives
    )
    anchor_scheme = lock["anchor_scheme"]
    n_anchor = 2 if anchor_scheme == "flat" else 4

    # Per-observation Case-A length using per-type mean vertex count. This is
    # an intermediate summary; Task 3c does the full per-cell aggregation
    # using the joint distribution.
    per_observation_tokens_per_type: dict[int, dict] = {}
    weighted_lengths: list[float] = []
    total_obs = 0
    for fc_key, stats in joint["per_feature_type"].items():
        n_obs = stats["n_observations"]
        v_mean = stats["vertex_count_mean"]
        v_p95 = stats["vertex_count_p95"]
        v_p99 = stats["vertex_count_p99"]
        v_max = stats["vertex_count_max"]
        if n_obs == 0:
            continue
        per_observation_tokens_per_type[int(fc_key)] = {
            "n_observations": n_obs,
            "case_a_tokens_at_v_mean": case_a_tokens(round(v_mean), n_anchor),
            "case_a_tokens_at_v_p95": case_a_tokens(int(v_p95), n_anchor) if v_p95 else None,
            "case_a_tokens_at_v_p99": case_a_tokens(int(v_p99), n_anchor) if v_p99 else None,
            "case_a_tokens_at_v_max": case_a_tokens(int(v_max), n_anchor),
        }
        # Weight by observation count.
        weighted_lengths.append(case_a_tokens(round(v_mean), n_anchor) * n_obs)
        total_obs += n_obs

    per_obs_mean = (sum(weighted_lengths) / total_obs) if total_obs else 0.0

    output = {
        "anchor_scheme_used": anchor_scheme,
        "n_anchor": n_anchor,
        "case_used": "A (uncrossed) — stage-4 cross-cell overhead added at Task 3c",
        "per_observation_tokens_mean_weighted": float(per_obs_mean),
        "per_observation_tokens_per_type": per_observation_tokens_per_type,
        "encoder_primitives_source": "configs/sub_f/encoding_primitives.yaml (Halt 2 LOCKED)",
        "direction_count": lock.get("direction_count"),
        "magnitude_quantum_m": lock.get("magnitude_quantum_m"),
        "note": (
            "Per-observation tokens, Case A only. Per-cell aggregation deferred "
            "to Task 3c; that's where the full 4D joint is composed with stage-4 "
            "overhead per spec §7.2/§7.4."
        ),
        "_status": "INTERMEDIATE — feeds Task 3c.",
    }
    out = ROOT / "configs" / "sub_f" / "stage_3_compound.yaml"
    out.write_text(yaml.safe_dump(output, sort_keys=True), encoding="utf-8")
    print(f"[stage 3 compound] wrote {out}")
    print(
        f"[stage 3 compound] anchor={anchor_scheme} n_anchor={n_anchor} "
        f"per_observation_mean_weighted={per_obs_mean:.2f} tokens"
    )
    for fc_key, row in sorted(per_observation_tokens_per_type.items()):
        print(
            f"  type {fc_key}: n={row['n_observations']} "
            f"v_mean→{row['case_a_tokens_at_v_mean']} "
            f"v_p95→{row['case_a_tokens_at_v_p95']} "
            f"v_p99→{row['case_a_tokens_at_v_p99']} "
            f"v_max→{row['case_a_tokens_at_v_max']}"
        )
    _ = mean  # keep import alive in case future revs use it
    return 0


if __name__ == "__main__":
    sys.exit(main())

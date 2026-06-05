#!/usr/bin/env python
"""Batch driver for the bounded multi-region extract (spec §3, F1).

Runs the per-city chain (``driver.run_batch``) for the given cities with sha-aware
resume via per-city state sidecars. ``--dry-run`` prints the stages that WOULD run
per city and submits nothing. The declared CPU ``--partition`` is asserted
non-boost up front (it must never be a GPU-billed partition) — fetch is in-process
and cache-hits, so a login-node pre-fetch (a separate Phase-G step) populates the
cache that the Slurm process run then reads with no egress. The rich roll-up (axis
coverage, tile/token budget) is assembled in Phase G from the ratified candidate
list + processed manifests; this CLI is the drive layer.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# src/ path inject (iCloud/editable-install mitigation; mirrors scripts/smoke.py).
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from cfm.data.multiregion import driver, state  # noqa: E402
from cfm.data.multiregion.partition import assert_cpu_partition  # noqa: E402
from cfm.data.multiregion.stages import STAGE_ORDER, StageContext  # noqa: E402

_log = logging.getLogger("cfm.data.multiregion.cli")
_DEFAULT_RELEASE = "2026-04-15.0"


def _city_ctx(region: str, release: str, repo_root: Path) -> StageContext:
    base = repo_root / "data" / "processed"
    return StageContext(
        region=region,
        release=release,
        repo_root=repo_root,
        commit_sha=driver._head_sha(repo_root),
        sub_c_dir=base / "sub_c" / release / region,
        sub_d_dir=base / "sub_d" / release / region,
        sub_e_dir=base / "sub_e" / release / region,
        sub_f_dir=base / "sub_f" / release / region,
        sub_g_dir=base / "sub_g" / release / region,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="Bounded multi-region extract batch driver")
    p.add_argument("--cities", nargs="+", required=True, help="Region names, e.g. berlin.")
    p.add_argument("--release", default=_DEFAULT_RELEASE)
    p.add_argument(
        "--partition",
        required=True,
        help="Slurm CPU partition (asserted non-boost; declared even in --dry-run).",
    )
    p.add_argument(
        "--authorized-boost-override",
        action="store_true",
        help="DELIBERATE single-run escape hatch: allow a boost (GPU) partition. "
        "PI-authorized for batch-2 preprocessing (2026-06-04, per-core billing). "
        "Guard stays default-deny; the override is logged loudly.",
    )
    p.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Per-city state sidecar dir (default <repo>/data/processed/multiregion/state).",
    )
    p.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    p.add_argument(
        "--dry-run", action="store_true", help="Print the per-city plan; submit nothing."
    )
    args = p.parse_args(argv)

    # Non-boost guard on the declared processing partition — fail loud BEFORE any work.
    # (--authorized-boost-override is the PI-authorized single-run escape hatch.)
    try:
        assert_cpu_partition(
            args.partition, authorized_boost_override=args.authorized_boost_override
        )
    except ValueError as exc:
        _log.error("%s", exc)
        return 2

    repo_root = args.repo_root.resolve()
    state_dir = args.state_dir or (repo_root / "data" / "processed" / "multiregion" / "state")
    contexts = [_city_ctx(c, args.release, repo_root) for c in args.cities]

    if args.dry_run:
        head = driver._head_sha(repo_root)
        for ctx in contexts:
            cs = state.load_city_state(state_dir / f"{ctx.region}.json", ctx.region)
            to_run = state.stages_to_run(cs, head, STAGE_ORDER, repo_root)
            print(f"[dry-run] {ctx.region}: would run {to_run or '(nothing — up to date)'}")
        return 0

    city_states = {
        ctx.region: state.load_city_state(state_dir / f"{ctx.region}.json", ctx.region)
        for ctx in contexts
    }
    results = driver.run_batch(contexts, city_states, repo_root)
    for ctx in contexts:
        state.save_city_state(state_dir / f"{ctx.region}.json", city_states[ctx.region])

    for r in results:
        _log.info("city=%s status=%s %s", r.region, r.status, r.detail)
    failed = [r.region for r in results if r.status == "failed"]
    if failed:
        _log.error("%d cities failed-needs-attention: %s", len(failed), failed)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

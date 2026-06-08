#!/usr/bin/env python
"""Re-stamp the resume-state sidecars of cities whose sub_c is already complete, so
a future driver run does NOT re-derive blessed/good cities just because a
byte-NEUTRAL source change (the v1.2 sub-F refactor + the io.py atomic-write fix)
moved HEAD past their recorded sha.

`state.stages_to_run` is git-sha based: it re-runs any stage whose source_globs
changed between the recorded sha and HEAD. io.py is in EVERY stage's globs, so
after the atomic-write commit every blessed city would re-derive on a driver run —
expensive sub_c on cities that are byte-identical. This stamps each COMPLETED stage
(by its on-disk marker) to HEAD, asserting "the on-disk artifact IS current" —
which is true: the only intervening changes are proven byte-neutral.

MARKER-BASED + SUB_C-GATED (safe by construction):
  - Only cities with sub_c/_SUCCESS are touched (the ones with expensive completed
    sub_c). The 9 timeout re-runs (partial sub_c, no _SUCCESS), welwyn (no sub_c),
    and paris/lyon/madrid (dropped, partial) are AUTOMATICALLY skipped.
  - Within a city, a stage is stamped ONLY if its marker exists. A backlog city
    with sub_c/d/e/f done but validate not-yet-passed gets 5 stages stamped, NOT
    validate — so a future run still (correctly, cheaply) re-runs only validation.

Read-mostly: only writes data/processed/multiregion/state/<city>.json. --dry-run
prints the plan and writes nothing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.multiregion import selection, state  # noqa: E402
from cfm.data.multiregion.stages import STAGE_ORDER  # noqa: E402


def _head_sha(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _marker_path(stage_name: str, city: str, release: str, proc: Path, cache: Path) -> Path:
    if stage_name == "fetch":
        return cache / release / city / "manifest.yaml"
    if stage_name == "validate":
        return proc / "sub_g" / release / city / "_PHASE1_VALIDATED"
    return proc / stage_name / release / city / "_SUCCESS"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", default="2026-04-15.0")
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--cache", default="data/cache/overture")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    proc, cache = Path(args.proc), Path(args.cache)
    sha = _head_sha(_REPO)
    state_dir = proc / "multiregion" / "state"

    cities = [
        c["name"]
        for c in selection.load_canary_manifest(
            _REPO / "configs" / "multiregion" / "canary_v1.yaml"
        )
        + selection.load_canary_manifest(_REPO / "configs" / "multiregion" / "batch2_v1.yaml")
    ]

    print(f"HEAD={sha} release={args.release}  (dry_run={args.dry_run})")
    touched = 0
    for city in sorted(cities):
        sub_c_ok = (proc / "sub_c" / args.release / city / "_SUCCESS").exists()
        if not sub_c_ok:
            continue  # partial/absent sub_c (9 timeouts, welwyn, dropped) — driver owns these
        present = [
            s.name
            for s in STAGE_ORDER
            if _marker_path(s.name, city, args.release, proc, cache).exists()
        ]
        cs = state.load_city_state(state_dir / f"{city}.json", city)
        old = {k: v.sha[:7] for k, v in cs.completions.items()}
        cs.completions = {name: state.StageCompletion(stage=name, sha=sha) for name in present}
        print(f"  {city:<16} stamp {present}  (was {old or '∅'})")
        touched += 1
        if not args.dry_run:
            state.save_city_state(state_dir / f"{city}.json", cs)
    print(f"\n{'WOULD re-stamp' if args.dry_run else 're-stamped'} {touched} cities to {sha[:7]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""scripts/derive_boundary_contracts.py

Run sub-E derivation for one region.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# iCloud-safe sys.path inject — matches scripts/smoke.py pattern.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.sub_e.pipeline import PipelineConfig, derive_region  # noqa: E402


def _git_commit_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO).decode().strip()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--release", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--sub-c-region-dir", type=Path, required=True)
    p.add_argument("--sub-d-region-dir", type=Path, required=True)
    p.add_argument("--output-region-dir", type=Path, required=True)
    p.add_argument(
        "--lever-3-collapse",
        action="store_true",
        help="Bypass class-precedence derivation; emit uniformly null boundary_class.",
    )
    args = p.parse_args()

    derive_region(
        PipelineConfig(
            release=args.release,
            region=args.region,
            sub_c_region_dir=args.sub_c_region_dir,
            sub_d_region_dir=args.sub_d_region_dir,
            output_region_dir=args.output_region_dir,
            commit_sha=_git_commit_sha(),
            lever_3_collapse=args.lever_3_collapse,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

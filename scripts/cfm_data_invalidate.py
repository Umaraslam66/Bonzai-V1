"""Invalidate (delete) a region's cached Overture data.

Usage:
    uv run python scripts/cfm_data_invalidate.py <region> [--release <version>] [--repo-root <path>]

Refuses to delete anything outside data/cache/overture/.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("region", help="Region name (e.g. 'singapore')")
    parser.add_argument(
        "--release",
        default=None,
        help="Specific release version to invalidate; defaults to currently pinned release.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Override repo root (used in tests).",
    )
    args = parser.parse_args()

    if "/" in args.region or "\\" in args.region or ".." in args.region:
        print(f"refusing to remove suspicious region name {args.region!r}", file=sys.stderr)
        return 2

    repo_root = Path(args.repo_root) if args.repo_root else _find_repo_root()
    release = args.release or _read_pinned_release(repo_root)

    target = repo_root / "data" / "cache" / "overture" / release / args.region
    safe_root = (repo_root / "data" / "cache" / "overture").resolve()
    try:
        resolved = target.resolve()
    except FileNotFoundError:
        resolved = target

    if not str(resolved).startswith(str(safe_root)):
        print(
            f"refusing to remove {resolved}: outside of {safe_root}",
            file=sys.stderr,
        )
        return 2

    if not target.exists():
        print(f"[overture] {target} not present; nothing to remove.")
        return 0

    size = _dir_size(target)
    shutil.rmtree(target)
    print(f"[overture] Removed {target.relative_to(repo_root)} ({size} bytes reclaimed).")
    return 0


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("could not locate repo root")


def _read_pinned_release(repo_root: Path) -> str:
    path = repo_root / "configs" / "data" / "overture_release.yaml"
    if not path.exists():
        print(f"missing release pin at {path}", file=sys.stderr)
        sys.exit(2)
    data = yaml.safe_load(path.read_text())
    return data["release"]


def _dir_size(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


if __name__ == "__main__":
    sys.exit(main())

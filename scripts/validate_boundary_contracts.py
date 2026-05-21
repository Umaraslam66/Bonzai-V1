#!/usr/bin/env python3
"""scripts/validate_boundary_contracts.py

Run sub-E cross-tile validator on an existing region.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.sub_e.validator_cross_tile import validate_extraction_cross_tile  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("region_dir", type=Path)
    args = p.parse_args()
    validate_extraction_cross_tile(args.region_dir)
    print(f"OK: {args.region_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

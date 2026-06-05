#!/usr/bin/env python3
"""scripts/sub_f/validate.py — sub-F region validator CLI (T12).

Validate an already-derived sub-F region: run the inline validator on every
tile, then the cross-tile validator over the region.

Two required inputs (the T10 two-arg fix): the cross-tile validator now takes
BOTH the sub-F region dir AND the sub-E region dir
(validate_cross_tile(sub_f_region_dir, sub_e_region_dir)) because its
cross-reference + coverage legs compare emitted brefs against sub-E's boundary
contract. The master-plan one-arg snippet would TypeError; this CLI surfaces
both args.

  - --region-dir         the sub-F region (with tile=*/cells.parquet + provenance.yaml)
  - --sub-e-region-dir   the sub-E region (with tile=*/boundary_contract.parquet)

Exit 0 when all checks pass; exit 1 (clear stderr) on any validation failure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# iCloud-safe sys.path inject — mirrors scripts/validate_boundary_contracts.py
# (parents[2] because this script sits one level deeper, under scripts/sub_f/).
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.sub_f.boundary_contract import SubEContractViolation  # noqa: E402
from cfm.data.sub_f.validator_cross_tile import (  # noqa: E402
    CrossTileValidationError,
    validate_cross_tile,
)
from cfm.data.sub_f.validator_inline import InlineValidationError, validate_inline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Validate a derived sub-F region (inline per tile + cross-tile)."
    )
    parser.add_argument("--region-dir", required=True, type=Path, help="sub-F region dir")
    parser.add_argument(
        "--sub-e-region-dir",
        required=True,
        type=Path,
        help="sub-E region dir (cross-tile legs cross-reference its boundary contract)",
    )
    parser.add_argument(
        "--sub-c-region-dir",
        required=True,
        type=Path,
        help="sub-C region dir (v1.2: symmetry + coverage legs read features.parquet "
        "for the road-edge-presence signal)",
    )
    args = parser.parse_args(argv)

    tile_paths = sorted(args.region_dir.glob("tile=*/cells.parquet"))
    if not tile_paths:
        print(f"[validate] no tile=*/cells.parquet under {args.region_dir}", file=sys.stderr)
        return 1

    try:
        for tile_path in tile_paths:
            validate_inline(tile_path)
        validate_cross_tile(args.region_dir, args.sub_e_region_dir, args.sub_c_region_dir)
    except (InlineValidationError, CrossTileValidationError, SubEContractViolation) as e:
        # Type name is kept in the message so callers (and tests) can tell which
        # layer failed — inline vs cross-tile vs sub-E contract.
        print(f"[validate] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(f"[validate] all checks passed for {args.region_dir} ({len(tile_paths)} tiles)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

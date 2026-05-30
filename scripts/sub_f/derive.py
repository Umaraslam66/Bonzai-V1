#!/usr/bin/env python3
"""scripts/sub_f/derive.py — sub-F region derivation CLI (T12).

Run the full sub-F derivation for one region (encode every tile → inline-validate
→ provenance → cross-tile-validate → manifest → _SUCCESS). Thin wrapper around
pipeline.derive_region; mirrors scripts/derive_boundary_contracts.py (sub-E).

Config is built by EXPLICIT field enumeration, NOT PipelineConfig(**vars(args)):
the **vars form only works while every arg dest exactly matches a field and no
other arg exists, so it breaks silently the first time a non-field flag (e.g.
--verbose) is added. Enumerating the fields survives that.

  - 6 required: --release, --region, --sub-c-region-dir, --sub-d-region-dir,
    --sub-e-region-dir, --output-region-dir
  - --extracted-utc   pin the wall-clock stamp for a BYTE-REPRODUCIBLE run
                      (excluded from the provenance sha but present in the
                      provenance.yaml bytes); default: live clock at run time.
  - --no-alpha-drop-report  skip the post-derive alpha-drop warning-band
                            diagnostic (on by default).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# iCloud-safe sys.path inject — mirrors scripts/derive_boundary_contracts.py
# (parents[2] because this script sits one level deeper, under scripts/sub_f/).
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.sub_f.pipeline import PipelineConfig, derive_region  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Derive a sub-F region (full pipeline).")
    parser.add_argument("--release", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--sub-c-region-dir", required=True, type=Path)
    parser.add_argument("--sub-d-region-dir", required=True, type=Path)
    parser.add_argument("--sub-e-region-dir", required=True, type=Path)
    parser.add_argument("--output-region-dir", required=True, type=Path)
    parser.add_argument(
        "--extracted-utc",
        default=None,
        help="pin the wall-clock stamp for a byte-reproducible run; default: live clock",
    )
    parser.add_argument(
        "--no-alpha-drop-report",
        dest="run_alpha_drop_report",
        action="store_false",
        help="skip the post-derive alpha-drop warning-band diagnostic",
    )
    args = parser.parse_args(argv)

    derive_region(
        PipelineConfig(
            release=args.release,
            region=args.region,
            sub_c_region_dir=args.sub_c_region_dir,
            sub_d_region_dir=args.sub_d_region_dir,
            sub_e_region_dir=args.sub_e_region_dir,
            output_region_dir=args.output_region_dir,
            extracted_utc=args.extracted_utc,
            run_alpha_drop_report=args.run_alpha_drop_report,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

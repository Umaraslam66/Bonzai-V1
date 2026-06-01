#!/usr/bin/env python3
"""scripts/sub_g/validate_phase1_region.py — sub-G validator-only CLI.

Validate an already-materialized region (sub-C/D/E/F present) WITHOUT running the
chain. Thin wrapper around cfm.data.sub_g.cli.validate_main; mirrors
scripts/sub_f/derive.py's iCloud-safe path inject.

Exit codes: 0 = clean, 1 = quarantine non-empty / sanity-floor breach,
2 = precondition failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

# iCloud-safe sys.path inject — mirrors scripts/sub_f/derive.py (parents[2] = repo root).
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.sub_g.cli import validate_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(validate_main(sys.argv[1:]))

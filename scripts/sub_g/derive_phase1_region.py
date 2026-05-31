#!/usr/bin/env python3
"""scripts/sub_g/derive_phase1_region.py — sub-G chain-runner CLI.

Run sub-E -> sub-F derivation on a region (resume-from-_SUCCESS), then the
cross-artifact validator. Thin wrapper around cfm.data.sub_g.cli.derive_main;
mirrors scripts/sub_f/derive.py's iCloud-safe path inject.

Exit codes: 0 = clean, 1 = quarantine non-empty / sanity-floor breach,
2 = precondition failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

# iCloud-safe sys.path inject — mirrors scripts/sub_f/derive.py (parents[2] = repo root).
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.sub_g.cli import derive_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(derive_main(sys.argv[1:]))

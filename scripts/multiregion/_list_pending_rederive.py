#!/usr/bin/env python3
"""List corpus cities whose sub_f is NOT yet re-derived to DERIVATION 1.2 (one per line).

Used by the wave-2 recovery driver to re-derive exactly what is left, self-correcting for
any partial/failed chunk (rather than assuming static chunk boundaries). Excludes the
non-corpus Phase-1/test cities and the .preserve11_* aside-copies.
"""

from __future__ import annotations

import glob
from pathlib import Path

import yaml

RELEASE = "2026-04-15.0"
PROC = Path("data/processed")
NOT_CORPUS = {"singapore", "berlin"}

for d in sorted(glob.glob(str(PROC / "sub_f" / RELEASE / "*"))):
    city = Path(d).name
    if city.startswith(".") or city in NOT_CORPUS:
        continue
    manifest = Path(d) / "manifest.yaml"
    if not manifest.exists():
        continue
    dv = str(yaml.safe_load(manifest.read_text()).get("sub_f_derivation_version"))
    if dv != "1.2":
        print(city)

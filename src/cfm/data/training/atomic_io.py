"""Atomic text writes for the training layer (known_issues F17).

Text twin of the canonical crash-safe parquet writer ``cfm.data.io.write_parquet``
(known_issues #18) and mirrors its exact shape: write to a per-pid dot-temp in the
SAME directory (same filesystem, so ``os.replace`` is an atomic rename), then
replace into place. A kill or write-failure between write and replace leaves the
destination untouched — a prior-good manifest is never truncated — and the temp
is cleaned up on failure. Output bytes are unchanged versus a plain ``write_text``,
so byte-determinism guarantees are preserved.
"""

from __future__ import annotations

import os
from pathlib import Path


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write *text* to *path* atomically (temp + ``os.replace``; see module docstring)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, path)  # atomic on same-fs; replaces any prior file in one step
    finally:
        if tmp.exists():
            tmp.unlink()  # only reached on failure (success already renamed it away)

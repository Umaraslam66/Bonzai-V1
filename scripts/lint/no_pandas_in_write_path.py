"""Lint rule: no `import pandas` or `from pandas` in sub-C write-path modules.

Per spec §14.7: pandas is forbidden in write-path modules because round-trip
through pandas can quietly coerce types (int8 → int64 in particular), which
breaks byte-determinism.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Write-path roots. Any Python file under these prefixes is scanned.
WRITE_PATH_PREFIXES = (Path("src/cfm/data/sub_c"),)

# Specific patterns to forbid.
FORBIDDEN_IMPORTS = (
    "import pandas",
    "from pandas",
)


def _is_write_path(path: Path) -> bool:
    """True iff path is under any WRITE_PATH_PREFIXES (relative to repo root)."""
    try:
        relative = path.resolve().relative_to(Path.cwd())
    except ValueError:
        return False
    return any(str(relative).startswith(str(prefix)) for prefix in WRITE_PATH_PREFIXES)


def _scan_file(path: Path) -> list[tuple[int, str]]:
    """Returns list of (line_no, offending_line) for any forbidden imports."""
    offences: list[tuple[int, str]] = []
    try:
        with open(path) as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.lstrip()
                # Skip comments / docstring lines
                if stripped.startswith("#"):
                    continue
                for forbidden in FORBIDDEN_IMPORTS:
                    if stripped.startswith(forbidden):
                        offences.append((line_no, line.rstrip()))
                        break
    except (OSError, UnicodeDecodeError):
        pass
    return offences


def main(argv: list[str] | None = None) -> int:
    """Pre-commit hook entry point. Args are the file paths to scan.
    Exits 0 if clean; 1 if any forbidden import found.
    """
    if argv is None:
        argv = sys.argv[1:]
    paths = [Path(p) for p in argv]
    failures: list[tuple[Path, int, str]] = []
    for path in paths:
        if not _is_write_path(path):
            continue
        if path.suffix != ".py":
            continue
        for line_no, line in _scan_file(path):
            failures.append((path, line_no, line))

    if failures:
        print(
            "Pandas is forbidden in sub-C write-path modules per spec §14.7:",
            file=sys.stderr,
        )
        for path, line_no, line in failures:
            print(f"  {path}:{line_no}: {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""BP2 encoder primitive candidate constants for Sub-F Halt 2."""

from __future__ import annotations

DIRECTION_COUNT_CANDIDATES: tuple[int, ...] = (8, 16, 24)
MAGNITUDE_QUANTUM_M_CANDIDATES: tuple[float, ...] = (0.25, 0.5, 1.0)
ANCHOR_SCHEMES: tuple[str, ...] = ("flat", "hierarchical")

BP2_PLACEHOLDER_START_ID = 300
BP2_PLACEHOLDER_END_ID = 1499
MAX_SEGMENT_CHUNK_M = 32.0
CELL_EXTENT_M = 250.0

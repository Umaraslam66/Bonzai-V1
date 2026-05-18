"""Per-quantity-type EPSILON constants for structural-boundary comparisons.

Per spec §4.3 / §14.4:
- Apply EPSILON at STRUCTURAL boundaries (0, 1, computed-value equality).
- Do NOT apply EPSILON at USER thresholds (500m, 0.01m²); use strict comparison.

See auto-memory feedback_epsilon_structural_vs_user_threshold.md.
"""

from __future__ import annotations

EPS_RATIO: float = 1e-9
"""For [0, 1] ratio comparisons: sea_water_fraction, water_fraction, sea_overlap_fraction."""

EPS_COORD_M: float = 1e-6
"""For SVY21 meter coordinate equality: bbox match validator, cross-run coord comparisons."""

EPS_AREA_M2: float = 1e-6
"""For m² area equality: area-weighted-mean validator, cell_area_admin_clipped_m2 > 0 check."""

EPS_LENGTH_M: float = 1e-6
"""For meter length equality (NOT for the 500m user threshold)."""

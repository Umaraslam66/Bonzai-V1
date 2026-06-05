"""Lossless-clip length-conservation guard — the v1.2 relax's must-distinguish twin.

The sub-F symmetry + coverage relax (validator v1.2) DELIBERATELY allows a road
that terminates at an internal cell boundary (§8.3) to emit on one side only.
That makes a real sub-C clip-DROP — a true crossing whose neighbour fragment was
dropped — indistinguishable from a termination to both relaxed legs (the dropped
road is simply absent on the neighbour side). This guard is the ONLY in-corpus
check that catches that drop: for each feature, the per-cell clipped fragment
lengths must sum (within tolerance) to the bbox-clipped source length. A shortfall
above tolerance is a dropped positive-length fragment.

Deliberately LENGTH-based and source-anchored — independent of the sub-C
`crossings.parquet` path whose `_both_cells_present`/`per_cell_pieces` step (taken
pre-discard) CARRIES the spurious touch-as-cross records this whole fix tolerates.
The independent source-trace corpus gate is the second, code-disjoint backstop.
See reports/2026-06-05-batch2-subf-symmetry-fp-investigation.md.
"""

from __future__ import annotations

from collections.abc import Sequence

# Drop-detection tolerance. sub-C discards <0.01 m LineString slivers and 0-d
# boundary-touch collapses BEFORE writing features (geom.py:200-202 / 224-225),
# so a legit §8.3 termination loses only zero / sub-sliver length. 0.1 m sits an
# order of magnitude above that floor and far below any real road-crossing
# fragment, so a shortfall above it is a genuine dropped fragment, not clip noise.
DROP_TOLERANCE_M = 0.1


class LosslessClipError(ValueError):
    """Raised when sub-C clipped fragments lost positive length vs the source."""


def assert_lossless_clip(
    feature_id: str,
    source_clipped_length_m: float,
    cell_fragment_lengths_m: Sequence[float],
    tol_m: float = DROP_TOLERANCE_M,
) -> None:
    """Raise LosslessClipError if the per-cell fragments dropped length vs source.

    Args:
      feature_id: the source feature id (for the error message).
      source_clipped_length_m: ``len(source_road ∩ bbox)`` — what sub-C SHOULD
        have distributed across cells (the bbox-clipped raw Overture geometry).
      cell_fragment_lengths_m: per-cell clipped fragment lengths actually written
        to features.parquet for this feature.
      tol_m: maximum tolerated shortfall (default DROP_TOLERANCE_M).

    The fragments must sum to the source length within ``tol_m``. A shortfall
    above tol = a dropped fragment — the drop mode the v1.2 symmetry/coverage
    relax allows past those legs, so this guard is the backstop. A legit §8.3
    termination conserves length (the neighbour clip is a zero-length boundary
    point), so it does NOT raise.
    """
    total = float(sum(cell_fragment_lengths_m))
    shortfall = float(source_clipped_length_m) - total
    if shortfall > tol_m:
        raise LosslessClipError(
            f"lossless-clip violation for feature {feature_id!r}: bbox-clipped "
            f"source length {source_clipped_length_m:.4f}m but per-cell fragments "
            f"sum to {total:.4f}m — a {shortfall:.4f}m fragment was dropped "
            f"(> tol {tol_m}m). A true crossing lost a fragment; the v1.2 "
            f"symmetry/coverage relax allows this past those legs, so this "
            f"length-conservation guard is the backstop. See "
            f"reports/2026-06-05-batch2-subf-symmetry-fp-investigation.md."
        )

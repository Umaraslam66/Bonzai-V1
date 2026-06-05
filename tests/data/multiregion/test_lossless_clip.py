"""Length-conservation drop-guard tests — the must-distinguish twin for the v1.2
sub-F symmetry/coverage relax.

The relax DELIBERATELY allows a one-sided emission (a road terminating at an
internal cell boundary, §8.3). That makes a real sub-C clip-DROP (a true crossing
whose neighbour fragment was dropped) INVISIBLE to both relaxed legs — the dropped
road is simply absent on the neighbour side, identical to a termination. This
guard is the ONLY in-corpus check that catches the drop: the per-cell clipped
fragment lengths must sum (within tolerance) to the bbox-clipped source length.
See reports/2026-06-05-batch2-subf-symmetry-fp-investigation.md.
"""

from __future__ import annotations

import pytest

from cfm.data.multiregion.lossless_clip import LosslessClipError, assert_lossless_clip


def test_dropped_crossing_fragment_raises():
    """Fixture (c): a true crossing A->B whose B fragment (40m) was dropped.

    source ∩ bbox = 100m but only the 60m A fragment survives -> a 40m shortfall
    -> LosslessClipError. This is the drop mode both relaxed legs allow past them;
    the guard MUST catch it.
    """
    with pytest.raises(LosslessClipError, match="dropped"):
        assert_lossless_clip(
            "fid-crossing", source_clipped_length_m=100.0, cell_fragment_lengths_m=[60.0]
        )


def test_legit_termination_conserves_length_passes():
    """Near-miss: a road TERMINATING at the internal boundary — wholly in A (50m);
    the neighbour clip is a degenerate boundary point (zero length, discarded by
    sub-C). Fragments sum to the source length -> no shortfall -> PASSES. The guard
    must NOT fire on the legit §8.3 termination it coexists with (else the relax is
    unsafe — it would re-break the 14 edges from a different leg).
    """
    assert_lossless_clip(
        "fid-termination", source_clipped_length_m=50.0, cell_fragment_lengths_m=[50.0]
    )  # must NOT raise


def test_sub_threshold_sliver_loss_within_tolerance_passes():
    """A legit <0.01m sliver / 0-d touch discarded by sub-C's filters loses only
    sub-tolerance length and must NOT raise — otherwise the guard re-fires on the
    very touch-as-cross the v1.2 relax tolerates."""
    assert_lossless_clip(
        "fid-sliver", source_clipped_length_m=50.005, cell_fragment_lengths_m=[50.0]
    )  # 5 mm shortfall < tol -> must NOT raise


def test_multi_cell_crossing_all_fragments_present_passes():
    """A road crossing three cells (40+35+25=100m) with every fragment present
    conserves length -> PASSES (the normal, non-dropped case)."""
    assert_lossless_clip(
        "fid-multi", source_clipped_length_m=100.0, cell_fragment_lengths_m=[40.0, 35.0, 25.0]
    )  # must NOT raise

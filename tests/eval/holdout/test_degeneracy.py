from __future__ import annotations

from cfm.eval.holdout import degeneracy
from cfm.eval.holdout.bref_rate import bref_placeholder_rate
from cfm.eval.holdout.sizing import DELTA_BREF_REGIME

# --- per-instance exclusion fixtures ---
_OUTBOUND_BREF_COLLAPSE = [509, 41, 300, 323, 363, 369, 1500, 510]  # body ends in bref
_COLLAPSE_GEOM = {"type": "LineString", "coordinates": [[0.0, 0.0], [0.0, 0.0]]}
# G-D1: a MODEL-style degenerate block with NO outbound bref (distinct semantic tag
# from sub-G's real-data fixture) - the model emitted a zero-length stub it should not.
_MODEL_DEGENERATE_NO_BREF = [509, 7, 300, 323, 363, 369, 511, 443, 510]
_OK_BLOCK = [509, 7, 300, 1, 50, 510]
_OK_GEOM = {"type": "LineString", "coordinates": [[0.0, 0.0], [10.0, 0.0]]}


def test_per_instance_excludes_outbound_bref_collapse():
    v = degeneracy.classify_block(_OUTBOUND_BREF_COLLAPSE, _COLLAPSE_GEOM)
    assert v is degeneracy.Verdict.EXCLUDED_BREF_PLACEHOLDER  # faithful model not penalized


def test_GD1_gate_fires_on_model_emitted_degeneracy_without_bref():
    """G-D1 RE-PROVEN on a MODEL-emitted fixture (not inherited from sub-G's real-data
    drill): identical zero-length symptom, no outbound bref -> MODEL_INVALID. Proves the
    exclusion keys on construction identity in the regime the model populates."""
    v = degeneracy.classify_block(_MODEL_DEGENERATE_NO_BREF, _COLLAPSE_GEOM)
    assert v is degeneracy.Verdict.MODEL_INVALID


def test_GD2_at_threshold_just_over_trips_just_under_passes():
    """G-D2 at the threshold (not at 2x): faithful rate r0; a model emitting just past
    r0+delta must TRIP; just under must PASS. delta is the single DELTA_BREF_REGIME."""
    r0 = 0.05  # round-tripped-real faithful rate in this stratum
    over = degeneracy.over_emission_verdict(
        model_rate=r0 + DELTA_BREF_REGIME + 0.005, faithful_rate=r0
    )
    under = degeneracy.over_emission_verdict(
        model_rate=r0 + DELTA_BREF_REGIME - 0.005, faithful_rate=r0
    )
    assert over is degeneracy.RateVerdict.OVER_EMITTING
    assert under is degeneracy.RateVerdict.WITHIN_TOLERANCE


def test_GD2_stratified_cancellation_global_matches_but_one_stratum_diverges():
    """G-D2 stratified (not global): the Singapore-wide rate matches round-tripped-real
    while a dense stratum over-emits and a sparse stratum under-emits. A global check
    passes (vacuous); the stratified check MUST trip on the diverging stratum.
    Strata = cell_density_bucket (density is an aggregate - labels.py)."""
    faithful = {0: 0.05, 3: 0.05}  # per-stratum faithful rates
    # model: stratum 3 over-emits (10/100=0.10), stratum 0 under-emits (0/100) -> global 0.05
    model_blocks = (
        [_OUTBOUND_BREF_COLLAPSE] * 10  # stratum 3: 10 collapse
        + [_OK_BLOCK] * 90  # stratum 3: 90 ok
        + [_OK_BLOCK] * 100  # stratum 0: 100 ok
    )
    model_geoms = [_COLLAPSE_GEOM] * 10 + [_OK_GEOM] * 90 + [_OK_GEOM] * 100
    model_strata = [3] * 100 + [0] * 100
    model_rate = bref_placeholder_rate(model_blocks, model_geoms, model_strata)

    report = degeneracy.stratified_over_emission(model_rate, faithful_rate=faithful)
    assert report.global_within_tolerance is True  # the vacuous-pass signal
    assert 3 in report.over_emitting_strata  # the stratified check fires
    assert report.over_emitting_strata[3] == 0.05

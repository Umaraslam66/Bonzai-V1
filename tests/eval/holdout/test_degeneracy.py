from __future__ import annotations

from cfm.eval.holdout import degeneracy
from cfm.eval.holdout.bref_rate import bref_placeholder_rate
from cfm.eval.holdout.sizing import over_emission_threshold

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


def test_GD2_at_threshold_relative_just_over_trips_just_under_passes():
    """G-D2 at the threshold: the trip boundary is the RELATIVE per-stratum threshold
    max(rho*faithful, delta_floor). Just past it must TRIP; just under must PASS."""
    r0 = 0.04
    thr = over_emission_threshold(r0)  # relative: 0.5*0.04 = 0.02 (above the 0.005 floor)
    over = degeneracy.over_emission_verdict(model_rate=r0 + thr + 0.005, faithful_rate=r0)
    under = degeneracy.over_emission_verdict(model_rate=r0 + thr - 0.005, faithful_rate=r0)
    assert over is degeneracy.RateVerdict.OVER_EMITTING
    assert under is degeneracy.RateVerdict.WITHIN_TOLERANCE


def test_GD2_GUARD_dense_bucket_doubling_trips_under_relative_but_absolute_missed_it():
    """REGIME-DISTINGUISHING GUARD (2026-06-01 δ review): the bug was that an ABSOLUTE
    δ=0.03 waved through a DOUBLING of the dense bucket's rate (faithful 2.33% -> 4.66%,
    excess 2.33pp < 3pp). The RELATIVE guard MUST fire on exactly that case - this is
    the specific failure the new form has to catch."""
    dense_faithful = 0.0233
    doubled = dense_faithful * 2.0  # the model emits twice the faithful degenerate rate
    excess = doubled - dense_faithful
    # the OLD absolute guard would have MISSED it:
    assert excess < 0.03
    # the NEW relative guard FIRES on it (threshold 0.5*0.0233 = 0.01165 < excess 0.0233):
    assert (
        degeneracy.over_emission_verdict(model_rate=doubled, faithful_rate=dense_faithful)
        is degeneracy.RateVerdict.OVER_EMITTING
    )


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

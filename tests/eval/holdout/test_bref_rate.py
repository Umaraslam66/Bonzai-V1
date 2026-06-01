from __future__ import annotations

from cfm.data.sub_g import seam_decodability
from cfm.eval.holdout import bref_rate


def test_bref_rate_overall_and_stratified():
    # block, geom pairs: two collapses in stratum 3, none in stratum 0.
    collapse_block = [509, 41, 300, 323, 363, 369, 1500, 510]  # body ends in bref (1500)
    collapse_geom = {"type": "LineString", "coordinates": [[0.0, 0.0], [0.0, 0.0]]}
    ok_block = [509, 41, 300, 323, 363, 369, 1, 50, 510]
    ok_geom = {"type": "LineString", "coordinates": [[0.0, 0.0], [10.0, 0.0]]}

    blocks = [collapse_block, collapse_block, ok_block]
    geoms = [collapse_geom, collapse_geom, ok_geom]
    strata = [3, 3, 0]

    res = bref_rate.bref_placeholder_rate(blocks, geoms, strata)
    assert res.overall_rate == 2 / 3
    assert res.per_stratum[3].n_total == 2 and res.per_stratum[3].n_collapse == 2
    assert res.per_stratum[3].rate == 1.0
    assert res.per_stratum[0].n_collapse == 0 and res.per_stratum[0].rate == 0.0


def test_GATE6_identity_lock_uses_sub_g_predicate_verbatim():
    """Gate 6: the eval must classify a block EXACTLY as sub-G does - same import,
    no reimplementation. Cross-reference: sub-G's own degenerate-no-bref fixture
    (a genuine defect with identical zero-length symptom but NO outbound bref) must
    NOT be counted as a placeholder collapse here, proving we key on construction
    identity, not magnitude."""
    # The eval's predicate IS sub-G's object (identity, not a copy).
    assert bref_rate._bref_predicate is seam_decodability._is_bref_placeholder_collapse

    degenerate_no_bref = [509, 41, 300, 323, 363, 369, 511, 443, 510]  # sub-G fixture shape
    zero_len_geom = {"type": "LineString", "coordinates": [[0.0, 0.0], [0.0, 0.0]]}
    res = bref_rate.bref_placeholder_rate([degenerate_no_bref], [zero_len_geom], [0])
    assert res.per_stratum[0].n_collapse == 0  # genuine defect is NOT excluded


def test_empty_input_is_zero_not_division_error():
    res = bref_rate.bref_placeholder_rate([], [], [])
    assert res.overall_rate == 0.0 and res.per_stratum == {}

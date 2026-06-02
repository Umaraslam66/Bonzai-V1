"""Task 9 discrimination tests for inference / decode.

The decode path REUSES the sealed sub-F decoder and sub-G splitter (one source,
never reimplemented) and is ROBUST: a block that survives the 509/510 split but
fails to decode is skipped (mirrors sub-G ``check_decodability``'s per-block
try/except), so a random untrained model never makes decode raise. Decodability
is then a measured RATE (decoded / attempted), not an exception.
"""

from __future__ import annotations

from cfm.data.sub_f import decoder as sub_f_decoder
from cfm.data.sub_g import seam_decodability as sub_g_seam
from cfm.data.training.conditioning import CONDITIONING_ID_BASE
from cfm.inference import generate as G
from cfm.models.micro_ar import MicroAR, MicroARConfig

# Known-good Case-A block (decodes to a 2-vertex LineString, verified) and a block
# that survives the 509/510 split but FAILS to decode (anchor underflow).
_GOOD = [509, 41, 300, 323, 363, 369, 1, 50, 510]
_BAD = [509, 7, 300, 1, 50, 510]
_N_SUBF, _N_COND = 1508, 8


def _model() -> MicroAR:
    return MicroAR(
        MicroARConfig(
            d_model=64, n_layers=2, n_heads=2, n_subf_vocab=_N_SUBF, n_cond=_N_COND, max_len=128
        )
    )


def test_decode_reuses_sealed_functions_one_source():
    assert G.split_cell_into_features is sub_g_seam.split_cell_into_features
    assert G.decode_feature is sub_f_decoder.decode_feature


def test_decode_known_good_cell_is_nonvacuous():
    geoms = G.decode_cell_to_geojson(_GOOD + _GOOD)
    assert len(geoms) == 2
    assert all(g["type"] == "LineString" for g in geoms)


def test_decode_skips_undecodable_block_without_raising():
    # 2 well-formed-by-marker blocks; only _GOOD decodes -> 1 geom, no raise.
    geoms = G.decode_cell_to_geojson(_GOOD + _BAD)
    assert len(geoms) == 1
    assert geoms[0]["type"] == "LineString"
    # the robust per-block primitive: None on failure, dict on success (twin)
    assert G.try_decode_block(_BAD) is None
    assert G.try_decode_block(_GOOD)["type"] == "LineString"


def test_generated_tokens_decode_through_sealed_decoder():
    m = _model()
    prefix = list(range(CONDITIONING_ID_BASE, CONDITIONING_ID_BASE + _N_COND))  # field-slot prefix
    tokens = G.generate_cell_tokens(m, prefix=prefix, max_new=64, seed=0)
    assert all(0 <= t < _N_SUBF for t in tokens)  # head emits sub-F range only
    geoms = G.decode_cell_to_geojson(tokens)  # may be empty (untrained), never raises
    assert isinstance(geoms, list)


def test_generation_is_seed_reproducible():
    m = _model()
    prefix = list(range(CONDITIONING_ID_BASE, CONDITIONING_ID_BASE + _N_COND))
    a = G.generate_cell_tokens(m, prefix=prefix, max_new=32, seed=7)
    b = G.generate_cell_tokens(m, prefix=prefix, max_new=32, seed=7)
    c = G.generate_cell_tokens(m, prefix=prefix, max_new=32, seed=8)
    assert a == b  # same seed -> identical (reproducible)
    assert a != c  # different seed -> different (sampling is actually stochastic)

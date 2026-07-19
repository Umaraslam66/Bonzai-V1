"""Task 9 discrimination tests for inference / decode.

The decode path REUSES the sealed sub-F decoder and sub-G splitter (one source,
never reimplemented) and is ROBUST: a block that survives the 509/510 split but
fails to decode is skipped (mirrors sub-G ``check_decodability``'s per-block
try/except), so a random untrained model never makes decode raise. Decodability
is then a measured RATE (decoded / attempted), not an exception.
"""

from __future__ import annotations

import torch

from cfm.data.sub_f import decoder as sub_f_decoder
from cfm.data.sub_f.vocab import CELL_END_TOKEN_ID
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


# ----------------------------------------------------------------------------- #
# Tooth 5 (cell-EOS): the generator BREAK isolated from model learning. Proves the
# break wiring with NO training, so a future Tooth-1 RED can be triaged:
# Tooth 5 green + Tooth 1 red => undertraining, not broken break-logic.
# ----------------------------------------------------------------------------- #

_OTHER_ID = 5  # a non-260 id the stub emits before the terminator (must be < _STUB_VOCAB)
_STUB_VOCAB = 512


class _ForcedEmitModel:
    """Controllable MicroAR stand-in: emits ``_OTHER_ID`` for the first k-1 sampled
    steps and <cell_end>=260 on the k-th step (and every step after, so WITHOUT the
    break it runs to max_new). Exposes only the interface generate_cell_tokens
    touches: ``.training``/``.eval()``/``.train()``/``.parameters()``/``__call__``.
    One-hot logits (favored=0.0, rest=-inf) make multinomial deterministic."""

    def __init__(self, *, k: int) -> None:
        self.k = k
        self.calls = 0
        self.training = False
        self._param = torch.zeros(1)  # the device probe reads next(model.parameters())

    def parameters(self):
        return iter([self._param])

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def __call__(self, ids: torch.Tensor, char_stats: torch.Tensor | None = None) -> torch.Tensor:
        self.calls += 1
        favored = CELL_END_TOKEN_ID if self.calls >= self.k else _OTHER_ID
        t = ids.shape[1]
        logits = torch.full((1, t, _STUB_VOCAB), float("-inf"))
        logits[0, -1, favored] = 0.0  # only the last position is read ([:, -1]) -> one-hot
        return logits


def test_generate_breaks_on_cell_end_and_keeps_it():
    """The model emits 260 on the k-th step; generation stops there (NOT at max_new),
    the trailing 260 is KEPT, and __call__ was invoked exactly k times (no over-run)."""
    k = 4
    max_new = 50
    model = _ForcedEmitModel(k=k)

    tail = G.generate_cell_tokens(model, prefix=[0, 1, 2], max_new=max_new, seed=0)

    assert len(tail) == k  # stops on the step that emitted 260, NOT at the cap
    assert tail[-1] == CELL_END_TOKEN_ID  # the trailing 260 is included (D.9-keep)
    assert tail[:-1] == [_OTHER_ID] * (k - 1)  # the preceding k-1 tokens
    assert len(tail) < max_new  # decisively unpinned from the cap
    assert model.calls == k  # halted at the break — no forward past the terminator


def test_generate_runs_to_cap_when_no_cell_end_emitted():
    """Contrast / non-vacuity: a model that NEVER emits 260 runs the full max_new
    (the pre-fix baseline shape). This is what Tooth 1 must move off of."""
    max_new = 16
    # k > max_new: the stub never reaches the 260 step within the budget.
    model = _ForcedEmitModel(k=max_new + 5)

    tail = G.generate_cell_tokens(model, prefix=[0, 1, 2], max_new=max_new, seed=0)

    assert len(tail) == max_new  # pinned at the cap
    assert CELL_END_TOKEN_ID not in tail
    assert model.calls == max_new

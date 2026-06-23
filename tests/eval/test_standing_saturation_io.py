"""Saturation IO (spec §3, D4): CSV reader + version_N->(backbone,seed) resolver.

D4 ruling: resolve from hparams.yaml on the full bake-off signature; FAIL LOUD on
ambiguity or no-match (do NOT guess) — the version dirs also hold old singapore smokes.
"""

from __future__ import annotations

import pytest

from cfm.eval.standing.saturation import read_loss_series, resolve_bakeoff_run


def test_read_loss_series_skips_blank_train_loss(tmp_path):
    csv = tmp_path / "metrics.csv"
    csv.write_text("epoch,step,train_loss,val_loss\n0,49,4.8,\n0,99,3.9,\n5,1249,2.7,2.6\n,,,\n")
    steps, losses = read_loss_series(csv)
    assert steps == [49, 99, 1249]
    assert losses == [4.8, 3.9, 2.7]


def _mk(tmp_path, ver, **hp):
    d = tmp_path / f"version_{ver}"
    d.mkdir()
    (d / "hparams.yaml").write_text("\n".join(f"{k}: {v}" for k, v in hp.items()) + "\n")
    (d / "metrics.csv").write_text("epoch,step,train_loss,val_loss\n0,1,3.0,\n")
    return d


_BAKEOFF = dict(d_model=512, train_set="eu-train-union", conditioning_scheme="value-char-v1")


def test_resolve_matches_full_signature_not_old_smoke(tmp_path):
    # an old singapore d256 smoke with the same backbone+seed must NOT match
    _mk(
        tmp_path,
        0,
        backbone="transformer-ar",
        seed=7,
        d_model=256,
        train_set="single",
        conditioning_scheme="value-char-v1",
    )
    good = _mk(tmp_path, 5, backbone="transformer-ar", seed=7, **_BAKEOFF)
    assert resolve_bakeoff_run(tmp_path, backbone="transformer-ar", seed=7) == good


def test_resolve_fails_loud_on_ambiguous(tmp_path):
    _mk(tmp_path, 1, backbone="mamba-hybrid", seed=13, **_BAKEOFF)
    _mk(tmp_path, 2, backbone="mamba-hybrid", seed=13, **_BAKEOFF)  # duplicate match
    with pytest.raises(ValueError, match=r"(?i)ambiguous|multiple"):
        resolve_bakeoff_run(tmp_path, backbone="mamba-hybrid", seed=13)


def test_resolve_fails_loud_on_no_match(tmp_path):
    _mk(tmp_path, 0, backbone="transformer-ar", seed=7, **_BAKEOFF)
    with pytest.raises(ValueError, match=r"(?i)no.*match|not found"):
        resolve_bakeoff_run(tmp_path, backbone="mamba-hybrid", seed=23)

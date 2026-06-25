"""Saturation IO (spec §3, D4): CSV reader + version_N->(backbone,seed) resolver.

D4 ruling: resolve from hparams.yaml on the full bake-off signature; FAIL LOUD on
ambiguity or no-match (do NOT guess) — the version dirs also hold old singapore smokes.
"""

from __future__ import annotations

import pytest

from cfm.eval.standing import saturation as sat_mod
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
    # transformer-ar seed=13 is NOT overridden, so two matching dirs are genuinely ambiguous.
    _mk(tmp_path, 1, backbone="transformer-ar", seed=13, **_BAKEOFF)
    _mk(tmp_path, 2, backbone="transformer-ar", seed=13, **_BAKEOFF)  # duplicate match
    with pytest.raises(ValueError, match=r"(?i)ambiguous|multiple"):
        resolve_bakeoff_run(tmp_path, backbone="transformer-ar", seed=13)


def test_resolve_fails_loud_on_no_match(tmp_path):
    _mk(tmp_path, 0, backbone="transformer-ar", seed=7, **_BAKEOFF)
    with pytest.raises(ValueError, match=r"(?i)no.*match|not found"):
        resolve_bakeoff_run(tmp_path, backbone="mamba-hybrid", seed=23)


# ── D4 committed lookup table (§9 teeth): the mamba logs are ambiguous (many crashed restarts
#    share the signature). The table pins the single COMPLETED run; a wrong pin must FAIL LOUD,
#    never silently map to a crashed-restart stub. ──────────────────────────────────────────────


def _mk_run(tmp_path, ver, *, final_step, **hp):
    """Like _mk but the metrics.csv's last logged step == final_step (3 rows)."""
    d = tmp_path / f"version_{ver}"
    d.mkdir()
    (d / "hparams.yaml").write_text("\n".join(f"{k}: {v}" for k, v in hp.items()) + "\n")
    s0, s1 = max(final_step - 2, 0), max(final_step - 1, 0)
    rows = "".join(f"0,{st},{tl},\n" for st, tl in ((s0, 3.2), (s1, 3.1), (final_step, 3.0)))
    (d / "metrics.csv").write_text("epoch,step,train_loss,val_loss\n" + rows)
    return d


def test_committed_override_table_pins_canonical_mamba_runs():
    # The decision: pin the COMPLETED canonical runs for the ambiguous mamba seeds (D4).
    assert sat_mod._BAKEOFF_VERSION_OVERRIDE[("mamba-hybrid", 7)] == "version_25"
    assert sat_mod._BAKEOFF_VERSION_OVERRIDE[("mamba-hybrid", 13)] == "version_27"


def test_override_resolves_ambiguous_mamba_seed7_to_completed_run(tmp_path):
    # Reproduce the real ambiguity: 5 dirs all match the mamba-hybrid seed7 signature; only
    # version_25 is the COMPLETED run (final_step≈112549), the rest are crashed restarts.
    _mk_run(tmp_path, 19, final_step=99, backbone="mamba-hybrid", seed=7, **_BAKEOFF)
    _mk_run(tmp_path, 20, final_step=2, backbone="mamba-hybrid", seed=7, **_BAKEOFF)
    _mk_run(tmp_path, 22, final_step=10232, backbone="mamba-hybrid", seed=7, **_BAKEOFF)
    _mk_run(tmp_path, 23, final_step=149, backbone="mamba-hybrid", seed=7, **_BAKEOFF)
    done = _mk_run(tmp_path, 25, final_step=112549, backbone="mamba-hybrid", seed=7, **_BAKEOFF)
    assert resolve_bakeoff_run(tmp_path, backbone="mamba-hybrid", seed=7) == done


def test_override_to_crashed_stub_fails_loud(tmp_path, monkeypatch):
    # A pin to a crashed-restart stub (final_step=10232, below the completion floor) must FAIL
    # LOUD — never silently read saturation off a stub.
    _mk_run(tmp_path, 22, final_step=10232, backbone="mamba-hybrid", seed=7, **_BAKEOFF)
    monkeypatch.setattr(sat_mod, "_BAKEOFF_VERSION_OVERRIDE", {("mamba-hybrid", 7): "version_22"})
    with pytest.raises(ValueError, match=r"(?i)stub|completion floor"):
        resolve_bakeoff_run(tmp_path, backbone="mamba-hybrid", seed=7)


def test_override_to_signature_mismatch_fails_loud(tmp_path, monkeypatch):
    # A typo'd pin to a COMPLETED run for a DIFFERENT seed must FAIL LOUD (refuse a mismatched pin).
    _mk_run(tmp_path, 3, final_step=112549, backbone="mamba-hybrid", seed=99, **_BAKEOFF)
    monkeypatch.setattr(sat_mod, "_BAKEOFF_VERSION_OVERRIDE", {("mamba-hybrid", 7): "version_3"})
    with pytest.raises(ValueError, match=r"(?i)mismatched pin"):
        resolve_bakeoff_run(tmp_path, backbone="mamba-hybrid", seed=7)

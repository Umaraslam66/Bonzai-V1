from __future__ import annotations

from cfm.eval.conditioning_gate import conditioning_discrimination_gate


def test_gate_passes_when_same_stratum_tiles_share_distributions():
    r = conditioning_discrimination_gate(
        {"glasgow_v_munich": 0.04, "krakow_v_munich": 0.05}, tolerance=0.08
    )
    assert r.passes is True


def test_gate_fails_on_residual_city_style_and_signals_reopen():
    r = conditioning_discrimination_gate({"glasgow_v_munich": 0.22}, tolerance=0.08)
    assert r.passes is False
    assert "REOPEN" in r.reason.upper()


# (A) fail-closed: a gate that cannot be discharged must NOT silently pass.
def test_gate_fail_closes_on_empty_input():
    r = conditioning_discrimination_gate({}, tolerance=0.08)
    assert r.passes is False


# (B) boundary discrimination: SAME shape flips outcome only on crossing tolerance.
def test_gate_discriminates_at_the_tolerance_boundary():
    # just below tolerance -> PASS; just above -> FIRE/REOPEN. Same key, only the value crosses.
    assert conditioning_discrimination_gate({"a_v_b": 0.09}, tolerance=0.10).passes is True
    fired = conditioning_discrimination_gate({"a_v_b": 0.11}, tolerance=0.10)
    assert fired.passes is False and "REOPEN" in fired.reason.upper()


# (B-edge) the `worst == tolerance` equality edge PASSES per the plan's `<=`.
def test_gate_equality_edge_passes():
    assert conditioning_discrimination_gate({"a_v_b": 0.10}, tolerance=0.10).passes is True

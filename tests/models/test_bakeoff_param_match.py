"""Param-match gate for the bake-off scale table (spec §3.3, §8 — a VERIFIED gate).

Architecture must be the ONLY variable: at each scale transformer-ar and mamba-hybrid must
land at the SAME parameter count (not the same depth) — equal-depth would be unequal
capacity (a Jamba 7:1 layer has different per-layer params than a pure-transformer layer),
confounding architecture with capacity and breaking the bake-off's validity condition. This
counts the ACTUAL built models (never the eyeballed mapping) and asserts each pair matches
within 2% per scale.

Marked ``slow``: constructing the 1B pair allocates a few GB and exceeds the default <5s
budget. ``importorskip`` so it runs only where the mamba env is present (the unified
Leonardo bake-off env), skipping the mamba-less default repo env.
"""

import pytest

pytest.importorskip("mamba_ssm")

from cfm.models.bakeoff_scales import BAKEOFF_SCALES, build_pair_for_scale
from cfm.models.mamba_hybrid import MambaHybrid
from cfm.models.micro_ar import MicroAR


def _params(m) -> int:
    return sum(p.numel() for p in m.parameters())


@pytest.mark.slow
@pytest.mark.parametrize("scale", BAKEOFF_SCALES)
def test_backbones_param_matched_within_tolerance(scale: str) -> None:
    tcfg, mcfg = build_pair_for_scale(scale)
    nt = _params(MicroAR(tcfg))
    nm = _params(MambaHybrid(mcfg))
    rel = abs(nt - nm) / nt
    assert rel <= 0.02, f"{scale}: transformer {nt:,} vs mamba {nm:,} = {rel:.1%} > 2% tol"

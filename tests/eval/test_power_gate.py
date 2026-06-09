"""T12: assert_coherence_power_sufficient — sole architecture-discrimination verdict gate.

Tests use INJECTED effect values; the real definition of model_vs_real_effect is an
OPEN first-model decision (see docstring of assert_coherence_power_sufficient).
"""

import pytest

from cfm.eval.resolution import CoherencePowerInsufficientError, assert_coherence_power_sufficient


def test_coherence_power_failloud_owns_the_swap():
    # INJECTED effect (NOT a real definition): finer than the train-resolved gap -> unresolvable
    with pytest.raises(CoherencePowerInsufficientError) as e:
        assert_coherence_power_sufficient(
            stratum="munich", usable_n=156, resolved_gap=0.10, model_vs_real_effect=0.04
        )
    assert "munich->manchester" in str(e.value)  # THIS gate owns the swap (KS-resolution does not)


def test_coherence_power_passes_when_resolvable():
    # INJECTED effect bigger than the resolved gap -> resolvable -> no raise
    assert_coherence_power_sufficient(
        stratum="munich", usable_n=156, resolved_gap=0.10, model_vs_real_effect=0.20
    )

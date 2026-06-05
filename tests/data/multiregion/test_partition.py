"""Must-distinguish pair for the non-boost guard: it must FIRE on GPU partitions
(boost_usr_prod AND any boost_* family name), not merely accept CPU ones. A guard
only ever seen passing is untested."""

from __future__ import annotations

import pytest

from cfm.data.multiregion import partition


@pytest.mark.parametrize("p", ["boost_usr_prod", "boost_qos_dbg", "boost_fancy_new"])
def test_gpu_partitions_rejected_loud(p):
    with pytest.raises(ValueError, match="GPU-billed"):
        partition.assert_cpu_partition(p)


@pytest.mark.parametrize("p", ["dcgp_usr_prod", "lrd_all_serial"])
def test_cpu_partitions_accepted(p):
    partition.assert_cpu_partition(p)  # must not raise


def test_boost_still_denied_by_default_when_override_false():
    # The override must be OPT-IN: default-deny is unchanged.
    with pytest.raises(ValueError, match="GPU-billed"):
        partition.assert_cpu_partition("boost_usr_prod", authorized_boost_override=False)


def test_authorized_override_allows_boost_and_logs_loud(caplog):
    # Must-distinguish: same call FIRES without the flag (above), PASSES with it.
    import logging

    with caplog.at_level(logging.WARNING, logger="cfm.data.multiregion.partition"):
        partition.assert_cpu_partition("boost_usr_prod", authorized_boost_override=True)  # no raise
    assert any("AUTHORIZED-OVERRIDE" in r.message for r in caplog.records), (
        "the override must log loudly so the exception is auditable"
    )


def test_override_does_not_affect_cpu_partitions():
    # The flag is inert on CPU partitions (no spurious allow-path behaviour).
    partition.assert_cpu_partition("lrd_all_serial", authorized_boost_override=True)

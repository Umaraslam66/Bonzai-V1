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

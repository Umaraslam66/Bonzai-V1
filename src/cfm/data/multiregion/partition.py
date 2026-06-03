"""Topology guard (spec §3.2): extraction is CPU-only and must NEVER run on a
GPU-billed Leonardo partition. The ``boost_*`` family bills a 4xA100 node for
CPU work and would silently consume the training budget — so this is a hard
runtime assertion on the Slurm PROCESSING-submission path only. Fetch runs on the
login node (no partition) and never calls this; the orchestrator driver is
partition-agnostic (it runs subprocesses wherever it is invoked).
"""

from __future__ import annotations

#: Leonardo GPU (A100) partition family — reject anything in it.
FORBIDDEN_PARTITION_PREFIXES = ("boost",)
#: Explicit known GPU partition (belt-and-suspenders alongside the prefix rule).
FORBIDDEN_PARTITIONS = frozenset({"boost_usr_prod"})
#: Known CPU partitions extraction may use (documentation; not an allowlist gate).
KNOWN_CPU_PARTITIONS = frozenset({"dcgp_usr_prod", "lrd_all_serial"})


def assert_cpu_partition(partition: str) -> None:
    """Fail loud if ``partition`` is a GPU-billed Leonardo partition.

    Rejects the ``boost_*`` family (current Leonardo GPU partitions). A new GPU
    partition outside that family must be added here. CPU partitions
    (dcgp_usr_prod / lrd_all_serial) pass.
    """
    if partition in FORBIDDEN_PARTITIONS or partition.startswith(FORBIDDEN_PARTITION_PREFIXES):
        raise ValueError(
            f"refusing to submit CPU extraction to GPU-billed partition "
            f"{partition!r}; use a CPU partition (e.g. dcgp_usr_prod / "
            f"lrd_all_serial). The boost_* family bills a 4xA100 node and would "
            f"eat the training budget."
        )

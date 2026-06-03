"""Guard the committed process-sbatch template against a copy-pasted GPU header
(the regression that would silently eat the training budget): its #SBATCH
partition must pass assert_cpu_partition, it must request no GPU, and it must
invoke the batch CLI."""

from __future__ import annotations

import re
from pathlib import Path

from cfm.data.multiregion.partition import assert_cpu_partition

_REPO = Path(__file__).resolve().parents[3]
_SBATCH = _REPO / "scripts" / "multiregion_process.sbatch"

# lrd_all_serial QoS per-user/per-job caps, verified on Leonardo 2026-06-04 via
# `sacctmgr show qos lrd_all_serial` (MaxTRES = cpu=8,mem=30800M,node=1; MaxWall
# 04:00:00). The original template asked 32cpu/120G and was REJECTED at submit
# (QOSMaxCpuPerUserLimit). This guard makes a too-big header fail in CI, not at
# 3am on the login node.
_QOS_MAX_CPUS = 8
_QOS_MAX_MEM_MB = 30800
_QOS_MAX_WALL_S = 4 * 3600


def _directive(flag: str) -> str | None:
    """Return the value of a `#SBATCH --flag=value` directive, or None."""
    for ln in _SBATCH.read_text().splitlines():
        m = re.match(rf"^#SBATCH\s+--{re.escape(flag)}=(.+?)\s*$", ln)
        if m:
            return m.group(1)
    return None


def _mem_to_mb(value: str) -> int:
    # Slurm --mem defaults to MB when unitless; G means GiB (x1024 MB).
    m = re.fullmatch(r"(\d+)([MG]?)", value.strip())
    assert m, f"unparseable --mem value: {value!r}"
    n, unit = int(m.group(1)), m.group(2)
    return n * (1024 if unit == "G" else 1)


def _wall_to_s(value: str) -> int:
    # Slurm HH:MM:SS (the form this template uses).
    h, mm, ss = (int(x) for x in value.split(":"))
    return h * 3600 + mm * 60 + ss


def test_sbatch_partition_is_cpu_not_boost():
    lines = [ln for ln in _SBATCH.read_text().splitlines() if ln.startswith("#SBATCH --partition=")]
    assert lines, "no #SBATCH --partition line in the template"
    partition = lines[0].split("=", 1)[1].strip()
    assert_cpu_partition(partition)  # raises if it is a boost_* / GPU partition


def test_sbatch_requests_no_gpu():
    # Check #SBATCH DIRECTIVE lines only — the explanatory comment legitimately
    # mentions "--gres=gpu" (as the thing NOT to request).
    gres = [
        ln
        for ln in _SBATCH.read_text().splitlines()
        if ln.startswith("#SBATCH") and "gres" in ln and "gpu" in ln
    ]
    assert not gres, f"template requests a GPU gres directive: {gres}"


def test_sbatch_invokes_the_batch_cli():
    assert "extract_region_batch.py" in _SBATCH.read_text()


def test_sbatch_fits_lrd_all_serial_qos():
    """The resource header must fit the budget-free lrd_all_serial QoS caps.

    Regression guard for the 32cpu/120G header that shipped 'green' (the old
    template never ran — it was rejected at submit with QOSMaxCpuPerUserLimit).
    This FAILS on that header and passes on the prague-validated 8cpu/30G/4h shape.
    """
    cpus = _directive("cpus-per-task")
    mem = _directive("mem")
    wall = _directive("time")
    assert cpus and mem and wall, f"missing directive(s): cpus={cpus} mem={mem} time={wall}"
    assert int(cpus) <= _QOS_MAX_CPUS, f"--cpus-per-task={cpus} exceeds QoS cap {_QOS_MAX_CPUS}"
    assert _mem_to_mb(mem) <= _QOS_MAX_MEM_MB, f"--mem={mem} exceeds QoS cap {_QOS_MAX_MEM_MB}M"
    assert _wall_to_s(wall) <= _QOS_MAX_WALL_S, f"--time={wall} exceeds QoS MaxWall 04:00:00"

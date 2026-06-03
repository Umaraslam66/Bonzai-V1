"""Guard the committed process-sbatch template against a copy-pasted GPU header
(the regression that would silently eat the training budget): its #SBATCH
partition must pass assert_cpu_partition, it must request no GPU, and it must
invoke the batch CLI."""

from __future__ import annotations

from pathlib import Path

from cfm.data.multiregion.partition import assert_cpu_partition

_REPO = Path(__file__).resolve().parents[3]
_SBATCH = _REPO / "scripts" / "multiregion_process.sbatch"


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

"""Task 12 end-to-end smoke (slow): proves the whole loop runs on a tiny budget
BEFORE the real run — build -> audit -> train >=1 step -> checkpoint -> reload
bit-identical -> generate + decode. Runs on CPU (devices=1); the real run uses
devices=4 on Leonardo (per feedback_leonardo_full_node)."""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_smoke_closes_the_loop():
    from scripts.train_scaffold import run_smoke

    result = run_smoke(devices=1)
    assert result["trained_steps"] >= 1
    assert result["checkpoint_written"] is True
    assert result["resumed_bit_identical"] is True  # on-disk ckpt == trained weights
    assert result["decoded_cells"] >= 0  # decode runs (may be 0 from an untrained model)

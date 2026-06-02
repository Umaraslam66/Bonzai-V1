"""Scaffold run config (pydantic). The experiment IS config + commit + snapshot.

Every field here is part of the reproducibility record dumped into ``reports/``.
The bake-off comparability lock (torch/lightning/pydantic versions) lives in
``cfm.training.env_lock`` and is asserted at the run entrypoint, not here.
"""

from __future__ import annotations

from pydantic import BaseModel


class ScaffoldConfig(BaseModel):
    # data snapshot
    release: str = "2026-04-15.0"
    region: str = "singapore"
    seed: int = 7

    # model (toy ~10-30M; the simplest block shared by bake-off candidates 1-3)
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    #: CELL token budget (sub-F P99.9 lock = 5760 tokens/cell). The model's positional
    #: capacity is this PLUS the conditioning prefix (n_cond) -- see lit_module.
    max_len: int = 5760

    # optimisation
    lr: float = 3e-4
    batch_size: int = 8
    max_steps: int = 2000

    # trainer / hardware
    precision: str = "bf16-mixed"
    #: "gpu" for real runs (Leonardo 4xA100); tests/CPU pass "cpu" so the Trainer is
    #: constructible where there is no GPU (login node + Mac dev both lack one).
    accelerator: str = "gpu"
    devices: int = 4
    compile: bool = True

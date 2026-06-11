"""Scaffold run config (pydantic). The experiment IS config + commit + snapshot.

Every field here is part of the reproducibility record dumped into ``reports/``.
The bake-off comparability lock (torch/lightning/pydantic versions) lives in
``cfm.training.env_lock`` and is asserted at the run entrypoint, not here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ScaffoldConfig(BaseModel):
    # data snapshot
    release: str = "2026-04-15.0"
    region: str = "singapore"
    seed: int = 7

    #: Which training corpus the datamodule consumes (F7). "single" = the per-region
    #: manifest for ``region`` (legacy path, builds on demand). "eu-train-union" = the
    #: UNION of per-city manifests for train_cities(G4 roll-up minus held-out cities);
    #: never rebuilds (the Task-8 sbatch builds them) and pins the multiregion
    #: holdout manifest + schema 2.0 directly. ``region`` is ignored for data
    #: selection on the union path (it still tags reports).
    train_set: Literal["single", "eu-train-union"] = "single"

    #: F16 generation-coherence tag: which conditioning-prefix scheme the model was trained
    #: under. "value-char-v1" (knob B, Task 24b) = value-bearing 9-id prefix + the continuous
    #: character position — ONE bump covering 24a+24b (version-fold: no checkpoint was blessed
    #: under the interim 24a layout, so the pre-24a "value" literal is retired with it).
    #: "slot" = the legacy constant field-slot block. A checkpoint loads ONLY under its own
    #: scheme - a mismatch is silent without this tag.
    conditioning_scheme: Literal["slot", "value-char-v1"] = "value-char-v1"

    #: Task 24a (spec §8): conditioning-ablation switch, applied IDENTICALLY to the
    #: train-side prefix build and the generation-side matched conditioning (the
    #: datamodule builds both). "no_city" = Lane S's scored-generalization instrument
    #: (city_identity slot forced to bucket 0); "no_character" is loud until Task 24b.
    conditioning_ablation: Literal["full", "no_city", "no_character"] = "full"

    # model
    #: Bake-off backbone: "transformer-ar" (today) | "mamba-hybrid" | "discrete-diffusion"
    #: (the latter two gated behind Task 5's mamba-ssm verify-before-lock).
    backbone: str = "transformer-ar"
    # toy ~10-30M; the simplest block shared by bake-off candidates 1-3
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    #: CELL token budget (sub-F P99.9 lock = 5760 tokens/cell). The model's positional
    #: capacity is this PLUS the conditioning prefix (CONDITIONING_PREFIX_LEN = 9 id
    #: positions + CHARACTER_PREFIX_POSITIONS = 1 continuous position since Task 24b)
    #: -- see models/backbone.py.
    max_len: int = 5760

    # optimisation
    lr: float = 3e-4
    batch_size: int = 8
    #: Gradient accumulation: holds the EFFECTIVE batch (batch_size * devices * grad_accum)
    #: constant across scales when per-GPU memory forces a smaller batch_size at 300M/1B
    #: (§10 comparability -- per-scale memory limits must not be an uncontrolled variable).
    grad_accum: int = 1
    max_steps: int = 2000

    # post-train eval generation (F15): these lengths are part of the EXPERIMENT
    # config (they land in hparams/checkpoints and the reports/ Config block via
    # model_dump), not run-invocation trivia — the emergence verdict's
    # commensurability check reads eval_max_new as the generated-length cap.
    #: cells to generate in the post-train eval (0 = generate nothing, no floor needed)
    eval_cells: int = 64
    #: tokens generated per cell — keep small at large model scales (AR generation cost)
    eval_max_new: int = 512

    # trainer / hardware
    precision: str = "bf16-mixed"
    #: "gpu" for real runs (Leonardo 4xA100); tests/CPU pass "cpu" so the Trainer is
    #: constructible where there is no GPU (login node + Mac dev both lack one).
    accelerator: str = "gpu"
    devices: int = 4
    compile: bool = True

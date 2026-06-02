"""Quarantined discrete-diffusion divergences (Phase-2 bake-off Task 8; §9).

The diffusion backbone shares the embedding / conditioning builder / sub-F head / eval
content with the AR backbones, and quarantines its three differences to this subpackage:
``mask`` (bidirectional absorbing-state masking, not causal), ``loss`` (denoising/ELBO,
not next-token CE), ``generate`` (T denoising passes, not autoregressive). Only ``mask``
is CPU-dependency-free and built now; ``loss``/``generate`` and the CUDA forward land
behind the Task-5 mamba-ssm verify-before-lock gate.
"""

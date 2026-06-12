# mamba-lock candidate pin — CPU half RESOLVES (2026-06-12, W4)

**Status: CANDIDATE, not locked.** The mamba-ssm verify-before-lock verdict is the
GPU half (forward/backward numerics on A100, post-renewal, on Umar's word). This
doc records what the CPU half proved and the preconditions the GPU half MUST
carry. Machine-produced evidence: `reports/2026-06-12-mamba-cpu-half-probe.json`
(Leonardo jobs 46014130 → 46019067 → 46024139 → 46030607; probe
`scripts/probe_mamba_cpu_half.sh`).

## The candidate pin

| component | version | note |
|---|---|---|
| torch | **2.5.1+cu121 — UNCHANGED** | comparability lock intact; NO re-lock-all event |
| triton | 3.1.0 | torch 2.5.1's own companion |
| causal-conv1d | 1.6.2.post1 | source-built (prebuilt wheel 404 for this combo) |
| mamba-ssm | 2.3.1 | resolver-backtracked from 2.3.2 under the torch constraint |

Proven on the CPU side: both packages source-build against the locked torch,
import cleanly, and a `Mamba(d_model=64)` module constructs (32,640 params).
`env_lock.py` extends with these pins ONLY at the GPU lock, not before.

## GPU-half PRECONDITIONS (not probe trivia — omit any one and it silently regresses)

1. **Build toolchain**: `module load gcc/12.2.0` with `CC`/`CXX`/`CUDAHOSTCXX`
   pointing at it. The default RHEL-8 GCC 8.5 hard-fails on torch 2.5.1 headers
   ("We need GCC 9 or later"); 12.2.0 is the newest module and the max host
   compiler CUDA 12.2 supports. (Run 46014130.)
2. **Install constraint**: a pip constraints file pinning `torch==2.5.1+cu121`
   and `triton==3.1.0` on EVERY install touching mamba packages. Without it pip
   SILENTLY upgraded torch to 2.12.0+cu130 to satisfy mamba-ssm 2.3.2 — which
   would have trained the mamba candidate under a DIFFERENT torch than the other
   two backbones, invalidating the bake-off comparison with no error raised.
   The probe's `torch_matches_lock` tooth caught it; the constraint converts the
   failure mode into resolver backtracking. (Run 46019067.)
3. **Import runtime**: the gcc-12 `libstdc++.so.6` (provides `GLIBCXX_3.4.29`)
   must be visible at import — `LD_PRELOAD` it (the probe's mechanism) or prepend
   its dir to `LD_LIBRARY_PATH` in every sbatch entry point that imports
   `mamba_ssm`/`causal_conv1d`. The spack python's library chain otherwise
   resolves the gcc-8.5 runtime and the import dies. (Run 46024139.)

## Parked (the GPU half, post-renewal, each on Umar's word)

Forward/backward kernel numerics on A100 = the verify-before-lock verdict;
`env_lock.py` pin extension at that lock; the mamba-hybrid backbone stays behind
`BackboneNotYetBuilt` until the verdict. The probe venv
(`/leonardo_work/AIFAC_P02_222/envs/mamba-cpu-probe`) stays in place — its pip
cache holds the built wheels, so the GPU half reuses them instead of recompiling.

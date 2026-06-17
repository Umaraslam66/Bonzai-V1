#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=mamba-gpu-half
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Step (b): mamba-ssm verify-before-lock GPU HALF — the actual verdict the W4 CPU
# half deferred. Forward/backward kernel NUMERICS on A100 under the EXACT locked
# stack (scripts/mamba_gpu_numerics.py). Reuses the W4 probe venv (built wheels:
# torch 2.5.1+cu121, triton 3.1.0, causal-conv1d 1.6.2.post1, mamba-ssm 2.3.1).
# Carries the THREE candidate-pin preconditions VERBATIM
# (reports/2026-06-12-mamba-candidate-pin.md). Cheap dbg job, 1 A100, account 548.

set -uo pipefail   # NOT -e: the python probe records every result; the JSON is the product

REPO=/leonardo_work/AIFAC_P02_222/Bonzai-OSM
ENVDIR=/leonardo_work/AIFAC_P02_222/envs/mamba-cpu-probe
REPORT=$REPO/reports/2026-06-17-mamba-gpu-half-probe.json

# --- Precondition 1: gcc/12.2.0 build toolchain (+ CC/CXX/CUDAHOSTCXX) -----------------
# The default RHEL-8 gcc 8.5 hard-fails on torch 2.5.1 headers; 12.2.0 is the newest
# module and the max host compiler CUDA 12.2 supports. (Candidate-pin precondition 1.)
module load python/3.11.7 cuda/12.2 gcc/12.2.0
export CC=$(command -v gcc) CXX=$(command -v g++) CUDAHOSTCXX=$(command -v g++)

# --- Reuse the W4 probe venv (NO install) ---------------------------------------------
# Precondition 2 (the constraints file pinning torch==2.5.1+cu121 + triton==3.1.0 on
# every install) is satisfied here by NOT INSTALLING: the venv already holds the built
# wheels, so pip never runs and torch cannot silently drift. The runtime tooth below
# re-checks torch==2.5.1+cu121 regardless.
if [ ! -d "$ENVDIR" ]; then echo "FATAL: probe venv $ENVDIR missing"; exit 66; fi
source "$ENVDIR/bin/activate"

# --- Precondition 3: gcc-12 libstdc++ (GLIBCXX_3.4.29) visible at import ---------------
# The .so files are gcc-12-compiled and need GLIBCXX_3.4.29; the spack python's chain
# otherwise resolves the gcc-8.5 runtime and the import dies. LD_PRELOAD it.
export LD_PRELOAD="$(dirname "$($CC -print-file-name=libstdc++.so.6)")/libstdc++.so.6"

cd "$REPO"
mkdir -p logs
echo "=== context ==="
echo "host=$(hostname) date=$(date -u +%FT%TZ) git_sha=$(git rev-parse HEAD)"
echo "toolchain: gcc=$($CC --version | head -1)"
echo "LD_PRELOAD=$LD_PRELOAD"
echo "venv=$VIRTUAL_ENV"
nvidia-smi -L

echo "=== mamba GPU numerics (fused vs reference, fwd+bwd) ==="
python scripts/mamba_gpu_numerics.py "$REPORT"
RC=$?
echo "PROBE_RC=$RC -> $REPORT"
exit $RC

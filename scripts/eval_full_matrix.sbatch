#!/bin/bash
# Standing eval — FULL 6-checkpoint matrix (spec 2026-06-23-standing-eval-harness).
# Build+verify the held-out cache ONCE, then eval all 6 reusing it, then aggregate. Single
# GPU; resumable (skips checkpoints whose JSON exists). Generous walltime (mamba gen is slow).
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=standing-eval-matrix
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7 cuda/12.2 gcc/12.2.0
export CC=$(command -v gcc) CXX=$(command -v g++) CUDAHOSTCXX=$(command -v g++)
export LD_PRELOAD="$(dirname "$($CC -print-file-name=libstdc++.so.6)")/libstdc++.so.6"
source .venv/bin/activate
mkdir -p logs reports/_standing_eval

echo "host=$(hostname) date=$(date -u +%FT%TZ) git_sha=$(git rev-parse HEAD)"
nvidia-smi -L

python -u scripts/eval_full_matrix.py --n-per-city 2000

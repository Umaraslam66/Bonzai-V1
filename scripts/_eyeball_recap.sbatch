#!/bin/bash
# Re-run ONLY the 2 capped dense cells at the full 13312 cap (cell-EOS completeness check
# in the dense regime). NON-scored. Single process, 1 GPU. Env mirrors mamba_rate_smoke.sbatch.
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=eyeball-recap
#SBATCH --time=00:25:00
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
mkdir -p logs

CKPT=/leonardo_work/AIFAC_P02_548/Bonzai-OSM/checkpoints/bakeoff/transformer-ar-53M/krakow-seed7/last.ckpt
JSON=reports/_eyeball_probe/gen_tokens.json

echo "host=$(hostname) date=$(date -u +%FT%TZ) git_sha=$(git rev-parse HEAD)"
nvidia-smi -L
python -u scripts/_eyeball_recap.py --ckpt "$CKPT" --json "$JSON" --max-new 13312

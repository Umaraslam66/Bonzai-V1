#!/bin/bash
# Standing methodology-eval — ONE checkpoint (spec 2026-06-23-standing-eval-harness).
# GPU for perplexity-gap (held-out NLL) + geometry-validity (generate probe); CPU for
# saturation. Same env as the eyeball probe. Pass CKPT (and optionally SMOKE=1) via --export.
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=standing-eval
#SBATCH --time=00:30:00
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

: "${CKPT:?set CKPT=/path/to/last.ckpt via --export}"
SMOKE_FLAG=""
[ "${SMOKE:-0}" = "1" ] && SMOKE_FLAG="--smoke"

echo "host=$(hostname) date=$(date -u +%FT%TZ) git_sha=$(git rev-parse HEAD) ckpt=$CKPT smoke=${SMOKE:-0}"
nvidia-smi -L

python -u scripts/eval_checkpoint.py --ckpt "$CKPT" $SMOKE_FLAG

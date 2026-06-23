#!/bin/bash
# Eyeball generation probe (PI-approved 2026-06-23) — NON-scored methodology validation.
# Loads ONE transformer-ar-53M checkpoint, generates ~21 cells across 3 hand-built density
# contexts, dumps tokens to JSON (decode+render happen locally). Single process, 1 GPU.
# NO scoring / floor / manifest / decide(). Env mirrors scripts/mamba_rate_smoke.sbatch.
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=eyeball-gen-probe
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
# mamba_ssm needs the gcc-12 libstdc++ at import; build_backbone's import chain may touch it.
export CC=$(command -v gcc) CXX=$(command -v g++) CUDAHOSTCXX=$(command -v g++)
export LD_PRELOAD="$(dirname "$($CC -print-file-name=libstdc++.so.6)")/libstdc++.so.6"
source .venv/bin/activate
mkdir -p logs reports/_eyeball_probe

CKPT=/leonardo_work/AIFAC_P02_548/Bonzai-OSM/checkpoints/bakeoff/transformer-ar-53M/krakow-seed7/last.ckpt
OUT=reports/_eyeball_probe/gen_tokens.json

echo "=== context ==="
echo "host=$(hostname) date=$(date -u +%FT%TZ) git_sha=$(git rev-parse HEAD)"
nvidia-smi -L
echo "ckpt=$CKPT"

python -u scripts/_eyeball_gen_probe.py \
  --ckpt "$CKPT" --out "$OUT" --max-new 1536 --cells-per-context 7

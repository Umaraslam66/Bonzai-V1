#!/bin/bash
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=cache-v2-cell-eos
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000M
#SBATCH --output=/leonardo_work/AIFAC_P02_222/cache_v2_%j.out
#SBATCH --error=/leonardo_work/AIFAC_P02_222/cache_v2_%j.err
set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7
source .venv/bin/activate
echo "JOB HEAD=$(git rev-parse HEAD)  derivation=$(grep 'SHARD_CACHE_DERIVATION_VERSION: str' src/cfm/data/training/shard_cache.py)"
SECONDS=0
python -u scripts/build_shard_cache.py --release 2026-04-15.0
echo "WALL_SECONDS=$SECONDS RC=0"

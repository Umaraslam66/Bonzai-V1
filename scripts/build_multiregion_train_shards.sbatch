#!/bin/bash
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=mr-train-shards
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000M
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Task 8 — build the multi-region TRAIN-city shards (CPU-only, cache-hit, no GPU).
# Driver loop over train_cities (validated minus the 4 held-out cities), writing a
# per-city training_manifest.yaml, then re-reading every manifest to verify end-state.
# NEVER boost_usr_prod: CPU work must not bill a 4xA100 node and eat the training
# budget (no --gres=gpu requested). Resources fit the lrd_all_serial QoS cap
# (cpu=8, mem<=30800M, node=1, MaxWall=04:00:00) verified on Leonardo 2026-06-04.
set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7
source .venv/bin/activate

echo "=== mr-train-shards context ==="
echo "host=$(hostname) sha=$(git rev-parse --short HEAD) $(date -u +%FT%TZ)"
echo "node=${SLURMD_NODENAME:-unknown} cpus=${SLURM_CPUS_PER_TASK:-unset} nproc=$(nproc)"

python -u scripts/build_multiregion_train_shards.py
echo "MR_TRAIN_SHARDS_RC=$?"

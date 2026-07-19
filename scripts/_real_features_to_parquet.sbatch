#!/bin/bash
# SCRATCH (uncommitted): convert the real-features YAML -> parquet + floor-reproduction teeth
# on the new format. CPU serial (lrd_all_serial); reads the 665MB YAML (~minutes) then rebuilds
# the 265 floor rows from the parquet and confirms exact match to the sha-locked floor.
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=real-features-parquet
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000M
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7
source .venv/bin/activate
echo "sha=$(git rev-parse --short HEAD) $(date -u +%FT%TZ) host=$(hostname)"
python -u scripts/_real_features_to_parquet.py
echo "PARQUET_RC=$?"

#!/bin/bash
# SCRATCH (uncommitted): dump real_by_city / real_train_by_city + VERIFY floor reproduction.
# CPU-only serial job (lrd_all_serial; bills CPU core-h, NOT GPU-h — ~5 core-h, trivial vs the
# 40,000 core-h grant). Reuses extract_features_by_city_stratum_metric, the SAME extractor that
# built the locked floor. Scale: 42 cities / 23,971 tiles (~40 min; gate-i did the 4 held-out in
# 3:45). Account 548 (222 compute expired 2026-06-11; the 222 FS tree still holds the corpus,
# read-only — lrd_all_serial AllowAccounts=ALL).
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=dump-real-features
#SBATCH --time=02:30:00
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
python -u scripts/_dump_real_features.py
echo "DUMP_RC=$?"

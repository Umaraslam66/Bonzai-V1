#!/bin/bash
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_222
#SBATCH --job-name=mr-process
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000M
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Per-city PROCESS job (sub_c..validate), CPU-only, cache-hit (no S3 egress).
# Fetch (stage 1a) runs on the LOGIN node BEFORE this is submitted (Phase-G G2:
# a load_region loop in tmux populates data/cache/overture/), so run_city's fetch
# cache-hits here. NEVER boost_usr_prod: CPU work must not bill a 4xA100 node and
# eat the training budget — extract_region_batch.py asserts this too, and no
# --gres=gpu is requested. Pass CITY via: sbatch --export=ALL,CITY=<region> ...
#
# Resources fit the lrd_all_serial QoS cap (verified on Leonardo 2026-06-04):
# per-user/per-job MaxTRES = cpu=8, mem=30800M, node=1, MaxWall=04:00:00. The
# original 32cpu/120G header was REJECTED (QOSMaxCpuPerUserLimit) — it never ran
# on this budget-free partition. The prague canary validated the 8cpu/30G/4h
# shape end-to-end (253 tiles, ~91min). test_sbatch_template.py guards these caps.
# Note the cpu=8 per-USER cap serialises a multi-city submit to one job at a time.
set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7
source .venv/bin/activate

echo "=== mr-process context ==="
echo "host=$(hostname) sha=$(git rev-parse --short HEAD) city=${CITY} $(date -u +%FT%TZ)"
echo "node=${SLURMD_NODENAME:-unknown} cpus=${SLURM_CPUS_PER_TASK:-unset} nproc=$(nproc)"

python -u scripts/extract_region_batch.py --cities "${CITY}" --partition lrd_all_serial
echo "MR_PROCESS_RC=$?"

#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=mr-proc-boost
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000M
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Batch-2 PROCESS on boost_usr_prod — PI-AUTHORIZED EXCEPTION (2026-06-04) to the
# CPU-only/no-boost rule, for batch-2 preprocessing ONLY. Pre-top-up; per-core
# billing CONFIRMED (Slurm billing=cpu: an 8-core job bills 8 local-h/h, NOT a
# whole 4xA100 node). The guard (assert_cpu_partition) stays default-DENY; this
# passes the explicit --authorized-boost-override flag so the CLI logs the
# exception (auditable). This is NEVER the default path — the canary/default
# process sbatch stays scripts/multiregion_process.sbatch (lrd_all_serial).
# Boost's 24h wall cap removes the lrd 4h-timeout risk, so boxes stay generous.
# No --gres=gpu requested: this is CPU work (we waste no GPU). If boost rejects a
# 0-GPU job, add `#SBATCH --gres=gpu:1` (billing stays =cpu=8).
set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7
source .venv/bin/activate

echo "=== mr-proc-boost context (PI-authorized boost exception, batch-2) ==="
echo "host=$(hostname) sha=$(git rev-parse --short HEAD) city=${CITY} $(date -u +%FT%TZ)"
echo "node=${SLURMD_NODENAME:-unknown} cpus=${SLURM_CPUS_PER_TASK:-unset} nproc=$(nproc)"

python -u scripts/extract_region_batch.py \
  --cities "${CITY}" --partition boost_usr_prod --authorized-boost-override
echo "MR_PROCESS_RC=$?"

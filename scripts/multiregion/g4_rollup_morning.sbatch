#!/bin/bash
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=g4-morning
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000M
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Armed with --dependency=afterany:<the 9 timeout jobs> so it runs ONCE they all
# reach a terminal state (success, fail, or 24h timeout). Read-only over data/:
# re-runs the G4 roll-up (firm validated total + coverage matrix + per-city table)
# and tees a dated morning report. lrd_all_serial (budget-free, fits in 1h).
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7
source .venv/bin/activate
echo "=== G4 morning roll-up $(date -u +%FT%TZ) sha=$(git rev-parse --short HEAD) ==="
python scripts/multiregion/build_g4_rollup.py 2>&1 | tee reports/2026-06-06-overnight-g4-rollup.txt
echo "G4_MORNING_RC=${PIPESTATUS[0]} $(date -u +%FT%TZ)"

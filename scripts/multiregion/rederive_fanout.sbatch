#!/bin/bash
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=mr-rederive
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
# 8G: the eindhoven gate's actual MaxRSS was ~0.6G, so 8G is generous. Sized so
# TWO concurrent tasks (2x8=16G) fit under the lrd_all_serial QOS mem cap (30800M)
# AND the cpu=8 cap (2x4) — else QOSMaxMemoryPerUser throttles to 1 lane, not 2.
#SBATCH --output=logs/rederive/%x-%A_%a.out
#SBATCH --error=logs/rederive/%x-%A_%a.err
#
# Phase-3 #19 corpus-wide re-derive WORKER — one city per array task.
# Submit as: sbatch --array=0-N%2 --export=ALL,WAVEFILE=<path> rederive_fanout.sbatch
# %2 respects the lrd_all_serial QOS cap (MaxJobsPU=2, cpu=8). Per-city lock
# (guarded_rederive, 03ba5b6) lets the two concurrent tasks hit different cities.
# Each task: guarded sub_f re-derive (--allow-content-change) -> sub_g validate
# (old verdict backed up) -> the projector reads ground truth from disk afterward.
set -uo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7
source .venv/bin/activate
export PYTHONPATH="src:${PYTHONPATH:-}"
R=2026-04-15.0
CITY=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$WAVEFILE")
if [ -z "$CITY" ]; then echo "no city at index $SLURM_ARRAY_TASK_ID in $WAVEFILE"; exit 2; fi
echo "=== REDERIVE $CITY START $(date -u +%FT%TZ) (sha=$(git rev-parse --short HEAD)) ==="

python -m scripts.multiregion.guarded_rederive --release $R --city "$CITY" --allow-content-change
echo "rederive_rc=$?"

# Fresh sub_g verdict (back up the stale one).
if [ -d "data/processed/sub_g/$R/$CITY" ]; then
  mv "data/processed/sub_g/$R/$CITY" "data/processed/sub_g/$R/${CITY}.bak_rd_$(date +%s)"
fi
python scripts/sub_g/validate_phase1_region.py --region "$CITY" --release $R \
  --sub-c-region-dir "data/processed/sub_c/$R/$CITY" \
  --sub-d-region-dir "data/processed/sub_d/$R/$CITY" \
  --sub-e-region-dir "data/processed/sub_e/$R/$CITY" \
  --sub-f-region-dir "data/processed/sub_f/$R/$CITY" \
  --output-dir "data/processed/sub_g/$R/$CITY"
echo "subg_rc=$?"
python scripts/multiregion/_gate_verdict.py "$CITY" || true
python scripts/multiregion/_gate_token_count.py "$CITY" || true
echo "=== REDERIVE $CITY DONE $(date -u +%FT%TZ) ==="

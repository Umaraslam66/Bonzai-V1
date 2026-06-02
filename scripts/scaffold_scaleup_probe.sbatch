#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=AIFAC_P02_222
#SBATCH --job-name=scaffold-scaleup-300m
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Bake-off de-risking probe: ONE run of the baseline decoder-only AR at the PRD's
# TOP scale (300M; PRD §6 ladder is 30M/100M/300M) to (1) confirm 300M runs on the
# node and (2) measure per-step node-h so the 4x3 bake-off can be sized against the
# PRD's ~375 node-h (1500 GPU-h) envelope. Wall-clock-bounded (max_time) so it always
# yields a cost number; reports per-step node-h + the loss/geometry trajectory.

set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7 cuda/12.2
source .venv/bin/activate
mkdir -p logs

echo "=== job context ==="
echo "host=$(hostname) date=$(date -u +%FT%TZ) git_sha=$(git rev-parse HEAD)"
echo "ntasks=$SLURM_NTASKS gpus=$(nvidia-smi -L | wc -l)"

echo "=== pre-build training manifest (single process) ==="
python -c "from cfm.data.training.build_shards import build_training_shards as b; s=b('2026-04-15.0','singapore'); print('built', len(s), 'shards')"

# 300M baseline AR (d_model=1024, 24 layers, 16 heads => ~311M params, verified).
# max_len 2048 caps the cell budget for memory headroom at 300M (p90=2409; full 5760
# is a follow-up if this fits comfortably). max_time leaves margin under the 30-min cap
# for the eval + report on rank 0. batch_size 2/GPU is conservative for 300M; raise if
# GPU memory allows on a clean run.
echo "=== 300M scale-up probe (4-GPU DDP, wall-bounded) ==="
srun --kill-on-bad-exit=1 python -u scripts/train_scaffold.py \
  --devices 4 --no-build --no-compile \
  --d-model 1024 --n-layers 24 --n-heads 16 \
  --max-len 2048 --batch-size 2 \
  --max-time "00:00:18:00"

echo "=== report ==="
cat reports/phase-1-training-scaffold/2026-04-15.0-singapore-loop-closed-scaleup-311M.md 2>/dev/null || echo "(report not found)"
echo "JOB_DONE"

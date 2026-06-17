#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=AIFAC_P02_548
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
# Fit bounded by max_steps=2000 (~5 min @ ~8 it/s) for a clean per-step cost number.
# Eval kept SMALL (8 cells x 256 tokens): autoregressive eval generation at 300M is
# per-token forwards and overran the first run (64x512 took >20 min on rank 0) -- the
# eval cost is itself a bake-off finding; the cost deliverable is TRAINING throughput.
# max_len 2048 caps the cell budget for memory headroom; batch 2/GPU conservative.
echo "=== 300M scale-up probe (4-GPU DDP) ==="
srun --kill-on-bad-exit=1 python -u scripts/train_scaffold.py \
  --devices 4 --no-build --no-compile \
  --d-model 1024 --n-layers 24 --n-heads 16 \
  --max-len 2048 --batch-size 2 --max-steps 2000 \
  --eval-cells 8 --eval-max-new 256

echo "=== report ==="
# F17 run-keyed report name: {release}-{region}-{backbone}-{params}M-seed{seed}-loop-closed.md
cat reports/phase-1-training-scaffold/2026-04-15.0-singapore-*-loop-closed.md 2>/dev/null || echo "(report not found)"
echo "JOB_DONE"

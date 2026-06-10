#!/bin/bash
# Phase-2 bake-off Task 4 -- the diagnostic (the HARD GATE before Task 5 / scored runs).
# Measures, at ~90M on real Singapore: (a) geometry-r = where train loss flattens (FREE
# from the CSV loss curve, no generation); (b) emergence = does generated output clear the
# real holdout density floor (1.96 polys/active-cell = 0.25 x measured 7.85); (c) stage-1
# truncation control = do generated streams contain building tokens at all; (d) per-cell
# EVAL node-h (priced explicitly via eval_seconds, NOT assumed free).
#
# RECORDED FLAG 1 (regime-transfer, §15, on the diagnostic's OWN output): the emergence
# FLOOR (1.96) is a real-holdout property -> scale-invariant, transfers to 1B cleanly. But
# geometry-r read HERE is MEASURED-AT-90M; applying it to size the 1B token budget is an
# ASSUMPTION, not a measured fact. A single 300M emergence check would bound the
# scale-dependence later (budget permitting) -- do NOT let the 90M r silently become the 1B r.
# RECORDED FLAG 2 (eval cost): the final 64-cell eval is timed (eval_seconds in the cost
# report); the emergence-vs-step TRAJECTORY is a PRICED follow-on (eval the saved
# step-checkpoints) sized by that measurement -- not 7 blind mid-training evals.
#
# Value-AGNOSTIC conditioning (existing datamodule) = Topic-3 STAGE 2 (train longer first).
# Only if buildings do NOT clear the floor by r~=40 do we build value-bearing (stage 3).
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=AIFAC_P02_222
#SBATCH --job-name=bakeoff-diagnostic-90m
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7 cuda/12.2
source .venv/bin/activate
mkdir -p logs
echo "=== job context ==="
echo "host=$(hostname) date=$(date -u +%FT%TZ) git_sha=$(git rev-parse HEAD)"
echo "ntasks=$SLURM_NTASKS gpus=$(nvidia-smi -L | wc -l)"

echo "=== pre-build training manifest (single process, no rank write-race) ==="
python -c "from cfm.data.training.build_shards import build_training_shards as b; b('2026-04-15.0','singapore'); print('manifest built')"

# ~90M (d_model 768 / 12L / 12H). tokens/step = batch 4 * 4 devices * 2048 = 32,768.
# --max-steps 110000 ~= r 40 (3.6B tokens / 90M). --no-compile (conservative; variable cell
# lengths). --emergence-floor 1.96 -> the §2 verdict. --ckpt-every-n-steps 25000 -> step
# checkpoints for the trajectory follow-on. --eval-max-new 2048 generous (truncation control).
srun --kill-on-bad-exit=1 python -u scripts/train_scaffold.py \
  --devices 4 --no-build --no-compile --backbone transformer-ar \
  --d-model 768 --n-layers 12 --n-heads 12 --max-len 2048 --batch-size 4 \
  --max-steps 110000 --ckpt-every-n-steps 25000 \
  --eval-cells 64 --eval-max-new 2048 --emergence-floor 1.96

echo "=== report ==="
cat reports/phase-1-training-scaffold/2026-04-15.0-singapore-loop-closed-scaleup-*.md 2>/dev/null | tail -60 || echo "(report not found)"
echo "JOB_DONE"

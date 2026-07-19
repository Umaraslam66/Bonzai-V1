#!/bin/bash
# SCRATCH (uncommitted, underscore-prefixed): Phase-2 pre-matrix LEARNING-RISK check.
#
# Question: does the LOCKED 53M mamba-hybrid (d512 / 24L / transformer_every=7 = 53,733,348,
# clean 1:7 = 21 mamba + 3 tf @ 8/16/24) LEARN city-like geometry on eu-train-union, or just
# flatten loss? transformer-ar is ALREADY retired by the 89M diagnostic (right_angle 0.9985);
# this is the mamba half — the one backbone with no informative geometry evidence.
#
# PRE-REGISTERED BAR (stated before the run; judged after):
#   right_angle_rate >= 0.50  -> learning      (>= 0.95 = strong; matches the reports' PoC bar)
#   right_angle_rate ~ 0 with n_polygons > 0   -> DEGENERATE = "trains but doesn't learn" -> HALT
#                                                  (mamba wiring/conditioning investigation, NOT a retry)
#   n_polygons ~ 0                              -> INCONCLUSIVE (didn't emit enough buildings to judge)
# This is the DEGENERACY TRIPWIRE, NOT full geometry validation: the building_area_m2 KS axis
# stays UNRETIRED for BOTH backbones until the scored eval (needs the real-reference dump).
#
# Budget: boost_qos_dbg (priority 80, 30-min wall, highly backfillable — normal/lprod estimate
# ~11h queue under current load). Fits the wall as ~18 min training (--max-time) + a small eval
# ~= 1.7 GPU-h total, squarely in the approved ~1-2 GPU-h target and well under the 10 GPU-h gate.
# NOT --scored-run: this is a diagnostic (max_len 2048, eval_max_new 1024), exactly like the 89M
# tf-ar diagnostic — not the 13,312 scored budget. --ckpt-every-n-steps gives a checkpoint as
# insurance: if the eval is killed at the 30-min wall, it can be re-evaluated without retraining.
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=mamba-geom-check
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

REGION=krakow                 # TAG only (held-out): eu-train-union EXCLUDES it from training;
RELEASE=2026-04-15.0          # the eval scores dm.val_cells (training-city val tiles), not krakow.
TRAIN_SET=eu-train-union
SHARD_CACHE=data/processed/training_cache

cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
# mamba_ssm / causal_conv1d need the gcc-12.2 libstdc++ (GLIBCXX_3.4.29) LD_PRELOAD or they
# ImportError (selective_scan_cuda was compiled with gcc-12). REQUIRED for mamba-hybrid — the
# tf-ar diagnostic template omits this; mamba does not tolerate its absence.
module load python/3.11.7 cuda/12.2 gcc/12.2.0
export CC=$(command -v gcc) CXX=$(command -v g++) CUDAHOSTCXX=$(command -v g++)
export LD_PRELOAD="$(dirname "$($CC -print-file-name=libstdc++.so.6)")/libstdc++.so.6"
source .venv/bin/activate
mkdir -p logs
echo "=== job context ==="
echo "host=$(hostname) date=$(date -u +%FT%TZ) git_sha=$(git rev-parse HEAD)"
echo "ntasks=$SLURM_NTASKS gpus=$(nvidia-smi -L | wc -l) region=$REGION release=$RELEASE backbone=mamba-hybrid"

# LOCKED 53M mamba-hybrid: d512/24L/n_heads8; transformer_every defaults to 7 in
# MambaHybridConfig (NOT a ScaffoldConfig field) -> 53,733,348, == bakeoff_scales.py["53M"].
# Confirm at runtime: the run prints n_params_M=53.7 (scale_label "54M").
# --max-steps 200000 is a NON-BINDING ceiling so --max-time (30 min) is the graceful stop;
# the ScaffoldConfig default max_steps=2000 would stop in minutes (too few for geometry).
# conditioning_scheme defaults to value-char-v1 + conditioning_ablation full (the locked recipe).
# eu-train-union NEVER rebuilds (--no-build); the union datamodule reads the Task-8 per-city
# manifests and runs the fail-closed holdout audit (now logs an explicit PASS line) at setup.
srun --kill-on-bad-exit=1 python -u scripts/train_scaffold.py \
  --devices 4 --no-build --no-compile --backbone mamba-hybrid \
  --region "$REGION" --release "$RELEASE" --train-set "$TRAIN_SET" --shard-cache "$SHARD_CACHE" \
  --d-model 512 --n-layers 24 --n-heads 8 --max-len 2048 --batch-size 4 \
  --max-steps 200000 --max-time 00:00:18:00 --ckpt-every-n-steps 2000 \
  --eval-cells 12 --eval-max-new 1024 --seed 7

echo "=== end-state verify (mamba report on disk) ==="
REPORT_GLOB="reports/phase-1-training-scaffold/${RELEASE}-${REGION}-mamba-hybrid-*-loop-closed.md"
if ! compgen -G "$REPORT_GLOB" > /dev/null; then
  echo "JOB_FAILED_ENDSTATE: no report matches $REPORT_GLOB"; exit 1
fi
cat $REPORT_GLOB | tail -55
echo "JOB_DONE"

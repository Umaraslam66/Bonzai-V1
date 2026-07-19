#!/bin/bash
# Mamba-hybrid GEN-RATE + full-config feasibility smoke (NON-scored). Resolves the last
# unknown before the scored matrix is authorized:
#   (1) the mamba gen-rate at the LOCKED 53.7M config (the 3.1%-of-grant matrix estimate is
#       TRANSFORMER-only; an early mixer/2048 smoke clocked mamba ~6.8x slower per token), and
#   (2) whether that config TRAINS + GENERATES at the full 13312-ctx at all — mamba geometry
#       was only ever validated at the mixer / 2048-ctx, never the full 53.7M/13312.
#
# HYPOTHESIS: mamba per-token gen-rate is materially slower than transformer's 0.0268 s/tok.
# At ~6.8x (~0.18 s/tok) the mamba half of the matrix (3 seeds x 5,705 cells x ~600 self-
# terminated tok) is ~524 GPU-h, pushing the COMBINED matrix from the 3.1% transformer-only
# figure toward ~12% of the 5,000 GPU-h grant. This smoke MEASURES the real rate so N + the
# matrix budget can be locked. EXPECTED PASS: builds, loss trends down, generates w/o OOM at
# 13312-ctx, emits token 260 (self-termination quality needs the trained model — deferred).
#
# CONFIG = the LOCKED bake-off "53M" mamba-hybrid (src/cfm/models/bakeoff_scales.py):
#   d_model 512 / 24L / 8H, transformer_every=7 (MambaHybridConfig DEFAULT — no CLI flag),
#   max_len 13312 (-> positional capacity 13312+9+1=13322), --no-compile.
# Specified with EXPLICIT FLAGS, not --config: the configs/experiments/*.yaml are untracked /
# undeployed (gated matrix YAMLs), so the experiment is fully self-described here. The rate-
# relevant architecture (d_state/d_conv/expand=16/4/2, dropout 0) are MambaHybridConfig
# defaults = identical to the scored config. batch_size 1 mirrors the scored run's peak memory.
#
# Mamba preconditions (verbatim from scripts/mamba_smoke.sbatch): gcc-12 + CC/CXX/CUDAHOSTCXX
# + LD_PRELOAD gcc-12 libstdc++ (mamba_ssm needs the gcc-12 libstdc++ at import).
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=mamba-rate-smoke
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7 cuda/12.2 gcc/12.2.0
export CC=$(command -v gcc) CXX=$(command -v g++) CUDAHOSTCXX=$(command -v g++)
export LD_PRELOAD="$(dirname "$($CC -print-file-name=libstdc++.so.6)")/libstdc++.so.6"
source .venv/bin/activate
mkdir -p logs reports

REGION=krakow            # held-out EU city; light calibration tag (train-set is the union)
RELEASE=2026-04-15.0
TRAIN_SET=eu-train-union # trains on the union (excludes held-out cities); the scored target
SHARD_CACHE=data/processed/training_cache  # the v2/260 rebuild (v1 backed up beside it)

echo "=== context ==="
echo "host=$(hostname) date=$(date -u +%FT%TZ) git_sha=$(git rev-parse HEAD)"
echo "toolchain: gcc=$($CC --version | head -1)  LD_PRELOAD=$LD_PRELOAD"
nvidia-smi -L

echo "=== mamba-hybrid 53.7M LOCKED-config smoke (train-trend + gen-rate) ==="
# --max-steps 150 (loss-trend only) bounded by --max-time 6min so the post-train EVAL is
# guaranteed to run (the eval is where gen_seconds_per_token is produced). --eval-cells 6
# x --eval-max-new 640 (~ the ~600-tok operating point; covers real p99~601) = a few
# thousand generated tokens, enough for a stable first per-token rate.
srun --kill-on-bad-exit=1 python -u scripts/train_scaffold.py \
  --devices 4 --no-build --no-compile --backbone mamba-hybrid \
  --region "$REGION" --release "$RELEASE" --train-set "$TRAIN_SET" --shard-cache "$SHARD_CACHE" \
  --d-model 512 --n-layers 24 --n-heads 8 --max-len 13312 \
  --batch-size 1 --grad-accum 1 \
  --max-steps 150 --max-time 00:00:06:00 \
  --eval-cells 6 --eval-max-new 640

echo "=== loop-closed report (gen_seconds_per_token + cost JSON live here) ==="
cat reports/phase-1-training-scaffold/${RELEASE}-${REGION}-mamba-hybrid-*-loop-closed.md | tail -90
echo "MAMBA_RATE_SMOKE_DONE"

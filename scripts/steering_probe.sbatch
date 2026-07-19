#!/bin/bash
# Generation-steering probe (SPEC 2026-07-17) — NON-scored methodology experiment.
# One node, 4 GPUs, 4 concurrent single-GPU workers (full-node discipline; never 1-GPU jobs).
#
# LAYOUT (simplest correct): the 400-item work list per checkpoint is partitioned into 4 shards
# (shard k/4). For each of the 3 checkpoints we launch 4 background python workers — worker k
# pinned to GPU k via CUDA_VISIBLE_DEVICES=k and generating shard k/4 — then `wait` for all 4
# before the next checkpoint. Every GPU is busy for the whole of each checkpoint; the only idle
# is the short per-checkpoint barrier. Budget: ~8 GPU-h billed (2 node-h), well under the
# 10-GPU-h small-spend line (spec §Budget).
#
# CHECKPOINTS (fill at submit time; the 6-run matrix final `last.ckpt`s on Leonardo):
#   transformer-ar seed7  -> $CKPT_TRANSFORMER_S7
#   transformer-ar seed13 -> $CKPT_TRANSFORMER_S13
#   mamba-hybrid   seed7  -> $CKPT_MAMBA_S7
# D4 version-dir convention (src/cfm/eval/standing/saturation.py _D4_LOOKUP): the ambiguous
# Lightning-logs runs map (mamba-hybrid,7)->version_25, (mamba-hybrid,13)->version_27; the
# transformer-ar seeds resolve by the bake-off signature. Those are the *training-log* dirs; the
# checkpoint artifacts themselves live under checkpoints/bakeoff/<dir>/krakow-seed<seed>/last.ckpt
# (the defaults below mirror scripts/_diag_perfield.sbatch). Override via env or sed if the paths
# move.
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=steering-probe
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7 cuda/12.2 gcc/12.2.0
# mamba_ssm needs the gcc-12 libstdc++ at import; build_backbone's import chain touches it.
export CC=$(command -v gcc) CXX=$(command -v g++) CUDAHOSTCXX=$(command -v g++)
export LD_PRELOAD="$(dirname "$($CC -print-file-name=libstdc++.so.6)")/libstdc++.so.6"
source .venv/bin/activate
mkdir -p logs reports/_steering

CKR=/leonardo_work/AIFAC_P02_548/Bonzai-OSM/checkpoints/bakeoff
CKPT_TRANSFORMER_S7=${CKPT_TRANSFORMER_S7:-$CKR/transformer-ar-53M/krakow-seed7/last.ckpt}
CKPT_TRANSFORMER_S13=${CKPT_TRANSFORMER_S13:-$CKR/transformer-ar-53M/krakow-seed13/last.ckpt}
CKPT_MAMBA_S7=${CKPT_MAMBA_S7:-$CKR/mamba-hybrid-54M/krakow-seed7/last.ckpt}

# (id, checkpoint) pairs. id -> output basename steering-<id>-shard<k>.json.
CKPTS=(
  "transformer-ar-seed7:$CKPT_TRANSFORMER_S7"
  "transformer-ar-seed13:$CKPT_TRANSFORMER_S13"
  "mamba-hybrid-seed7:$CKPT_MAMBA_S7"
)

echo "=== context ==="
echo "host=$(hostname) date=$(date -u +%FT%TZ) git_sha=$(git rev-parse HEAD)"
nvidia-smi -L

N_SHARDS=4
for pair in "${CKPTS[@]}"; do
  id=${pair%%:*}
  ckpt=${pair#*:}
  echo "=== checkpoint $id ($ckpt) ==="
  pids=()
  for k in $(seq 0 $((N_SHARDS - 1))); do
    out="reports/_steering/steering-${id}-shard${k}.json"
    CUDA_VISIBLE_DEVICES=$k python -u scripts/steering_probe_gen.py \
      --ckpt "$ckpt" \
      --out "$out" \
      --max-new 1536 \
      --shard "$k/$N_SHARDS" \
      --cache reports/_standing_eval/heldout_cache.json \
      > "logs/steering-${id}-shard${k}.out" 2>&1 &
    pids+=($!)
  done
  # Bare `wait` returns 0 even when a worker died — a silent failure would let the ALL_DONE
  # marker lie (no-marker-without-endstate-verification). Reap each PID and check its status,
  # then require every shard's artifact AND its in-log sentinel before moving on.
  fail=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "WORKER FAILED: $id shard $i (see logs/steering-${id}-shard${i}.out)" >&2
      fail=1
    fi
  done
  [ "$fail" -eq 0 ] || exit 1
  for k in $(seq 0 $((N_SHARDS - 1))); do
    out="reports/_steering/steering-${id}-shard${k}.json"
    [ -s "$out" ] || { echo "MISSING/EMPTY $out" >&2; exit 1; }
    grep -q "STEERING_PROBE_SHARD_DONE" "logs/steering-${id}-shard${k}.out" \
      || { echo "NO SENTINEL in shard $k log for $id" >&2; exit 1; }
  done
  echo "  [$id] all $N_SHARDS shards done (status + artifact + sentinel verified)"
done

echo "STEERING_PROBE_ALL_DONE"

#!/bin/bash
# Steering-probe REPLICATION (pre-registered addendum, reports/2026-07-19-steering-probe-result.md):
# independent disjoint seeds 3000..3159 (n=160 pairs), contrasts C1 (confirm), C4 (validity),
# C5 (the product-steering claim). Judged on THIS sample alone — never pooled with the main run.
# Layout identical to steering_probe.sbatch (4 GPU lanes, per-worker status+artifact+sentinel).
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=steering-repl
#SBATCH --time=02:00:00
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
export CC=$(command -v gcc) CXX=$(command -v g++) CUDAHOSTCXX=$(command -v g++)
export LD_PRELOAD="$(dirname "$($CC -print-file-name=libstdc++.so.6)")/libstdc++.so.6"
source .venv/bin/activate
mkdir -p logs reports/_steering_repl

CKR=/leonardo_work/AIFAC_P02_548/Bonzai-OSM/checkpoints/bakeoff
CKPTS=(
  "transformer-ar-seed7:$CKR/transformer-ar-53M/krakow-seed7/last.ckpt"
  "transformer-ar-seed13:$CKR/transformer-ar-53M/krakow-seed13/last.ckpt"
  "mamba-hybrid-seed7:$CKR/mamba-hybrid-54M/krakow-seed7/last.ckpt"
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
    out="reports/_steering_repl/steering-repl-${id}-shard${k}.json"
    CUDA_VISIBLE_DEVICES=$k python -u scripts/steering_probe_gen.py \
      --ckpt "$ckpt" \
      --out "$out" \
      --max-new 1536 \
      --shard "$k/$N_SHARDS" \
      --seed-start 3000 --n-seeds 160 --contrasts C1,C4,C5 \
      --cache reports/_standing_eval/heldout_cache.json \
      > "logs/steering-repl-${id}-shard${k}.out" 2>&1 &
    pids+=($!)
  done
  fail=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "WORKER FAILED: $id shard $i (see logs/steering-repl-${id}-shard${i}.out)" >&2
      fail=1
    fi
  done
  [ "$fail" -eq 0 ] || exit 1
  for k in $(seq 0 $((N_SHARDS - 1))); do
    out="reports/_steering_repl/steering-repl-${id}-shard${k}.json"
    [ -s "$out" ] || { echo "MISSING/EMPTY $out" >&2; exit 1; }
    grep -q "STEERING_PROBE_SHARD_DONE" "logs/steering-repl-${id}-shard${k}.out" \
      || { echo "NO SENTINEL in shard $k log for $id" >&2; exit 1; }
  done
  echo "  [$id] all $N_SHARDS shards done (status + artifact + sentinel verified)"
done

echo "STEERING_REPL_ALL_DONE"

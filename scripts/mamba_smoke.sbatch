#!/bin/bash
# Phase-2 bake-off Task 7 — the NON-scored mamba-hybrid GPU smoke (spec §3.5/§5). The
# pre-scored gate: mamba-hybrid must (1) TRAIN (finite grads, loss decreasing), (2) EVAL
# clean (generate->decode->score completes — the FIRST real check of the gcc/compile
# eval-path fix on the mamba backbone, COMPILED), and (3) be COMPILE-STABLE (the §5
# quantified verdict). This is NEVER folded into a scored run.
#
# Carries the THREE mamba preconditions VERBATIM (reports/2026-06-12-mamba-candidate-pin.md,
# probe_mamba_gpu_half.sh): gcc/12.2.0 + CC/CXX/CUDAHOSTCXX + LD_PRELOAD gcc-12 libstdc++.
#
# Submit with (SHARD_CACHE strongly recommended — a cold shard walk is ~40min and overruns
# the 30min dbg window; the sealed cache loads in ~7min):
#   REGION=singapore RELEASE=2026-04-15.0 SHARD_CACHE=<cache_root> sbatch scripts/mamba_smoke.sbatch
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=mamba-smoke
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
# --- the 3 mamba preconditions (spec §3.5/§4.1): gcc-12 toolchain + libstdc++ preload.
module load python/3.11.7 cuda/12.2 gcc/12.2.0
export CC=$(command -v gcc) CXX=$(command -v g++) CUDAHOSTCXX=$(command -v g++)
export LD_PRELOAD="$(dirname "$($CC -print-file-name=libstdc++.so.6)")/libstdc++.so.6"
source .venv/bin/activate
mkdir -p logs reports

# EU multiregion bake-off data: train on the eu-train-union (the sealed EU cache, fast),
# eval against a held-out EU city's floor. REGION=krakow (a held-out EU city, LIGHT enough
# to fit the 30min dbg window — NOT dense munich, whose core would burn the budget; the
# smoke is path-validation, not cost measurement). singapore was Phase-1 de-risking and is
# NOT in the corpus.
: "${REGION:=krakow}"
: "${RELEASE:=2026-04-15.0}"
: "${TRAIN_SET:=eu-train-union}"
: "${SHARD_CACHE:=data/processed/training_cache}"
SHARD_CACHE_ARG=""
[ -n "${SHARD_CACHE:-}" ] && SHARD_CACHE_ARG="--shard-cache ${SHARD_CACHE}"

echo "=== context ==="
echo "host=$(hostname) date=$(date -u +%FT%TZ) git_sha=$(git rev-parse HEAD) region=$REGION release=$RELEASE train_set=$TRAIN_SET shard_cache=${SHARD_CACHE:-<none>}"
echo "toolchain: gcc=$($CC --version | head -1)  LD_PRELOAD=$LD_PRELOAD"
nvidia-smi -L

# eu-train-union NEVER rebuilds — the per-city manifests are built by the dedicated Task-8
# sbatch and the union path reads them (a missing one fails loud in setup). Only the
# single-region, no-cache path pre-builds the manifest once (single process, no DDP write
# race). The smoke's default (eu-train-union + cache) skips this entirely.
if [ "$TRAIN_SET" = "single" ] && [ -z "${SHARD_CACHE:-}" ]; then
  echo "=== pre-build manifest (single-region, no shard cache) ==="
  python -c "from cfm.data.training.build_shards import build_training_shards as b; b('${RELEASE}','${REGION}'); print('manifest built')"
fi

# --- Teeth 1+2: trains (finite grads, loss decreasing) + evals (generate->decode->score
# on the COMPILED mamba backbone — proves the gcc/compile eval-path on mamba). NON-scored:
# small budget, the smallest (30M) param-matched mamba shape (d512 / 12L / 8H), COMPILED
# (NO --no-compile). max_len 2048 keeps the smoke fast; this is a buildability/runnability
# gate, not a budget run.
echo "=== smoke train+eval (mamba-hybrid, compiled, 30M, short) ==="
srun --kill-on-bad-exit=1 python -u scripts/train_scaffold.py \
  --devices 4 --no-build --backbone mamba-hybrid \
  --region "$REGION" --release "$RELEASE" --train-set "$TRAIN_SET" $SHARD_CACHE_ARG \
  --d-model 512 --n-layers 12 --n-heads 8 --max-len 2048 --batch-size 4 \
  --max-steps 100 --eval-cells 4 --eval-max-new 64

# --- Tooth 3: the §5 quantified compile-stability verdict over REAL variable-length
# batches. Single-GPU (shape-driven recompiles are per-process — representative of the DDP
# scored regime). Emits the compile-on/off verdict JSON.
echo "=== compile-stability probe (spec §5) ==="
CUDA_VISIBLE_DEVICES=0 python scripts/measure_compile_stability.py \
  --region "$REGION" --release "$RELEASE" --train-set "$TRAIN_SET" $SHARD_CACHE_ARG \
  --steps 250 --batch-size 4 --scale 30M \
  --out reports/2026-06-17-mamba-smoke-compile.json

echo "SMOKE_DONE"

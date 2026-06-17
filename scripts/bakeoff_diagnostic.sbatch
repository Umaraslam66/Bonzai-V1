#!/bin/bash
# Phase-2 bake-off Task 4 -- the diagnostic (the HARD GATE before Task 5 / scored runs).
# Measures, at ~90M on the real target region: (a) geometry-r = where train loss flattens (FREE
# from the CSV loss curve, no generation); (b) emergence = does generated output clear the
# region's real holdout density floor (resolved per-region from configs/eval/
# emergence_floors.yaml by --region; frac 0.25 x measured holdout density); (c) stage-1
# truncation control = do generated streams contain building tokens at all; (d) per-cell
# EVAL node-h (priced explicitly via eval_seconds, NOT assumed free).
#
# RECORDED FLAG 1 (regime-transfer, §15, on the diagnostic's OWN output): the emergence
# FLOOR is a real-holdout property -> scale-invariant, transfers to 1B cleanly. But
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
#SBATCH --account=AIFAC_P02_548
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

# Region/release are env-driven (no hardcoded region; F1/F7). A job without them dies here.
# Submit with: REGION=<region> RELEASE=<release> sbatch scripts/bakeoff_diagnostic.sbatch
: "${REGION:?set REGION (e.g. krakow)}"
: "${RELEASE:?set RELEASE (e.g. 2026-04-15.0)}"
# EU multiregion diagnostic: geometry-r is training-data-dependent, so the diagnostic TRAINS
# on the eu-train-union (what the scored runs train on) and evals against REGION's held-out
# emergence floor (REGION=krakow). The sealed EU cache makes the union load fast; the per-
# token eval cost it measures is region-agnostic, so krakow is a fine floor/calibration tag.
: "${TRAIN_SET:=eu-train-union}"
: "${SHARD_CACHE:=data/processed/training_cache}"
SHARD_CACHE_ARG=""
[ -n "${SHARD_CACHE:-}" ] && SHARD_CACHE_ARG="--shard-cache ${SHARD_CACHE}"

cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
# gcc/12.2.0 is PRECAUTIONARY (toolchain consistency only): this job runs --no-compile,
# so it cannot hit the torch.inductor crash that needs gcc. No CC/CXX/LD_PRELOAD exports
# here — this job needs neither mamba nor compile.
module load python/3.11.7 cuda/12.2 gcc/12.2.0
source .venv/bin/activate
mkdir -p logs
echo "=== job context ==="
echo "host=$(hostname) date=$(date -u +%FT%TZ) git_sha=$(git rev-parse HEAD)"
echo "ntasks=$SLURM_NTASKS gpus=$(nvidia-smi -L | wc -l) region=$REGION release=$RELEASE"

# --- F8 verified end-state: JOB_DONE is a TRUSTED marker, so it must follow PROOF the
# run's artifacts exist on disk, never mere control-flow reaching the last line. This
# job's backbone is hardcoded transformer-ar (see the srun below); the checkpoint scale
# label is DERIVED from the real param count and the seed lives in config, so globs.
# compgen -G returns nonzero on no match (inside `if` so set -e doesn't fire early).
verify_endstate() {
  local ckpt_glob="${WORK:-checkpoints}/Bonzai-OSM/checkpoints/bakeoff/transformer-ar-*/${REGION}-seed*/last.ckpt"
  if ! compgen -G "$ckpt_glob" > /dev/null; then
    echo "JOB_FAILED_ENDSTATE: no checkpoint matches glob: $ckpt_glob"
    exit 1
  fi
  local report_glob="reports/phase-1-training-scaffold/${RELEASE}-${REGION}-transformer-ar-*-loop-closed.md"
  if ! compgen -G "$report_glob" > /dev/null; then
    echo "JOB_FAILED_ENDSTATE: no report matches glob: $report_glob"
    exit 1
  fi
  echo "end-state verified: checkpoint + report artifacts on disk"
}

# eu-train-union NEVER rebuilds (the per-city manifests are built by the Task-8 sbatch; the
# union path reads them, fail-loud if missing). Only the single-region path pre-builds the
# manifest once (single process, no DDP write-race). REGION=krakow is HELD-OUT, so a union
# diagnostic must NOT pre-build krakow's shards — it trains the union, not krakow.
if [ "$TRAIN_SET" = "single" ]; then
  echo "=== pre-build training manifest (single-region, no rank write-race) ==="
  python -c "from cfm.data.training.build_shards import build_training_shards as b; b('${RELEASE}','${REGION}'); print('manifest built')"
fi

# ~90M (d_model 768 / 12L / 12H). tokens/step = batch 4 * 4 devices * 2048 = 32,768.
# --max-steps 110000 ~= r 40 (3.6B tokens / 90M). --no-compile (conservative; variable cell
# lengths). The §2 emergence floor resolves from configs/eval/emergence_floors.yaml by
# --region (the run FAILS at config time if the region has no derived floor entry —
# run scripts/measure_emergence_floor.py first). --ckpt-every-n-steps 25000 -> step
# checkpoints for the trajectory follow-on. --eval-max-new 2048 generous (truncation control).
#
# F15 COMMENSURABILITY NOTE (readiness Task 14): --eval-max-new 2048 < the floor's
# locked-budget derivation regime (13,312 since the 2026-06-11 decision; was 5760), so
# a run of THIS script renders emergence_verdict=INCOMMENSURATE by
# construction. That is HONEST, not a bug. Positional capacity = max_len +
# CONDITIONING_PREFIX_LEN + CHARACTER_PREFIX_POSITIONS = 2048 + 9 + 1 = 2058 POSITIONS
# since Task 24b (n_cond=576 is embedding-table rows on the token-id axis, NOT
# positions). --eval-max-new 2048 yields (9 ids + 1 character placeholder) + 2048 =
# 2058 tokens — exactly fills capacity, zero headroom, fits by construction (boundary pinned
# in tests/models/test_micro_ar.py); bumping it toward the budget would CRASH generation, not
# fix the verdict. Making the verdict scoreable requires max_len >= the floor regime —
# a re-planned model budget, i.e. a Phase-6 / PI decision; do NOT silently edit these
# numbers.
# --max-time bounds TRAINING so fit stops GRACEFULLY before the 8h Slurm wall, leaving room
# for the post-train eval to RUN — the eval is where gen_seconds_per_token (the §6 budget
# input) and the emergence verdict are produced. Without it, an 8h-wall SIGTERM mid-training
# would skip the eval (no per-token cost) AND fail verify_endstate (no report). r is read
# from the loss CSV up to the stop, so a graceful stop short of --max-steps still yields r
# (or, if loss hasn't flattened, the FIXED_SCALE_PLUS_S13 ladder-collapse verdict). 7h30m
# train + ~20min eval (transformer-ar 90M, 64x2048 batch-1) fits the 8h wall.
srun --kill-on-bad-exit=1 python -u scripts/train_scaffold.py \
  --devices 4 --no-build --no-compile --backbone transformer-ar \
  --region "$REGION" --release "$RELEASE" --train-set "$TRAIN_SET" $SHARD_CACHE_ARG \
  --d-model 768 --n-layers 12 --n-heads 12 --max-len 2048 --batch-size 4 \
  --max-steps 110000 --ckpt-every-n-steps 25000 --max-time 00:07:30:00 \
  --eval-cells 64 --eval-max-new 2048

verify_endstate

echo "=== report ==="
# F17 run-keyed report name: {release}-{region}-{backbone}-{params}M-seed{seed}-loop-closed.md
# verify_endstate just proved the report exists; a cat failure here is LOUD by design
# (set -euo pipefail) -- a missing report must fail the job, not print a shrug (F8).
cat "reports/phase-1-training-scaffold/${RELEASE}-${REGION}"-transformer-ar-*-loop-closed.md | tail -60
echo "JOB_DONE"

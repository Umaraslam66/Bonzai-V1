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

# Region/release are env-driven (no hardcoded region; F1/F7). A job without them dies here.
# Submit with: REGION=<region> RELEASE=<release> sbatch scripts/bakeoff_diagnostic.sbatch
: "${REGION:?set REGION (e.g. krakow)}"
: "${RELEASE:?set RELEASE (e.g. 2026-04-15.0)}"

cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7 cuda/12.2
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

echo "=== pre-build training manifest (single process, no rank write-race) ==="
python -c "from cfm.data.training.build_shards import build_training_shards as b; b('${RELEASE}','${REGION}'); print('manifest built')"

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
srun --kill-on-bad-exit=1 python -u scripts/train_scaffold.py \
  --devices 4 --no-build --no-compile --backbone transformer-ar \
  --region "$REGION" --release "$RELEASE" \
  --d-model 768 --n-layers 12 --n-heads 12 --max-len 2048 --batch-size 4 \
  --max-steps 110000 --ckpt-every-n-steps 25000 \
  --eval-cells 64 --eval-max-new 2048

verify_endstate

echo "=== report ==="
# F17 run-keyed report name: {release}-{region}-{backbone}-{params}M-seed{seed}-loop-closed.md
# verify_endstate just proved the report exists; a cat failure here is LOUD by design
# (set -euo pipefail) -- a missing report must fail the job, not print a shrug (F8).
cat "reports/phase-1-training-scaffold/${RELEASE}-${REGION}"-transformer-ar-*-loop-closed.md | tail -60
echo "JOB_DONE"

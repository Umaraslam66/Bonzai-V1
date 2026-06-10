#!/bin/bash
# Phase-2 bake-off run with across-job-boundary resume (Task 10 / §10).
# A long 1B run spans multiple jobs: on approaching the wall-clock limit Slurm sends
# USR1, the trap resubmits THIS script, and the relaunched job resumes from the latest
# checkpoint on $WORK (cfm.training.resume.resume_ckpt_path). Checkpoints live on $WORK
# so they survive allocation expiry / the renewal gap -- "resume when the new allocation
# lands", never "start over".
#
# HARD ORDERING (plan Task 12 Step 2b): prove this relaunch path on a SHORT---time 300M
# job (kill -> resubmit -> continues from $WORK's last.ckpt, not step 0) BEFORE submitting
# any 1B job. Discovering a broken resume mid-1B-run costs ~159 node-h.
#
# Submit with:
#   BACKBONE=<name> SCALE=<30M|100M|300M|1B> REGION=<region> RELEASE=<release> \
#     [TRAIN_SET=single|eu-train-union] sbatch scripts/bakeoff_run.sbatch
# TRAIN_SET defaults to single (per-region corpus). eu-train-union consumes the
# Task-8 per-city manifests (built by build_multiregion_train_shards.sbatch) and
# NEVER rebuilds them -- the preamble only asserts they exist.
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=AIFAC_P02_222
#SBATCH --job-name=bakeoff-run
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@120
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

: "${BACKBONE:?set BACKBONE}" "${SCALE:?set SCALE}"
: "${REGION:?set REGION (e.g. krakow)}"
: "${RELEASE:?set RELEASE (e.g. 2026-04-15.0)}"
: "${TRAIN_SET:=single}"
echo "=== job context ==="
echo "host=$(hostname) date=$(date -u +%FT%TZ) git_sha=$(git rev-parse HEAD) backbone=$BACKBONE scale=$SCALE region=$REGION release=$RELEASE train_set=$TRAIN_SET"

# On wall-clock signal: resubmit THIS script (continues from $WORK on relaunch), then exit cleanly.
trap 'echo "USR1: wall-clock approaching; resubmitting for across-job resume"; sbatch "$0"; exit 0' USR1

if [ "$TRAIN_SET" = "single" ]; then
  # Pre-build the manifest ONCE (single process) before the DDP ranks read it (no write race).
  python -c "from cfm.data.training.build_shards import build_training_shards as b; b('${RELEASE}','${REGION}')"
else
  # eu-train-union: the per-city manifests are built ONLY by the dedicated Task-8 sbatch
  # (scripts/build_multiregion_train_shards.sbatch). NO rebuild here -- verify_union_manifests
  # is the tested fail-loud gate (strict held_out_cities read; RAISES listing every missing
  # manifest) and shares the one-source DEFAULT_G4_ROLLUP constant with the training process,
  # so this preamble verifies the SAME city set training consumes.
  python -c "from cfm.data.training.build_shards import DEFAULT_G4_ROLLUP, verify_union_manifests; from cfm.eval.holdout.paths import multiregion_holdout_manifest_path; cities = verify_union_manifests('${RELEASE}', g4_rollup=DEFAULT_G4_ROLLUP, holdout_manifest=multiregion_holdout_manifest_path('${RELEASE}')); print(f'eu-train-union preamble: {len(cities)} per-city manifests verified on disk')"
fi

# TODO(Task 12): --config is NOT wired yet (no per-run yaml loader on the CLI; see the
# xfail in tests/training/test_cli_contract.py). Task 12 either adds a --config loader OR
# expands this to individual --d-model/--n-layers/... flags like bakeoff_diagnostic.sbatch.
# The per-run config (configs/experiments/bakeoff-$BACKBONE-$SCALE.yaml) carries r-derived
# max_steps, the locked recipe, grad_accum, and the $WORK checkpoint dir; run_short resumes
# from resume_ckpt_path() and checkpoints to work_checkpoint_dir(BACKBONE, SCALE).
srun --kill-on-bad-exit=1 python -u scripts/train_scaffold.py \
  --devices 4 --no-build --backbone "$BACKBONE" \
  --region "$REGION" --release "$RELEASE" --train-set "$TRAIN_SET" \
  --config "configs/experiments/bakeoff-${BACKBONE}-${SCALE}.yaml" &
wait
echo "JOB_DONE"

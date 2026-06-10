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

# --- F8 verified end-state: JOB_DONE is a TRUSTED marker (future sessions and the
# orchestration believe it), so it must follow PROOF the run's artifacts exist on disk,
# never mere control-flow reaching the last line. Globs, not exact paths, PERMANENTLY:
# the checkpoint scale label is DERIVED from the real param count (e.g. "311M", not
# nominal $SCALE) and the seed lives in the YAML config -- neither is knowable to bash
# even after Task 26 of the readiness-closure plan (= the bake-off plan's Task-12 TODO
# below) wires --config, since --config hands the YAML to python; bash never parses it.
# compgen -G returns nonzero on no match (used inside `if` so set -e doesn't fire early).
verify_endstate() {
  local ckpt_glob="${WORK:-checkpoints}/Bonzai-OSM/checkpoints/bakeoff/${BACKBONE}-*/${REGION}-seed*/last.ckpt"
  if ! compgen -G "$ckpt_glob" > /dev/null; then
    echo "JOB_FAILED_ENDSTATE: no checkpoint matches glob: $ckpt_glob"
    exit 1
  fi
  local report_glob="reports/phase-1-training-scaffold/${RELEASE}-${REGION}-${BACKBONE}-*-loop-closed.md"
  if ! compgen -G "$report_glob" > /dev/null; then
    echo "JOB_FAILED_ENDSTATE: no report matches glob: $report_glob"
    exit 1
  fi
  echo "end-state verified: checkpoint + report artifacts on disk"
}

# On wall-clock USR1 (Slurm --signal=B:USR1@120 delivers ONLY to this batch shell):
# (1) FORWARD the signal to the srun ranks so they checkpoint + exit cleanly (otherwise
#     Slurm SIGKILLs them at the wall and the last checkpoint window is lost);
# (2) resubmit VERIFIED -- sbatch --parsable emits a bare job id (no "Submitted batch
#     job NNNN" prose) and the check demands a NUMERIC id (verify the KIND of yes, not
#     just any output); a non-numeric/empty result means the resume chain is BROKEN and
#     must fail loudly (exit 1), never exit 0 (this project's false-completion class).
#     Even on that failure, grace-wait for the ranks first: the in-flight last.ckpt
#     flush is exactly what the operator's manual resubmit will resume from;
# (3) exit 0 WITHOUT requiring end-state: the run is mid-flight by definition; the
#     relaunched job continues from $WORK's last.ckpt.
on_usr1() {
  echo "USR1: wall-clock approaching; forwarding to srun + resubmitting for across-job resume"
  if [[ -n "${SRUN_PID:-}" ]]; then kill -USR1 "$SRUN_PID" 2>/dev/null || true; fi
  JID=$(sbatch --parsable --export=ALL "$0") || JID=""
  if [[ "$JID" =~ ^[0-9]+ ]]; then
    echo "resubmitted for across-job resume: $JID"
  else
    echo "JOB_FAILED_RESUBMIT: sbatch returned no numeric job id; resume chain broken"
    [[ -n "${SRUN_PID:-}" ]] && wait "$SRUN_PID" || true
    exit 1
  fi
  # Grace window (@120): let the ranks finish flushing last.ckpt before the job ends.
  if [[ -n "${SRUN_PID:-}" ]]; then wait "$SRUN_PID" || true; fi
  exit 0
}
trap on_usr1 USR1

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
# from resume_ckpt_path() and checkpoints to work_checkpoint_dir(backbone, param-derived
# scale label, region, seed). NOTE: the scale label is derived from the param count, NOT
# from the nominal $SCALE env (e.g. "311M" vs nominal "300M") — resume stability comes
# from the label being deterministic from cfg, not from matching $SCALE.
srun --kill-on-bad-exit=1 python -u scripts/train_scaffold.py \
  --devices 4 --no-build --backbone "$BACKBONE" \
  --region "$REGION" --release "$RELEASE" --train-set "$TRAIN_SET" \
  --config "configs/experiments/bakeoff-${BACKBONE}-${SCALE}.yaml" &
SRUN_PID=$!
wait "$SRUN_PID"

verify_endstate
echo "JOB_DONE"

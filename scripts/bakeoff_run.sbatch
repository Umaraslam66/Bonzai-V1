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
#SBATCH --account=AIFAC_P02_548
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
# gcc/12.2.0 toolchain (spec §4.1): the RHEL-8 gcc-8.5 crashed torch.compile inductor CPU
# codegen in eval at Step 18.5; 12.2.0 is the newest module CUDA 12.2 supports. The same
# load satisfies the mamba run's gcc-12 libstdc++ (GLIBCXX_3.4.29) precondition #3 — the
# LD_PRELOAD is REQUIRED for the mamba_ssm/causal_conv1d import and HARMLESS for
# transformer-ar, so it rides this shared entry unconditionally (mirrors the GPU-verified
# scripts/probe_mamba_gpu_half.sh). Verified by `sbatch --test-only` on AIFAC_P02_548.
module load python/3.11.7 cuda/12.2 gcc/12.2.0
export CC=$(command -v gcc) CXX=$(command -v g++) CUDAHOSTCXX=$(command -v g++)
export LD_PRELOAD="$(dirname "$($CC -print-file-name=libstdc++.so.6)")/libstdc++.so.6"
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

# --config IS wired (readiness Task 26; the former Task-12 xfail in
# tests/training/test_cli_contract.py flipped to a positive guard). The per-run config
# (configs/experiments/bakeoff-$BACKBONE-$SCALE.yaml) carries r-derived max_steps, the
# locked recipe and grad_accum; explicit CLI flags below override its values; run_short
# resumes from resume_ckpt_path() and checkpoints to work_checkpoint_dir(backbone,
# param-derived scale label, region, seed). NOTE: the scale label is derived from the
# param count, NOT from the nominal $SCALE env (e.g. "311M" vs nominal "300M") — resume
# stability comes from the label being deterministic from cfg, not from matching $SCALE.
RUN_CONFIG="configs/experiments/bakeoff-${BACKBONE}-${SCALE}.yaml"

# Buildability gate (readiness A-3 / Task 26): prove the per-run YAML parses AND the
# backbone BUILDS, on CPU, in this compute-node preamble (it runs INSIDE the
# allocation, before srun — what it saves is the srun/DDP spin-up and a doomed
# multi-rank launch, not the allocation itself). An unbuildable config (typo'd field,
# missing mamba-ssm kernels, bad shape) fails HERE, never minutes into a 4-rank
# launch. The YAML goes through the CLI's OWN fail-closed loader,
# _load_config_yaml (Task-26 spec review #2): bare ScaffoldConfig(**yaml) silently
# IGNORES unknown kwargs, so a typo'd recipe key would survive the preamble and train
# the wrong experiment; the loader refuses unknown AND topology-owned keys — the
# preamble enforces literally the SAME rules the srun's --config path will.
#
# The dry-run builds the TRAINED backbone (Task-26 quality review #1): the srun below
# passes --backbone "$BACKBONE", an explicit CLI flag that OVERRIDES the YAML — so the
# gate applies the SAME override. Without it, a YAML omitting `backbone` would gate
# the DEFAULT transformer while the job trains mamba: the exact missing-kernels regime
# this gate exists to catch. And a YAML naming a DIFFERENT backbone than $BACKBONE is
# a config/submission mismatch — CONFIG_BACKBONE_MISMATCH refuses loudly; the override
# is for the omitted case only, never a silent stomp of a contradicting YAML.
python -c "
from scripts.train_scaffold import _load_config_yaml
from cfm.models.backbone import build_backbone
from cfm.training.config import ScaffoldConfig
loaded = _load_config_yaml('${RUN_CONFIG}')
declared = loaded.get('backbone')
if declared is not None and declared != '${BACKBONE}':
    raise SystemExit(
        'CONFIG_BACKBONE_MISMATCH: ${RUN_CONFIG} names backbone ' + repr(declared)
        + ' but the job was submitted with BACKBONE=${BACKBONE} — a config/submission'
        + ' mismatch is a refusal, never a silent stomp; fix the YAML or the submit env.'
    )
cfg = ScaffoldConfig(**{**loaded, 'backbone': '${BACKBONE}'})
build_backbone(cfg.backbone, cfg)
print(f'buildability dry-run OK: {cfg.backbone} d_model={cfg.d_model} n_layers={cfg.n_layers}')
"

# --scored-run: this entry point feeds the bake-off decision, so the W1
# commensurability gate (max_len == locked budget, eval_max_new >= it) fires at
# the entrypoint — an opt-down recipe dies named, before any GPU-hour burns.
srun --kill-on-bad-exit=1 python -u scripts/train_scaffold.py \
  --devices 4 --no-build --scored-run --backbone "$BACKBONE" \
  --region "$REGION" --release "$RELEASE" --train-set "$TRAIN_SET" \
  --config "$RUN_CONFIG" &
SRUN_PID=$!
wait "$SRUN_PID"

verify_endstate
echo "JOB_DONE"

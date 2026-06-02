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
# Submit with: BACKBONE=<name> SCALE=<30M|100M|300M|1B> sbatch scripts/bakeoff_run.sbatch
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
echo "=== job context ==="
echo "host=$(hostname) date=$(date -u +%FT%TZ) git_sha=$(git rev-parse HEAD) backbone=$BACKBONE scale=$SCALE"

# On wall-clock signal: resubmit THIS script (continues from $WORK on relaunch), then exit cleanly.
trap 'echo "USR1: wall-clock approaching; resubmitting for across-job resume"; sbatch "$0"; exit 0' USR1

# Pre-build the manifest ONCE (single process) before the DDP ranks read it (no write race).
python -c "from cfm.data.training.build_shards import build_training_shards as b; b('2026-04-15.0','singapore')"

# The per-run config (configs/experiments/bakeoff-$BACKBONE-$SCALE.yaml) carries r-derived
# max_steps, the locked recipe, grad_accum, and the $WORK checkpoint dir; run_short resumes
# from resume_ckpt_path() and checkpoints to work_checkpoint_dir(BACKBONE, SCALE).
srun --kill-on-bad-exit=1 python -u scripts/train_scaffold.py \
  --devices 4 --no-build --backbone "$BACKBONE" \
  --config "configs/experiments/bakeoff-${BACKBONE}-${SCALE}.yaml" &
wait
echo "JOB_DONE"

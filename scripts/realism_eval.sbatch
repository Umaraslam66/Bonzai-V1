#!/bin/bash
# Scored realism-eval GENERATION — one checkpoint per job (Task 4 launcher for
# scripts/realism_eval_gen.py). Full-node 4-GPU torchrun job (boost_usr_prod bills per
# node; never a 1-GPU job). This is the golden torchrun launch shape
# (eval_sharding_golden.sbatch:35), NOT the steering-probe 4-worker bash fan-out.
#
# PARAMETERIZED VIA ENVIRONMENT (sbatch --export). Required: CKPT, OUT. Optional:
# MANIFEST (defaults to the sealed Lane-S manifest), BASE_SEED (python owns the default
# if unset), MAX_NEW (default 4096 — orchestrator decision 2026-07-20; overridable).
#
#   # derive a self-describing job name from the checkpoint path, then submit:
#   CKPT=/leonardo_work/AIFAC_P02_548/Bonzai-OSM/checkpoints/bakeoff/transformer-ar-53M/krakow-seed7/last.ckpt
#   OUT=/leonardo_work/AIFAC_P02_222/Bonzai-OSM/reports/_realism_eval/gen-transformer-ar-53M-krakow-seed7.json
#   CKID="$(basename "$(dirname "$(dirname "$CKPT")")")-$(basename "$(dirname "$CKPT")")"
#   sbatch --job-name="realism-eval-$CKID" \
#          --export=ALL,CKPT="$CKPT",OUT="$OUT" scripts/realism_eval.sbatch
#
# One checkpoint per job — run 6 jobs for the 6-run matrix. The rank-0 write-once
# artifact + REALISM_EVAL_GEN_DONE sentinel are re-verified below before this job's own
# REALISM_EVAL_CKPT_DONE marker is echoed (no marker without end-state verification).
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=realism-eval
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# QOS: deliberately NOT --qos=boost_qos_lprod. The golden sbatch carries lprod, but
# steering_probe.sbatch runs the same 4-GPU/04:00:00 shape on the DEFAULT qos with no
# lprod and completed fine (jobs 49835918 et al.); this job is that same runtime class,
# so the default qos suffices. Add lprod at submit (sbatch --qos=...) only if the queue
# demands it.
set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM

# Fail loud on the two required parameters BEFORE loading modules / touching a GPU.
# (set -u would abort on bare $CKPT/$OUT, so guard via ${VAR:-} and exit 2 with a message.)
if [ -z "${CKPT:-}" ]; then
  echo "FATAL: CKPT unset. Submit with --export=ALL,CKPT=<path/to/last.ckpt>,OUT=<artifact.json>" >&2
  exit 2
fi
if [ -z "${OUT:-}" ]; then
  echo "FATAL: OUT unset. Submit with --export=ALL,CKPT=<path/to/last.ckpt>,OUT=<artifact.json>" >&2
  exit 2
fi

# MANIFEST default = the SEALED Lane-S manifest (a single sampler-manifest.yaml FILE, not a
# dir — realism_eval_gen.py --manifest -> load_verified_manifest_or_raise(Path) loads one file).
MANIFEST="${MANIFEST:-/leonardo_work/AIFAC_P02_222/Bonzai-OSM/data/processed/lane_s_sampler/2026-04-15.0/sampler-manifest.yaml}"
# MAX_NEW default 4096 (orchestrator decision 2026-07-20; matches DEFAULT_MAX_NEW). BASE_SEED,
# if unset, is left to the python default (DEFAULT_BASE_SEED) — no duplicated constant here.
MAX_NEW="${MAX_NEW:-4096}"

module load python/3.11.7 cuda/12.2 gcc/12.2.0
# mamba_ssm needs the gcc-12 libstdc++ at import; build_backbone's import chain touches it
# even for the transformer-ar checkpoints (steering_probe.sbatch:35-39).
export CC=$(command -v gcc) CXX=$(command -v g++) CUDAHOSTCXX=$(command -v g++)
export LD_PRELOAD="$(dirname "$($CC -print-file-name=libstdc++.so.6)")/libstdc++.so.6"
source .venv/bin/activate
export PYTHONPATH="src:${PYTHONPATH:-}"  # match the golden torchrun launch (eval_sharding_golden.sbatch:29)
mkdir -p logs "$(dirname "$OUT")"

# Self-describing checkpoint identity for the context echo (e.g. transformer-ar-53M-krakow-seed7).
CKID="$(basename "$(dirname "$(dirname "$CKPT")")")-$(basename "$(dirname "$CKPT")")"

echo "=== context ==="
echo "host=$(hostname) date=$(date -u +%FT%TZ) git_sha=$(git rev-parse HEAD)"
echo "ckid=$CKID ckpt=$CKPT"
echo "out=$OUT manifest=$MANIFEST max_new=$MAX_NEW base_seed=${BASE_SEED:-<python-default>}"
nvidia-smi -L

# Build the python arg list; only pass --base-seed when the submitter set it.
GEN_ARGS=(--ckpt "$CKPT" --manifest "$MANIFEST" --out "$OUT" --max-new "$MAX_NEW")
if [ -n "${BASE_SEED:-}" ]; then
  GEN_ARGS+=(--base-seed "$BASE_SEED")
fi

# torchrun spawns 4 ranks (one per GPU) on this single node. --standalone = single-node
# rendezvous; WORLD_SIZE/RANK/LOCAL_RANK are set by torchrun and read by the script's guard.
# Tee to a dedicated per-job log so the sentinel grep below is independent of %x/%j resolution.
RUN_LOG="logs/realism-eval-${SLURM_JOB_ID:-local}.gen.out"
set +e
torchrun --standalone --nproc_per_node=4 scripts/realism_eval_gen.py "${GEN_ARGS[@]}" 2>&1 | tee "$RUN_LOG"
status=${PIPESTATUS[0]}
set -e

# Verification discipline: a bare success must never lie. Require (1) torchrun exit 0,
# (2) a non-empty rank-0 artifact on disk, (3) the in-log REALISM_EVAL_GEN_DONE sentinel —
# only then echo this job's own done marker (steering_probe.sbatch:58-95 pattern).
if [ "$status" -ne 0 ]; then
  echo "TORCHRUN FAILED for $CKID (exit $status); see $RUN_LOG" >&2
  exit 1
fi
if [ ! -s "$OUT" ]; then
  echo "MISSING/EMPTY artifact for $CKID: $OUT" >&2
  exit 1
fi
if ! grep -q "REALISM_EVAL_GEN_DONE" "$RUN_LOG"; then
  echo "NO SENTINEL (REALISM_EVAL_GEN_DONE) in $RUN_LOG for $CKID" >&2
  exit 1
fi

echo "[$CKID] gen done (exit + non-empty artifact + sentinel verified) -> $OUT"
echo "REALISM_EVAL_CKPT_DONE"

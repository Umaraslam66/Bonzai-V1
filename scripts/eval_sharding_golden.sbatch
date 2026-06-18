#!/bin/bash
# EXECUTION-DEFERRED — eval-sharding GPU equivalence golden (plan Task 11, Step 4).
# ⛔ Do NOT submit during the CINECA $WORK outage. Submit once $WORK heavy I/O has recovered
# (bundled with the 50M param-match verify), then gate the scored runs on the _SHARDING_GOLDEN_PASS
# marker this writes. Mirrors scripts/bakeoff_run.sbatch's verified env preamble; one node, 4 GPUs,
# torchrun (4 ranks) so the golden actually exercises the 4-GPU sharded path vs the rank-0 baseline.
#   Submit with:  sbatch scripts/eval_sharding_golden.sbatch
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=eval-sharding-golden
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
# Same toolchain + libstdc++ preload as the verified bakeoff entry (gcc/12.2.0; the LD_PRELOAD is
# REQUIRED for the mamba_ssm/causal_conv1d import and harmless for transformer-ar — this golden
# builds transformer-ar but rides the shared preamble unchanged).
module load python/3.11.7 cuda/12.2 gcc/12.2.0
export CC=$(command -v gcc) CXX=$(command -v g++) CUDAHOSTCXX=$(command -v g++)
export LD_PRELOAD="$(dirname "$($CC -print-file-name=libstdc++.so.6)")/libstdc++.so.6"
source .venv/bin/activate
export PYTHONPATH="src:${PYTHONPATH:-}"
mkdir -p logs

# torchrun spawns the 4 ranks (one per GPU) on this single node. --standalone = single-node
# rendezvous (no external etcd); WORLD_SIZE/RANK/LOCAL_RANK are set by torchrun and read by the
# script's hard guard.
torchrun --standalone --nproc_per_node=4 scripts/eval_sharding_golden.py

# F8 end-state: trust the marker only after proving it exists on disk (set -e makes a missing
# marker fail the job loudly, never a shrug).
MARKER="reports/phase-2-bakeoff/_SHARDING_GOLDEN_PASS"
if [ ! -f "$MARKER" ]; then
  echo "JOB_FAILED_ENDSTATE: golden did not write $MARKER (a tooth failed or it never reached rank-0 write)"
  exit 1
fi
echo "end-state verified: $MARKER on disk"
cat "$MARKER"
echo "JOB_DONE"

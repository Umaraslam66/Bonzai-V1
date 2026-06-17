#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=sdpa-window-probe
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# SDPA flash-attention / memory probe at the LOCKED window 13,312 -- execution
# obligation #2 of the token-budget coupled decision (decision memo 2026-06-11).
# NON-SCORED. Single GPU, single process, minutes of walltime. NOT training (the
# full-node 4-GPU rule is a training-job rule; billing is per-node either way and
# this is a kernel/memory measurement, deliberately the smallest footprint that
# answers the question before the allocation lapses).

set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7 cuda/12.2
source .venv/bin/activate
mkdir -p logs

echo "=== job context ==="
echo "host=$(hostname) date=$(date -u +%FT%TZ) git_sha=$(git rev-parse HEAD)"
echo "probe_script_sha256=$(sha256sum scripts/probe_sdpa_window.py | cut -d' ' -f1)"
nvidia-smi -L

# capture rc explicitly (set -e would otherwise abort before logging it);
# rc=3 == probe ran cleanly and the verdict is HALT (YAML still written)
python -u scripts/probe_sdpa_window.py && rc=0 || rc=$?
echo "probe_rc=$rc"
exit "$rc"

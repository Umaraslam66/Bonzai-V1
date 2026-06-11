#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=AIFAC_P02_222
#SBATCH --job-name=scaffold-ddp-short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# The real short run on 4xA100: closes the loop on Singapore and writes the
# reports/ summary. Manifest is pre-built once here; the 4 srun ranks only read it.

set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7 cuda/12.2
source .venv/bin/activate
mkdir -p logs

echo "=== job context ==="
echo "host=$(hostname) date=$(date -u +%FT%TZ) git_sha=$(git rev-parse HEAD)"
echo "ntasks=$SLURM_NTASKS gpus=$(nvidia-smi -L | wc -l)"

echo "=== pre-build training manifest (single process) ==="
python -c "from cfm.data.training.build_shards import build_training_shards as b; s=b('2026-04-15.0','singapore'); print('built', len(s), 'shards')"

echo "=== short run (4-GPU DDP; world_size==4 guard; report on rank 0) ==="
srun --kill-on-bad-exit=1 python -u scripts/train_scaffold.py \
  --devices 4 --max-steps 500 --max-len 2048 --no-build

echo "=== report ==="
# F17 run-keyed report name: {release}-{region}-{backbone}-{params}M-seed{seed}-loop-closed.md
cat reports/phase-1-training-scaffold/2026-04-15.0-singapore-*-loop-closed.md 2>/dev/null || echo "report MISSING"
echo "JOB_DONE"

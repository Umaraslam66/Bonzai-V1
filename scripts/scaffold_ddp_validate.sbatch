#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=scaffold-ddp-validate
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# DDP validation: world_size==4 all-ranks audit-halt + bit-identical 4->4 resume.
# The training manifest is built ONCE here (single process, before any srun) so the
# 4 DDP ranks only READ it (no write race).

set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7 cuda/12.2
source .venv/bin/activate

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
export MASTER_PORT=29501
export SHARED_DIR="$SLURM_SUBMIT_DIR/reports/ddp-resume-$SLURM_JOB_ID"
mkdir -p "$SHARED_DIR" logs

echo "=== job context ==="
echo "host=$(hostname) date=$(date -u +%FT%TZ) git_sha=$(git rev-parse HEAD)"
echo "nodes=$SLURM_NNODES ntasks=$SLURM_NTASKS gpus=$(nvidia-smi -L | wc -l)"

echo "=== env lock ==="
python -c "from cfm.training.env_lock import assert_training_env_locked as a; a(); print('env lock OK')"

echo "=== pre-build training manifest (single process) ==="
python -c "from cfm.data.training.build_shards import build_training_shards as b; s=b('2026-04-15.0','singapore'); print('built', len(s), 'shards')"

echo "=== [1/4] all-ranks audit-halt (world_size==4) ==="
srun --kill-on-bad-exit=1 bash -lc 'RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID python -u scripts/ddp_audit_halt_check.py'

echo "=== [2/4] resume reference (4-GPU fit; world_size==4 guard) ==="
srun --kill-on-bad-exit=1 python -u scripts/ddp_resume_check.py --phase reference

echo "=== [3/4] resume part1 (checkpoint) ==="
srun --kill-on-bad-exit=1 python -u scripts/ddp_resume_check.py --phase part1

echo "=== [4/4] resume part2 (resume + bit-identical compare) ==="
srun --kill-on-bad-exit=1 python -u scripts/ddp_resume_check.py --phase part2

echo "JOB_DONE"

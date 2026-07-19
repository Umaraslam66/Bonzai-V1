#!/bin/bash
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=heldout-cell-count
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000M
#SBATCH --output=/leonardo_work/AIFAC_P02_222/heldout_count_%j.out
#SBATCH --error=/leonardo_work/AIFAC_P02_222/heldout_count_%j.err
set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
echo "HEAD=$(git rev-parse HEAD)"
SECONDS=0
python -u /leonardo_work/AIFAC_P02_222/_heldout_cell_count.py
echo "COUNT_WALL_SECONDS=$SECONDS RC=$?"

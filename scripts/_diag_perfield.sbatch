#!/bin/bash
# READ-ONLY per-macro-field ablation diagnostic (NOT scored). Single GPU, forward-NLL only.
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=diag-perfield
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7 cuda/12.2 gcc/12.2.0
export CC=$(command -v gcc) CXX=$(command -v g++) CUDAHOSTCXX=$(command -v g++)
export LD_PRELOAD="$(dirname "$($CC -print-file-name=libstdc++.so.6)")/libstdc++.so.6"
source .venv/bin/activate
mkdir -p logs reports/_diag

echo "host=$(hostname) date=$(date -u +%FT%TZ)"
CKR=/leonardo_work/AIFAC_P02_548/Bonzai-OSM/checkpoints/bakeoff
for spec in "transformer-ar-53M:7" "transformer-ar-53M:13" "mamba-hybrid-54M:7"; do
  dir=${spec%:*}; seed=${spec#*:}
  id=$(echo "$dir" | sed 's/-5[0-9]M//')-seed$seed
  echo "=== perfield $id ==="
  python -u scripts/_diag_perfield_ablation.py \
    --ckpt "$CKR/$dir/krakow-seed$seed/last.ckpt" \
    --out "reports/_diag/perfield-$id.json" \
    --n-per-city 500
done
echo "DIAG_PERFIELD_DONE"

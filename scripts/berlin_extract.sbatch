#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=AIFAC_P02_222
#SBATCH --job-name=subc-berlin
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Berlin pilot sub_c extraction (Q1 multi-region). Mirrors extract_singapore.sbatch
# (the proven 120G/32-cpu pattern). The Berlin fetch is already cached, so
# load_region cache-hits (no S3 egress); this is pure CPU+memory work — it OOM'd
# on the login node, hence a real compute node here. NOTE: boost_usr_prod bills a
# full GPU node for CPU-only work; the bounded multi-city extract should later move
# to a CPU partition (dcgp_usr_prod). One-off pilot job (~<6 GPU-h) is acceptable.
set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7
source .venv/bin/activate

echo "=== job context ==="
echo "host=$(hostname) date=$(date -u +%FT%TZ) pid=$$"
echo "git_branch=$(git rev-parse --abbrev-ref HEAD)"
echo "git_sha=$(git rev-parse HEAD)"
echo "node=${SLURMD_NODENAME:-unknown} cpus=${SLURM_CPUS_PER_TASK:-unset} nproc=$(nproc)"

rm -rf data/processed/sub_c/2026-04-15.0/berlin
mkdir -p data/processed/sub_c/2026-04-15.0

echo "=== running extract_tiles.py --region berlin --pool-size 32 ==="
time python -u scripts/extract_tiles.py --region berlin --pool-size 32

echo "=== output check ==="
OUT=data/processed/sub_c/2026-04-15.0/berlin
test -f "$OUT/_SUCCESS" && echo "_SUCCESS present" || echo "_SUCCESS MISSING (validator halt or error)"
test -f "$OUT/manifest.yaml" && echo "manifest.yaml present"
tile_count=$(ls -d "$OUT"/tile=EPSG25833_* 2>/dev/null | wc -l)
echo "BERLIN_SUBC_TILES=$tile_count"
du -sh "$OUT" 2>/dev/null || true

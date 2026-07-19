#!/bin/bash
# Lane-S held-out CELL SAMPLER — build job (serial, NO GPU).
#
# Two read-only steps, then one write-once seal:
#   1. per-stratum cell-key census over the held-out tiles (scripts/_heldout_cell_count.py --emit)
#   2. build + seal the sha-locked sampler manifest (scripts/build_lane_s_sampler.py), which
#      verified-loads the 95abb88 floor for lineage, computes census_sha256, and prints the
#      §6 cost re-derivation.
#
# DECISIONS (mirror the proven scripts/heldout_cell_count.sbatch, not the plan snippet):
#   - module load + .venv activate (NOT `uv run`): the established serial pattern on Leonardo.
#   - in-repo script paths (scripts/...), NOT the external /leonardo_work copy the old job used.
#   - .out/.err to the parent work dir, NOT inside the repo: keeps the working tree clean so
#     future write-once git-bundle deploys never hit an untracked-file collision.
#   - NO gcc-12 libstdc++ LD_PRELOAD: neither script imports torch/mamba (pure pyarrow/yaml/cfm).
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=lane-s-sampler
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000M
#SBATCH --output=/leonardo_work/AIFAC_P02_222/lane_s_sampler_%j.out
#SBATCH --error=/leonardo_work/AIFAC_P02_222/lane_s_sampler_%j.err
set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
echo "HEAD=$(git rev-parse HEAD)"

RELEASE=2026-04-15.0
OUTDIR=data/processed/lane_s_sampler/$RELEASE
mkdir -p "$OUTDIR"

# 1. per-stratum cell-key census (read-only over the 1,952 held-out manifest tiles)
SECONDS=0
python -u scripts/_heldout_cell_count.py --emit "$OUTDIR/heldout-cell-census.parquet"
echo "CENSUS_WALL_SECONDS=$SECONDS"

# 2. build + seal the manifest (verified-load floor for lineage; computes census_sha256;
#    seal is write-once + round-trip verified inside the CLI; prints §6 cost re-derivation)
SECONDS=0
python -u scripts/build_lane_s_sampler.py \
  --floor reports/conditioning_floor/$RELEASE/conditioning-floor.yaml \
  --census "$OUTDIR/heldout-cell-census.parquet" \
  --out "$OUTDIR/sampler-manifest.yaml" \
  --release "$RELEASE" --seed 7 --target-features 50 --headroom 2.0
echo "BUILD_WALL_SECONDS=$SECONDS"
echo "LANE_S_BUILD_DONE"

#!/bin/bash
#SBATCH --partition=lrd_all_serial
#SBATCH --account=AIFAC_P02_548
#SBATCH --job-name=mr-eindhoven-gate
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Phase-2 eindhoven GATE for the #19 de-densify fix (sha 507e4a9). Confirms ALL FOUR
# before any fan-out (PI, 2026-06-07):
#   1. #19 clears        — sub_g passes, decoded_vertex_within_cell_bound == 0
#   2. sub_c UNTOUCHED    — PRE vs POST sub_c/d/e fingerprint identical (sub_f-only)
#   3. token yield        — MEASURED post-fix count vs the ~26.9M (inflated) baseline
#   4. VALID geometry     — no OGC-invalid / collapsed rings (decodable_to_valid_geojson)
#
# Budget-free serial partition (no S3 egress needed; sub_c/d/e are cached locally).
# guarded_rederive does sub_f-only (reads sub_c/d/e); sub_g validates separately.
set -uo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7
source .venv/bin/activate
export PYTHONPATH="src:${PYTHONPATH:-}"
R=2026-04-15.0
C=eindhoven
B=data/processed
echo "=== EINDHOVEN GATE START $(date -u +%FT%TZ) sha=$(git rev-parse --short HEAD) ==="

echo "--- PRE sub_c/d/e fingerprint (checkpoint #2 baseline) ---"
echo "    expected: sub_c 5c462db1.. / sub_d a41b0a6b.. / sub_e 419124ba.."
for s in sub_c sub_d sub_e; do
  find $B/$s/$R/$C -type f -printf '%p %s %T@\n' 2>/dev/null | sort | sha256sum \
    | awk -v s=$s '{print "    "s" PRE  "$1}'
done

echo "=== STEP 1: guarded_rederive sub_f (content-changing, --allow-content-change) ==="
python -m scripts.multiregion.guarded_rederive --release $R --city $C --allow-content-change
echo "rederive_rc=$?"

echo "=== STEP 2: sub_g validate (fresh; old verdict backed up) ==="
if [ -d $B/sub_g/$R/$C ]; then
  mv $B/sub_g/$R/$C "$B/sub_g/$R/${C}.bak_prefix19_$(date +%s)"
fi
python scripts/sub_g/validate_phase1_region.py --region $C --release $R \
  --sub-c-region-dir $B/sub_c/$R/$C \
  --sub-d-region-dir $B/sub_d/$R/$C \
  --sub-e-region-dir $B/sub_e/$R/$C \
  --sub-f-region-dir $B/sub_f/$R/$C \
  --output-dir $B/sub_g/$R/$C
echo "subg_rc=$?"

echo "=== STEP 3: MEASURED post-fix token count (checkpoint #3) ==="
python scripts/multiregion/_gate_token_count.py $C

echo "=== STEP 4a: POST sub_c/d/e fingerprint (must match PRE — checkpoint #2) ==="
for s in sub_c sub_d sub_e; do
  find $B/$s/$R/$C -type f -printf '%p %s %T@\n' 2>/dev/null | sort | sha256sum \
    | awk -v s=$s '{print "    "s" POST "$1}'
done

echo "=== STEP 4b: verdict + decoded_vertex / OGC-validity counts (checkpoints #1 + #4) ==="
ls $B/sub_g/$R/$C/
python scripts/multiregion/_gate_verdict.py $C
echo "=== EINDHOVEN GATE DONE $(date -u +%FT%TZ) ==="

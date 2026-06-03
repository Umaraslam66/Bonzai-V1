#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=AIFAC_P02_222
#SBATCH --job-name=berlin-chain
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Berlin sub_d -> sub_e -> sub_f -> sub_g chain (Q1 multi-region tokenizer-on-Europe).
# Halts at the first failing stage (NO set -e on the python calls; explicit checks)
# so a surfaced regime is captured at its stage. sub_d applies the LOCKED Singapore
# macro vocab; sub_f applies the LOCKED Singapore sub_f vocab -> watch the unknown rate.
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM || exit 2
module load python/3.11.7
source .venv/bin/activate

REL=2026-04-15.0
SUBC=data/processed/sub_c/$REL/berlin
SUBD=data/processed/sub_d/$REL/berlin
SUBE=data/processed/sub_e/$REL/berlin
SUBF=data/processed/sub_f/$REL/berlin
SUBG=data/processed/sub_g/$REL/berlin
VOCAB=configs/macro_plan/v1/macro_plan_vocab.yaml

echo "=== context: host=$(hostname) sha=$(git rev-parse --short HEAD) $(date -u +%FT%TZ) ==="
rm -rf "$SUBD" "$SUBE" "$SUBF" "$SUBG"

echo "############ STAGE sub_d (derive_macro_plan; locked SG vocab) ############"
time python -u scripts/derive_macro_plan.py --region berlin --release $REL \
    --sub-c-dir "$SUBC" --output-dir "$SUBD" --macro-vocab "$VOCAB" \
    --commit-sha "$(git rev-parse HEAD)"
RC=$?; echo "SUBD_RC=$RC"; [ $RC -ne 0 ] && { echo "HALT at sub_d"; exit 10; }

echo "############ STAGE sub_e (derive_boundary_contracts) ############"
time python -u scripts/derive_boundary_contracts.py --release $REL --region berlin \
    --sub-c-region-dir "$SUBC" --sub-d-region-dir "$SUBD" --output-region-dir "$SUBE"
RC=$?; echo "SUBE_RC=$RC"; [ $RC -ne 0 ] && { echo "HALT at sub_e"; exit 11; }

echo "############ STAGE sub_f (derive; locked SG sub_f vocab) ############"
time python -u scripts/sub_f/derive.py --release $REL --region berlin \
    --sub-c-region-dir "$SUBC" --sub-d-region-dir "$SUBD" --sub-e-region-dir "$SUBE" \
    --output-region-dir "$SUBF"
RC=$?; echo "SUBF_RC=$RC"; [ $RC -ne 0 ] && { echo "HALT at sub_f"; exit 12; }

echo "############ VOCAB UNKNOWN-RATE on Berlin sub_f ############"
python -u scripts/diag_berlin_vocab.py "$SUBF" || echo "(vocab diag failed, non-fatal)"

echo "############ STAGE sub_g (validate cross-artifact) ############"
python -u -c "import sys; from cfm.data.sub_g.cli import validate_main; sys.exit(validate_main(sys.argv[1:]))" \
    --region berlin --release $REL --sub-c-region-dir "$SUBC" --sub-d-region-dir "$SUBD" \
    --sub-e-region-dir "$SUBE" --sub-f-region-dir "$SUBF" --output-dir "$SUBG"
RC=$?; echo "SUBG_RC=$RC"

echo "############ TILE COUNTS ############"
for S in sub_d sub_e sub_f; do
  echo "$S tiles: $(ls -d data/processed/$S/$REL/berlin/tile=EPSG25833_* 2>/dev/null | wc -l)"
done
echo "=== BERLIN CHAIN END $(date -u +%FT%TZ) ==="
touch berlin_chain_DONE

#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_P02_222
#SBATCH --job-name=mr-timeout-24h
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000M
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Overnight 24h extraction of ONE PI-greenlit non-pathological timeout city.
# Boost is REQUIRED because the 24h wall exceeds lrd_all_serial's 4h cap; the
# PI-authorized batch-2 boost exception (2026-06-04, per-core billing) applies via
# --authorized-boost-override. The destructive sub-C write is guarded by the
# DRIVER (extract_region_batch.py -> run_city): a per-city fcntl.flock (a second
# concurrent extract of this city refuses) + atomic write_parquet (temp+os.replace,
# crash-safe). One job per city => continue-but-loud + per-city logs for free.
set -uo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7
source .venv/bin/activate
echo "=== mr-timeout-24h (overnight, PI-greenlit) ==="
echo "host=$(hostname) sha=$(git rev-parse --short HEAD) city=${CITY} $(date -u +%FT%TZ)"
echo "node=${SLURMD_NODENAME:-unknown} cpus=${SLURM_CPUS_PER_TASK:-unset} wall=24h full-box"
rc=0
python -u scripts/extract_region_batch.py \
  --cities "${CITY}" --partition boost_usr_prod --authorized-boost-override || rc=$?
echo "MR_TIMEOUT_RC=${rc} city=${CITY} $(date -u +%FT%TZ)"
exit ${rc}

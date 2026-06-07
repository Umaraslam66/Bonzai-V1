#!/bin/bash
# SUPERSEDED 2026-06-07 by wave2b_driver.sh — DO NOT USE. This version false-completed:
# it submitted the next chunk while the previous one still held the MaxSubmit=10 quota, the
# sbatch was rejected (empty jid), and it did not guard for that — logged a false "DRAINED"
# and wrote a false DONE, so chunks 3+4 never ran. wave2b adds an sbatch-success guard, a
# submit-quota gate (wait for the user's queue to clear before each chunk), and a
# double-confirmed drain. Kept for the record only.
#
# Wave-2 chunking driver for the #19 corpus-wide re-derive.
# The lrd_all_serial QOS caps SUBMITTED jobs at 10 (MaxSubmitPU=10), so a 33-task
# array is rejected. This submits the wave in chunks of <=9 (under the cap), each as
# a %2-throttled array (MaxJobsPU=2 / cpu=8), and fully drains each chunk before the
# next. Marker-based + setsid-detached so it survives SSH socket drops.
set -uo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
LOG=logs/rederive/wave2_driver.log
WF=logs/rederive/wave2.txt
CHUNK=9
rm -f logs/rederive/wave2_driver.DONE
mapfile -t CITIES < "$WF"
N=${#CITIES[@]}
echo "$(date -u +%FT%TZ) wave2 driver START: $N cities, chunks of $CHUNK" | tee -a "$LOG"
i=0
ci=0
while [ $i -lt "$N" ]; do
  ci=$((ci + 1))
  cf="logs/rederive/wave2_chunk_${ci}.txt"
  printf "%s\n" "${CITIES[@]:$i:$CHUNK}" > "$cf"
  n=$(grep -c . "$cf")
  jid=$(sbatch --parsable --array=0-$((n - 1))%2 \
    --export=ALL,WAVEFILE=$PWD/$cf scripts/multiregion/rederive_fanout.sbatch)
  echo "$(date -u +%FT%TZ) chunk $ci ($n cities) array=$jid" | tee -a "$LOG"
  while squeue -h -j "$jid" -t PENDING,RUNNING -o %T 2>/dev/null | grep -q .; do sleep 60; done
  echo "$(date -u +%FT%TZ) chunk $ci DRAINED" | tee -a "$LOG"
  i=$((i + CHUNK))
done
echo "$(date -u +%FT%TZ) wave2 driver COMPLETE ($ci chunks)" | tee -a "$LOG"
touch logs/rederive/wave2_driver.DONE

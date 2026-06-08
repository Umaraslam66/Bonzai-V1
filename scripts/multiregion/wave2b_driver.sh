#!/bin/bash
# Wave-2 RECOVERY driver. The first wave2_driver.sh false-completed: it submitted chunk 3
# while chunk 2 still held the 10-job submit quota, sbatch was REJECTED (empty jid), and
# the script did not guard for that — it saw no job in squeue, logged "DRAINED", and wrote
# a false DONE. Chunks 3+4 (15 cities) never ran.
#
# This driver fixes all three faults:
#   1. SBATCH-SUCCESS GUARD — retry until sbatch returns a non-empty jid (never proceed on
#      a rejected submit).
#   2. SUBMIT-QUOTA GATE — wait until THIS user's whole queue is empty before each chunk
#      (double-confirmed), so a chunk can never be rejected by the MaxSubmit=10 cap.
#   3. ROBUST DRAIN — double-confirm the chunk array is gone (a single transient squeue
#      hiccup can return empty; one check is not enough — that caused the false drain).
# Idempotent: re-derives only cities still at DERIVATION 1.1 (per-city lock + the
# pending-lister), so it self-corrects for whatever chunk 2 did/did not finish.
set -uo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7 >/dev/null 2>&1
source .venv/bin/activate
export PYTHONPATH="src:${PYTHONPATH:-}"
LOG=logs/rederive/wave2b_driver.log
SBATCH=scripts/multiregion/rederive_fanout.sbatch
CHUNK=9
rm -f logs/rederive/wave2_all.DONE

queue_empty() {
  [ "$(squeue -h -u uaslam00 -t PENDING,RUNNING -o '%i' 2>/dev/null | grep -c .)" -eq 0 ]
}
wait_queue_empty() {  # double-confirmed (guards transient squeue hiccups)
  while :; do
    if queue_empty; then sleep 15; queue_empty && return 0; fi
    sleep 60
  done
}

echo "$(date -u +%FT%TZ) wave2b START (recovery)" | tee -a "$LOG"
ci=0
while :; do
  wait_queue_empty
  mapfile -t REMAIN < <(python scripts/multiregion/_list_pending_rederive.py)
  if [ "${#REMAIN[@]}" -eq 0 ]; then break; fi
  ci=$((ci + 1))
  cf="logs/rederive/wave2b_chunk_${ci}.txt"
  printf "%s\n" "${REMAIN[@]:0:$CHUNK}" > "$cf"
  n=$(grep -c . "$cf")
  echo "$(date -u +%FT%TZ) chunk $ci: $n cities (${#REMAIN[@]} pending total)" | tee -a "$LOG"
  jid=""
  while [ -z "$jid" ]; do
    jid=$(sbatch --parsable --array=0-$((n - 1))%2 --export=ALL,WAVEFILE=$PWD/$cf "$SBATCH" 2>>"$LOG")
    if [ -z "$jid" ]; then
      echo "$(date -u +%FT%TZ) chunk $ci sbatch EMPTY (quota?); wait+retry" | tee -a "$LOG"
      sleep 120
      wait_queue_empty
    fi
  done
  echo "$(date -u +%FT%TZ) chunk $ci array=$jid" | tee -a "$LOG"
  while squeue -h -j "$jid" -t PENDING,RUNNING -o %T 2>/dev/null | grep -q .; do sleep 60; done
  sleep 15  # double-confirm drain
  while squeue -h -j "$jid" -t PENDING,RUNNING -o %T 2>/dev/null | grep -q .; do sleep 60; done
  echo "$(date -u +%FT%TZ) chunk $ci DRAINED" | tee -a "$LOG"
done

pend=$(python scripts/multiregion/_list_pending_rederive.py | grep -c . || true)
if [ "$pend" -eq 0 ]; then
  echo "$(date -u +%FT%TZ) wave2b COMPLETE — 0 pending; all corpus cities at 1.2" | tee -a "$LOG"
  touch logs/rederive/wave2_all.DONE
else
  echo "$(date -u +%FT%TZ) wave2b WARN: $pend still pending (investigate)" | tee -a "$LOG"
fi

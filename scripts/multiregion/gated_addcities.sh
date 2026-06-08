#!/bin/bash
# One-city-GATED add-cities launch (2026-06-06). Proves the just-debugged path on ONE
# city before committing 14h of fan-out: pre-fetch eindhoven on the LOGIN node (egress)
# -> boost job must CACHE-HIT that warm cache (the exact handoff that bit the first
# launch: no-egress on compute) -> sub_c must start PRODUCING tiles. Only then fan out
# the other 6. HALTS + writes a self-evident status on any failure. setsid-detached, so
# it survives the recurring laptop-sleep SSH-socket drops.
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7
source .venv/bin/activate
R=2026-04-15.0
ST=logs/addcities_status.txt
SB=scripts/multiregion_process_boost_24h.sbatch
GATE=eindhoven
OTHERS="tilburg wolfsburg telford szczecin linz debrecen"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ST"; }

prefetch() {  # $1 city ; returns 0 iff cache manifest lands
  python -u -c "import sys; sys.path.insert(0,'src'); from pathlib import Path; from cfm.data.overture import load_region; load_region(sys.argv[1], confirm=True, repo_root=Path('.'))" "$1" >>"logs/prefetch_$1.log" 2>&1
  [ -f "data/cache/overture/$R/$1/manifest.yaml" ]
}
submit() { sbatch --parsable --job-name="mr-add-$1" --export=ALL,CITY="$1" "$SB"; }

log "GATED-ADDCITIES START gate=$GATE others=[$OTHERS]"

# 1. gate city: prefetch (login egress) + submit ONE boost job
if ! prefetch "$GATE"; then log "HALT: $GATE prefetch FAILED (logs/prefetch_$GATE.log)"; exit 1; fi
log "GATE $GATE prefetched OK (cache manifest present)"
gjid=$(submit "$GATE"); log "GATE $GATE boost submitted jid=$gjid"

# 2. GATE CHECK: cache-HIT (no S3 egress error) AND sub_c producing >=20 tiles; else HALT
gate_ok=0
for _ in $(seq 1 120); do  # up to ~2h
  elog=$(ls -t logs/mr-add-$GATE-*.err 2>/dev/null | head -1)
  if [ -n "$elog" ] && grep -qa "Could not connect to server" "$elog"; then
    log "HALT: $GATE boost hit S3 EGRESS error -> cache MISS / key mismatch. NOT fanning out."; exit 1
  fi
  nt=$(ls -d data/processed/sub_c/$R/$GATE/tile=* 2>/dev/null | wc -l | tr -d ' ')
  if [ "${nt:-0}" -ge 20 ]; then gate_ok=1; log "GATE PASSED: $GATE cache-HIT + sub_c producing ($nt tiles)"; break; fi
  st=$(sacct -j "$gjid" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
  case "$st" in FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)
    log "HALT: $GATE boost state=$st before producing tiles. NOT fanning out."; exit 1;; esac
  sleep 60
done
[ "$gate_ok" = 1 ] || { log "HALT: $GATE gate not passed within 2h. NOT fanning out."; exit 1; }

# 3. fan out the other 6 (prefetch on login -> submit boost)
ids=("$gjid")
for c in $OTHERS; do
  if prefetch "$c"; then jid=$(submit "$c"); log "FANOUT $c boost=$jid"; ids+=("$jid")
  else log "SKIP $c (prefetch FAILED)"; fi
done

# 4. arm the final shipped-corpus G4 rollup (afterany all submitted jobs)
dep=$(IFS=:; echo "${ids[*]}")
roll=$(sbatch --parsable --dependency=afterany:"$dep" scripts/multiregion/g4_rollup_morning.sbatch)
log "FANOUT_DONE jobs=${#ids[@]} rollup=$roll"

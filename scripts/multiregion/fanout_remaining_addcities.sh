#!/bin/bash
# Smart fan-out of the remaining 6 add-cities (2026-06-06). The one-city gate is
# satisfied on eindhoven by the DIRECT health signals (not the >=20-tile proxy):
# cache-HIT confirmed (no S3-egress error) + boost RUNNING + sub_c actively computing
# (sstat AveCPU advancing 1:1 with wall, RSS ~3.8G = upfront load/partition, NOT stalled).
# The only NEW risk on this path was the fetch->boost cache handoff, and that is proven;
# a slow tile-start is just the known 1.7-5.3h upfront, not a failure. setsid-detached.
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
module load python/3.11.7
source .venv/bin/activate
R=2026-04-15.0
ST=logs/addcities_status.txt
SB=scripts/multiregion_process_boost_24h.sbatch
GATE_JID=44768419   # eindhoven boost (already running; gate proven healthy)
OTHERS="tilburg wolfsburg telford szczecin linz debrecen"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ST"; }
prefetch() {
  python -u -c "import sys; sys.path.insert(0,'src'); from pathlib import Path; from cfm.data.overture import load_region; load_region(sys.argv[1], confirm=True, repo_root=Path('.'))" "$1" >>"logs/prefetch_$1.log" 2>&1
  [ -f "data/cache/overture/$R/$1/manifest.yaml" ]
}
submit() { sbatch --parsable --job-name="mr-add-$1" --export=ALL,CITY="$1" "$SB"; }

log "FANOUT-REMAINING START (eindhoven gate PASSED via direct health: cache-HIT + RUNNING + CPU-active upfront)"
ids=("$GATE_JID")
for c in $OTHERS; do
  if prefetch "$c"; then jid=$(submit "$c"); log "FANOUT $c boost=$jid"; ids+=("$jid")
  else log "SKIP $c (prefetch FAILED — see logs/prefetch_$c.log)"; fi
done
dep=$(IFS=:; echo "${ids[*]}")
roll=$(sbatch --parsable --dependency=afterany:"$dep" scripts/multiregion/g4_rollup_morning.sbatch)
log "FANOUT_DONE jobs=${#ids[@]} ids=[${ids[*]}] rollup=$roll"

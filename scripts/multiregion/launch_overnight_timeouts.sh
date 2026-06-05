#!/bin/bash
# Overnight launcher: the PI-greenlit 9 NON-pathological timeout cities (2026-06-05).
# - HARD-EXCLUDES paris/lyon/madrid (DROPPED: FR/FR/ES already covered, pathological
#   wall) and refuses if any excluded city slips into the run list.
# - Submits ONE guarded 24h-boost sbatch per city (driver = per-city lock + atomic
#   write). Independent jobs => continue-but-loud + per-city logs.
# - Arms a G4 auto-roll-up with afterany dependency on all 9 (runs when they land).
# NO globs, NO hand-rolled execution loop against the corpus — only sbatch submission.
set -euo pipefail
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM

NINE=(amsterdam budapest hamburg helsinki lisbon rotterdam valencia vienna warsaw)
EXCLUDE=(paris lyon madrid)

# --- exclusion guard: a dropped city must NEVER enter a run list ---
for c in "${NINE[@]}"; do
  for x in "${EXCLUDE[@]}"; do
    if [ "$c" = "$x" ]; then
      echo "FATAL: excluded/dropped city '$c' present in run list — refusing to launch." >&2
      exit 1
    fi
  done
done
if [ "${#NINE[@]}" -ne 9 ]; then
  echo "FATAL: expected 9 cities, got ${#NINE[@]}" >&2; exit 1
fi
echo "Launching 9: ${NINE[*]}"
echo "Hard-excluded (DROPPED): ${EXCLUDE[*]}"

ids=()
for c in "${NINE[@]}"; do
  jid=$(sbatch --parsable --job-name="mr-to-${c}" --export=ALL,CITY="${c}" \
        scripts/multiregion_process_boost_24h.sbatch)
  echo "  submitted ${c} -> job ${jid}"
  ids+=("${jid}")
done
if [ "${#ids[@]}" -ne 9 ]; then
  echo "FATAL: only ${#ids[@]}/9 jobs submitted — NOT arming roll-up" >&2; exit 1
fi

dep=$(IFS=:; echo "${ids[*]}")
roll=$(sbatch --parsable --dependency=afterany:"${dep}" scripts/multiregion/g4_rollup_morning.sbatch)
echo "G4 auto-roll-up armed -> job ${roll} (afterany:${dep})"
echo "OVERNIGHT_LAUNCH_DONE timeouts=[${ids[*]}] rollup=${roll}"

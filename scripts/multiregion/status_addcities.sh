#!/bin/bash
# One-glance TRUE terminal status of the add-cities, from ON-DISK MARKERS (not the
# SSH socket, not an RC echo a killed job skips — the RC=137 lesson). Run on re-auth.
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM
R=2026-04-15.0
echo "=== add-cities terminal status (markers = truth) $(date -u +%FT%TZ) ==="
for c in eindhoven tilburg wolfsburg telford szczecin linz debrecen; do
  if   [ -f data/processed/sub_g/$R/$c/_PHASE1_VALIDATED ]; then s="VALIDATED"
  elif [ -f data/processed/sub_g/$R/$c/quarantine_report.yaml ]; then s="FAILED-VALIDATION"
  elif [ -f data/processed/sub_f/$R/$c/_SUCCESS ]; then s="sub_f-done(not-validated)"
  elif [ -f data/processed/sub_c/$R/$c/_SUCCESS ]; then s="sub_c-done-only"
  elif [ -d data/processed/sub_c/$R/$c ]; then s="sub_c-PARTIAL($(ls -d data/processed/sub_c/$R/$c/tile=* 2>/dev/null | wc -l | tr -d ' ')t)"
  else s="NOT-STARTED"; fi
  # IN-FLIGHT state only (squeue = active jobs); markers above are the terminal truth.
  # Do NOT use sacct --name|tail -1 — it surfaces the STALE old failed job (the scare).
  js=$(squeue -h -n "mr-add-$c" -o "%T" 2>/dev/null | head -1)
  printf "  %-12s %-26s live=%s\n" "$c" "$s" "${js:-no-active-job(marker=truth)}"
done
echo "=== gated-chain status log ==="; tail -20 logs/addcities_status.txt 2>/dev/null
echo "=== final G4 (if rollup ran) ==="; sed -n '/cities=/p; /DoD PASS/p' reports/2026-06-06-overnight-g4-rollup.txt 2>/dev/null

#!/bin/bash
# Berlin multi-region pilot (Q1 CRS parameterization).
# STEP 1: fetch Berlin (confirm=True -> COUNT(*) optimization), timed.
# STEP 2: sub_c extract_tiles --region berlin -> tiles/city + first regimes.
# No `set -e`: we WANT to see how far the chain gets and capture any regime
# failure (the pilot's purpose), not abort the whole script.
cd /leonardo_work/AIFAC_P02_222/Bonzai-OSM || exit 2
module load python/3.11.7 2>/dev/null
source .venv/bin/activate
mkdir -p logs

echo "=================== BERLIN PILOT START $(date -u +%FT%TZ) ==================="

echo "=== STEP 1: fetch Berlin (confirm=True, optimized) ==="
python - <<'PY'
import time, json
from cfm.data.overture import load_region
t0 = time.time()
r = load_region("berlin", confirm=True)
dt = time.time() - t0
out = {
    "fetch_seconds": round(dt, 1),
    "region": r.name,
    "projected_crs": r.projected_crs,
    "themes": {k: r.themes[k].num_rows for k in r.themes},
}
print("BERLIN_FETCH_RESULT", json.dumps(out))
with open("berlin_fetch_result.json", "w") as f:
    json.dump(out, f, indent=2)
PY
echo "STEP1_EXIT=$?"

echo "=== STEP 2: sub_c extract_tiles --region berlin ==="
python scripts/extract_tiles.py --region berlin
echo "STEP2_EXIT=$?"

echo "=== STEP 3: Berlin tile count (tiles/city) ==="
BERLIN_TILES=$(ls -d data/processed/sub_c/2026-04-15.0/berlin/tile=EPSG25833_* 2>/dev/null | wc -l)
echo "BERLIN_SUBC_TILES=$BERLIN_TILES"

echo "=================== BERLIN PILOT END $(date -u +%FT%TZ) ==================="
touch berlin_pilot_DONE

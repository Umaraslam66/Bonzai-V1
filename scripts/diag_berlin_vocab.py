"""Berlin sub_f vocab unknown-tag rate.

Pilot question (2026-06-03): does the Singapore-derived sub_f vocab floor
under-cover European morphology? A HIGH unknown rate on Berlin => the floor
needs re-derivation before a multi-city extract; a low rate => acceptable
graceful degradation. Reports the rate as a NUMBER.

Each encoded feature is `[<feature>=509, semantic_id, ...]`, so the token right
after each <feature> marker is the semantic tag id. We classify it against the
BP1 (semantic) vs BP4 (<unknown_*>) id sets from the locked vocab.
"""

from __future__ import annotations

import glob
import sys

import pyarrow.parquet as pq

from cfm.data.sub_f.encoder import _FEATURE_TOKEN_ID
from cfm.data.sub_f.vocab import load_sub_f_vocab

subf_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed/sub_f/2026-04-15.0/berlin"

slots = load_sub_f_vocab()
unknown_ids = {s.token_id for s in slots if s.family == "unknown"}
semantic_ids = {s.token_id for s in slots if s.family == "semantic"}
print(f"vocab: {len(semantic_ids)} BP1 semantic + {len(unknown_ids)} BP4 unknown ids")

tiles = sorted(glob.glob(f"{subf_dir}/tile=*/cells.parquet"))
print(f"sub_f tiles with cells.parquet: {len(tiles)}")
if not tiles:
    sys.exit("no sub_f cells.parquet found")

t0 = pq.read_table(tiles[0])
print(f"cells.parquet columns: {t0.column_names}")
tok_col = next((c for c in t0.column_names if "token" in c.lower()), None)
if tok_col is None:
    sys.exit(f"no token column found in {t0.column_names}")
print(f"using token column: {tok_col}")

n_feat = n_unknown = n_bp1 = n_other = 0
for tp in tiles:
    col = pq.read_table(tp, columns=[tok_col]).column(tok_col).to_pylist()
    for row in col:
        if not row:
            continue
        toks = list(row)
        for i, tk in enumerate(toks):
            if tk == _FEATURE_TOKEN_ID and i + 1 < len(toks):
                sem = toks[i + 1]
                n_feat += 1
                if sem in unknown_ids:
                    n_unknown += 1
                elif sem in semantic_ids:
                    n_bp1 += 1
                else:
                    n_other += 1

rate = (n_unknown / n_feat * 100) if n_feat else 0.0
print(f"BERLIN_VOCAB features={n_feat} BP1={n_bp1} unknown={n_unknown} other={n_other}")
print(f"BERLIN_UNKNOWN_RATE_PCT={rate:.2f}")

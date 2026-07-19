"""Re-run ONLY the capped dense cells at the full 13312 cap to confirm cell_end fires.

Reads the probe's gen_tokens.json, picks the dense_urban cells that hit the 1536 probe cap,
and regenerates them with IDENTICAL prefix/char_stats/seed at max_new=13312 (seeded -> exact
continuation of the same cell). Reports whether <cell_end>=260 fires in the dense regime.
NON-scored; no fixes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cfm.inference.generate import generate_cell_tokens
from cfm.models.backbone import build_backbone
from cfm.training.config import ScaffoldConfig


def load_model(ckpt: str, device: torch.device) -> torch.nn.Module:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ScaffoldConfig(**ck["hyper_parameters"])
    model = build_backbone(cfg.backbone, cfg)
    sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss or unexp:
        raise SystemExit(f"state_dict mismatch missing={list(miss)} unexpected={list(unexp)}")
    model.eval().to(device)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--max-new", type=int, default=13312)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[recap] device={dev}", flush=True)
    model = load_model(a.ckpt, dev)
    data = json.loads(Path(a.json).read_text())
    targets = [r for r in data["records"] if r["context"] == "dense_urban" and r["hit_cap"]]
    print(f"[recap] re-running {len(targets)} capped dense cells at max_new={a.max_new}", flush=True)
    for r in targets:
        toks = generate_cell_tokens(
            model, prefix=r["prefix"], max_new=a.max_new, seed=r["gen_seed"], char_stats=r["char_stats"]
        )
        fired = 260 in toks
        hit = len(toks) >= a.max_new
        print(f"  cell {r['cell_index']} seed={r['gen_seed']}: n_tokens={len(toks)} "
              f"cell_end_fired={fired} hit_cap={hit}", flush=True)
    print("EYEBALL_RECAP_DONE", flush=True)


if __name__ == "__main__":
    main()

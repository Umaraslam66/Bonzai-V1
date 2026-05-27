"""Snapshot taginfo global key+value raw counts for sub-F BP1 floor.

Per Bug 1 fix: snapshot writes raw counts only (count_all, count_ways,
count_nodes, count_relations). Fraction derivation happens in
floor_analysis.py with consistent denominator (BP1 fix 2:
fraction-of-feature-bearing-elements within element type).

Schema:
  key,value,count_all,count_ways,count_nodes,count_relations,row_type,parent_key

  - row_type = "key": from /api/4/keys/all; value="" parent_key=""
  - row_type = "value": from /api/4/key/values for each parent key;
    count_ways/nodes/relations = 0 (taginfo /key/values doesn't return
    per-element-type breakdown for values — value rows inherit parent
    key's dominant ET distribution as documented approximation in
    floor_analysis.py). Per cascade #6, building is paginated because it is
    in cascade #4 Singapore X scope; non-scope keys are capped at first
    999 results where applicable per spec §12 #12.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import json

TAGINFO_KEYS_URL = "https://taginfo.openstreetmap.org/api/4/keys/all"
TAGINFO_VALUES_URL = (
    "https://taginfo.openstreetmap.org/api/4/key/values"
    "?key={key}&page={page}&rp=999&sortname=count&sortorder=desc"
)

USER_AGENT = "bonzai-osm-sub-f-snapshot/1.0 (research)"


def _fetch_json(url: str) -> Any:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def fetch_all_values_for_key(key: str, max_total: int = 20000) -> list[dict]:
    """Fetch all (or up to max_total) values for key, paginating taginfo."""
    values: list[dict] = []
    page = 1
    while True:
        url = TAGINFO_VALUES_URL.format(key=key, page=page)
        resp = _fetch_json(url)
        page_data = resp.get("data", [])
        values.extend(page_data)
        if len(page_data) < 999 or len(values) >= max_total:
            return values[:max_total]
        page += 1


def snapshot(out_csv: Path, top_n_keys: int = 200) -> None:
    """Snapshot taginfo raw counts into out_csv. Re-runnable but idempotent."""
    if out_csv.exists():
        print(f"[snapshot_taginfo] {out_csv} exists; skipping (delete to re-snapshot)")
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    keys_resp = _fetch_json(TAGINFO_KEYS_URL)
    keys_sorted = sorted(keys_resp["data"], key=lambda r: -r["count_all"])[:top_n_keys]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "key", "value", "count_all", "count_ways", "count_nodes",
                "count_relations", "row_type", "parent_key",
            ]
        )
        for key_row in keys_sorted:
            key = key_row["key"]
            # Key row: raw counts per ET from /keys/all
            writer.writerow(
                [
                    key, "", key_row["count_all"], key_row["count_ways"],
                    key_row["count_nodes"], key_row["count_relations"],
                    "key", "",
                ]
            )
            # Value rows: only count_all from /key/values; per-ET breakdown not
            # provided by this endpoint. Value rows inherit parent key's
            # dominant ET in floor_analysis.py (documented approximation).
            try:
                if key == "building":  # cascade #4: building needs full coverage
                    values_data = fetch_all_values_for_key(key)
                else:
                    values_resp = _fetch_json(TAGINFO_VALUES_URL.format(key=key, page=1))
                    values_data = values_resp.get("data", [])
                for v in values_data:
                    writer.writerow(
                        [
                            key, v["value"], v["count"], 0, 0, 0,
                            "value", key,
                        ]
                    )
            except Exception as exc:  # noqa: BLE001 — best-effort per key
                print(f"[snapshot_taginfo] warning: {key}: {exc}", file=sys.stderr)

    print(f"[snapshot_taginfo] wrote {out_csv}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="2026-04-15.0")
    args = parser.parse_args()
    out = Path(__file__).resolve().parents[2] / "configs" / "sub_f" / "taginfo" / f"{args.release}.csv"
    snapshot(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

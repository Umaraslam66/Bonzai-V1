"""Snapshot OSM wiki Map_features page (wikitext + revision_id + sha256).

Per BP1 fix B: HTML hashing is fragile (MediaWiki embeds timestamps in
rendered markup). Snapshot the raw wikitext via MediaWiki API; hash the
wikitext bytes; pin the MediaWiki revision ID as the canonical reproducibility
anchor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

WIKI_API_URL = (
    "https://wiki.openstreetmap.org/w/api.php"
    "?action=query&prop=revisions&titles=Map_features"
    "&rvprop=content|ids&rvslots=main&format=json&formatversion=2"
)

USER_AGENT = "bonzai-osm-sub-f-snapshot/1.0 (research)"


def snapshot(out_dir: Path, release: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    wikitext_path = out_dir / f"{release}.wikitext"
    sha256_path = out_dir / f"{release}.sha256"
    revid_path = out_dir / f"{release}.revision_id"

    if wikitext_path.exists() and sha256_path.exists() and revid_path.exists():
        print(f"[snapshot_wiki] {release}.* exists in {out_dir}; skipping")
        return

    req = Request(WIKI_API_URL, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read())

    page = payload["query"]["pages"][0]
    rev = page["revisions"][0]
    revision_id = rev["revid"]
    wikitext_bytes = rev["slots"]["main"]["content"].encode("utf-8")

    sha = hashlib.sha256(wikitext_bytes).hexdigest()

    wikitext_path.write_bytes(wikitext_bytes)
    sha256_path.write_text(f"{sha}\n", encoding="utf-8")
    revid_path.write_text(f"{revision_id}\n", encoding="utf-8")

    print(f"[snapshot_wiki] wrote {wikitext_path}, sha={sha[:16]}…, rev={revision_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="2026-04-15.0")
    args = parser.parse_args()
    out_dir = Path(__file__).resolve().parents[2] / "configs" / "sub_f" / "wiki_map_features"
    snapshot(out_dir, args.release)
    return 0


if __name__ == "__main__":
    sys.exit(main())

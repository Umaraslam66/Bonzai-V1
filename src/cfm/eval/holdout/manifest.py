"""Frozen holdout manifest - the lock artifact (spec §F), written once.

Region-keyed (region is a first-class partition key, spec §B): adding a held-out
region D later is adding a ``regions[D]`` entry, with zero change to the freeze or
audit logic. Mirrors sub-C/sub-D: canonical YAML, a manifest_sha256 that EXCLUDES
itself (the ``*_sha256`` exclusion grammar), and write-once semantics - a
contaminated or re-derived holdout invalidates every eval number, so the artifact
never moves once frozen.
"""

from __future__ import annotations

from pathlib import Path

from cfm.data.determinism import compute_sha256
from cfm.data.io import canonicalize_yaml

MANIFEST_SCHEMA_VERSION: str = "1.0"

TileKey = tuple[int, int]


def build_holdout_manifest(
    *,
    region: str,
    selected_tiles: list[TileKey],
    per_tile_provenance: dict[TileKey, dict],
) -> dict:
    """Build the (unfrozen) manifest dict for one region."""
    tiles = []
    for ti, tj in sorted(selected_tiles):
        prov = per_tile_provenance[(ti, tj)]
        tiles.append(
            {
                "tile_i": int(ti),
                "tile_j": int(tj),
                "provenance_sha256": prov["provenance_sha256"],
                "macro_vocab_sha256": prov.get("macro_vocab_sha256"),
            }
        )
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "regions": {
            region: {
                "partition_path": f"holdout/region={region}",
                "tiles": tiles,
            }
        },
    }


def manifest_sha256(data: dict) -> str:
    """SHA over the canonical manifest EXCLUDING the manifest_sha256 field itself."""
    payload = {k: v for k, v in data.items() if k != "manifest_sha256"}
    return compute_sha256(canonicalize_yaml(payload).encode("utf-8"))


def freeze_holdout_manifest(data: dict, path: Path) -> None:
    """Stamp the manifest SHA and write ONCE. Refuses to overwrite a locked manifest."""
    if path.exists():
        raise FileExistsError(
            f"holdout manifest already locked at {path}; it is written once and never "
            "regenerated (spec §F). Delete deliberately only to re-lock the eval set."
        )
    frozen = dict(data)
    frozen["manifest_sha256"] = manifest_sha256(frozen)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonicalize_yaml(frozen), encoding="utf-8")

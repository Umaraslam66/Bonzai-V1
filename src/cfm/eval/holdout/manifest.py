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

# Single-region / Singapore LEGACY schema version. FROZEN — never bump: the frozen SG
# holdout manifest on disk carries manifest_schema_version: '1.0' and its manifest_sha256
# is computed OVER that field, so a bump would silently break the locked SG eval set
# (spec §2.3: "the SG set is untouched"). The multi-region builder pins its own version
# INDEPENDENTLY via MULTIREGION_MANIFEST_SCHEMA_VERSION below.
MANIFEST_SCHEMA_VERSION: str = "1.0"

# Multi-region (EU corpus) schema version, used ONLY by build_holdout_manifest_multiregion.
# Independent by construction from MANIFEST_SCHEMA_VERSION so neither edit can leak across.
MULTIREGION_MANIFEST_SCHEMA_VERSION: str = "2.0"

TileKey = tuple[int, int]


class HoldoutDeclarationError(Exception):
    """A whole-city holdout declaration that cannot be proven correct-by-construction."""


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


def build_holdout_manifest_multiregion(
    regions_payload: dict[str, dict],
    *,
    corpus_release: str,
    derivation_version: str,
    train_cities: set[str],
    corpus_tile_counts: dict[str, int],
) -> dict:
    """Build the (unfrozen) multi-region whole-city holdout manifest (spec §2.1).

    The whole-city declaration is proven correct-by-construction (spec §2.2), not merely
    claimed: a faithful §6 city-guard over a WRONG declaration still leaks, so we assert
    at build time. ``manifest_sha256`` and ``totals.train_tokens`` are added later at the
    real build/freeze (T6), NOT here.

    Raises:
        HoldoutDeclarationError: if a city appears on both sides (§2.2a), or if an
            enumerated tile set does not match the frozen-corpus tile count (§2.2b).
    """
    held_out: list[str] = sorted(regions_payload)

    # §2.2(a): no city on both the holdout and the train side.
    both = set(held_out) & set(train_cities)
    if both:
        raise HoldoutDeclarationError(
            f"cities on both sides (holdout AND train): {sorted(both)} — a held-out city "
            "must never appear in the train split (§2.2a)."
        )

    regions: dict[str, dict] = {}
    for city in held_out:
        p = regions_payload[city]
        tiles = [
            {
                "tile_i": int(t["tile_i"]),
                "tile_j": int(t["tile_j"]),
                "provenance_sha256": t["provenance_sha256"],
                "macro_vocab_sha256": t.get("macro_vocab_sha256"),
            }
            for t in sorted(p["tiles"], key=lambda t: (t["tile_i"], t["tile_j"]))
        ]
        # §2.2(b): the enumerated tiles MATCH the frozen-corpus tile set (count vs G4).
        # WORDING IS LOAD-BEARING: "matches frozen-corpus tile set", NEVER "fully
        # enumerated" — munich is whole-CORPUS-city but inner-core by extent (#21), so
        # "fully enumerated" would invite mis-verifying (b) as geographic completeness
        # (the false-DONE class).
        if len(tiles) != corpus_tile_counts[city]:
            raise HoldoutDeclarationError(
                f"{city}: enumerated {len(tiles)} tiles but frozen corpus has "
                f"{corpus_tile_counts[city]} — manifest tile set must equal the "
                f"frozen-corpus tile set, but this set does not. The check is whether the "
                f"enumeration matches frozen-corpus tile set (no invented/dropped tiles). "
                f"NB this is corpus-tile-set match, NOT geographic completeness "
                f"(e.g. munich is inner-core by extent, #21)."
            )
        regions[city] = {
            "partition_path": f"holdout/region={city}",
            "holdout_kind": "whole_city",
            "morphology": p["morphology"],
            "density": p["density"],
            "geography": p["geography"],
            "crs": p["crs"],
            "n_tiles": len(tiles),
            "n_usable_tiles": p.get("n_usable_tiles"),
            "tokens": int(p["tokens"]),
            "tiles": tiles,
        }

    held_tok = sum(r["tokens"] for r in regions.values())
    return {
        "manifest_schema_version": MULTIREGION_MANIFEST_SCHEMA_VERSION,
        "corpus_release": corpus_release,
        "derivation_version": derivation_version,
        "held_out_cities": held_out,
        "regions": regions,
        "totals": {"held_out_tokens": held_tok},
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

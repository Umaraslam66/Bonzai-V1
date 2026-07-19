"""Held-out cell loader for the perplexity-gap (spec §2, D2/D3).

Builds per-cell examples for the held-out cities EXACTLY as training does — via
``build_shards_in_memory`` + ``flatten_shards_to_cells`` (the trigger-2 single source) —
so the matched conditioning prefix and char_stats are identical to what the model trained
under. The shuffled prefix is a within-city DONOR cell's whole value-bearing prefix
(within-city derangement, D3): same region => differs only in the macro buckets. Macro-only
keeps the cell's OWN char_stats; full uses the donor's.

Requires the sub-C/sub-D/sub-F tile artifacts on Leonardo $WORK (Leonardo-only).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import yaml

from cfm.data.training.build_shards import build_shards_in_memory
from cfm.data.training.datamodule import flatten_shards_to_cells
from cfm.eval.holdout.paths import holdout_manifest_for_region
from cfm.eval.standing.shuffle import macro_deranged_donor


@dataclass(frozen=True)
class HeldoutCell:
    region: str
    body_tokens: list[int]
    own_prefix: list[int]
    own_char: list[float]
    donor_prefix: list[int]  # within-city donor's whole value-bearing prefix
    donor_char: list[float]


def _heldout_tile_ids(release: str, region: str) -> list[tuple[int, int]]:
    manifest = yaml.safe_load(holdout_manifest_for_region(release, region).read_text())
    tiles = manifest["regions"][region]["tiles"]
    return [(int(t["tile_i"]), int(t["tile_j"])) for t in tiles]


def _sample(examples: list, n_per_city: int | None, seed: int) -> list:
    """Deterministic sample of n_per_city examples (examples are already (tile,cell)-sorted)."""
    if n_per_city is None or n_per_city >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(examples)), n_per_city))
    return [examples[i] for i in idx]


def load_heldout_cells(
    release: str,
    regions: list[str],
    *,
    n_per_city: int | None,
    sample_seed: int = 1234,
    shuffle_seed: int = 5678,
    conditioning_seed: int = 0,
    max_tiles_per_city: int | None = None,
) -> list[HeldoutCell]:
    """Load held-out gap cells across cities. n_per_city None = all cells in the city.

    ``max_tiles_per_city`` (smoke only) caps how many tiles are read before sampling — the
    full run reads every held-out tile.
    """
    sampled_all = []  # CellExamples across all cities (each carries .region)
    for r_idx, region in enumerate(regions):
        tiles = _heldout_tile_ids(release, region)
        if max_tiles_per_city is not None:
            tiles = tiles[:max_tiles_per_city]
        shards = build_shards_in_memory(release, region, tile_ids=tiles)
        # ablation="full" == the bake-off training scheme (value-char-v1 full).
        examples, _ = flatten_shards_to_cells(shards, seed=conditioning_seed, ablation="full")
        # stable per-region sample seed (NOT hash(region) — PYTHONHASHSEED-robust)
        sampled_all.extend(_sample(examples, n_per_city, sample_seed * 1000 + r_idx))

    # MACRO-deranged donor: same city, DIFFERENT macro tuple (the value-bearing prefix);
    # fails loud per city if it can't (< 2 distinct macro tuples). The prefix tuple is the
    # macro identity — it differs iff the macro buckets differ (region/city constant in-city).
    cities = [ex.region for ex in sampled_all]
    macros = [tuple(ex.prefix_ids) for ex in sampled_all]
    donor = macro_deranged_donor(cities, macros, seed=shuffle_seed)
    out: list[HeldoutCell] = []
    for i, ex in enumerate(sampled_all):
        d = sampled_all[donor[i]]
        out.append(
            HeldoutCell(
                region=ex.region,
                body_tokens=list(ex.tokens),
                own_prefix=list(ex.prefix_ids),
                own_char=list(ex.character_stats),
                donor_prefix=list(d.prefix_ids),
                donor_char=list(d.character_stats),
            )
        )
    return out


def write_heldout_cache(cells: list[HeldoutCell], path: Path) -> None:
    """Persist loaded held-out cells once so the full 6-checkpoint run reuses ONE tile read.

    The cells are checkpoint-independent; a byte-identical read-back (read_heldout_cache) is the
    cached≡uncached guarantee — the optimization never changes the gap inputs.
    """
    Path(path).write_text(json.dumps([c.__dict__ for c in cells]))


def read_heldout_cache(path: Path) -> list[HeldoutCell]:
    """Reconstruct held-out cells from a cache written by write_heldout_cache (exact)."""
    return [HeldoutCell(**d) for d in json.loads(Path(path).read_text())]


def build_and_verify_cache(
    release: str,
    regions: list[str],
    *,
    n_per_city: int | None,
    cache_path: Path,
    sample_seed: int = 1234,
    shuffle_seed: int = 5678,
) -> int:
    """Load held-out cells fresh, persist, read back, and ASSERT byte-identical.

    This is the cached≡uncached gate: the cells are checkpoint-independent and the gap is a
    deterministic function of (cells, model), so a byte-identical read-back guarantees the
    cache never changes the gap inputs for ANY of the 6 checkpoints. Raises on mismatch.
    """
    fresh = load_heldout_cells(
        release,
        regions,
        n_per_city=n_per_city,
        sample_seed=sample_seed,
        shuffle_seed=shuffle_seed,
    )
    write_heldout_cache(fresh, cache_path)
    back = read_heldout_cache(cache_path)
    if back != fresh:
        raise SystemExit(
            "cache verification FAILED: read-back != fresh load — the optimization would "
            "silently change the gap inputs; refusing to run the full matrix"
        )
    return len(fresh)

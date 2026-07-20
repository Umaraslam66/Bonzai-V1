"""Manifest -> matched training-style conditioning join (realism-eval Task 1).

Maps the sealed Lane-S sampler manifest's cells to the EXACT conditioning the
model trained under: per held-out city, rebuild the training-path shards for the
manifest's tiles, flatten them through the SAME ``flatten_shards_to_cells`` the
datamodule uses (value-bearing 10-id prefix incl. the character placeholder,
ablation-applied char stats), then left-join the manifest's cells against the
flattened ``CellExample`` index by 5-tuple key — in manifest ORDER.

Import discipline: this module must import WITHOUT torch (the real shard builder
/ flattener live in ``cfm.data.training`` whose ``datamodule`` imports torch and
lightning at module top). The injectable ``shard_builder``/``flattener`` params
therefore default to ``None`` sentinels resolved by a LOCAL import only when the
caller does not inject fakes — unit tests never touch torch or Leonardo parquet.

Call shapes (content-anchored against the real modules, 2026-07-20 — the
upstream params are KEYWORD-ONLY, so the callables are invoked as):

  shard_builder(release, city, tile_ids=[(ti, tj), ...]) -> list[TrainingShard]
      (real: cfm.data.training.build_shards.build_shards_in_memory)
  flattener(shards, seed=conditioning_seed, ablation=ablation)
      -> (list[CellExample], dropped_counts)
      (real: cfm.data.training.datamodule.flatten_shards_to_cells)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cfm.eval.lane_s_sampler import load_verified_manifest

logger = logging.getLogger(__name__)

#: Pinned lineage of the ONE sealed Lane-S manifest this driver may consume
#: (GROUND_TRUTH: floor 95abb88 locked 2026-06-21; census + counts sealed
#: 2026-06-22, 5,705 cells across 146 floored strata). LOCK-AND-GUARDS-TRAVEL-
#: TOGETHER: a re-derived manifest must update these constants (and the tests
#: asserting them) in the SAME commit.
EXPECTED_FLOOR_SHA256 = "95abb88bfaf0a79d4254883478aa5e5b558ed63c27a3c0a5845e8bb65f3a6be6"
EXPECTED_CENSUS_SHA256 = "236cea99dc370021113352c9c737da2404791ad200ca6d8d7e908e81ca6cb373"
EXPECTED_N_CELLS = 5705
EXPECTED_N_STRATA = 146


class ConditioningJoinError(RuntimeError):
    """A manifest cell has no matching flattened CellExample (A2: expect none).

    Every manifest cell was sampled from the census of conditionable held-out
    cells, so an unmatched key means release/tile-set/flatten-drop drift between
    the sealed manifest and the shards being rebuilt — fail loud, never skip."""


@dataclass(frozen=True)
class ConditionedCell:
    """One manifest cell joined to its matched training-style conditioning."""

    cell_key: tuple[str, int, int, int, int]  # (city, tile_i, tile_j, cell_i, cell_j) — A2
    density_bucket: int
    prefix_ids: tuple[int, ...]  # 10 ids incl. char placeholder (A1/A6)
    char_stats: tuple[float, ...]  # 7 floats (A1)
    real_body_tokens: tuple[int, ...]  # CellExample.tokens (real cell body, for real-features)


def load_verified_manifest_or_raise(path: Path) -> dict:
    """``lane_s_sampler.load_verified_manifest`` + pinned-lineage asserts.

    The seal verification (absent/unsealed/sha-mismatch/version-skew ->
    ``SamplerArtifactError``) is delegated to the lane_s reader; THIS layer pins
    which sealed manifest it is: floor_sha256, census_sha256, cell count and
    stratum count must match the locked lineage constants above."""
    manifest = load_verified_manifest(path)
    assert manifest["floor_sha256"] == EXPECTED_FLOOR_SHA256, (
        f"manifest floor_sha256={manifest['floor_sha256']!r} != pinned "
        f"{EXPECTED_FLOOR_SHA256!r} — not the locked Lane-S lineage; refusing."
    )
    assert manifest["census_sha256"] == EXPECTED_CENSUS_SHA256, (
        f"manifest census_sha256={manifest['census_sha256']!r} != pinned "
        f"{EXPECTED_CENSUS_SHA256!r} — not the locked Lane-S lineage; refusing."
    )
    n_cells = len(manifest["cells"])
    assert n_cells == EXPECTED_N_CELLS, (
        f"manifest carries {n_cells} cells, expected {EXPECTED_N_CELLS}; refusing."
    )
    n_strata = len(manifest["strata"])
    assert n_strata == EXPECTED_N_STRATA, (
        f"manifest carries {n_strata} strata, expected {EXPECTED_N_STRATA}; refusing."
    )
    logger.info(
        "lane-s manifest verified: %d cells / %d strata, floor %.8s, census %.8s",
        n_cells,
        n_strata,
        manifest["floor_sha256"],
        manifest["census_sha256"],
    )
    return manifest


def _cell_key(cell: dict) -> tuple[str, int, int, int, int]:
    return (
        cell["city"],
        int(cell["tile_i"]),
        int(cell["tile_j"]),
        int(cell["cell_i"]),
        int(cell["cell_j"]),
    )


def build_conditioned_cells(
    manifest: dict,
    *,
    release: str,
    ablation: str,  # A3 — MUST match the checkpoint's training scheme (Task 3 reads it)
    shard_builder: Callable[..., list] | None = None,
    flattener: Callable[..., tuple[list, object]] | None = None,
    conditioning_seed: int = 0,
) -> list[ConditionedCell]:
    """Join every manifest cell to its matched training-style conditioning.

    Per held-out city named in ``manifest["cells"]``: rebuild that city's shards
    for exactly the manifest's tiles, flatten with the checkpoint's ``ablation``
    (threaded, NEVER hardcoded — A3), index the resulting examples by ``.key``,
    then left-join the manifest cells IN MANIFEST ORDER. Raises
    :class:`ConditioningJoinError` listing every manifest cell_key absent from
    the index (A2: expect none).

    ``shard_builder``/``flattener`` default to the real training-path functions
    via a lazy LOCAL import (they pull torch); tests inject fakes.
    ``conditioning_seed`` threads to the flattener's ``seed`` (constant-bucketed
    in the prefix — its value is inert in the ids; datamodule contract)."""
    if shard_builder is None:
        from cfm.data.training.build_shards import build_shards_in_memory

        shard_builder = build_shards_in_memory
    if flattener is None:
        from cfm.data.training.datamodule import flatten_shards_to_cells

        flattener = flatten_shards_to_cells

    manifest_cells: list[dict] = manifest["cells"]

    # Per-city tile sets, derived from the manifest itself (sorted -> deterministic).
    tiles_by_city: dict[str, set[tuple[int, int]]] = {}
    for cell in manifest_cells:
        tiles_by_city.setdefault(cell["city"], set()).add(
            (int(cell["tile_i"]), int(cell["tile_j"]))
        )

    index: dict[tuple[str, int, int, int, int], object] = {}
    for city in sorted(tiles_by_city):
        tile_ids = sorted(tiles_by_city[city])
        shards = shard_builder(release, city, tile_ids=tile_ids)
        examples, dropped = flattener(shards, seed=conditioning_seed, ablation=ablation)
        logger.info(
            "conditioning join: %s — %d tiles -> %d examples (dropped: %s)",
            city,
            len(tile_ids),
            len(examples),
            dropped,
        )
        for ex in examples:
            index[ex.key] = ex

    out: list[ConditionedCell] = []
    missing: list[tuple[str, int, int, int, int]] = []
    for cell in manifest_cells:
        key = _cell_key(cell)
        ex = index.get(key)
        if ex is None:
            missing.append(key)
            continue
        out.append(
            ConditionedCell(
                cell_key=key,
                density_bucket=int(cell["density_bucket"]),
                prefix_ids=tuple(ex.prefix_ids),
                char_stats=tuple(ex.character_stats),
                real_body_tokens=tuple(ex.tokens),
            )
        )
    if missing:
        raise ConditioningJoinError(
            f"{len(missing)} manifest cell(s) have no matching flattened CellExample "
            f"(release/tile/flatten-drop drift vs the sealed manifest): "
            f"{[str(k) for k in missing]}"
        )
    return out

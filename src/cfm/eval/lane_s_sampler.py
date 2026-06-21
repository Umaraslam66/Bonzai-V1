# src/cfm/eval/lane_s_sampler.py
"""Lane-S held-out CELL SAMPLER (spec 2026-06-21).

A budget-bounded stratified DOWN-sampler over the held-out cell pool. Picks
which held-out cells the bake-off backbones generate so generated-side feature
distributions clear the conditioning floor's min_n per scored stratum.

UNIT DISCIPLINE (spec, protocol §10.3): the obligation is FEATURES (>= min_n per
floored (city, metric, stratum)); the lever is CELLS (per (city, 4-tuple)). The
scarce floored metric (building_area where owed) binds n_cells.

The artifact is sha-locked write-once via cfm.data.locked_yaml, mirroring the
conditioning floor's grammar (sha excludes itself; a _LANE_S_SAMPLER_LOCKED
marker beside the file; reader refuses absent/unsealed/sha-mismatch/skew).
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from cfm.data.locked_yaml import stamp_and_seal, verify_sealed_yaml

logger = logging.getLogger(__name__)

SAMPLER_SCHEMA_VERSION = "1.0"
SAMPLER_LOCK_NAME = "_LANE_S_SAMPLER_LOCKED"
SAMPLER_SHA_FIELD = "sampler_sha256"

#: Metric token strings as the floor freezes them (conditioning_discrimination._tile_features).
#: Used by binding_metric() to identify the scarce metric that drives n_cells sizing.
BUILDING_METRIC = "building_area_m2"
ROAD_METRIC = "road_length_m"

#: LOCK-AND-GUARDS-TRAVEL-TOGETHER (spec invariant, PI 2026-06-21): floor_n is READ from THIS
#: locked floor (sha 95abb88), NEVER recomputed. The build CLI fails loud if the loaded floor's
#: sha differs (a re-derived floor could change n_a/n_b silently); Task 4's external-SoT test
#: RED-flags a change so the guard + this constant update in the SAME commit as the floor.
EXPECTED_FLOOR_SHA256 = "95abb88bfaf0a79d4254883478aa5e5b558ed63c27a3c0a5845e8bb65f3a6be6"

#: DECISION: default target = the floor's locked min_n (the obligation unit). Revisit only if
#: the floor's min_n changes (then cells re-derive automatically). Spec §6.
DEFAULT_TARGET_FEATURES = 50
#: DECISION: headroom=2.0 default (spec Gate 5 + R3: 6/119 ceiling-bound at 2.0, glasgow-
#: concentrated, #21 risk negligible). Config knob; refined after first generation. Spec §6.
DEFAULT_HEADROOM = 2.0


class SamplerArtifactError(RuntimeError):
    """The sampler manifest failed verification (absent / unsealed / tampered / skewed)."""


def seal_manifest(payload: dict, path: Path) -> None:
    """Stamp the sha, write canonical YAML ONCE, touch the lock marker."""
    stamp_and_seal(payload, path, sha_field=SAMPLER_SHA_FIELD, lock_name=SAMPLER_LOCK_NAME)


def load_verified_manifest(path: Path) -> dict:
    """Verified read; refuses absent/unsealed/sha-mismatch/version-skew (fail-closed)."""
    return verify_sealed_yaml(
        path,
        sha_field=SAMPLER_SHA_FIELD,
        lock_name=SAMPLER_LOCK_NAME,
        schema_field="sampler_schema_version",
        schema_version=SAMPLER_SCHEMA_VERSION,
        required_key="strata",
        error=SamplerArtifactError,
    )


# ---------------------------------------------------------------------------
# Sizing: scarce-metric binding + ceiling-bound (spec §6, R3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
# NOT order=True: bare sort would include density_bucket and diverge from _cell_sort_key
# (manifest order). Always sort cells via _cell_sort_key.
class SampledCell:
    """One held-out cell to condition generation on. Identity = grid coordinate (cell_i,
    cell_j) within (city, tile_i, tile_j); density_bucket is the conditioned stratum dim."""

    city: str
    tile_i: int
    tile_j: int
    cell_i: int
    cell_j: int
    density_bucket: int


@dataclass(frozen=True)
class SizingResult:
    n_cells_target: int  # raw demand = ceil(target/real_fpc * headroom)
    n_cells_selected: int  # min(raw, available_cells)
    ceiling_bound: bool  # raw > available_cells (pool exhausted at the target)


def binding_metric(owed_metrics: frozenset[str]) -> str:
    """The SCARCE floored metric that binds n_cells: building_area where owed (it emits
    ~0-1/cell vs roads ~5-15/cell), else road_length. building_area is never the sole owed
    metric (the floor guarantees building_area ⊆ road_length at the stratum level — Gate 1),
    so this is total over the floored set."""
    if BUILDING_METRIC in owed_metrics:
        return BUILDING_METRIC
    if ROAD_METRIC in owed_metrics:
        return ROAD_METRIC
    raise ValueError(f"no known floored metric in owed set {sorted(owed_metrics)}")


def size_stratum(
    *, target_features: int, headroom: float, floor_n_binding: int, available_cells: int
) -> SizingResult:
    """n_cells = ceil(target / real_fpc[binding] * headroom), real_fpc = floor_n/available.

    Algebraically raw = ceil(target * headroom * available / floor_n); `available` cancels in
    the ceiling test, so ceiling_bound <=> floor_n < target*headroom (independent of available —
    why R3 was computable from floor_n alone). floor_n must be >= 1 (a floored stratum has
    n >= the floor's min_n by construction).
    """
    if floor_n_binding < 1:
        raise ValueError(
            f"floor_n_binding must be >= 1 (got {floor_n_binding}); a floored "
            "stratum has n >= min_n by construction"
        )
    if available_cells < 1:
        raise ValueError(f"available_cells must be >= 1 (got {available_cells})")
    raw = math.ceil(target_features * headroom * available_cells / floor_n_binding)
    selected = min(raw, available_cells)
    return SizingResult(
        n_cells_target=raw,
        n_cells_selected=selected,
        ceiling_bound=raw > available_cells,
    )


# ---------------------------------------------------------------------------
# Selection: blake2b hash-rank (PYTHONHASHSEED-proof, input-order-independent)
# ---------------------------------------------------------------------------


def _cell_sort_key(c: SampledCell) -> tuple:
    """Canonical sort key for manifest output: (city, tile_i, tile_j, cell_i, cell_j).
    density_bucket excluded — identity is the grid coordinate."""
    return (c.city, c.tile_i, c.tile_j, c.cell_i, c.cell_j)


def _rank_digest(seed: int, c: SampledCell) -> str:
    """blake2b hexdigest over the cell identity + seed. stdlib hashlib is byte-stable across
    Python/numpy versions and PYTHONHASHSEED-independent, giving a total order on cells."""
    # Assumes held-out city names contain no ':' (true for the locked held-out set);
    # the digest keys on cell identity.
    raw = f"{seed}:{c.city}:{c.tile_i}:{c.tile_j}:{c.cell_i}:{c.cell_j}"
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()


def select_cells(cells: list[SampledCell], n: int, *, seed: int) -> list[SampledCell]:
    """Deterministically select <= n cells by blake2b hash-rank of the cell identity.

    stdlib hashlib is byte-stable across Python/numpy versions (a seeded numpy shuffle is
    not) and order-independent (the digest is a total order), so the result is reproducible
    for a sha-locked write-once manifest. Take-all when n >= len(cells). Output is sorted
    canonically by _cell_sort_key (not by digest) so manifest bytes are stable.

    n == 0 returns an empty list by design (no cells selected).
    """
    if n < 0:
        raise ValueError(f"select_cells: n must be >= 0 (got {n})")
    if n >= len(cells):
        chosen = list(cells)
    else:
        chosen = sorted(cells, key=lambda c: _rank_digest(seed, c))[:n]
    return sorted(chosen, key=_cell_sort_key)

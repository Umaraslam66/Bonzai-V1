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


# ---------------------------------------------------------------------------
# Floor adapter: floored targets + held-out feature counts (Task 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlooredTarget:
    """One (city, 4-tuple) targeted stratum: the set of owed floored metrics and the binding
    (scarce) metric that drives n_cells sizing. The 4-tuple dims are
    (zoning, skeleton, density_bucket, coastal)."""

    city: str
    stratum: tuple
    owed_metrics: frozenset[str]
    binding_metric: str


def floored_targets(floor_payload: dict) -> dict[tuple[str, tuple], FlooredTarget]:
    """Per (city, 4-tuple) targeted stratum: the owed floored metrics + the binding metric.

    Target set = the distinct (city, 4-tuple) carrying >= 1 floored metric row (spec Gate 1).
    Reads ``floor_payload["floors"]`` — the per-(city, metric, stratum) floor rows emitted by
    ``conditioning_floor.build_floor()``.
    """
    owed: dict[tuple[str, tuple], set[str]] = {}
    for rec in floor_payload["floors"]:
        key = (rec["city"], tuple(rec["stratum"]))
        owed.setdefault(key, set()).add(rec["metric"])
    return {
        (city, stratum): FlooredTarget(
            city=city,
            stratum=stratum,
            owed_metrics=frozenset(ms),
            binding_metric=binding_metric(frozenset(ms)),
        )
        for (city, stratum), ms in owed.items()
    }


def heldout_feature_counts(floor_payload: dict) -> dict[tuple[str, str, tuple], int]:
    """Per (held-out city, metric, stratum): the real feature count n, read from the floor's
    pair records (n_a/n_b) across BOTH families (``pairs`` + ``cross_pairs``).

    These ARE the floor's qualify counts (same extraction as ``conditioning_floor``), so
    n >= min_n for any floored stratum. Equals the optimistic gen ceiling at gen_ratio=1,
    full draw (= available_cells * real_fpc).

    Source: conditioning_floor n_a/n_b (locked floor artifact — NEVER recomputed).
    ASSUMPTION (verified against locked floor 95abb88, 2026-06-21): n is consistent for a
    given (city, metric, stratum) across all pair records in BOTH families — n =
    len(qualifying_features) is a property of the city's sample, not of any pair — so
    last-write-wins dict assignment is safe (0 conflicts confirmed over 312 keys; the 47
    cross-only keys are never floored, so Task 6's join hits no KeyError). Re-verify if a
    future floor re-derivation changes the qualify rule.
    LOCK-AND-GUARDS-TRAVEL-TOGETHER: reads THIS floor (sha EXPECTED_FLOOR_SHA256); the
    build CLI and the Task-4 SoT test enforce the sha invariant jointly.
    """
    held = set(floor_payload["held_out_cities"])
    out: dict[tuple[str, str, tuple], int] = {}
    for table in ("pairs", "cross_pairs"):
        for p in floor_payload.get(table, []):
            stratum = tuple(p["stratum"])
            for city, n in ((p["city_a"], p["n_a"]), (p["city_b"], p["n_b"])):
                if city in held:
                    out[(city, p["metric"], stratum)] = int(n)
    return out


# ---------------------------------------------------------------------------
# Census: per-cell parquet emit + read (pyarrow imports are LOCAL to keep
# the pure-logic core import-light — only materialised when census IO is used)
# ---------------------------------------------------------------------------

#: Census parquet column order. Sorted-row determinism relies on this fixed schema.
_CENSUS_COLS: tuple[str, ...] = (
    "city",
    "tile_i",
    "tile_j",
    "cell_i",
    "cell_j",
    "zoning",
    "skeleton",
    "density",
    "coastal",
)


def write_cell_census(
    cells: list[SampledCell],
    tile_strata: dict[tuple[str, int, int], tuple],
    path: Path,
) -> None:
    """Write the per-cell census parquet: one row per conditionable held-out cell, carrying
    the cell's density and its tile's (zoning, skeleton, coastal). Rows are sorted
    canonically before writing to give byte-deterministic output.

    ``tile_strata[(city, tile_i, tile_j)] = (zoning, skeleton, coastal)``
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows: list[tuple] = []
    for c in cells:
        key = (c.city, c.tile_i, c.tile_j)
        stratum_triple = tile_strata.get(key)
        if stratum_triple is None:
            raise ValueError(
                f"write_cell_census: no tile stratum recorded for cell's tile "
                f"{c.city} ({c.tile_i},{c.tile_j})"
            )
        z, sk, co = stratum_triple
        rows.append((c.city, c.tile_i, c.tile_j, c.cell_i, c.cell_j, z, sk, c.density_bucket, co))
    rows.sort()  # canonical order => byte-deterministic parquet
    col_data = {name: [r[i] for r in rows] for i, name in enumerate(_CENSUS_COLS)}
    table = pa.table({k: col_data[k] for k in _CENSUS_COLS})
    pq.write_table(table, str(path))


def read_cell_census(path: Path) -> dict[tuple[str, tuple], list[SampledCell]]:
    """Read the census back, grouped by (city, 4-tuple stratum).

    The 4-tuple is (zoning, skeleton, density_bucket, coastal) — the floor's grammar.
    """
    import pyarrow.parquet as pq

    tbl = pq.ParquetFile(str(path)).read()
    col = {n: tbl.column(n).to_pylist() for n in tbl.column_names}
    pool: dict[tuple[str, tuple], list[SampledCell]] = {}
    for i in range(tbl.num_rows):
        city = col["city"][i]
        density = int(col["density"][i])
        stratum = (col["zoning"][i], col["skeleton"][i], density, col["coastal"][i])
        cell = SampledCell(
            city,
            int(col["tile_i"][i]),
            int(col["tile_j"][i]),
            int(col["cell_i"][i]),
            int(col["cell_j"][i]),
            density,
        )
        pool.setdefault((city, stratum), []).append(cell)
    return pool


# ---------------------------------------------------------------------------
# Build orchestrator: manifest assembly (Task 6)
# ---------------------------------------------------------------------------


def build_manifest(
    *,
    floor_payload: dict,
    floor_sha256: str,
    census_sha256: str,
    cell_pool: dict[tuple[str, tuple], list[SampledCell]],
    release: str,
    seed: int,
    target_features: int = DEFAULT_TARGET_FEATURES,
    headroom: float = DEFAULT_HEADROOM,
) -> dict:
    """Assemble the (unsealed) manifest payload: size + select per floored (city, 4-tuple).

    ``census_sha256`` is the sha256 hex digest of the census parquet bytes — it pins the
    exact input cell pool unambiguously (spec §7, PI ratification 2026-06-22). Pass
    ``compute_sha256(args.census.read_bytes())`` from the build CLI.

    NOTE: the sha guard (floor_sha == EXPECTED_FLOOR_SHA256) lives in the BUILD CLI, NOT here.
    Fixture tests pass floor_sha256="abc123" and census_sha256="c0ffee" and must work without
    hitting the guard.
    """
    targets = floored_targets(floor_payload)
    counts = heldout_feature_counts(floor_payload)
    strata_records: list[dict] = []
    all_cells: list[SampledCell] = []
    # DECISION: str()-based sort. density_bucket is single-digit (0-3, locked by the 4-bucket
    # schema in conditioning_floor); str("0") < str("1") < ... matches integer order for single
    # digits, so this is safe. Re-check if density buckets ever exceed 9 (str("10") < str("9")).
    for city, stratum in sorted(targets, key=lambda k: (k[0], tuple(map(str, k[1])))):
        t = targets[(city, stratum)]
        available = cell_pool.get((city, stratum), [])
        if not available:
            logger.warning(
                "lane-s sampler: no census cells for floored stratum %s %s; skipping "
                "(census/floor lineage mismatch)",
                city,
                stratum,
            )
            continue
        floor_n = counts[(city, t.binding_metric, stratum)]
        sizing = size_stratum(
            target_features=target_features,
            headroom=headroom,
            floor_n_binding=floor_n,
            available_cells=len(available),
        )
        chosen = select_cells(available, sizing.n_cells_selected, seed=seed)
        all_cells.extend(chosen)
        strata_records.append(
            {
                "city": city,
                "stratum": list(stratum),
                "owed_metrics": sorted(t.owed_metrics),
                "binding_metric": t.binding_metric,
                "floor_n_binding": floor_n,
                "available_cells": len(available),
                "real_fpc_binding": floor_n / len(available),
                "n_cells_target": sizing.n_cells_target,
                "n_cells_selected": sizing.n_cells_selected,
                "ceiling_bound": sizing.ceiling_bound,
            }
        )
    all_cells.sort(key=_cell_sort_key)
    return {
        "sampler_schema_version": SAMPLER_SCHEMA_VERSION,
        "release": release,
        "floor_sha256": floor_sha256,
        "census_sha256": census_sha256,
        "methodology": {
            "target_features": target_features,
            "headroom": headroom,
            "seed": seed,
            "selection": "blake2b_hash_rank",
            "sizing": "ceil(target_features * headroom * available / floor_n_binding)",
            "binding_rule": "scarce_floored_metric_building_area_else_road_length",
            "real_fpc_source": "heldout_floor_n_a_n_b_div_census_cells",
            "gen_ratio": "training_city_informed_proxy_validated_at_first_generation",
        },
        "held_out_cities": sorted(floor_payload["held_out_cities"]),
        "strata": strata_records,
        "cells": [
            {
                "city": c.city,
                "tile_i": c.tile_i,
                "tile_j": c.tile_j,
                "cell_i": c.cell_i,
                "cell_j": c.cell_j,
                "density_bucket": c.density_bucket,
            }
            for c in all_cells
        ],
    }


# ---------------------------------------------------------------------------
# Consumer-side coverage check — §9 ceiling-bound split (Task 7)
# ---------------------------------------------------------------------------


class SamplerCoverageError(RuntimeError):
    """A floored (metric, stratum) is below min_n on the GENERATED side WITHOUT being
    ceiling-bound — a sampler sizing / headroom bug, never hidden behind the ceiling
    exclusion (spec Gate 5 / protocol §9 regime-distinguishing guard)."""


@dataclass(frozen=True)
class CoverageReport:
    """Result of verify_gen_coverage: three disjoint buckets over floored (city, metric, stratum).

    ok                   — achieved >= min_n on the generated side.
    ceiling_bound_excluded — short on the binding metric AND the stratum was ceiling-bound:
                             a data limit (mirrors floor's 'report, do NOT coarsen'); these
                             strata are dropped from Lane-S scoring (#21 demotion downstream).
    unexpected_short     — always empty on successful return; we raise SamplerCoverageError
                           first, so this field is reserved for future batch-collect mode.
    """

    ok: list[tuple[str, str, tuple]]
    ceiling_bound_excluded: list[tuple[str, str, tuple]]
    unexpected_short: list[tuple[str, str, tuple]]


def verify_gen_coverage(
    gen_by_city: dict[str, dict[tuple[str, tuple], list]],
    manifest: dict,
    *,
    min_n: int | None = None,
) -> CoverageReport:
    """Per floored (city, metric, stratum) in the manifest: assert achieved gen features >=
    min_n on the ACTUAL generated set (spec Gate 5, protocol §10.3 correct unit).

    §9 split: a short metric that is the binding metric of a CEILING-BOUND stratum is a data
    limit -> exclude-and-report (mirrors the floor's 'report, do NOT coarsen'; #21 demotion +
    SECOND_REGION downstream). Any other short -> FAIL LOUD (sampler under-sized).

    ``gen_by_city[city][(metric, stratum)]`` is the list of generated features for that slot.
    ``min_n`` defaults to ``manifest["methodology"]["target_features"]``.
    """
    min_n = manifest["methodology"]["target_features"] if min_n is None else min_n
    ok: list[tuple[str, str, tuple]] = []
    excluded: list[tuple[str, str, tuple]] = []
    for s in manifest["strata"]:
        city = s["city"]
        stratum = tuple(s["stratum"])
        binding = s["binding_metric"]
        ceiling = bool(s["ceiling_bound"])
        for metric in s["owed_metrics"]:
            key = (city, metric, stratum)
            achieved = len(gen_by_city.get(city, {}).get((metric, stratum), []))
            if achieved >= min_n:
                ok.append(key)
            elif metric == binding and ceiling:
                # Construction-identity exclusion (spec §9): the binding metric is short
                # because the pool was exhausted (ceiling-bound), not because of under-sizing.
                # Report it; do NOT coarsen; downstream demotes this stratum (#21).
                logger.warning(
                    "lane-s coverage: ceiling-bound exclusion %s — achieved %d < min_n=%d "
                    "(pool exhausted at generation; binding=%s, ceiling_bound=True)",
                    key,
                    achieved,
                    min_n,
                    binding,
                )
                excluded.append(key)
            else:
                # NOT ceiling-bound for this metric: the sampler should have provided enough
                # cells but didn't. This is a sizing/headroom bug — fail loud so it cannot
                # be silently hidden by a symptom-keyed "skip if thin" guard.
                #
                # NOTE (spec gap, do not 'fix' silently): a ceiling-bound stratum's
                # NON-binding metric falling short is ALSO a data limit (the cell pool
                # was exhausted — all cells taken), yet this else-branch currently raises
                # it as a sampler bug. UNREACHABLE under R3 (0/119 ceiling-bound at
                # headroom <= 2.0; roads are plentiful), so safe for the first generation.
                # REVISIT this branch before increasing headroom > 2.0 or applying to a
                # corpus where R3's zero-ceiling-bound guarantee does not hold — the
                # binding-vs-non-binding semantics in a ceiling-bound stratum are a PI
                # decision (widen the exclusion to any-metric-when-ceiling vs keep loud
                # as a model-pathology signal).
                raise SamplerCoverageError(
                    f"lane-s coverage: {key} has {achieved} gen features < min_n={min_n} but the "
                    f"stratum is not ceiling-bound for this metric (binding={binding}, "
                    f"ceiling_bound={ceiling}) — the sampler under-sized it; re-derive headroom, "
                    "do not exclude. (spec Gate 5 / protocol §9)"
                )
    return CoverageReport(ok=ok, ceiling_bound_excluded=excluded, unexpected_short=[])

"""Sub-D frequency analysis artifacts (spec section 6, Task 6).

Produces reviewer-facing proposal sections from per-tile sub-C inputs:

- ``zoning_proposal`` — feature-class distribution over active cells, with a
  ``candidate_strategies`` series (each entry carries its full bucket
  definition + coverage + marginal_cost).
- ``cell_density_proposal`` — building-footprint-ratio bucketing over active
  cells, with a ``candidate_strategies`` series.
- ``road_skeleton_proposal`` — road-crossing-count bucketing over **active
  internal edges only**. Edge scope is recomputed per-tile from sub-C
  ``cells.parquet`` (via ``derive_internal_edge_scope`` on each endpoint
  cell's scope) and the distribution restricts to ``Scope.ACTIVE`` edges so
  vocab candidates are not contaminated by ``SCOPE_BOUNDARY`` or
  ``FULLY_MASKED`` slots.
- ``zoning_orthogonality`` — Pearson correlation between per-cell zoning
  signal (building count) and density signal (building footprint ratio) so
  reviewers can judge whether the two evidence streams encode redundant
  information.

Outputs are byte-deterministic: same inputs -> same dict and same
``canonicalize_yaml`` bytes. Use ``write_frequency_analysis`` to serialise.
Use ``validate_frequency_analysis`` to enforce shape invariants before
handing the artifact to reviewers.

Gate 2 reviewer workflow
------------------------

The proposal YAML is a reviewable document. The Gate 2 flow is:

1. Task 7 writes ``reports/phase-1-sub-D/macro_vocab_proposal.yaml`` with
   ``locked_buckets`` pre-filled to the most-granular candidate (the default
   recommendation).
2. The reviewer inspects ``candidate_strategies`` in each section to see the
   alternative bucketings (categories, coverage, marginal_cost,
   bucket_boundaries/bucket_lower_bounds). If the reviewer prefers a
   different strategy, they hand-edit ``locked_buckets`` in the proposal
   YAML to match one of the candidates' bucket definitions.
3. Reviewer's edit is auditable via ``git diff`` on the proposal file.
4. Task 8's ``scripts/promote_macro_vocab.py`` consumes the edited proposal
   and writes ``configs/macro_plan/v1/macro_plan_vocab.yaml`` with
   byte-identity to the proposal modulo only the ``status: proposal`` ->
   ``status: locked`` marker. No other edits between proposal and locked
   are permitted; the byte-identity test in Task 8 enforces this.

Monotonicity of ``marginal_cost`` is the typical heavy-tail pattern but not
universal (bimodal evidence distributions can violate it).
``validate_frequency_analysis`` reports the values without enforcing
monotonicity; a non-monotonic series is a signal for the Gate 2 reviewer to
investigate, not a hard failure.
"""

from __future__ import annotations

import math
from pathlib import Path

from cfm.data.io import canonicalize_yaml
from cfm.data.sub_d.enums import FeatureClass, Scope
from cfm.data.sub_d.errors import SubDValidationError
from cfm.data.sub_d.evidence import (
    DERIVATION_VERSION,
    derive_cell_scope_metrics,
    derive_density_evidence,
    derive_road_skeleton_evidence,
    derive_zoning_evidence,
)
from cfm.data.sub_d.lattice import (
    derive_internal_edge_scope,
    iter_internal_edge_slots,
)
from cfm.data.sub_d.sub_c_reader import SubCTileInputs

ANALYSIS_VERSION: str = "1.0"

# Density bucket boundaries proposed for review. Open lower bound is 0.0; the
# rest are explicit cut points. The proposal lists strategies with 2-5 buckets
# so the marginal-cost-of-cut sequence shows a non-trivial elbow.
_DENSITY_CANDIDATE_BUCKETS: list[list[float]] = [
    [0.0, 0.1, 0.3, 0.5, 1.0],  # 4 buckets
    [0.0, 0.1, 0.3, 1.0],        # 3 buckets
    [0.0, 0.3, 1.0],             # 2 buckets
    [0.0, 1.0],                  # 1 bucket
]

# Road skeleton candidate bucketings on road_crossing_count.
_ROAD_CANDIDATE_BUCKETS: list[list[int]] = [
    [0, 1, 3, 6],  # 4 open-ended buckets: [0], [1,2], [3,5], [6,inf)
    [0, 1, 3],     # 3 buckets
    [0, 1],        # 2 buckets
    [0],           # 1 bucket
]


def build_frequency_analysis(inputs: list[SubCTileInputs]) -> dict:
    """Compute the reviewer-facing frequency analysis dict.

    Determinism: iterates ``inputs`` in their argument order for per-tile
    aggregation, but every emitted list is sorted on a canonical key before
    return. The output dict is intended for ``canonicalize_yaml``.
    """
    zoning_counts: dict[str, int] = {
        FeatureClass.ROAD.name.lower(): 0,
        FeatureClass.BUILDING.name.lower(): 0,
        FeatureClass.POI.name.lower(): 0,
        FeatureClass.BASE.name.lower(): 0,
    }
    density_values: list[float] = []
    road_counts_active: list[int] = []

    # Per-cell zoning signal (building count) and density signal aligned for
    # the orthogonality comparison.
    ortho_building_counts: list[float] = []
    ortho_density_ratios: list[float] = []

    edge_scope_counts: dict[str, int] = {
        Scope.ACTIVE.name.lower(): 0,
        Scope.SCOPE_BOUNDARY.name.lower(): 0,
        Scope.FULLY_MASKED.name.lower(): 0,
    }

    for tile in inputs:
        scope = derive_cell_scope_metrics(tile.cells)
        zoning_metrics = derive_zoning_evidence(tile.features, tile.cells)
        density_metrics = derive_density_evidence(tile.features, tile.cells)
        road_metrics = derive_road_skeleton_evidence(tile.crossings, tile.features)

        for m in zoning_metrics:
            cls = m.metric_name.removeprefix("feature_count_")
            zoning_counts[cls] += int(m.value)

        # Build a per-cell building count for orthogonality (active cells only).
        building_count_by_cell: dict[tuple[int, int], int] = {}
        for m in zoning_metrics:
            if m.metric_name == "feature_count_building":
                # slot_index = cell_i * 8 + cell_j
                ci, cj = divmod(m.slot_index, 8)
                building_count_by_cell[(ci, cj)] = int(m.value)

        for m in density_metrics:
            density_values.append(float(m.value))
            ci, cj = divmod(m.slot_index, 8)
            ortho_density_ratios.append(float(m.value))
            ortho_building_counts.append(float(building_count_by_cell.get((ci, cj), 0)))

        # Road skeleton: filter to ACTIVE edges using each endpoint's scope.
        # Cache (slot_index -> EdgeSlot) once for the join.
        edge_lookup = {s.slot_index: s for s in iter_internal_edge_slots()}
        for m in road_metrics:
            edge = edge_lookup[m.slot_index]
            lower_active = scope[(edge.lower_cell_i, edge.lower_cell_j)]
            if edge.axis == 0:
                upper_active = scope[(edge.lower_cell_i + 1, edge.lower_cell_j)]
            else:
                upper_active = scope[(edge.lower_cell_i, edge.lower_cell_j + 1)]
            edge_scope = derive_internal_edge_scope(lower_active, upper_active)
            edge_scope_counts[edge_scope.name.lower()] += 1
            if edge_scope == Scope.ACTIVE:
                road_counts_active.append(int(m.value))

    zoning_proposal = _zoning_proposal_section(zoning_counts)
    density_proposal = _density_proposal_section(density_values)
    road_proposal = _road_proposal_section(road_counts_active, edge_scope_counts)
    orthogonality = _orthogonality_section(ortho_building_counts, ortho_density_ratios)

    return {
        "analysis_version": ANALYSIS_VERSION,
        "derivation_version": DERIVATION_VERSION,
        "tile_count": len(inputs),
        "input_digests": sorted(
            [{"tile_i": t.paths.tile_i, "tile_j": t.paths.tile_j, **t.digests} for t in inputs],
            key=lambda e: (e["tile_i"], e["tile_j"]),
        ),
        "zoning_proposal": zoning_proposal,
        "cell_density_proposal": density_proposal,
        "road_skeleton_proposal": road_proposal,
        "zoning_orthogonality": orthogonality,
    }


def write_frequency_analysis(analysis: dict, path: Path) -> None:
    """Serialise *analysis* to *path* using the neutral canonical YAML helper."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonicalize_yaml(analysis), encoding="utf-8")


def validate_frequency_analysis(analysis: dict) -> None:
    """Raise ``SubDValidationError`` on missing sections or empty proposals.

    The check enforces the non-empty-locked-buckets invariant so a section
    that failed to compute any cuts cannot reach reviewers (or worse, Gate 2)
    silently.
    """
    required_top = {
        "analysis_version",
        "derivation_version",
        "tile_count",
        "input_digests",
        "zoning_proposal",
        "cell_density_proposal",
        "road_skeleton_proposal",
        "zoning_orthogonality",
    }
    missing = required_top - analysis.keys()
    if missing:
        raise SubDValidationError(
            f"frequency analysis missing required top-level sections: {sorted(missing)}"
        )
    for section_name in ("zoning_proposal", "cell_density_proposal", "road_skeleton_proposal"):
        section = analysis[section_name]
        if not section.get("locked_buckets"):
            raise SubDValidationError(
                f"frequency analysis section {section_name!r} has empty locked_buckets; "
                "every proposal must recommend at least one bucket/token for review"
            )
        if not section.get("candidate_strategies"):
            raise SubDValidationError(
                f"frequency analysis section {section_name!r} has empty "
                "candidate_strategies series"
            )


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _zoning_proposal_section(zoning_counts: dict[str, int]) -> dict:
    sorted_pairs = sorted(zoning_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    sorted_counts = [c for _, c in sorted_pairs]
    sorted_names = [n for n, _ in sorted_pairs]
    candidate_strategies = _zoning_candidate_strategies(sorted_counts, sorted_names)
    # Default locked_buckets: the most-granular strategy (top entry of the
    # candidate list). Reviewer may overwrite this at Gate 2 by copying any
    # other candidate's tokens.
    locked_buckets = [
        {"token_id": idx, "token_name": name, "count": zoning_counts[name]}
        for idx, name in enumerate(sorted_names)
    ]
    return {
        "feature_class_distribution": dict(sorted(zoning_counts.items())),
        "locked_buckets": locked_buckets,
        "candidate_strategies": candidate_strategies,
    }


def _density_proposal_section(values: list[float]) -> dict:
    distribution = _summarise_distribution(values)
    candidate_strategies: list[dict] = []
    for buckets in _DENSITY_CANDIDATE_BUCKETS:
        counts_per_bucket = _bucket_count_floats(values, buckets)
        coverage = (
            sum(counts_per_bucket) / len(values) if values else 1.0
        )
        candidate_strategies.append(
            {
                "strategy": f"{len(counts_per_bucket)}_buckets",
                "categories": len(counts_per_bucket),
                "bucket_boundaries": list(buckets),
                "bucket_counts": counts_per_bucket,
                "coverage": float(coverage),
                "marginal_cost": None,
            }
        )
    _fill_marginal_cost(candidate_strategies)
    # Recommend the most granular (least-aggressive cut) as the locked default.
    locked_buckets = [
        {"token_id": idx, "lower_inclusive": lo, "upper_exclusive": hi}
        for idx, (lo, hi) in enumerate(
            zip(_DENSITY_CANDIDATE_BUCKETS[0], _DENSITY_CANDIDATE_BUCKETS[0][1:])
        )
    ]
    return {
        "ratio_distribution": distribution,
        "locked_buckets": locked_buckets,
        "candidate_strategies": candidate_strategies,
    }


def _road_proposal_section(
    active_counts: list[int],
    edge_scope_counts: dict[str, int],
) -> dict:
    candidate_strategies: list[dict] = []
    for lower_bounds in _ROAD_CANDIDATE_BUCKETS:
        counts_per_bucket = _bucket_count_ints(active_counts, lower_bounds)
        coverage = (
            sum(counts_per_bucket) / len(active_counts) if active_counts else 1.0
        )
        candidate_strategies.append(
            {
                "strategy": f"{len(counts_per_bucket)}_buckets",
                "categories": len(counts_per_bucket),
                "bucket_lower_bounds": list(lower_bounds),
                "bucket_counts": counts_per_bucket,
                "coverage": float(coverage),
                "marginal_cost": None,
            }
        )
    _fill_marginal_cost(candidate_strategies)
    locked_buckets = [
        {
            "token_id": idx,
            "lower_inclusive": lo,
            "upper_exclusive": (hi if hi is not None else None),
        }
        for idx, (lo, hi) in enumerate(_open_ended_int_pairs(_ROAD_CANDIDATE_BUCKETS[0]))
    ]
    return {
        "active_edge_count": len(active_counts),
        "edge_scope_distribution": dict(sorted(edge_scope_counts.items())),
        "count_distribution": _summarise_distribution([float(c) for c in active_counts]),
        "locked_buckets": locked_buckets,
        "candidate_strategies": candidate_strategies,
    }


def _orthogonality_section(
    building_counts: list[float], density_ratios: list[float]
) -> dict:
    correlation = _pearson(building_counts, density_ratios)
    return {
        "building_count_vs_density_ratio": {
            "correlation": float(correlation),
            "sample_size": len(building_counts),
            "note": (
                "Pearson correlation between per-active-cell building-count "
                "(zoning evidence) and building_footprint_ratio (density "
                "evidence). High |correlation| means the two evidence streams "
                "encode redundant signals; reviewer should consider whether "
                "both deserve separate vocab axes."
            ),
        }
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zoning_candidate_strategies(
    sorted_counts: list[int], sorted_names: list[str]
) -> list[dict]:
    """Zoning strategies dropping one category at a time from rarest to most common.

    Returns a list ordered from most-categories (least-aggressive cut) to
    fewest-categories (most-aggressive cut). Each entry carries the kept and
    merged token names so the reviewer can read the bucket definition
    directly without consulting source code. ``marginal_cost`` is the
    coverage loss per category dropped relative to the prior strategy.
    """
    total = sum(sorted_counts)
    entries: list[dict] = []
    for k in range(len(sorted_counts), 0, -1):
        covered = sum(sorted_counts[:k])
        coverage = covered / total if total > 0 else 1.0
        entries.append(
            {
                "strategy": f"top_{k}_categories",
                "categories": k,
                "kept_tokens": list(sorted_names[:k]),
                "merged_tokens": list(sorted_names[k:]),
                "coverage": float(coverage),
                "marginal_cost": None,
            }
        )
    _fill_marginal_cost(entries)
    return entries


def _fill_marginal_cost(entries: list[dict]) -> None:
    for i in range(1, len(entries)):
        prev = entries[i - 1]
        curr = entries[i]
        delta_cov = prev["coverage"] - curr["coverage"]
        delta_cat = prev["categories"] - curr["categories"]
        curr["marginal_cost"] = (
            float(delta_cov / delta_cat) if delta_cat > 0 else 0.0
        )


def _summarise_distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": float(ordered[0]),
        "p50": float(_percentile(ordered, 0.50)),
        "p90": float(_percentile(ordered, 0.90)),
        "p99": float(_percentile(ordered, 0.99)),
        "max": float(ordered[-1]),
    }


def _percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    idx = int(round(q * (len(ordered) - 1)))
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def _bucket_count_floats(values: list[float], bucket_edges: list[float]) -> list[int]:
    """Count values per half-open bucket [edges[i], edges[i+1])."""
    counts = [0] * (len(bucket_edges) - 1)
    if not counts:
        return counts
    for v in values:
        for i in range(len(bucket_edges) - 1):
            lo = bucket_edges[i]
            hi = bucket_edges[i + 1]
            in_bucket = lo <= v < hi if i < len(bucket_edges) - 2 else lo <= v <= hi
            if in_bucket:
                counts[i] += 1
                break
    return counts


def _bucket_count_ints(values: list[int], bucket_lower_bounds: list[int]) -> list[int]:
    """Count integers per open-ended bucket.

    bucket_lower_bounds = [a0, a1, ..., a_{n-1}] -> buckets
    [a0, a1), [a1, a2), ..., [a_{n-1}, +inf).
    """
    n = len(bucket_lower_bounds)
    counts = [0] * n
    for v in values:
        for i in range(n):
            lo = bucket_lower_bounds[i]
            hi = bucket_lower_bounds[i + 1] if i + 1 < n else None
            if v >= lo and (hi is None or v < hi):
                counts[i] += 1
                break
    return counts


def _open_ended_int_pairs(bucket_lower_bounds: list[int]) -> list[tuple[int, int | None]]:
    pairs: list[tuple[int, int | None]] = []
    n = len(bucket_lower_bounds)
    for i in range(n):
        lo = bucket_lower_bounds[i]
        hi = bucket_lower_bounds[i + 1] if i + 1 < n else None
        pairs.append((lo, hi))
    return pairs


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)

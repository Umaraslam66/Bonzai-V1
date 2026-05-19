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

from cfm.data.determinism import compute_sha256
from cfm.data.io import canonicalize_yaml
from cfm.data.sub_d.enums import FeatureClass, Scope
from cfm.data.sub_d.errors import SubDValidationError
from cfm.data.sub_d.evidence import (
    CELL_DENSITY_DERIVATION_VERSION,
    ROAD_SKELETON_DERIVATION_VERSION,
    TILE_POPULATION_DENSITY_DERIVATION_VERSION,
    ZONING_DERIVATION_VERSION,
    derive_cell_scope_metrics,
    derive_density_evidence,
    derive_road_skeleton_evidence,
    derive_tile_population_density_evidence,
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

    # Per-tile rows of tile_population_density proxies keyed by proxy name.
    tile_pop_density_by_proxy: dict[str, list[float]] = {}

    for tile in inputs:
        scope = derive_cell_scope_metrics(tile.cells)
        zoning_metrics = derive_zoning_evidence(tile.features, tile.cells)
        density_metrics = derive_density_evidence(tile.features, tile.cells)
        road_metrics = derive_road_skeleton_evidence(tile.crossings, tile.features)
        tile_pop_density_metrics = derive_tile_population_density_evidence(
            tile.cells, tile.features
        )
        for m in tile_pop_density_metrics:
            tile_pop_density_by_proxy.setdefault(m.metric_name, []).append(float(m.value))

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
    tile_pop_density_proposal = _tile_population_density_proposal_section(
        tile_pop_density_by_proxy
    )
    orthogonality = _orthogonality_section(ortho_building_counts, ortho_density_ratios)
    per_tile = _per_tile_evidence_summary(inputs)

    return {
        "analysis_version": ANALYSIS_VERSION,
        "derivation_versions": {
            "zoning": ZONING_DERIVATION_VERSION,
            "cell_density": CELL_DENSITY_DERIVATION_VERSION,
            "tile_population_density": TILE_POPULATION_DENSITY_DERIVATION_VERSION,
            "road_skeleton": ROAD_SKELETON_DERIVATION_VERSION,
        },
        "tile_count": len(inputs),
        "input_digests": sorted(
            [{"tile_i": t.paths.tile_i, "tile_j": t.paths.tile_j, **t.digests} for t in inputs],
            key=lambda e: (e["tile_i"], e["tile_j"]),
        ),
        "per_tile_evidence": per_tile,
        "zoning_proposal": zoning_proposal,
        "cell_density_proposal": density_proposal,
        "tile_population_density_proposal": tile_pop_density_proposal,
        "road_skeleton_proposal": road_proposal,
        "zoning_orthogonality": orthogonality,
    }


def write_frequency_analysis(analysis: dict, path: Path) -> None:
    """Serialise *analysis* to *path* using the neutral canonical YAML helper.

    Default-mode write (non-proposal). For the Gate 2 reviewer-facing 4+1
    artifact layout, use :func:`write_proposal_artifacts`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonicalize_yaml(analysis), encoding="utf-8")


#: Filename mapping from internal section key -> on-disk namespace file.
#: The four namespace files plus one index file are the Gate 2 layout.
NAMESPACE_ARTIFACT_FILENAMES: dict[str, str] = {
    "zoning_proposal": "zoning_analysis.yaml",
    "cell_density_proposal": "cell_density_analysis.yaml",
    "tile_population_density_proposal": "tile_population_density_analysis.yaml",
    "road_skeleton_proposal": "road_skeleton_analysis.yaml",
}

INDEX_ARTIFACT_FILENAME: str = "macro_vocab_proposal.yaml"


def write_proposal_artifacts(
    analysis: dict,
    output_dir: Path,
    *,
    layer3_subset: list[dict] | None = None,
    status: str = "proposal",
) -> dict:
    """Split *analysis* into 4 namespace files + 1 index file.

    The four namespace files (``zoning_analysis.yaml``,
    ``cell_density_analysis.yaml``,
    ``tile_population_density_analysis.yaml``,
    ``road_skeleton_analysis.yaml``) carry distributions, candidate
    strategies, and (where applicable) candidate proxies. They are
    content-pinned via sha256 references in the index. The reviewer does
    not edit namespace files at Gate 2 — if a different cut is wanted,
    the reviewer edits ``locked_buckets`` in the index.

    The index file (``macro_vocab_proposal.yaml``) carries the
    reviewer-editable bits: ``status``, ``locked_buckets`` per namespace,
    ``locked_proxy`` for tile_population_density, ``namespace_files``
    (name + sha256), ``per_tile_evidence``, ``zoning_orthogonality``,
    ``input_digests``, and (when supplied) ``selected_layer3_tiles``.
    Task 8's promote-script flips ``status: proposal`` ->
    ``status: locked`` with byte-identity to the rest of the file.

    Returns the index dict for inspection.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build and write the four namespace files first; their sha256s feed
    # the index. We strip ``locked_buckets``/``locked_proxy`` from each
    # namespace file because those reviewer-editable choices live in the
    # index — keeping them out of the digest-pinned content prevents drift
    # between locked-buckets edits and the namespace-file sha.
    namespace_file_records: list[dict] = []
    for section_key, filename in NAMESPACE_ARTIFACT_FILENAMES.items():
        section = analysis[section_key]
        namespace_content = _build_namespace_file_content(section_key, section, analysis)
        text = canonicalize_yaml(namespace_content)
        (output_dir / filename).write_text(text, encoding="utf-8")
        namespace_file_records.append(
            {
                "filename": filename,
                "section_key": section_key,
                "sha256": compute_sha256(text.encode("utf-8")),
            }
        )
    namespace_file_records.sort(key=lambda r: r["filename"])

    index = _build_index_content(
        analysis=analysis,
        namespace_file_records=namespace_file_records,
        layer3_subset=layer3_subset,
        status=status,
    )
    index_path = output_dir / INDEX_ARTIFACT_FILENAME
    index_path.write_text(canonicalize_yaml(index), encoding="utf-8")

    return index


def _build_namespace_file_content(
    section_key: str, section: dict, analysis: dict
) -> dict:
    """Strip reviewer-editable fields from a namespace section.

    Namespace files are content-pinned; the reviewer must not edit them
    between Task 7's CLI run and Task 8's promote. Reviewer-editable
    fields (``locked_buckets``, ``locked_proxy``) move to the index.
    """
    namespace_name = section_key.removesuffix("_proposal")
    derivation_version = analysis.get("derivation_versions", {}).get(namespace_name)
    pruned = {k: v for k, v in section.items() if k not in {"locked_buckets", "locked_proxy"}}
    return {
        "analysis_version": analysis["analysis_version"],
        "namespace": namespace_name,
        "derivation_version": derivation_version,
        "input_digests": analysis["input_digests"],
        "tile_count": analysis["tile_count"],
        **pruned,
    }


def _build_index_content(
    *,
    analysis: dict,
    namespace_file_records: list[dict],
    layer3_subset: list[dict] | None,
    status: str,
) -> dict:
    """Build the Gate 2 reviewer-facing index file.

    Locked-bucket choices live here (per namespace + locked_proxy for
    tile_population_density). Reviewer edits this file; Task 8's promote
    flips the ``status`` marker only.
    """
    locked_buckets_by_namespace: dict[str, list[dict]] = {}
    locked_proxy_by_namespace: dict[str, str | None] = {}
    for section_key in NAMESPACE_ARTIFACT_FILENAMES:
        section = analysis[section_key]
        namespace_name = section_key.removesuffix("_proposal")
        locked_buckets_by_namespace[namespace_name] = list(section.get("locked_buckets", []))
        if "locked_proxy" in section:
            locked_proxy_by_namespace[namespace_name] = section["locked_proxy"]

    index = {
        "status": status,
        "analysis_version": analysis["analysis_version"],
        "derivation_versions": dict(analysis["derivation_versions"]),
        "tile_count": analysis["tile_count"],
        "input_digests": list(analysis["input_digests"]),
        "per_tile_evidence": list(analysis["per_tile_evidence"]),
        "zoning_orthogonality": dict(analysis["zoning_orthogonality"]),
        "namespace_files": namespace_file_records,
        "locked_buckets": locked_buckets_by_namespace,
        "locked_proxy": locked_proxy_by_namespace,
        # Append-only-within-phase flag per dynamic vocab (spec §11.7). Sub-D
        # never reorders or deletes tokens within a phase; new tokens are
        # appended at the end. The flag is constant ``true`` here for every
        # reviewer-locked enum; promote_macro_vocab carries it verbatim from
        # proposal to locked artifact.
        "append_only_within_phase": {
            "cell_density": True,
            "road_skeleton": True,
            "tile_population_density": True,
            "zoning": True,
        },
    }
    if layer3_subset is not None:
        index["selected_layer3_tiles"] = list(layer3_subset)
    return index


def validate_proposal_index(index: dict) -> None:
    """Validate the Gate 2 index file shape (separate from
    :func:`validate_frequency_analysis` which validates the consolidated
    in-memory analysis dict).
    """
    required = {
        "status",
        "analysis_version",
        "derivation_versions",
        "tile_count",
        "input_digests",
        "per_tile_evidence",
        "zoning_orthogonality",
        "namespace_files",
        "locked_buckets",
        "locked_proxy",
    }
    missing = required - index.keys()
    if missing:
        raise SubDValidationError(
            f"proposal index missing required keys: {sorted(missing)}"
        )
    if index["status"] not in {"proposal", "locked"}:
        raise SubDValidationError(
            f"proposal index status must be 'proposal' or 'locked'; got {index['status']!r}"
        )
    # Every namespace must have non-empty locked_buckets in the index.
    for namespace_name in (
        "zoning",
        "cell_density",
        "tile_population_density",
        "road_skeleton",
    ):
        if not index["locked_buckets"].get(namespace_name):
            raise SubDValidationError(
                f"proposal index has empty locked_buckets for namespace {namespace_name!r}"
            )
    # tile_population_density also needs a locked_proxy.
    if not index["locked_proxy"].get("tile_population_density"):
        raise SubDValidationError(
            "proposal index has empty locked_proxy for tile_population_density"
        )


def validate_frequency_analysis(analysis: dict) -> None:
    """Raise ``SubDValidationError`` on missing sections or empty proposals.

    The check enforces the non-empty-locked-buckets invariant so a section
    that failed to compute any cuts cannot reach reviewers (or worse, Gate 2)
    silently.
    """
    required_top = {
        "analysis_version",
        "derivation_versions",
        "tile_count",
        "input_digests",
        "per_tile_evidence",
        "zoning_proposal",
        "cell_density_proposal",
        "tile_population_density_proposal",
        "road_skeleton_proposal",
        "zoning_orthogonality",
    }
    missing = required_top - analysis.keys()
    if missing:
        raise SubDValidationError(
            f"frequency analysis missing required top-level sections: {sorted(missing)}"
        )
    # Sections that carry a flat top-level candidate_strategies list.
    for section_name in (
        "zoning_proposal",
        "cell_density_proposal",
        "road_skeleton_proposal",
    ):
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
    # tile_population_density carries one candidate_strategies list PER PROXY
    # (under candidate_proxies[]), plus a locked_proxy + locked_buckets. There
    # is no flat top-level candidate_strategies; consumers look up the locked
    # proxy in candidate_proxies[] to get its strategies.
    tpd = analysis["tile_population_density_proposal"]
    if not tpd.get("locked_buckets"):
        raise SubDValidationError(
            "tile_population_density_proposal has empty locked_buckets"
        )
    if not tpd.get("locked_proxy"):
        raise SubDValidationError(
            "tile_population_density_proposal has empty locked_proxy; "
            "reviewer must lock a proxy name (e.g. mean_building_footprint_ratio)"
        )
    if not tpd.get("candidate_proxies"):
        raise SubDValidationError(
            "tile_population_density_proposal has empty candidate_proxies; "
            "Layer 1 must emit at least one proxy"
        )
    for entry in tpd["candidate_proxies"]:
        if not entry.get("candidate_strategies"):
            raise SubDValidationError(
                f"tile_population_density_proposal proxy {entry.get('proxy_name')!r} "
                "has empty candidate_strategies series"
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
        {
            "token_id": idx,
            "token_name": f"bucket_{idx}",
            "lower_inclusive": lo,
            "upper_exclusive": hi,
        }
        for idx, (lo, hi) in enumerate(
            zip(_DENSITY_CANDIDATE_BUCKETS[0], _DENSITY_CANDIDATE_BUCKETS[0][1:])
        )
    ]
    return {
        "ratio_distribution": distribution,
        "locked_buckets": locked_buckets,
        "candidate_strategies": candidate_strategies,
    }


def _tile_population_density_proposal_section(by_proxy: dict[str, list[float]]) -> dict:
    """Build the tile_population_density proposal section.

    Records per-candidate-proxy distribution and bucket strategies plus a
    default ``locked_proxy`` (first proxy alphabetically) and
    ``locked_buckets`` (most-granular cut of the default proxy). The
    reviewer hand-edits both fields at Gate 2.
    """
    proxy_names = sorted(by_proxy.keys())
    candidate_proxies: list[dict] = []
    for proxy_name in proxy_names:
        values = by_proxy[proxy_name]
        per_proxy_strategies: list[dict] = []
        for buckets in _DENSITY_CANDIDATE_BUCKETS:
            counts_per_bucket = _bucket_count_floats(values, buckets)
            coverage = sum(counts_per_bucket) / len(values) if values else 1.0
            per_proxy_strategies.append(
                {
                    "strategy": f"{len(counts_per_bucket)}_buckets",
                    "categories": len(counts_per_bucket),
                    "bucket_boundaries": list(buckets),
                    "bucket_counts": counts_per_bucket,
                    "coverage": float(coverage),
                    "marginal_cost": None,
                }
            )
        _fill_marginal_cost(per_proxy_strategies)
        candidate_proxies.append(
            {
                "proxy_name": proxy_name,
                "value_distribution": _summarise_distribution(values),
                "candidate_strategies": per_proxy_strategies,
            }
        )

    # Default locked_proxy: first proxy alphabetically. Reviewer edits at
    # Gate 2 to switch proxies. locked_buckets uses the most-granular cut
    # of that proxy.
    locked_proxy = proxy_names[0] if proxy_names else None
    locked_buckets: list[dict] = []
    if locked_proxy is not None:
        locked_buckets = [
            {
                "token_id": idx,
                "token_name": f"bucket_{idx}",
                "lower_inclusive": lo,
                "upper_exclusive": hi,
            }
            for idx, (lo, hi) in enumerate(
                zip(_DENSITY_CANDIDATE_BUCKETS[0], _DENSITY_CANDIDATE_BUCKETS[0][1:])
            )
        ]
    # NOTE: no top-level ``candidate_strategies`` mirror. Consumers read
    # ``locked_proxy`` from the index and look up that proxy's full
    # candidate_strategies inside ``candidate_proxies[]``. Mirroring at the
    # section level would create two sources of truth for the same data and
    # invite the namespace-file sha to silently include reviewer-editable
    # state from the index.
    return {
        "candidate_proxies": candidate_proxies,
        "locked_proxy": locked_proxy,
        "locked_buckets": locked_buckets,
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
            "token_name": f"bucket_{idx}",
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


def _per_tile_evidence_summary(inputs: list[SubCTileInputs]) -> list[dict]:
    """Per-tile evidence summary used by ``select_layer3_subset``.

    Sorted by (tile_i, tile_j) for byte-determinism. Each entry is a small
    aggregate over the per-tile Layer-1 metrics; ``select_layer3_subset``
    consumes these aggregates rather than re-running evidence derivation.
    """
    edge_lookup = {s.slot_index: s for s in iter_internal_edge_slots()}
    entries: list[dict] = []
    for tile in inputs:
        scope = derive_cell_scope_metrics(tile.cells)
        zoning_metrics = derive_zoning_evidence(tile.features, tile.cells)
        density_metrics = derive_density_evidence(tile.features, tile.cells)
        road_metrics = derive_road_skeleton_evidence(tile.crossings, tile.features)

        active_cell_count = sum(1 for v in scope.values() if v)
        zoning_signal = {
            FeatureClass.ROAD.name.lower(): 0,
            FeatureClass.BUILDING.name.lower(): 0,
            FeatureClass.POI.name.lower(): 0,
            FeatureClass.BASE.name.lower(): 0,
        }
        for m in zoning_metrics:
            cls = m.metric_name.removeprefix("feature_count_")
            zoning_signal[cls] += int(m.value)

        density_values = [float(m.value) for m in density_metrics]
        density_signal = {
            "cell_count": len(density_values),
            "max": float(max(density_values)) if density_values else 0.0,
            "mean": (
                float(sum(density_values) / len(density_values))
                if density_values
                else 0.0
            ),
        }

        active_road_counts: list[int] = []
        for m in road_metrics:
            edge = edge_lookup[m.slot_index]
            lower_active = scope[(edge.lower_cell_i, edge.lower_cell_j)]
            if edge.axis == 0:
                upper_active = scope[(edge.lower_cell_i + 1, edge.lower_cell_j)]
            else:
                upper_active = scope[(edge.lower_cell_i, edge.lower_cell_j + 1)]
            if derive_internal_edge_scope(lower_active, upper_active) == Scope.ACTIVE:
                active_road_counts.append(int(m.value))
        road_skeleton_signal = {
            "active_edge_count": len(active_road_counts),
            "mean_count": (
                float(sum(active_road_counts) / len(active_road_counts))
                if active_road_counts
                else 0.0
            ),
        }

        conditioning = tile.meta.get("conditioning_per_tile", {}) or {}
        coastal_inland_river = int(conditioning.get("coastal_inland_river", 0))

        entries.append(
            {
                "tile_i": tile.paths.tile_i,
                "tile_j": tile.paths.tile_j,
                "active_cell_count": active_cell_count,
                "zoning_signal": dict(sorted(zoning_signal.items())),
                "density_signal": density_signal,
                "road_skeleton_signal": road_skeleton_signal,
                "coastal_inland_river": coastal_inland_river,
            }
        )
    return sorted(entries, key=lambda e: (e["tile_i"], e["tile_j"]))


# Dimension list for deterministic subset selection. Each dimension is a
# (name, key_fn) pair; the selector picks the unselected tile maximising
# ``key_fn`` and tie-breaks lexicographically by (tile_i, tile_j). A tile
# already selected for a prior dimension accumulates additional rationale
# strings if it remains the maximiser, so the reviewer sees every dimension
# the tile covers.
_SUBSET_DIMENSIONS: list[tuple[str, "callable[[dict], float]"]] = [
    ("zoning_road_dominant", lambda e: e["zoning_signal"]["road"]),
    ("zoning_building_dominant", lambda e: e["zoning_signal"]["building"]),
    ("zoning_poi_dominant", lambda e: e["zoning_signal"]["poi"]),
    ("zoning_base_present", lambda e: e["zoning_signal"]["base"]),
    ("density_high", lambda e: e["density_signal"]["max"]),
    ("density_low", lambda e: -e["density_signal"]["max"]),
    ("road_skeleton_high_density", lambda e: e["road_skeleton_signal"]["mean_count"]),
    ("road_skeleton_sparse", lambda e: -e["road_skeleton_signal"]["active_edge_count"]),
    ("scope_full_tile", lambda e: e["active_cell_count"]),
    ("scope_sparse_tile", lambda e: -e["active_cell_count"]),
    ("coastal_present", lambda e: 1 if e["coastal_inland_river"] == 1 else 0),
    ("riverside_present", lambda e: 1 if e["coastal_inland_river"] in (2, 3) else 0),
    ("inland_present", lambda e: 1 if e["coastal_inland_river"] == 0 else 0),
]


def is_eligible_for_subset(entry: dict) -> bool:
    """Eligibility predicate for Layer-3 subset inclusion.

    A tile is eligible iff it has at least one active cell
    (``active_cell_count > 0``). Tiles with no active cells carry no
    derivation evidence — zoning, density, and road-skeleton signals are all
    zero — and provide nothing meaningful for a Layer-3 reviewer to inspect,
    even when meta-level dimensions like ``coastal_inland_river`` would
    otherwise rank them highly.

    Keeping the predicate explicit and separate from dimension scoring is
    defense in depth: a future dimension that scores positively for sparse
    or empty tiles (e.g. a hypothetical ``scope_completely_masked`` dim)
    cannot bypass eligibility through ranking alone.
    """
    return int(entry.get("active_cell_count", 0)) > 0


def select_layer3_subset(analysis: dict, max_tiles: int = 12) -> list[dict]:
    """Deterministically pick up to ``max_tiles`` tiles for Layer-3 review.

    Two-layered filter:

    1. Eligibility predicate (``is_eligible_for_subset``) removes tiles with
       no derivation evidence before any ranking applies. This catches
       active_cell_count==0 tiles regardless of how dimension scores might
       rank them.
    2. Dimension-driven ranking over eligible tiles. For each dimension in
       ``_SUBSET_DIMENSIONS``, picks the eligible tile that maximises that
       dimension's signal; ties break lexicographically on
       ``(tile_i, tile_j)``. A tile already selected accumulates additional
       rationale strings as more dimensions favour it.

    Returns a list sorted by ``(tile_i, tile_j)``. Each entry has::

        {"tile_i": int, "tile_j": int, "rationale": "selected for X, Y, Z"}

    ``max_tiles`` defaults to 12 to match the plan's Task 7 spec; the CLI
    exposes ``--max-subset-tiles`` for override.
    """
    per_tile = analysis.get("per_tile_evidence", [])
    if not per_tile or max_tiles <= 0:
        return []

    eligible = [e for e in per_tile if is_eligible_for_subset(e)]
    if not eligible:
        return []

    selected_keys: set[tuple[int, int]] = set()
    rationales: dict[tuple[int, int], list[str]] = {}

    for dim_name, key_fn in _SUBSET_DIMENSIONS:
        ordered = sorted(eligible, key=lambda e: (-key_fn(e), e["tile_i"], e["tile_j"]))
        top = ordered[0]
        top_key = (top["tile_i"], top["tile_j"])
        top_score = key_fn(top)

        if len(selected_keys) >= max_tiles:
            # Cap reached: only accumulate additional rationale for already-
            # selected tiles. A zero-signal top is meaningless either way.
            if top_score > 0 and top_key in selected_keys:
                rationales[top_key].append(dim_name)
            continue

        # A dimension with a non-positive top score among eligible tiles
        # does not justify picking a NEW tile. It can still accumulate
        # rationale for a tile already selected on stronger grounds.
        if top_score <= 0:
            if top_key in selected_keys:
                rationales[top_key].append(dim_name)
            continue

        if top_key in selected_keys:
            rationales[top_key].append(dim_name)
        else:
            selected_keys.add(top_key)
            rationales[top_key] = [dim_name]

    return [
        {
            "tile_i": key[0],
            "tile_j": key[1],
            "rationale": "selected for " + ", ".join(rationales[key]),
        }
        for key in sorted(selected_keys)
    ]


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

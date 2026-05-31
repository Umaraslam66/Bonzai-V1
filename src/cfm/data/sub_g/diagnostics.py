"""Diagnostic record, signature grouping, and byte-deterministic report writers.

A Diagnostic is one cross-artifact seam failure. Diagnostics are grouped by
(invariant_name, signature) where the *signature* is the failure PATTERN, not
the per-tile values (spec Decision 6). Reports are written every run; the empty
record is explicit `groups: []` (positive meaning — run completed, found
nothing), never file-absence (spec Decision 7).

Byte-determinism: the report BODY (groups, sorted) is deterministic; volatile
run-metadata (timestamp/host/uuid/commit) lives in a clearly-marked block
excluded from any digest (sub-C §12.4 + sub-E §9.2 EXCLUDED_FROM_SHA).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from cfm.data.io import canonicalize_yaml


@dataclass(frozen=True)
class Diagnostic:
    tile_id: str
    invariant_name: str
    artifact_left: str
    observed_left: object
    artifact_right: str
    observed_right: object
    expected_relationship: str
    spec_clause_citation: str
    signature: str  # the failure PATTERN; see spec Decision 6


@dataclass(frozen=True)
class DiagnosticGroup:
    invariant_name: str
    signature: str
    instance_count: int
    tile_ids: list[str]
    value_summary: dict
    spec_clause_citation: str
    hypothesis: str | None  # optional; empty is honest (spec Decision 6 obligation 4)


def _summarize(values: list[object]) -> dict:
    """min/max/median for numerics; value:count distribution for categoricals."""
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if nums and len(nums) == len(values):
        return {"min": min(nums), "max": max(nums), "median": median(nums)}
    dist: dict[str, int] = {}
    for v in values:
        dist[str(v)] = dist.get(str(v), 0) + 1
    return {"distribution": dict(sorted(dist.items()))}


def group_by_signature(diags: list[Diagnostic]) -> list[DiagnosticGroup]:
    """Collapse Diagnostics into groups keyed by (invariant_name, signature).

    Sort: instance_count desc, then invariant_name asc (deterministic tiebreak,
    spec Decision 6 obligation 2).
    """
    buckets: dict[tuple[str, str], list[Diagnostic]] = {}
    for d in diags:
        buckets.setdefault((d.invariant_name, d.signature), []).append(d)
    groups: list[DiagnosticGroup] = []
    for (inv, sig), members in buckets.items():
        groups.append(
            DiagnosticGroup(
                invariant_name=inv,
                signature=sig,
                instance_count=len(members),
                tile_ids=sorted(m.tile_id for m in members),
                value_summary={
                    "observed_left": _summarize([m.observed_left for m in members]),
                    "observed_right": _summarize([m.observed_right for m in members]),
                },
                spec_clause_citation=members[0].spec_clause_citation,
                hypothesis=None,
            )
        )
    groups.sort(key=lambda g: (-g.instance_count, g.invariant_name))
    return groups


def _group_to_dict(g: DiagnosticGroup) -> dict:
    d = {
        "invariant_name": g.invariant_name,
        "signature": g.signature,
        "instance_count": g.instance_count,
        "tile_ids": g.tile_ids,
        "value_summary": g.value_summary,
        "spec_clause_citation": g.spec_clause_citation,
    }
    if g.hypothesis is not None:
        d["hypothesis"] = g.hypothesis
    return d


def render_quarantine_report(
    groups: list[DiagnosticGroup], region: str, release: str, validator_version: str
) -> str:
    """Byte-deterministic YAML. `groups: []` is written explicitly when empty.

    `run_metadata` carries only the STABLE identity here (region/release/
    validator_version). Volatile fields (timestamp/host/uuid/commit) are added
    by the caller into a separate top-level block documented as excluded from
    any digest (see versions.py, a later task).
    """
    body = {
        "run_metadata": {
            "region": region,
            "release": release,
            "validator_version": validator_version,
        },
        "groups": [_group_to_dict(g) for g in groups],
    }
    return canonicalize_yaml(body)

"""Sub-G validator version + _PHASE1_VALIDATED / accuracy-baseline writers.

Byte-determinism follows sub-C §12.4 + sub-E §9.2 EXCLUDED_FROM_SHA: the stable
identity (region / release / validator_version) plus the content_digest are
digest-relevant; the `volatile` block (run_timestamp / host / run_uuid /
sub_g_commit_sha) is recorded for reproducibility + log-tracing but is EXCLUDED
from the digest, so two re-runs of the same validator_version over the same
artifacts produce identical digests. validator_version is the
VersionNamespace.VALIDATOR axis (reused from sub-D); it bumps on a semantic
behavior change.
"""

from __future__ import annotations

import math

from cfm.data.io import canonicalize_yaml
from cfm.data.sub_d.versions import VersionNamespace

# 1.1.0 (sub-G T11 H1, 2026-06-01): seam-3 accuracy is now geometry-aware
# (multi-part-paired symmetric Hausdorff vs the CANONICAL original) and splits
# into core (gated; excludes the v1-unencoded outbound bref vertex) vs full
# (reported). Replaces the index-positional metric whose Multi*/canonicalization/
# chunking drift produced the 318m/180deg first-measurement artifact.
VALIDATOR_VERSION = "1.1.0"


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile; 0.0 for an empty input.

    Non-finite values (NaN/inf) are EXCLUDED first: they are not orderable, so
    leaving them in corrupts ``sorted()`` and yields a non-monotonic result. A NaN
    can arise from a shapely Hausdorff over a degenerate decoded geometry (e.g. a
    zero-length bref-placeholder LineString in the *full* distribution); such a
    geometry is already caught by the decodability gate, so dropping it from the
    percentile is correct, not a silent narrowing.
    """
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return 0.0
    s = sorted(finite)
    idx = max(0, min(len(s) - 1, math.ceil(p / 100.0 * len(s)) - 1))
    return float(s[idx])


def render_validated_marker(
    region: str, release: str, content_digest: str, volatile: dict[str, str]
) -> str:
    """Render the _PHASE1_VALIDATED marker YAML.

    `content_digest` is computed by the caller over the validated content +
    stable identity (NOT the volatile block). This writer just records it.
    """
    body = {
        "run_metadata": {
            "region": region,
            "release": release,
            "validator_version": VALIDATOR_VERSION,
            "version_namespace": VersionNamespace.VALIDATOR.value,
        },
        "content_digest": content_digest,
        "volatile": dict(volatile),  # EXCLUDED from content_digest (sub-E §9.2 precedent)
    }
    return canonicalize_yaml(body)


def render_accuracy_baseline(
    position_core: list[float],
    position_full: list[float],
    angle_core: list[float],
    region: str,
    release: str,
    structural_bound_breaches: int,
) -> str:
    """Render _PHASE1_ACCURACY_BASELINE.yaml (written every run; spec Decision 3c).

    Reports BOTH the core (sanity-floor-gated) and full distributions. ``core``
    excludes the v1-by-design unencoded outbound bref edge-crossing vertex, by
    CONSTRUCTION IDENTITY (not error magnitude); ``full`` includes it so the bref
    residual on crossing roads stays visible (reviewer guard 2026-06-01). Angle is
    defined only where decoded/canonical vertex counts match (no chunking), so
    ``n_angle_features`` <= ``n_features``.
    """
    body = {
        "run_metadata": {
            "region": region,
            "release": release,
            "validator_version": VALIDATOR_VERSION,
        },
        "n_features": len(position_core),
        "n_angle_features": len(angle_core),
        "gated_metric": "core (sanity floor: position_core p99.9 / angle_core p95)",
        "core_excludes": (
            "v1-unencoded outbound bref edge-crossing vertex (Case B/D last vertex); "
            "v2-scoped per decoder.py:13-22"
        ),
        "position_core_p99_9": _percentile(position_core, 99.9),
        "position_core_p95": _percentile(position_core, 95.0),
        "position_full_p99_9": _percentile(position_full, 99.9),
        "position_full_p95": _percentile(position_full, 95.0),
        "angle_core_p99_9": _percentile(angle_core, 99.9),
        "angle_core_p95": _percentile(angle_core, 95.0),
        "structural_bound_breaches": structural_bound_breaches,
    }
    return canonicalize_yaml(body)

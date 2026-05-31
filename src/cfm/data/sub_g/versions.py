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

VALIDATOR_VERSION = "1.0.0"


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile; 0.0 for an empty input."""
    if not values:
        return 0.0
    s = sorted(values)
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
    position_errors: list[float],
    angle_errors: list[float],
    region: str,
    release: str,
    structural_bound_breaches: int,
) -> str:
    """Render _PHASE1_ACCURACY_BASELINE.yaml (written every run; spec Decision 3c)."""
    body = {
        "run_metadata": {
            "region": region,
            "release": release,
            "validator_version": VALIDATOR_VERSION,
        },
        "n_features": len(position_errors),
        "position_p99_9": _percentile(position_errors, 99.9),
        "position_p95": _percentile(position_errors, 95.0),
        "angle_p99_9": _percentile(angle_errors, 99.9),
        "angle_p95": _percentile(angle_errors, 95.0),
        "structural_bound_breaches": structural_bound_breaches,
    }
    return canonicalize_yaml(body)

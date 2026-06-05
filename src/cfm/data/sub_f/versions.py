"""Sub-F six-axis version manifest helpers for Halt 6.

SOURCE ``VersionRef.value`` canonical format:
``overture=<release>;subc_schema=<ver>;subc_commit=<full_sha>``.
The string is semicolon-delimited and emitted in fixed component order.
Consumers parsing component-level drift must use this format; component fields
are also available structurally in ``manifest["sub_f_source_version"]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from cfm.data.sub_d.versions import VersionNamespace, VersionRef

# 1.1: multi-region region_crs field added to the region manifest (provenance +
# cross-stage CRS consistency for the multi-region corpus). Manifest-FORMAT change
# only — cells.parquet data shape is unchanged, so SCHEMA stays 1.0. The
# ARTIFACT_FORMAT axis distinguishes a pre-region_crs manifest from a post one.
SUB_F_ARTIFACT_FORMAT_VERSION = "1.1"
SUB_F_SCHEMA_VERSION = "1.0"
SUB_F_VOCAB_VERSION = "1.0"
# 1.1: the cycle-1 N/S endpoint-direction fix (commit 98cdeb0,
# encoder._classify_feature_for_bref) changed bref direction output for the same
# input. Bumping the DERIVATION axis distinguishes pre/post-cycle-1 sub-F
# artifacts so a stale 1.0 cache can never silently compare equal to a 1.1 one.
SUB_F_DERIVATION_VERSION = "1.1"
# 1.1: cycle-3 validator fix — feature_key resolution now resolves BP4
# <unknown_*> tokens to their semantic key (vocab.unknown_family_tag_to_key),
# so unknown-subtype highways no longer false-positive the non-road-emission
# leg. Verdict-only change (no cells.parquet bytes change); the VALIDATOR axis
# distinguishes a 1.0-blessed cache from a 1.1-blessed one.
# 1.2: §8.3 termination relax — the symmetry (leg 2) and coverage (leg 4) legs
# are now road-presence-conditioned (road_edge_presence, derived from sub-C
# features.parquet via the shared encoder.endpoint_edge_direction authority).
# A road that TERMINATES at an internal cell boundary (present on one side only)
# no longer false-positives; an under-emission (road endpoint present on both
# sides, one side silent) still raises. Verdict-only change (no cells.parquet
# bytes change — the encoder/tokens are untouched); the VALIDATOR axis
# distinguishes a 1.1-blessed cache from a 1.2-blessed one. See
# reports/2026-06-05-batch2-subf-symmetry-fp-investigation.md.
SUB_F_VALIDATOR_VERSION = "1.2"

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_OVERTURE_PIN_PATH = _REPO_ROOT / "configs" / "data" / "overture_release.yaml"
_DEFAULT_SUB_C_REGION = "singapore"

SubFSourceVersion = dict[str, str]


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _default_sub_c_manifest_path(overture_release: str) -> Path:
    return (
        _REPO_ROOT
        / "data"
        / "processed"
        / "sub_c"
        / overture_release
        / _DEFAULT_SUB_C_REGION
        / "manifest.yaml"
    )


def load_sub_f_source_version(
    *,
    overture_pin_path: Path = _DEFAULT_OVERTURE_PIN_PATH,
    sub_c_manifest_path: Path | None = None,
) -> SubFSourceVersion:
    """Load the composite SOURCE identity consumed by sub-F.

    Sub-F reads sub-C output, so SOURCE records both the Overture release pin
    and the sub-C output identity. The corresponding ``VersionRef.value`` is
    scalar-only in sub-D, so callers that need a ``VersionRef`` must pass this
    mapping through ``encode_sub_f_source_version``.
    """

    overture_pin = _load_yaml(overture_pin_path)
    overture_release = str(overture_pin["release"])
    manifest_path = sub_c_manifest_path or _default_sub_c_manifest_path(overture_release)
    sub_c_manifest = _load_yaml(manifest_path)

    return {
        "overture_release": overture_release,
        "sub_c_schema_version": str(sub_c_manifest["sub_c_schema_version"]),
        "sub_c_commit_sha": str(sub_c_manifest["initial_extraction"]["commit_sha"]),
    }


def encode_sub_f_source_version(source_version: SubFSourceVersion) -> str:
    """Encode composite SOURCE as a deterministic scalar for VersionRef."""

    return (
        f"overture={source_version['overture_release']};"
        f"subc_schema={source_version['sub_c_schema_version']};"
        f"subc_commit={source_version['sub_c_commit_sha']}"
    )


def sub_f_version_manifest() -> dict[VersionNamespace, VersionRef]:
    """Return sub-F's six-axis version manifest as sub-D VersionRefs."""

    source_version = load_sub_f_source_version()
    return {
        VersionNamespace.ARTIFACT_FORMAT: VersionRef(
            VersionNamespace.ARTIFACT_FORMAT, SUB_F_ARTIFACT_FORMAT_VERSION
        ),
        VersionNamespace.DATA_SHAPE: VersionRef(VersionNamespace.DATA_SHAPE, SUB_F_SCHEMA_VERSION),
        VersionNamespace.VOCAB: VersionRef(VersionNamespace.VOCAB, SUB_F_VOCAB_VERSION),
        VersionNamespace.DERIVATION: VersionRef(
            VersionNamespace.DERIVATION, SUB_F_DERIVATION_VERSION
        ),
        VersionNamespace.VALIDATOR: VersionRef(VersionNamespace.VALIDATOR, SUB_F_VALIDATOR_VERSION),
        VersionNamespace.SOURCE: VersionRef(
            VersionNamespace.SOURCE, encode_sub_f_source_version(source_version)
        ),
    }

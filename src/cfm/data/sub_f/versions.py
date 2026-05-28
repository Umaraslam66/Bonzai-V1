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

SUB_F_ARTIFACT_FORMAT_VERSION = "1.0"
SUB_F_SCHEMA_VERSION = "1.0"
SUB_F_VOCAB_VERSION = "1.0"
SUB_F_DERIVATION_VERSION = "1.0"
SUB_F_VALIDATOR_VERSION = "1.0"

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
        VersionNamespace.DATA_SHAPE: VersionRef(
            VersionNamespace.DATA_SHAPE, SUB_F_SCHEMA_VERSION
        ),
        VersionNamespace.VOCAB: VersionRef(VersionNamespace.VOCAB, SUB_F_VOCAB_VERSION),
        VersionNamespace.DERIVATION: VersionRef(
            VersionNamespace.DERIVATION, SUB_F_DERIVATION_VERSION
        ),
        VersionNamespace.VALIDATOR: VersionRef(
            VersionNamespace.VALIDATOR, SUB_F_VALIDATOR_VERSION
        ),
        VersionNamespace.SOURCE: VersionRef(
            VersionNamespace.SOURCE, encode_sub_f_source_version(source_version)
        ),
    }

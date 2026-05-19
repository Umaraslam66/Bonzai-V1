"""Sub-D: macro-plan derivation sidecar over immutable sub-C outputs.

Phase A produces reviewable evidence and the locked macro vocab/config.
Phase B writes the digest-anchored sidecar dataset under
``data/processed/sub_d/<release>/<region>/``.
"""

from __future__ import annotations

from cfm.data.sub_d.errors import (
    SubDValidationError,
    VersionMismatchError,
    VersionNamespaceError,
)
from cfm.data.sub_d.versions import (
    VersionNamespace,
    VersionRef,
    compare_version,
)

__all__ = [
    "SubDValidationError",
    "VersionMismatchError",
    "VersionNamespace",
    "VersionNamespaceError",
    "VersionRef",
    "compare_version",
]

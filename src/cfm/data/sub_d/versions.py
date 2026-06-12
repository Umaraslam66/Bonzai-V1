"""Sub-D version namespace helper.

The sub-D contract carries six disjoint version namespaces so that, for
example, a vocab-version drift can never silently match an artifact-format
string with the same numeric value. ``compare_version`` is the single
sanctioned equality path; validator code MUST go through it rather than
comparing version strings directly with ``==``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cfm.data.sub_d.errors import VersionMismatchError, VersionNamespaceError


class VersionNamespace(str, Enum):  # noqa: UP042 — sealed sub-D: StrEnum changes str(member)
    """Disjoint version namespaces tracked by sub-D artefacts."""

    ARTIFACT_FORMAT = "artifact_format"
    DATA_SHAPE = "data_shape"
    VOCAB = "vocab"
    DERIVATION = "derivation"
    VALIDATOR = "validator"
    SOURCE = "source"


@dataclass(frozen=True)
class VersionRef:
    """A namespaced version reference (namespace + opaque string value)."""

    namespace: VersionNamespace
    value: str


def compare_version(
    namespace: VersionNamespace,
    expected: VersionRef,
    actual: VersionRef,
) -> None:
    """Assert ``expected`` and ``actual`` agree under ``namespace``.

    Raises ``VersionNamespaceError`` if either side belongs to a different
    namespace, and ``VersionMismatchError`` if the namespaces line up but the
    values diverge.
    """
    if expected.namespace != namespace or actual.namespace != namespace:
        raise VersionNamespaceError(
            f"version namespace mismatch: comparison={namespace.value}, "
            f"expected={expected.namespace.value}, actual={actual.namespace.value}"
        )
    if expected.value != actual.value:
        raise VersionMismatchError(
            f"version mismatch in {namespace.value}: expected {expected.value}, got {actual.value}"
        )

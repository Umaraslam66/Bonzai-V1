from __future__ import annotations

import pytest

from cfm.data.sub_d.errors import VersionNamespaceError
from cfm.data.sub_d.versions import VersionNamespace, VersionRef, compare_version
from cfm.data.sub_e.versions import (
    BOUNDARY_DERIVATION_NAMESPACE,
    BOUNDARY_DERIVATION_VERSION,
    BOUNDARY_VOCAB_NAMESPACE,
    BOUNDARY_VOCAB_VERSION,
    SUB_E_SCHEMA_NAMESPACE,
    SUB_E_SCHEMA_VERSION,
)


def test_initial_version_values() -> None:
    assert SUB_E_SCHEMA_VERSION == "1.0"
    assert BOUNDARY_VOCAB_VERSION == "1.0"
    assert BOUNDARY_DERIVATION_VERSION == "1.1"


def test_version_namespaces_use_subd_concept_enum() -> None:
    """Sub-E's three axes map to sub-D's DATA_SHAPE / VOCAB / DERIVATION."""
    assert SUB_E_SCHEMA_NAMESPACE is VersionNamespace.DATA_SHAPE
    assert BOUNDARY_VOCAB_NAMESPACE is VersionNamespace.VOCAB
    assert BOUNDARY_DERIVATION_NAMESPACE is VersionNamespace.DERIVATION


def test_compare_version_within_vocab_namespace_passes() -> None:
    """compare_version with matched-namespace refs and equal values must not raise."""
    expected = VersionRef(namespace=BOUNDARY_VOCAB_NAMESPACE, value=BOUNDARY_VOCAB_VERSION)
    actual = VersionRef(namespace=BOUNDARY_VOCAB_NAMESPACE, value=BOUNDARY_VOCAB_VERSION)
    compare_version(BOUNDARY_VOCAB_NAMESPACE, expected, actual)  # no raise


def test_compare_version_cross_namespace_rejects() -> None:
    """Mixing sub-E's vocab namespace with sub-E's derivation namespace must raise."""
    expected = VersionRef(namespace=BOUNDARY_VOCAB_NAMESPACE, value="1.0")
    actual = VersionRef(namespace=BOUNDARY_DERIVATION_NAMESPACE, value="1.0")
    with pytest.raises(VersionNamespaceError, match="namespace"):
        compare_version(BOUNDARY_VOCAB_NAMESPACE, expected, actual)

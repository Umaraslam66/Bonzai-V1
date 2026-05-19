"""Tests for the sub-D version namespace helper.

The version helper pins five disjoint namespaces (artifact_format, data_shape,
vocab, derivation, validator) and forces every equality comparison through
``compare_version`` so a vocab-version drift can never be checked against an
artifact-format string.
"""

from __future__ import annotations

import pytest

from cfm.data.sub_d.versions import (
    VersionMismatchError,
    VersionNamespace,
    VersionNamespaceError,
    VersionRef,
    compare_version,
)


def test_compare_version_accepts_same_namespace_and_value():
    expected = VersionRef(VersionNamespace.DATA_SHAPE, "1.0")
    actual = VersionRef(VersionNamespace.DATA_SHAPE, "1.0")
    compare_version(VersionNamespace.DATA_SHAPE, expected, actual)


def test_compare_version_rejects_value_mismatch():
    expected = VersionRef(VersionNamespace.VOCAB, "1.0")
    actual = VersionRef(VersionNamespace.VOCAB, "1.1")
    with pytest.raises(VersionMismatchError):
        compare_version(VersionNamespace.VOCAB, expected, actual)


def test_compare_version_rejects_cross_namespace_expected():
    expected = VersionRef(VersionNamespace.ARTIFACT_FORMAT, "1.0")
    actual = VersionRef(VersionNamespace.DATA_SHAPE, "1.0")
    with pytest.raises(VersionNamespaceError):
        compare_version(VersionNamespace.DATA_SHAPE, expected, actual)


def test_compare_version_rejects_cross_namespace_actual():
    expected = VersionRef(VersionNamespace.DATA_SHAPE, "1.0")
    actual = VersionRef(VersionNamespace.ARTIFACT_FORMAT, "1.0")
    with pytest.raises(VersionNamespaceError):
        compare_version(VersionNamespace.DATA_SHAPE, expected, actual)

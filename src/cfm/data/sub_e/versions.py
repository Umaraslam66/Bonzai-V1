"""Sub-E version constants.

Sub-E has three version axes: schema (data-shape), vocab, and derivation.
Each maps to a sub-D `VersionNamespace` concept (DATA_SHAPE, VOCAB,
DERIVATION). Sub-D's `compare_version(namespace, expected, actual)` is the
canonical namespace-aware equality path; sub-E does not introduce a new
helper.

Version constants are plain strings for ergonomic YAML serialization;
namespace constants pair each version with its sub-D namespace.
"""

from __future__ import annotations

from typing import Final

from cfm.data.sub_d.versions import VersionNamespace


# Schema version: governs on-disk parquet schema + YAML structure.
SUB_E_SCHEMA_VERSION: Final[str] = "1.0"
SUB_E_SCHEMA_NAMESPACE: Final[VersionNamespace] = VersionNamespace.DATA_SHAPE

# Vocab version: governs boundary_vocab.yaml token domain.
BOUNDARY_VOCAB_VERSION: Final[str] = "1.0"
BOUNDARY_VOCAB_NAMESPACE: Final[VersionNamespace] = VersionNamespace.VOCAB

# Derivation version: governs class-grouping + multi-crossing tie-break.
BOUNDARY_DERIVATION_VERSION: Final[str] = "1.0"
BOUNDARY_DERIVATION_NAMESPACE: Final[VersionNamespace] = VersionNamespace.DERIVATION

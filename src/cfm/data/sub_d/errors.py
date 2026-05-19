"""Sub-D error hierarchy.

All sub-D validation failures inherit from ``SubDValidationError`` so callers
can catch any sub-D contract violation with a single ``except`` clause while
still distinguishing specific failure modes (version namespace vs. value).
"""

from __future__ import annotations


class SubDValidationError(Exception):
    """Base class for any sub-D contract violation."""


class VersionNamespaceError(SubDValidationError):
    """Raised when a version is compared across disjoint namespaces."""


class VersionMismatchError(SubDValidationError):
    """Raised when two versions in the same namespace have different values."""

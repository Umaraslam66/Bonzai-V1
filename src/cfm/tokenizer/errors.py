from __future__ import annotations


class TokenizerError(ValueError):
    """Base class for all tokenizer failures. Inherits from ValueError."""


class UnsupportedFeatureClass(TokenizerError):
    """A feature's class is not present in the active vocabulary."""


class UnsupportedGeometry(TokenizerError):
    """A geometry shape is not handled by the current tokenizer (e.g. non-rectangular building,
    diagonal road in Phase 0)."""


class FeatureOutOfBounds(TokenizerError):
    """A feature's geometry extends outside the cell bounds (except as a deliberate boundary
    exit)."""


class VocabularyMismatch(TokenizerError):
    """Decoded token IDs are not present in the active vocabulary."""

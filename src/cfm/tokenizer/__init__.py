"""Tokenizer for cell-local GeoJSON ↔ token-ID round-trip."""

from cfm.tokenizer.encode import CellTokens, encode_cell
from cfm.tokenizer.errors import (
    FeatureOutOfBounds,
    TokenizerError,
    UnsupportedFeatureClass,
    UnsupportedGeometry,
    VocabularyMismatch,
)
from cfm.tokenizer.geometry import geometric_equal
from cfm.tokenizer.vocabulary import Vocabulary

__all__ = [
    "CellTokens",
    "FeatureOutOfBounds",
    "TokenizerError",
    "UnsupportedFeatureClass",
    "UnsupportedGeometry",
    "Vocabulary",
    "VocabularyMismatch",
    "encode_cell",
    "geometric_equal",
]

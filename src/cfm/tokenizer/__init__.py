"""Tokenizer for cell-local GeoJSON ↔ token-ID round-trip."""

from cfm.tokenizer.decode import decode_cell
from cfm.tokenizer.encode import CellTokens, encode_cell
from cfm.tokenizer.errors import (
    FeatureOutOfBounds,
    LoaderError,
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
    "LoaderError",
    "TokenizerError",
    "UnsupportedFeatureClass",
    "UnsupportedGeometry",
    "Vocabulary",
    "VocabularyMismatch",
    "decode_cell",
    "encode_cell",
    "geometric_equal",
]

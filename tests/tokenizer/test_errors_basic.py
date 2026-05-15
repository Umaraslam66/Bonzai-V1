from __future__ import annotations

import pytest

from cfm.tokenizer.errors import (
    FeatureOutOfBounds,
    TokenizerError,
    UnsupportedFeatureClass,
    UnsupportedGeometry,
    VocabularyMismatch,
)


def test_all_subclasses_inherit_from_tokenizer_error() -> None:
    for cls in (
        UnsupportedFeatureClass,
        UnsupportedGeometry,
        FeatureOutOfBounds,
        VocabularyMismatch,
    ):
        assert issubclass(cls, TokenizerError)


def test_tokenizer_error_is_value_error() -> None:
    assert issubclass(TokenizerError, ValueError)


def test_raising_and_catching_specific_subclass() -> None:
    with pytest.raises(UnsupportedFeatureClass):
        raise UnsupportedFeatureClass("unknown class 'X'")


def test_catching_as_tokenizer_error_catches_subclasses() -> None:
    with pytest.raises(TokenizerError):
        raise UnsupportedGeometry("triangle building")

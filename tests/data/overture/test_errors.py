from __future__ import annotations

import pytest

from cfm.data.overture.errors import (
    CacheCorrupt,
    OversizedFetch,
    OvertureError,
    OvertureSchemaMismatch,
    OvertureUnreachable,
    RegionNotFound,
    ReleaseNotConfigured,
)


def test_base_class_is_runtime_error() -> None:
    assert issubclass(OvertureError, RuntimeError)


@pytest.mark.parametrize(
    "cls",
    [
        OvertureUnreachable,
        OvertureSchemaMismatch,
        RegionNotFound,
        ReleaseNotConfigured,
        OversizedFetch,
        CacheCorrupt,
    ],
)
def test_each_subclass_inherits_from_base(cls: type[Exception]) -> None:
    assert issubclass(cls, OvertureError)


def test_subclasses_can_be_caught_as_base() -> None:
    with pytest.raises(OvertureError):
        raise CacheCorrupt("sha mismatch")


def test_six_distinct_subclasses() -> None:
    subclasses = {
        OvertureUnreachable,
        OvertureSchemaMismatch,
        RegionNotFound,
        ReleaseNotConfigured,
        OversizedFetch,
        CacheCorrupt,
    }
    assert len(subclasses) == 6

"""Tombstone guard: conditioning_gate is the F10 dead twin (readiness audit 2026-06-10)."""

from __future__ import annotations

import pytest


def test_conditioning_gate_module_is_tombstoned():
    with pytest.raises(ImportError, match=r"superseded by cfm\.eval\.conditioning_discrimination"):
        import cfm.eval.conditioning_gate  # noqa: F401

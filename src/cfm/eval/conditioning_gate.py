"""TOMBSTONE — superseded 2026-06-10 by ``cfm.eval.conditioning_discrimination``.

This module was the originally-planned tolerance-based conditioning-discrimination
gate (delta-spec T5). It never ran in production: the gate that actually ran on
Leonardo 2026-06-10 (and produced the on-disk FAIL verdict) is the BH-corrected
``cfm.eval.conditioning_discrimination``. Importing this module is the F10
dead-twin hazard — see
``reports/2026-06-10-readiness-audit-failure-class-enumeration.md`` class F10.
"""

from __future__ import annotations

raise ImportError(
    "cfm.eval.conditioning_gate is superseded by cfm.eval.conditioning_discrimination "
    "(the BH-corrected gate that actually ran 2026-06-10). "
    "Wiring this module is the F10 dead-twin hazard - see the readiness enumeration."
)

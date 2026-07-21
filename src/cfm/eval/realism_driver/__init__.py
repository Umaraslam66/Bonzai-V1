"""Scored realism-eval driver (glue over existing tested eval components).

Lives at ``cfm.eval.realism_driver`` — NOT ``cfm.eval.realism``, which is the
bake-off per-feature KS-distance module (13 importers; orchestrator decision
2026-07-20: leave that load-bearing module untouched days before a scored run).
"""

from __future__ import annotations

"""Bounded multi-region extract orchestrator (Phase-2).

A thin Python state machine that fetches, processes (sub_c→sub_g), and validates
many cities into one diversity-spanning corpus. See
docs/superpowers/specs/2026-06-03-phase-2-multiregion-extract-orchestrator-design.md.
"""

from __future__ import annotations

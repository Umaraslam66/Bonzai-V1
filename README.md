# Bonzai-OSM

A generative foundation model for city geometry. See `PRD.md` for goals and `CLAUDE.md` for collaboration rules.

## Phase 0 quick start

Phase 0 ships a single-cell GeoJSON tokenizer with a geometric-equivalence round-trip, the canonical Phase-0 vocabulary, and the three negative-fixture failure paths.

```bash
uv sync --all-extras
uv run ruff check .
uv run pytest -v
uv run python scripts/smoke.py
```

Phase 0 is "done" when all four commands succeed on a clean Mac or Linux machine and CI is green.

## Repository layout

See `docs/superpowers/specs/2026-05-15-phase-0-tokenizer-roundtrip-design.md` for the locked Phase-0 design and `docs/superpowers/plans/2026-05-15-phase-0-tokenizer-roundtrip.md` for the implementation plan.

## Next

Phase 1 takes over once Phase 0 is signed off: Overture loading, multi-cell tiles, boundary contracts, deterministic stitching, and the full ~100-class vocabulary derived from frequency analysis.

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

## Phase 1 sub-project A: Overture loader

```python
from cfm.data.overture import load_region

singapore = load_region("singapore")
print(singapore.release)              # "2026-04-15.0"
print(list(singapore.themes))         # ["divisions", "buildings", "places", "transportation", "base"]
print(singapore.themes["buildings"].num_rows)
```

First call fetches from public S3 and caches to `data/cache/overture/<release>/<region>/`. Subsequent calls verify sha256 and read from cache.

See `docs/data/overture_pinning_policy.md` for the re-pin procedure.

## Next

Phase 1 takes over once Phase 0 is signed off: Overture loading, multi-cell tiles, boundary contracts, deterministic stitching, and the full ~100-class vocabulary derived from frequency analysis.

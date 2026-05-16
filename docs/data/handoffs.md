# Sub-project handoff contracts

Phase 1 is decomposed into sub-projects A–G. Each sub-project has a contract with its downstream consumers: what is guaranteed, and what the consumer must still do. This document is the canonical record of those contracts.

## A → C: bbox-filtered themes, polygon for downstream clipping

**Sub-project A** (Overture loader) returns a `Region` object with:

- `themes: dict[str, pyarrow.Table]` — five Overture themes (buildings, places, transportation, base, divisions), **filtered ONLY by `fetch_bbox`** at fetch time.
- `fetch_bbox: BboxScope` — the bounding box actually used as the filter.
- `geometry: RegionGeometry` — the precise admin polygon for the region, surfaced as a *handoff record*. **Not applied at fetch time.**
- `manifest_path: Path` — the cache manifest, recording release version, sha256s, and source URLs.

**Sub-project C** (tile extraction) is contractually obligated to:

1. **Apply `region.admin_polygon` to clip themes** before partitioning into tiles. Failing this means open sea — which falls inside `region.fetch_bbox` but outside `region.admin_polygon` for Singapore — silently enters the training set.
2. Use `region.themes["divisions"]` as the source of truth for the precise polygon if the bbox-as-polygon placeholder in `region.geometry` is still in use. C may choose to compute its own polygon by dissolving rows from `region.themes["divisions"]` matching the country/locality.
3. Reproject from `EPSG:4326` to a local metric frame before tokenisation.

The Phase 1 simplification in sub-project A — using `box(fetch_bbox)` as a placeholder for the admin polygon — exists because the polygon is genuinely not needed for fetching, only for clipping. C-stage either uses the precise polygon from `themes["divisions"]` or, in a future iteration, A is upgraded to a two-pass fetch (divisions first, polygon-derived filter applied to the other four themes). Either is acceptable; A's contract holds either way.

## Why an explicit handoff doc

The risk of the bbox-only fetch is silent: A succeeds, C runs, and sea contamination only becomes visible far downstream when training metrics misbehave. Documenting the contract here lets every consumer of A know the obligation up front and lets a code reviewer flag a C-stage PR that skips the clip.

## Future handoffs

This document grows as more sub-projects ship. Each sub-project adds a section describing what its output guarantees and what consumers must do.

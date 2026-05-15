# Degenerate fixtures

One file per failure mode the Phase-0 tokenizer must surface as a specific `TokenizerError` subclass. Used exclusively by `tests/tokenizer/test_errors.py`.

| File | Trigger | Expected error |
|---|---|---|
| `non_rectangular_building.geojson` | L-shaped axis-aligned building polygon (6 vertices) | `UnsupportedGeometry` |
| `unknown_class.geojson` | feature with `class: "B_castle"` (not in Phase-0 vocab) | `UnsupportedFeatureClass` |
| `out_of_bounds.geojson` | POI at (300, 300) inside a 250 × 250 cell | `FeatureOutOfBounds` |

"""Building-emergence instrumentation for the Phase-2 bake-off.

Two jobs, both feeding the §5 diagnostic and the §2 slice-eval guard from ONE source:

1. The truncation discriminator: did a generated sequence contain building-class
   tokens AT ALL (vs. it never tried, vs. it tried but they did not close into
   polygons)?  ``sequence_has_building_tokens``.
2. The emergence floor: "buildings emerged" means the model reliably produces
   buildings WHERE REAL DATA HAS THEM -- a per-active-cell polygon rate tied to the
   holdout's real building density, NOT ``n_polygons > 0`` (one stray polygon is
   noise).  ``buildings_emerged`` / ``emergence_floor_polygons_per_cell``.

Building-token authority: the BP1 L1 key ``building`` (parallel to
``ROAD_L1_KEY="highway"``), resolved via ``semantic_tag_to_l1_key`` -- which also
covers the unknown-family ``<unknown_building>``. Verified against the sealed sub-F
vocab (77 ids; the raw Phase-0 ``B_`` prefix scheme is a DIFFERENT tokenizer and is
not present here).
"""

from __future__ import annotations

from functools import cache

from cfm.data.sub_f.vocab import semantic_tag_to_l1_key, vocab_tag_to_id

# Authority for "is this a building feature token": the BP1 L1 key, parallel to
# ROAD_L1_KEY. Covers building=* / building=<value> AND <unknown_building>.
# NOT a string prefix (sub-F tags are <key=value>, never "B_...").
BUILDING_L1_KEY = "building"

# DECISION: emergence floor = a fraction of the holdout's real polygons-per-active-cell
# density. Tied to real data (one stray polygon != emergence), RELATIVE not absolute so
# the threshold's meaning transfers across scales. frac is a recorded PI choice; default
# 0.25. Revisit if the Task-4 diagnostic shows it admits roads-only runs. One source with
# the §2 slice-eval guard.
EMERGENCE_FRAC_OF_HOLDOUT_DENSITY: float = 0.25


@cache
def building_token_ids() -> frozenset[int]:
    """Token ids whose sub-F tag resolves to the ``building`` BP1 L1 key."""
    return frozenset(
        i for tag, i in vocab_tag_to_id().items() if semantic_tag_to_l1_key(tag) == BUILDING_L1_KEY
    )


def sequence_has_building_tokens(tokens: list[int]) -> bool:
    """True iff the generated token sequence contains ANY building-class token.

    The §5 stage-1 truncation discriminator: NO building tokens at all means the
    model never tried to emit buildings (or generation truncated before reaching
    them); building tokens present but ``n_polygons == 0`` means they did not CLOSE
    into polygons -- a different cause than truncation.
    """
    ids = building_token_ids()
    return any(t in ids for t in tokens)


def emergence_floor_polygons_per_cell(
    *, holdout_polys_per_cell: float, frac: float = EMERGENCE_FRAC_OF_HOLDOUT_DENSITY
) -> float:
    """The per-active-cell polygon-count floor a run must clear to count as 'emerged'."""
    return frac * holdout_polys_per_cell


def buildings_emerged(*, n_polygons: int, n_cells: int, floor_per_cell: float) -> bool:
    """True iff the run's polygons-per-cell clears the holdout-density-tied floor.

    NOT ``n_polygons > 0``: a non-vacuous, density-tied threshold so a single stray
    polygon is never mistaken for emergence.
    """
    if n_cells <= 0:
        return False
    return (n_polygons / n_cells) >= floor_per_cell


def holdout_polygons_per_active_cell(*, release: str, region: str) -> float:
    """Measure the holdout's real polygons-per-active-cell by round-tripping real cells.

    The single source of ``holdout_polys_per_cell`` for the emergence floor. Reuses the
    SEALED sub-F decoder + sub-G splitter (one source; never reimplemented). Requires the
    real sub-F tile data (Leonardo ``$WORK``); the local checkout has only the manifest,
    so the calling test is ``@pytest.mark.slow`` and runs during the diagnostic.
    """
    import yaml
    from shapely.geometry import shape

    from cfm.data.sub_f.decoder import decode_feature
    from cfm.data.sub_g.readers import read_sub_f_cells
    from cfm.data.sub_g.seam_decodability import split_cell_into_features
    from cfm.eval.holdout.paths import holdout_manifest_path, sub_f_region_dir, tile_dirname

    manifest = yaml.safe_load(holdout_manifest_path(release).read_text(encoding="utf-8"))
    tiles = manifest["regions"][region]["tiles"]
    sub_f_dir = sub_f_region_dir(release, region)

    n_polygons = 0
    n_active_cells = 0
    for tile in tiles:
        cells_path = sub_f_dir / tile_dirname(tile["tile_i"], tile["tile_j"]) / "cells.parquet"
        if not cells_path.exists():
            continue
        for tokens in read_sub_f_cells(cells_path).values():
            if not tokens:
                continue
            n_active_cells += 1
            for block in split_cell_into_features(tokens):
                try:
                    geom = decode_feature(block)
                except Exception:
                    continue
                if shape(geom).geom_type in ("Polygon", "MultiPolygon"):
                    n_polygons += 1
    return n_polygons / n_active_cells if n_active_cells else 0.0

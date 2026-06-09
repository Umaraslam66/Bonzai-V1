"""One-source path resolution for the eval-set substrate.

Every on-disk location the eval-set reads or writes is built here so no
other module hard-codes a path. Layout verified 2026-06-01:
- sub-{c,d,f,g} region dirs: data/processed/sub_X/<release>/<region>/
- per-tile dir: tile=EPSG3414_i{i}_j{j}/ (sub_d/pipeline.py:156)
- _PHASE1_VALIDATED marker: data/processed/sub_g/<release>/<region>/
"""

from __future__ import annotations

from pathlib import Path

import yaml

#: Phase-1 validated Singapore release (sub-G _PHASE1_VALIDATED, 494 tiles).
DEFAULT_RELEASE: str = "2026-04-15.0"
DEFAULT_REGION: str = "singapore"

#: Default projection label embedded in sub-D tile dir names (EPSG:3414 -> EPSG3414).
_EPSG_LABEL: str = "EPSG3414"


def _repo_root() -> Path:
    # src/cfm/eval/holdout/paths.py -> repo root is four parents up from src/cfm.
    return Path(__file__).resolve().parents[4]


def _data_processed() -> Path:
    return _repo_root() / "data" / "processed"


def tile_dirname(tile_i: int, tile_j: int, epsg_label: str = _EPSG_LABEL) -> str:
    """Per-tile directory name, identical to sub-D (sub_d/pipeline.py:156)."""
    return f"tile={epsg_label}_i{int(tile_i)}_j{int(tile_j)}"


def _region_config_path(region: str) -> Path:
    return _repo_root() / "configs" / "data" / "regions" / f"{region}.yaml"


def epsg_label_for_region(region: str) -> str:
    """CRS label embedded in a region's sub-D tile dir names, from its config's
    projected_crs (e.g. 'EPSG:25832' -> 'EPSG25832')."""
    cfg = yaml.safe_load(_region_config_path(region).read_text(encoding="utf-8"))
    return cfg["projected_crs"].replace(":", "")


def sub_c_region_dir(release: str, region: str) -> Path:
    return _data_processed() / "sub_c" / release / region


def sub_d_region_dir(release: str, region: str) -> Path:
    return _data_processed() / "sub_d" / release / region


def sub_f_region_dir(release: str, region: str) -> Path:
    return _data_processed() / "sub_f" / release / region


def sub_g_region_dir(release: str, region: str) -> Path:
    return _data_processed() / "sub_g" / release / region


def phase1_validated_marker(release: str, region: str) -> Path:
    return sub_g_region_dir(release, region) / "_PHASE1_VALIDATED"


def macro_vocab_path() -> Path:
    """Locked sub-D macro vocab (one source for density/skeleton/zoning buckets)."""
    return _repo_root() / "configs" / "macro_plan" / "v1" / "macro_plan_vocab.yaml"


def eval_set_dir(release: str) -> Path:
    return _data_processed() / "eval_set" / release


# Distinct multi-region eval-set dir — the SG eval-set already occupies
# eval_set/<release>/ (frozen 2026-06-01, write-once), so the EU set CANNOT reuse it.
def multiregion_eval_set_dir(release: str) -> Path:
    return eval_set_dir(release) / "multiregion"


def multiregion_holdout_manifest_path(release: str) -> Path:
    return multiregion_eval_set_dir(release) / "holdout_manifest.yaml"


def multiregion_eval_set_locked_marker(release: str) -> Path:
    return multiregion_eval_set_dir(release) / "_EVAL_SET_LOCKED"


#: The EU multiregion held-out cities (the multiregion manifest's ``held_out_cities``).
#: Hard-coded as the schema/manifest selector (NOT read from the manifest) so the
#: SELECTION is independent of the file it selects — a region picks SG-vs-EU before any
#: manifest is opened. The single source of truth for WHICH tiles each city holds out
#: stays the manifest itself; this set only routes region -> (manifest, schema).
_EU_HELD_OUT_CITIES: frozenset[str] = frozenset({"eisenhuttenstadt", "glasgow", "krakow", "munich"})


def holdout_manifest_for_region(release: str, region: str) -> Path:
    """REGION-AWARE holdout manifest (obligation (a), delta-spec §3 CORRECTION).

    The holdout-manifest readers are dual-region — reached with ``region="singapore"``
    (the local test fixture) AND the 4 EU held-out cities (runtime). The ``region`` arg
    selects the manifest:

      - ``"singapore"``               -> the SG single-region manifest (schema 1.0)
      - one of the 4 EU held-out cities -> the multiregion manifest (schema 2.0)
      - anything else                 -> raise (fail-closed; never silently mis-route)

    Manifest-path and schema selection travel together: ``expected_holdout_schema_for_region``
    returns the matching schema for the SAME region, so a flip-ahead-of-repoint (or a
    repoint-ahead-of-flip) cannot happen.
    """
    if region == DEFAULT_REGION:
        return holdout_manifest_path(release)
    if region in _EU_HELD_OUT_CITIES:
        return multiregion_holdout_manifest_path(release)
    raise ValueError(
        f"holdout_manifest_for_region: unknown region {region!r}; expected "
        f"{DEFAULT_REGION!r} (SG) or one of the EU held-out cities "
        f"{sorted(_EU_HELD_OUT_CITIES)}"
    )


def expected_holdout_schema_for_region(region: str) -> str:
    """REGION-AWARE holdout schema version, the twin of ``holdout_manifest_for_region``.

    SG -> ``"1.0"`` (the frozen, immutable Singapore manifest); the 4 EU held-out cities
    -> ``"2.0"`` (the multiregion manifest). Unknown region -> raise. Manifest and schema
    selection MUST travel together (delta-spec §3 CORRECTION)."""
    if region == DEFAULT_REGION:
        return "1.0"
    if region in _EU_HELD_OUT_CITIES:
        return "2.0"
    raise ValueError(
        f"expected_holdout_schema_for_region: unknown region {region!r}; expected "
        f"{DEFAULT_REGION!r} (SG) or one of the EU held-out cities "
        f"{sorted(_EU_HELD_OUT_CITIES)}"
    )


def holdout_partition_dir(release: str, region: str) -> Path:
    """spec §F: region-keyed holdout partition the training loader excludes."""
    return eval_set_dir(release) / "holdout" / f"region={region}"


def holdout_manifest_path(release: str) -> Path:
    return eval_set_dir(release) / "holdout_manifest.yaml"


def eval_set_locked_marker(release: str) -> Path:
    return eval_set_dir(release) / "_EVAL_SET_LOCKED"

"""One-source path resolution for the eval-set substrate.

Every on-disk location the eval-set reads or writes is built here so no
other module hard-codes a path. Layout verified 2026-06-01:
- sub-{c,d,f,g} region dirs: data/processed/sub_X/<release>/<region>/
- per-tile dir: tile=EPSG3414_i{i}_j{j}/ (sub_d/pipeline.py:156)
- _PHASE1_VALIDATED marker: data/processed/sub_g/<release>/<region>/
"""

from __future__ import annotations

from pathlib import Path

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


def eval_set_dir(release: str) -> Path:
    return _data_processed() / "eval_set" / release


def holdout_partition_dir(release: str, region: str) -> Path:
    """spec §F: region-keyed holdout partition the training loader excludes."""
    return eval_set_dir(release) / "holdout" / f"region={region}"


def holdout_manifest_path(release: str) -> Path:
    return eval_set_dir(release) / "holdout_manifest.yaml"


def eval_set_locked_marker(release: str) -> Path:
    return eval_set_dir(release) / "_EVAL_SET_LOCKED"

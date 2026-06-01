"""One-source path resolution for the training-shard layer.

Reuses cfm.eval.holdout.paths for all sealed inputs (sub-C/D/F dirs, the frozen
holdout manifest, the eval-set marker) so there is a single path authority.
"""

from __future__ import annotations

from pathlib import Path

from cfm.eval.holdout.paths import (  # noqa: F401  (re-exported for one-source reuse)
    _data_processed,
    eval_set_locked_marker,
    holdout_manifest_path,
    sub_d_region_dir,
    sub_f_region_dir,
    sub_g_region_dir,
    tile_dirname,
)


def training_region_dir(release: str, region: str) -> Path:
    """Output dir for materialized per-tile training shards (gitignored)."""
    return _data_processed() / "training" / release / region


def training_manifest_path(release: str, region: str) -> Path:
    return training_region_dir(release, region) / "training_manifest.yaml"

"""Eval-set generation orchestrator (spec §3 dependency web + §6 sequencing).

Corrected sequencing (plan decision 6): BUILD the fresh selector -> G measures
THROUGH it -> (N, selection) -> F freezes the manifest. sub-D's #11 selector is
untouched. delta is the single DELTA_BREF_REGIME (sizing.py).

The freeze (holdout manifest + _EVAL_SET_LOCKED marker) is a WRITE-ONCE,
point-of-no-return act (spec §F: "locked at the start of the project and never
regenerated"). So ``generate_eval_set`` defaults to ``lock=False`` (a dry-run that
measures + proposes the (N, selection) and the per-stratum bref-rate/floors). The
caller freezes deliberately with ``lock=True`` only after reviewing the numbers.

This module reads sealed sub-C/sub-D/sub-F outputs and writes only under
data/processed/eval_set/<release>/ + reports/. Model-facing scoring is deferred
(spec §7).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from cfm.data.sub_d.enums import SlotKind
from cfm.data.sub_d.io import read_macro_core_parquet
from cfm.data.sub_g.readers import read_sub_f_cells
from cfm.eval.holdout import baselines, manifest, paths, selector
from cfm.eval.holdout.bref_rate import bref_placeholder_rate
from cfm.eval.holdout.labels import TileLabels, read_tile_labels
from cfm.eval.holdout.roundtrip import decode_region_blocks
from cfm.eval.holdout.selector import SelectionResult, _tile_stratum
from cfm.eval.holdout.sizing import (
    DELTA_BREF_REGIME,
    ks_two_sample_floor,
    rate_detection_floor,
)

#: The documented step order (corrected sequencing). Asserted in tests.
SEQUENCE: tuple[str, ...] = (
    "labels",
    "bref_rate",
    "baselines",
    "build_selector",
    "size_through_selector",
    "run_degeneracy_guards",
    "freeze_manifest",
    "write_partition_and_marker",
)

# DECISION: v1 distributional effect size for the KS per-stratum floor = 0.15. A KS
# gap of 0.15 between model and real per-stratum distributions is the smallest we aim
# to resolve in v1; revisit against the slow-run populations (a stratum that cannot
# reach this floor is reported UNDERPOWERED, never silently passed). Not a round 0.1.
_KS_EFFECT: float = 0.15

# DECISION: leave at least half the 494-tile pool for training (residual ceiling,
# spec §G). If per-stratum floors need more than this, the binding strata are
# reported UNDERPOWERED rather than consuming the whole pool (degradation option 2).
_DEFAULT_N_CAP_FRACTION: float = 0.5


@dataclass
class EvalSetResult:
    n: int
    proposed_selection: list[tuple[int, int]]
    per_stratum_bref_rate: dict[int, float]
    per_stratum_cell_floor: dict[int, int]
    per_stratum_cell_population: dict[int, int]
    underpowered_cell_density_strata: list[int]
    ceiling_overall: float
    residual: int
    locked: bool
    manifest_path: Path | None
    marker_written: bool
    report_path: Path | None
    degradation_log: list[str] = field(default_factory=list)


def co_optimize(
    tile_labels: list[TileLabels],
    cell_floor: dict[int, int],
    n_cap: int | None = None,
) -> SelectionResult:
    """Grow a uniform per-tile-stratum quota until every cell-density floor is met,
    the pool is exhausted, or N hits the residual cap. Any still-unmet floor is
    reported UNDERPOWERED by the selector (never silently satisfied)."""
    strata = sorted({_tile_stratum(tl) for tl in tile_labels})
    pool_sizes = Counter(_tile_stratum(tl) for tl in tile_labels)
    max_pool = max(pool_sizes.values(), default=0)

    best = selector.select_holdout_tiles(tile_labels, {s: 0 for s in strata}, cell_floor)
    for quota in range(1, max_pool + 1):
        best = selector.select_holdout_tiles(tile_labels, {s: quota for s in strata}, cell_floor)
        if not best.underpowered_cell_density_strata:
            break
        if n_cap is not None and len(best.selected) >= n_cap:
            break
    return best


def _load_inventory(release: str, region: str) -> list[dict]:
    """The 494-tile inventory from the sub-D manifest (authoritative, one source)."""
    md = yaml.safe_load(
        (paths.sub_d_region_dir(release, region) / "manifest.yaml").read_text(encoding="utf-8")
    )
    return md["tiles"]


def _cell_density_by_cell(tile_dir: Path) -> dict[tuple[int, int], int]:
    rows = read_macro_core_parquet(tile_dir / "macro_core.parquet")
    return {
        (int(r.cell_i), int(r.cell_j)): int(r.cell_density_bucket)
        for r in rows
        if r.slot_kind == SlotKind.CELL and r.cell_density_bucket is not None
    }


def _cell_populations(tile_labels: list[TileLabels]) -> dict[int, int]:
    pops: Counter[int] = Counter()
    for tl in tile_labels:
        pops.update(tl.cell_density_buckets)
    return dict(pops)


def generate_eval_set(
    *,
    release: str,
    region: str,
    lock: bool = False,
    n_cap_fraction: float = _DEFAULT_N_CAP_FRACTION,
) -> EvalSetResult:
    """Measure + propose (lock=False) or measure + freeze (lock=True) the eval set.

    Steps follow SEQUENCE: read labels + decode round-tripped-real per tile; compute
    the §2 shared bref-rate ONCE (stratified by cell_density_bucket); the ceiling;
    per-stratum cell floors (rate + KS); co-optimize (N, selection) through the fresh
    selector; freeze the manifest only when lock=True.
    """
    inventory = _load_inventory(release, region)
    sub_d_dir = paths.sub_d_region_dir(release, region)
    sub_f_dir = paths.sub_f_region_dir(release, region)

    tile_labels: list[TileLabels] = []
    blocks: list[list[int]] = []
    geoms: list[dict] = []
    strata: list[int] = []
    provenance_by_tile: dict[tuple[int, int], dict] = {}

    for entry in inventory:
        ti, tj = int(entry["tile_i"]), int(entry["tile_j"])
        dirname = paths.tile_dirname(ti, tj)
        tile_dir = sub_d_dir / dirname

        tile_labels.append(read_tile_labels(tile_dir, tile_i=ti, tile_j=tj))

        cdbc = _cell_density_by_cell(tile_dir)
        tokens = read_sub_f_cells(sub_f_dir / dirname / "cells.parquet")
        tb, tg, ts = decode_region_blocks(tokens, cdbc)
        blocks.extend(tb)
        geoms.extend(tg)
        strata.extend(ts)

        prov = yaml.safe_load((tile_dir / "provenance.yaml").read_text(encoding="utf-8"))
        provenance_by_tile[(ti, tj)] = {
            "provenance_sha256": entry["provenance_sha256"],
            "macro_vocab_sha256": prov.get("inputs", {}).get("macro_vocab_sha256"),
        }

    rate = bref_placeholder_rate(blocks, geoms, strata)
    ceiling = baselines.geometric_validity_ceiling(rate)

    cell_floor: dict[int, int] = {}
    for bucket, sr in rate.per_stratum.items():
        p = max(sr.rate, DELTA_BREF_REGIME)  # avoid a degenerate p=0 floor
        cell_floor[bucket] = max(
            rate_detection_floor(p=p, delta=DELTA_BREF_REGIME),
            ks_two_sample_floor(effect=_KS_EFFECT),
        )

    n_cap = int(len(inventory) * n_cap_fraction)
    selection = co_optimize(tile_labels, cell_floor, n_cap=n_cap)
    n = len(selection.selected)
    residual = len(inventory) - n

    degradation_log: list[str] = []
    if selection.underpowered_cell_density_strata:
        for bucket, sf in sorted(selection.underpowered_cell_density_strata.items()):
            degradation_log.append(
                f"UNDERPOWERED cell_density_bucket={bucket}: "
                f"have {sf.available} cells, floor {sf.floor} (reported, not passed)"
            )

    result = EvalSetResult(
        n=n,
        proposed_selection=selection.selected,
        per_stratum_bref_rate={b: sr.rate for b, sr in rate.per_stratum.items()},
        per_stratum_cell_floor=cell_floor,
        per_stratum_cell_population=_cell_populations(tile_labels),
        underpowered_cell_density_strata=sorted(selection.underpowered_cell_density_strata),
        ceiling_overall=ceiling.overall,
        residual=residual,
        locked=False,
        manifest_path=None,
        marker_written=False,
        report_path=None,
        degradation_log=degradation_log,
    )

    if lock:
        man = manifest.build_holdout_manifest(
            region=region,
            selected_tiles=selection.selected,
            per_tile_provenance={k: provenance_by_tile[k] for k in selection.selected},
        )
        man_path = paths.holdout_manifest_path(release)
        manifest.freeze_holdout_manifest(man, man_path)
        marker = paths.eval_set_locked_marker(release)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            yaml.safe_dump(
                {
                    "release": release,
                    "region": region,
                    "n_held_out": n,
                    "training_residual": residual,
                    "ceiling_overall": ceiling.overall,
                    "delta_bref_regime": DELTA_BREF_REGIME,
                }
            ),
            encoding="utf-8",
        )
        result.locked = True
        result.manifest_path = man_path
        result.marker_written = True

    return result

"""Eval-set generation orchestrator (spec §3 dependency web + §6 sequencing).

Corrected sequencing (plan decision 6): BUILD the fresh selector -> G measures
THROUGH it -> (N, selection) -> F freezes the manifest. sub-D's #11 selector is
untouched. The over-emission threshold is relative-to-base-rate (sizing.py
over_emission_threshold). MEASURED FINDING (2026-06-01): D's rate-detection floor is
in FEATURES and feature populations are abundant (~873k across the pool), so it is
NON-BINDING - contradicting the spec's "D's stratified floor is the binding one".
What drives N is the PROVISIONAL cell-density reference target (KS distance deferred,
spec §7). Both reported; D's feature power is verified per stratum.

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

import math
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
    DELTA_FLOOR_BREF,
    RHO_BREF_REGIME,
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

# DECISION: KS target gap = 0.08 (freeze sizing, 2026-06-01 PI call). N is sized to
# resolve a per-stratum KS distributional gap of 0.08 between model and real. This is
# an EXPLICIT, RECORDED choice (not a hidden default) on a single-region tradeoff
# curve bounded at ~0.049 (the finest gap this region can EVER resolve - the binding
# cell_density_bucket has the fewest cells). Chosen toward over-provisioning because
# the freeze is WRITE-ONCE: under-provisioning is unrecoverable (can shrink, never
# grow), over-provisioning costs ~11% training tiles that degrade all architectures
# EQUALLY (so it does not distort the bake-off COMPARISON), and capable models on the
# same data commonly differ by sub-0.10 gaps - exactly where discrimination matters.
# The gap the frozen set ACTUALLY resolves is reported; the eval-harness asserts the
# model's needed gap >= the frozen resolved gap and FAILS LOUD otherwise (-> the
# documented second-region trigger, never silent under-power).
KS_TARGET_GAP: float = 0.08

# DECISION: leave at least half the 494-tile pool for training (residual ceiling,
# spec §G). If per-stratum floors need more than this, the binding strata are
# reported UNDERPOWERED rather than consuming the whole pool (degradation option 2).
_DEFAULT_N_CAP_FRACTION: float = 0.5


def _gap_from_cells(cells: int) -> float:
    """Finest KS two-sample gap resolvable from ``cells`` (inverse of the KS floor)."""
    return 1.358 * math.sqrt(2.0 / cells) if cells > 0 else float("inf")


@dataclass
class EvalSetResult:
    n: int
    proposed_selection: list[tuple[int, int]]
    per_stratum_bref_rate: dict[int, float]
    per_stratum_cell_floor: dict[
        int, int
    ]  # PROVISIONAL cell-density reference target (KS deferred)
    per_stratum_cell_population: dict[int, int]
    underpowered_cell_density_strata: list[int]
    ceiling_overall: float
    residual: int
    locked: bool
    manifest_path: Path | None
    marker_written: bool
    report_path: Path | None
    degradation_log: list[str] = field(default_factory=list)
    # D's rate-detection power (FEATURES) - measured non-binding, but verified per stratum.
    per_stratum_feature_population: dict[int, int] = field(default_factory=dict)  # full pool
    per_stratum_feature_floor: dict[int, int] = field(default_factory=dict)  # to detect rho-excess
    held_out_feature_population: dict[int, int] = field(default_factory=dict)  # in selected tiles
    underpowered_feature_strata: list[int] = field(default_factory=list)
    # KS distributional resolution (write-once sizing). target = the chosen gap N is
    # sized to; resolved = the gap the frozen set ACTUALLY resolves (binding stratum);
    # single_region_floor = the finest gap this region can EVER resolve (full pool).
    ks_target_gap: float = 0.0
    ks_resolved_gap_binding: float = 0.0
    ks_single_region_floor: float = 0.0
    held_out_cell_population: dict[int, int] = field(default_factory=dict)


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
    ks_target_gap: float = KS_TARGET_GAP,
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
    per_tile_stratum_features: dict[tuple[int, int], Counter[int]] = {}

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
        per_tile_stratum_features[(ti, tj)] = Counter(ts)  # features per stratum, this tile

        prov = yaml.safe_load((tile_dir / "provenance.yaml").read_text(encoding="utf-8"))
        provenance_by_tile[(ti, tj)] = {
            "provenance_sha256": entry["provenance_sha256"],
            "macro_vocab_sha256": prov.get("inputs", {}).get("macro_vocab_sha256"),
        }

    rate = bref_placeholder_rate(blocks, geoms, strata)
    ceiling = baselines.geometric_validity_ceiling(rate)

    # D's rate-detection floor is in FEATURES (each feature is a Bernoulli collapse/not)
    # and detects a RELATIVE excess rho around the faithful rate. MEASURED non-binding:
    # feature populations dwarf these floors. Computed + verified, not used to drive N.
    feature_floor: dict[int, int] = {
        b: rate_detection_floor(p=sr.rate, delta=RHO_BREF_REGIME * sr.rate)
        for b, sr in rate.per_stratum.items()
        if sr.rate > 0
    }
    feature_population = {b: sr.n_total for b, sr in rate.per_stratum.items()}

    # Selection driver: per-stratum cell-density reference target sized to the EXPLICIT
    # ks_target_gap. The KS distance itself is model-facing and DEFERRED (spec §7); this
    # sizes the write-once set to resolve a chosen gap. N is an explicit function of the
    # recorded gap, never a hidden default.
    cell_ref_floor: dict[int, int] = {
        b: ks_two_sample_floor(effect=ks_target_gap) for b in rate.per_stratum
    }

    n_cap = int(len(inventory) * n_cap_fraction)
    selection = co_optimize(tile_labels, cell_ref_floor, n_cap=n_cap)
    n = len(selection.selected)
    residual = len(inventory) - n
    selected_set = set(selection.selected)

    # Held-out CELL populations + the KS gap the frozen set ACTUALLY resolves (the
    # binding/coarsest stratum). single_region_floor = finest gap this region can EVER
    # resolve using the full pool (the binding cell_density_bucket). Both reported so
    # the eval-harness can assert model-needed-gap >= resolved gap and fail loud.
    full_pool_cells = _cell_populations(tile_labels)
    held_out_cells: Counter[int] = Counter()
    for tl in tile_labels:
        if (tl.tile_i, tl.tile_j) in selected_set:
            held_out_cells.update(tl.cell_density_buckets)
    resolved_binding = (
        max(_gap_from_cells(held_out_cells[b]) for b in held_out_cells)
        if held_out_cells
        else float("inf")
    )
    single_region_floor = (
        max(_gap_from_cells(c) for c in full_pool_cells.values())
        if full_pool_cells
        else float("inf")
    )

    # Held-out feature populations (verify D's rate-detection power in the SELECTED set,
    # so the vacuous pass cannot hide in the sample size - the 2026-06-01 review point).
    held_out_features: Counter[int] = Counter()
    for key in selected_set:
        held_out_features.update(per_tile_stratum_features.get(key, Counter()))
    underpowered_feature = sorted(
        b for b, floor in feature_floor.items() if held_out_features.get(b, 0) < floor
    )

    degradation_log: list[str] = []
    for bucket, sf in sorted(selection.underpowered_cell_density_strata.items()):
        degradation_log.append(
            f"UNDERPOWERED cell-density REFERENCE (provisional) bucket={bucket}: "
            f"have {sf.available} cells, target {sf.floor} (reported, not passed)"
        )
    for bucket in underpowered_feature:
        degradation_log.append(
            f"UNDERPOWERED rate-detection bucket={bucket}: held-out features "
            f"{held_out_features.get(bucket, 0)} < floor {feature_floor[bucket]} (reported)"
        )

    result = EvalSetResult(
        n=n,
        proposed_selection=selection.selected,
        per_stratum_bref_rate={b: sr.rate for b, sr in rate.per_stratum.items()},
        per_stratum_cell_floor=cell_ref_floor,
        per_stratum_cell_population=_cell_populations(tile_labels),
        underpowered_cell_density_strata=sorted(selection.underpowered_cell_density_strata),
        ceiling_overall=ceiling.overall,
        residual=residual,
        locked=False,
        manifest_path=None,
        marker_written=False,
        report_path=None,
        degradation_log=degradation_log,
        per_stratum_feature_population=feature_population,
        per_stratum_feature_floor=feature_floor,
        held_out_feature_population=dict(held_out_features),
        underpowered_feature_strata=underpowered_feature,
        ks_target_gap=ks_target_gap,
        ks_resolved_gap_binding=resolved_binding,
        ks_single_region_floor=single_region_floor,
        held_out_cell_population=dict(held_out_cells),
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
                    "rho_bref_regime": RHO_BREF_REGIME,
                    "delta_floor_bref": DELTA_FLOOR_BREF,
                    "ks_target_gap": ks_target_gap,
                    "ks_resolved_gap_binding": resolved_binding,
                    "ks_single_region_floor": single_region_floor,
                }
            ),
            encoding="utf-8",
        )
        result.locked = True
        result.manifest_path = man_path
        result.marker_written = True

    return result

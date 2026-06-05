"""Read-only diagnostic for the batch-2 sub-F BP7 symmetry failures.

NOT a fix and NOT a validator change — pure instrumentation (Phase-1
evidence gathering per systematic-debugging). Reconstructs the exact
`_check_symmetry` disagreement set the validator would raise on, WITHOUT
raising, then classifies each disagreeing internal edge against sub-C's
crossings/features/cells to bucket false-positive (road terminates at the
internal cell boundary, or neighbour segment sea-dropped) vs real defect
(road crosses, neighbour segment present but no bref / genuinely missing).

Also audits the "symmetric-but-wrong" worry on passing cities: every
contract MINOR_ROAD/MAJOR_ROAD internal edge must trace to a genuine road
crossing in sub-C (class_raw in the road set) — if a non-road feature
produced the active class, the blast radius is the sub-E contract (all 40
cities), not just emission asymmetry.

Usage:
  python symmetry_probe.py --release 2026-04-15.0 --city bruges [--max-examples 8]
  python symmetry_probe.py --release 2026-04-15.0 --audit-contract --city bologna
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from cfm.data.sub_f.boundary_contract import load_boundary_contract
from cfm.data.sub_f.rotation import cell_edge_directions
from cfm.data.sub_f.validator_cross_tile import (
    _OPPOSITE_DIRECTION,
    _bref_id_to_dir_class,
    _emitted_brefs_by_cell,
    _neighbour_cell,
    _semantic_id_to_tag,
)

# sub-E §5.1 road class_raw set (MAJOR + MINOR). Everything else (water, rail,
# null) → NONE, so a MINOR/MAJOR contract edge MUST trace to one of these.
ROAD_CLASS_RAW = {
    "primary",
    "trunk",
    "secondary",  # MAJOR
    "tertiary",
    "residential",
    "service",
    "unclassified",
    "footway",
    "steps",
    "cycleway",  # explicit MINOR
}
# Plus the default-bucket rule: any other non-null value also → MINOR_ROAD,
# but ONLY if it is a transportation feature (sub-E reads crossings, which are
# road-derived). We treat presence in crossings.parquet as the road witness.

# Non-road LineString classes (waterways + rail). They cross internal edges but
# per §5.1 NEVER emit brefs and NEVER set the edge class (→ NONE in sub-E), so a
# crossing record for one is NOT a road point-crossing and must be excluded from
# the touch-as-cross / drop census. SHARED by classify_point_crossings,
# touch_census, and touch_detail so the exclusion set cannot drift between them.
NONROAD_CLASS_RAW = {
    "river",
    "canal",
    "stream",
    "drain",
    "ditch",
    "rail",
    "railway",
    "light_rail",
    "tram",
    "subway",
    "monorail",
    "funicular",
}


def _read(tile_dir: Path, name: str) -> list[dict]:
    f = tile_dir / name
    if not f.exists():
        return []
    return pq.ParquetFile(f).read().to_pylist()


def _by_cell_dir(emitted_by_cell: dict) -> dict:
    out: dict[tuple[tuple[int, int], str], set[str]] = {}
    for cell, emitted in emitted_by_cell.items():
        for direction, cls, _fk in emitted:
            out.setdefault((cell, direction), set()).add(cls)
    return out


def scan_city(release: str, city: str, max_examples: int) -> None:
    base = Path("data/processed")
    sub_f = base / "sub_f" / release / city
    sub_c = base / "sub_c" / release / city
    bref_decode = _bref_id_to_dir_class()
    sem_id_to_tag = _semantic_id_to_tag()

    tiles = sorted(sub_f.glob("tile=*/cells.parquet"))
    label_counts: Counter[str] = Counter()
    tiles_with_disagreement = 0
    total_disagreements = 0
    examples: list[str] = []

    for tile_path in tiles:
        tile = tile_path.parent.name
        cells_rows = pq.ParquetFile(tile_path).read().to_pylist()
        emitted = _emitted_brefs_by_cell(cells_rows, bref_decode, sem_id_to_tag)
        bcd = _by_cell_dir(emitted)

        # find disagreements exactly as _check_symmetry does
        disagreements = []
        for (cell, direction), classes in bcd.items():
            nb = _neighbour_cell(cell, direction)
            if nb is None:
                continue
            opp = _OPPOSITE_DIRECTION[direction]
            nb_classes = bcd.get((nb, opp), set())
            if classes != nb_classes:
                disagreements.append((cell, direction, nb, opp, classes, nb_classes))
        if not disagreements:
            continue
        tiles_with_disagreement += 1
        total_disagreements += len(disagreements)

        crossings = _read(sub_c / tile, "crossings.parquet")
        feats = _read(sub_c / tile, "features.parquet")
        cells_c = {(r["cell_i"], r["cell_j"]): r for r in _read(sub_c / tile, "cells.parquet")}
        feats_by_cell: dict[tuple[int, int], list[dict]] = {}
        for r in feats:
            feats_by_cell.setdefault((r["cell_i"], r["cell_j"]), []).append(r)
        # ROAD witness only: a bref is emitted by a road LineString, and only
        # road crossings set the edge class (§5.1: non-road -> NONE). A polygon
        # crossing the same edge is NOT a road witness — exclude it so a
        # co-located building doesn't masquerade as 'road present both sides'.
        from shapely import wkb as _wkb

        # Non-road LineStrings (waterways/rail) DON'T emit brefs and DON'T set the
        # edge class (§5.1: non-road crossing -> NONE). Exclude them so a river/
        # canal/stream sharing the edge isn't counted as 'road present both sides'.
        nonroad_class = {
            "river",
            "canal",
            "stream",
            "drain",
            "ditch",
            "rail",
            "railway",
            "light_rail",
            "tram",
            "subway",
            "monorail",
            "funicular",
        }
        line_fids = {
            r["source_feature_id"]
            for r in feats
            if _wkb.loads(r["geometry"]).geom_type in ("LineString", "MultiLineString")
            and str(r["class_raw"]) not in nonroad_class
        }

        for cell, direction, nb, opp, classes, nb_classes in disagreements:
            emit_cell = cell if classes else nb
            miss_cell = nb if classes else cell
            emit_dir = direction if classes else opp
            li, lj, axis, _kind = cell_edge_directions(cell[0], cell[1])[direction]
            edge_cr = [
                r
                for r in crossings
                if r["lower_cell_i"] == li and r["lower_cell_j"] == lj and r["axis"] == axis
            ]
            fids = {r["source_feature_id"] for r in edge_cr}
            road_fids = fids & line_fids  # road witnesses only (exclude polygons)
            ev = sorted(r["event_type"] for r in edge_cr)
            fid_in_miss = any(
                r["source_feature_id"] in road_fids for r in feats_by_cell.get(miss_cell, [])
            )
            mc = cells_c.get(miss_cell, {})
            water = float(mc.get("water_fraction", 0.0) or 0.0)
            sea = float(mc.get("sea_water_fraction", 0.0) or 0.0)
            kept = mc.get("kept_features_count", None)

            if classes and nb_classes and classes != nb_classes:
                label = "class_mismatch(MAJORvsMINOR)"
            elif fid_in_miss:
                label = "road_present_both_no_bref(encoder/float)"
            elif sea >= 0.5 or water >= 0.5:
                label = "neighbour_sea/water_dropped(>=0.5)"
            elif sea > 0.0 or water > 0.0:
                label = "neighbour_partial_water(<0.5)"
            else:
                label = "termination/touch_dry(FP candidate)"
            label_counts[label] += 1

            if len(examples) < max_examples:
                miss_dir = _OPPOSITE_DIRECTION[emit_dir]
                examples.append(
                    f"  {tile} edge {emit_cell}.{emit_dir}->{miss_cell}.{miss_dir} "
                    f"| emit={sorted(classes or nb_classes)} miss=[] "
                    f"| crossings_ev={ev} road_fids={len(road_fids)} fid_in_miss={fid_in_miss} "
                    f"| miss water={water:.2f} sea={sea:.2f} kept={kept} | {label}"
                )

    print(f"\n========== {city} ==========")
    print(
        f"tiles total={len(tiles)}  tiles_with_disagreement={tiles_with_disagreement}  "
        f"total_disagreeing_edges={total_disagreements}"
    )
    print("LABEL BREAKDOWN:")
    for lab, n in label_counts.most_common():
        print(f"  {n:6d}  {lab}")
    print("EXAMPLES:")
    for e in examples:
        print(e)


def audit_contract(release: str, city: str) -> None:
    """Passing-city check: every active (MAJOR/MINOR) internal edge must trace
    to a road crossing in sub-C. If an active edge has NO crossing, sub-E
    invented an active class (the symmetric-but-wrong blast-radius case)."""
    base = Path("data/processed")
    sub_e = base / "sub_e" / release / city
    sub_c = base / "sub_c" / release / city
    tiles = sorted(sub_e.glob("tile=*/boundary_contract.parquet"))
    active_edges = 0
    active_no_crossing = 0
    bad_examples: list[str] = []
    for bc_path in tiles:
        tile = bc_path.parent.name
        contract = load_boundary_contract(bc_path)
        crossings = _read(sub_c / tile, "crossings.parquet")
        cr_edges = {(r["lower_cell_i"], r["lower_cell_j"], r["axis"]) for r in crossings}
        seen_edge: set[tuple[int, int, int]] = set()
        for (ci, cj), edges in contract.items():
            for direction, cls in edges.items():
                if cls not in ("MAJOR_ROAD", "MINOR_ROAD"):
                    continue
                li, lj, axis, _kind = cell_edge_directions(ci, cj)[direction]
                key = (li, lj, axis)
                if key in seen_edge:
                    continue  # internal edge shared — count once
                seen_edge.add(key)
                active_edges += 1
                if key not in cr_edges:
                    active_no_crossing += 1
                    if len(bad_examples) < 8:
                        bad_examples.append(f"  {tile} edge {key} cls={cls} has NO sub-C crossing")
    print(f"\n===== CONTRACT AUDIT {city} =====")
    print(f"active internal edges={active_edges}  active_without_crossing={active_no_crossing}")
    for e in bad_examples:
        print(e)
    if active_no_crossing == 0:
        print(
            "  OK: every active edge traces to a real sub-C road crossing (no non-road mislabel)."
        )


def explain_edge(release: str, city: str, tile: str, ci: int, cj: int, direction: str) -> None:
    """Dump the geometry of the crossing road(s) on one disagreeing edge in BOTH
    cells, with each endpoint/vertex distance to the shared edge line, to tell
    'along-edge run (endpoints elsewhere)' from 'endpoint off by float'."""
    from shapely import wkb

    base = Path("data/processed")
    sub_c = base / "sub_c" / release / city / tile
    feats = _read(sub_c, "features.parquet")
    crossings = _read(sub_c, "crossings.parquet")
    nb = _neighbour_cell((ci, cj), direction)
    opp = _OPPOSITE_DIRECTION[direction]
    li, lj, axis, _k = cell_edge_directions(ci, cj)[direction]
    coord_idx = 0 if axis == 0 else 1  # axis0=E/W=vertical(x); axis1=N/S=horizontal(y)
    edge_cr = [
        r
        for r in crossings
        if r["lower_cell_i"] == li and r["lower_cell_j"] == lj and r["axis"] == axis
    ]
    fids = {r["source_feature_id"] for r in edge_cr}
    print(f"\n--- EXPLAIN {city}/{tile} ({ci},{cj}).{direction} <-> {nb}.{opp} axis={axis} ---")
    edge_summary = [{"fid": r["source_feature_id"][:8], "ev": r["event_type"]} for r in edge_cr]
    print(f"crossings on edge: {edge_summary}")

    def cell_feats(cell):
        return [
            r
            for r in feats
            if (r["cell_i"], r["cell_j"]) == cell and r["source_feature_id"] in fids
        ]

    print(f"(coord axis index={coord_idx}; the on-edge coordinate should match between cells)")
    for cell, _lbl in [((ci, cj), "EMIT"), (nb, "MISS")]:
        for r in cell_feats(cell):
            g = wkb.loads(r["geometry"])
            if g.geom_type != "LineString":
                print(f"  cell {cell} fid={r['source_feature_id'][:8]} gtype={g.geom_type}")
                continue
            cs = list(g.coords)
            ev = [round(c[coord_idx], 4) for c in cs]
            fid8 = r["source_feature_id"][:8]
            print(
                f"  cell {cell} fid={fid8} class={r['class_raw']} nverts={len(cs)} "
                f"axis-coords(first,last)=({ev[0]},{ev[-1]}) all={ev}"
            )


def _find_disagreements(release: str, city: str) -> list[dict]:
    """Re-derive the validator's symmetry-disagreement set for a city, with the
    road witnesses and miss/emit cells, for source-tracing."""
    from shapely import wkb as _wkb

    base = Path("data/processed")
    sub_f = base / "sub_f" / release / city
    sub_c = base / "sub_c" / release / city
    bref_decode = _bref_id_to_dir_class()
    sem_id_to_tag = _semantic_id_to_tag()
    nonroad = {
        "river",
        "canal",
        "stream",
        "drain",
        "ditch",
        "rail",
        "railway",
        "light_rail",
        "tram",
        "subway",
        "monorail",
        "funicular",
    }
    out: list[dict] = []
    for tile_path in sorted(sub_f.glob("tile=*/cells.parquet")):
        tile = tile_path.parent.name
        cells_rows = pq.ParquetFile(tile_path).read().to_pylist()
        emitted = _emitted_brefs_by_cell(cells_rows, bref_decode, sem_id_to_tag)
        bcd = _by_cell_dir(emitted)
        diss = []
        for (cell, d), classes in bcd.items():
            nb = _neighbour_cell(cell, d)
            if nb is None:
                continue
            if classes != bcd.get((nb, _OPPOSITE_DIRECTION[d]), set()):
                diss.append((cell, d, nb))
        if not diss:
            continue
        feats = _read(sub_c / tile, "features.parquet")
        crossings = _read(sub_c / tile, "crossings.parquet")
        line_fids = {
            r["source_feature_id"]
            for r in feats
            if _wkb.loads(r["geometry"]).geom_type in ("LineString", "MultiLineString")
            and str(r["class_raw"]) not in nonroad
        }
        feats_by_cell: dict[tuple[int, int], set[str]] = {}
        for r in feats:
            feats_by_cell.setdefault((r["cell_i"], r["cell_j"]), set()).add(r["source_feature_id"])
        for cell, d, nb in diss:
            emit_cell = cell if (bcd.get((cell, d))) else nb
            miss_cell = nb if (bcd.get((cell, d))) else cell
            li, lj, axis, _k = cell_edge_directions(cell[0], cell[1])[d]
            edge_cr = [
                r
                for r in crossings
                if r["lower_cell_i"] == li and r["lower_cell_j"] == lj and r["axis"] == axis
            ]
            road_fids = {r["source_feature_id"] for r in edge_cr} & line_fids
            emit_road_fids = road_fids & feats_by_cell.get(emit_cell, set())
            out.append(
                {
                    "tile": tile,
                    "emit_cell": emit_cell,
                    "miss_cell": miss_cell,
                    "direction": d if emit_cell == cell else _OPPOSITE_DIRECTION[d],
                    "axis": axis,
                    "road_fids": sorted(road_fids),
                    "emit_road_fids": sorted(emit_road_fids),
                    "miss_has_fid": bool(road_fids & feats_by_cell.get(miss_cell, set())),
                }
            )
    return out


def source_trace(release: str, city: str) -> None:
    """THIRD-AUTHORITY check: for each disagreeing edge, pull the emitting road's
    UNCLIPPED Overture source geometry, replicate sub-C's bbox clip independently
    (keep = road ∩ bbox), and test whether `keep` enters the MISS cell. If it does
    and sub-C has no fragment there -> sub-C DROP (real defect). If `keep` ends at
    the edge -> termination (FP). Independent of the sub-C clip output."""
    import yaml
    from shapely import wkb as _wkb
    from shapely.geometry import box as _box

    from cfm.data.sub_c.coords import region_coords

    cache = Path("data/cache/overture") / release / city
    man = yaml.safe_load((cache / "manifest.yaml").read_text())
    w, s, e, n = man["scope"]["bbox"]
    trans = pq.ParquetFile(cache / "transportation.parquet").read()
    ids = trans.column("id").to_pylist()
    geoms = trans.column("geometry").to_pylist()
    id2geom = dict(zip(ids, geoms, strict=True))

    diss = _find_disagreements(release, city)
    print(f"\n========== SOURCE-TRACE {city}: {len(diss)} disagreeing edges ==========")
    verdicts: list[str] = []
    for dd in diss:
        tile = dd["tile"]
        crs = "EPSG:" + tile.split("=")[1].split("_")[0].replace("EPSG", "")
        rc = region_coords(crs)
        bbox_proj = rc.reproject_geometry(_box(w, s, e, n).segmentize(0.01))
        ti = int(tile.split("_i")[1].split("_")[0])
        tj = int(tile.split("_j")[1])
        bcx, bcy = dd["miss_cell"]
        ecx, ecy = dd["emit_cell"]
        miss_sq = _box(
            ti * 2000 + bcx * 250,
            tj * 2000 + bcy * 250,
            ti * 2000 + (bcx + 1) * 250,
            tj * 2000 + (bcy + 1) * 250,
        )
        emit_sq = _box(
            ti * 2000 + ecx * 250,
            tj * 2000 + ecy * 250,
            ti * 2000 + (ecx + 1) * 250,
            tj * 2000 + (ecy + 1) * 250,
        )
        for fid in dd["emit_road_fids"] or dd["road_fids"]:
            raw = id2geom.get(fid)
            if raw is None:
                print(f"  {tile} {dd['emit_cell']}.{dd['direction']} fid={fid[:8]} SOURCE-MISSING")
                continue
            road = rc.reproject_geometry(_wkb.loads(raw))
            keep = road.intersection(bbox_proj)
            len_emit = keep.intersection(emit_sq).length
            len_miss = keep.intersection(miss_sq).length
            total = road.length
            kept = keep.length
            verdict = "DROP(REAL-DEFECT)" if len_miss > 1.0 else "termination/clip(FP)"
            verdicts.append(verdict)
            print(
                f"  {tile} {dd['emit_cell']}.{dd['direction']}->{dd['miss_cell']} fid={fid[:8]} "
                f"| src_len={total:.0f}m kept(after bbox)={kept:.0f}m "
                f"| len_in_EMIT={len_emit:.1f}m len_in_MISS={len_miss:.1f}m "
                f"miss_has_fid={dd['miss_has_fid']} => {verdict}"
            )
    from collections import Counter as _C

    print("  VERDICTS:", dict(_C(verdicts)))


def classify_point_crossings(
    crossings_rows: list[dict],
    features_rows: list[dict],
    nonroad: set[str] = NONROAD_CLASS_RAW,
) -> dict[str, int]:
    """Bucket each ROAD point-crossing record by how many flanking cells carry it.

    For one tile: a sub-C ``crossings.parquet`` record with
    ``edge_extent_length_m == 0`` is a point-crossing on an internal cell edge.
    For a ROAD one (its feature's ``class_raw`` not in ``nonroad``), look up
    whether the road's clipped fragment is present in each of the two flanking
    cells' ``features.parquet`` rows and bucket:

    - ``real_cross``    — fragment in BOTH cells (a genuine through-crossing).
    - ``touch_as_cross``— fragment in EXACTLY ONE cell (the §8.3 touch recorded
      as a cross; the spurious-but-tolerated population the v1.2 relax accepts).
    - ``anomaly``       — fragment in NEITHER cell. This is an ORPHANED crossing
      record: a road crosses the edge per ``crossings.parquet`` yet sub-C wrote
      no fragment on either side — the DROP signature. ``anomaly`` is the SOLE
      corpus-wide drop check for cities the source-trace gate cannot reach
      (passing cities have no symmetry disagreement to trace), so it is
      load-bearing. Its teeth (that a real orphan is COUNTED here, not silently
      filtered) are pinned by tests/data/multiregion/test_touch_census_teeth.py.

    ``point_cross`` is the total of the three. Pure over row dicts (no I/O) so the
    teeth test exercises THIS exact classifier, not a reimplementation.
    """
    fids_by_cell: dict[tuple[int, int], set[str]] = {}
    fid_class: dict[str, str] = {}
    for r in features_rows:
        fids_by_cell.setdefault((r["cell_i"], r["cell_j"]), set()).add(r["source_feature_id"])
        fid_class[r["source_feature_id"]] = str(r["class_raw"])
    tot = {"point_cross": 0, "real_cross": 0, "touch_as_cross": 0, "anomaly": 0}
    for r in crossings_rows:
        if r["edge_extent_length_m"] != 0.0:
            continue  # interval (polygon edge-interval), not a road point-crossing
        fid = r["source_feature_id"]
        if fid_class.get(fid) in nonroad:
            continue  # waterway/rail crossing -> NONE in sub-E, never MINOR_ROAD
        li, lj, ax = r["lower_cell_i"], r["lower_cell_j"], r["axis"]
        cA, cB = ((li, lj), (li + 1, lj)) if ax == 0 else ((li, lj), (li, lj + 1))
        in_a = fid in fids_by_cell.get(cA, set())
        in_b = fid in fids_by_cell.get(cB, set())
        tot["point_cross"] += 1
        if in_a and in_b:
            tot["real_cross"] += 1
        elif in_a or in_b:
            tot["touch_as_cross"] += 1
        else:
            tot["anomaly"] += 1
    return tot


def touch_census(release: str) -> None:
    """Corpus-wide §8.3 touch-as-cross census. A road point-crossing record
    (edge_extent_length_m==0) whose road has a real feature-row fragment in only
    ONE flanking cell = a touch recorded as a cross (spec §8.3: 'feature wholly in
    one cell' must produce 0 crossing records). Counts the FULL population, not
    just the asymmetric subset the symmetry leg catches. Validated-city counts =
    the silent (symmetric-manifestation) population that passed validation. The
    ``anom`` column is the load-bearing corpus-wide DROP check (see
    classify_point_crossings); ``anomaly==0`` means orphaned crossings were
    looked for and none found, NOT that none could be detected."""
    base = Path("data/processed/sub_c") / release
    cities = sorted(p.name for p in base.glob("*") if p.is_dir())
    grand = {"point_cross": 0, "real_cross": 0, "touch_as_cross": 0, "anomaly": 0}
    print(f"{'city':<18} {'pt_cross':>9} {'real(both)':>11} {'TOUCH(one)':>11} {'anom':>6}  tiles")
    for city in cities:
        cdir = base / city
        tot = {"point_cross": 0, "real_cross": 0, "touch_as_cross": 0, "anomaly": 0}
        ntiles = 0
        for tile_dir in cdir.glob("tile=*"):
            cr = _read(tile_dir, "crossings.parquet")
            fe = _read(tile_dir, "features.parquet")
            if not fe:
                continue
            ntiles += 1
            tile_counts = classify_point_crossings(cr, fe)
            for k in tot:
                tot[k] += tile_counts[k]
        if ntiles == 0:
            continue
        for k in grand:
            grand[k] += tot[k]
        print(
            f"{city:<18} {tot['point_cross']:>9} {tot['real_cross']:>11} "
            f"{tot['touch_as_cross']:>11} {tot['anomaly']:>6}  {ntiles}"
        )
    print(
        f"{'TOTAL':<18} {grand['point_cross']:>9} {grand['real_cross']:>11} "
        f"{grand['touch_as_cross']:>11} {grand['anomaly']:>6}"
    )


def touch_detail(release: str, city: str) -> None:
    """For each §8.3 touch-as-cross edge in a city, report sub-F emission on BOTH
    flanking cells — to confirm the symmetric-manifestation MECHANISM (both emit
    via two opposite-side roads, vs neither emits) rather than assume it."""
    nonroad = NONROAD_CLASS_RAW
    base = Path("data/processed")
    sub_c = base / "sub_c" / release / city
    sub_f = base / "sub_f" / release / city
    bref_decode = _bref_id_to_dir_class()
    sem_id_to_tag = _semantic_id_to_tag()
    cnt = {"both_emit(sym)": 0, "one_emit(asym)": 0, "neither_emit(sym)": 0}
    examples: list[str] = []
    for tile_dir in sorted(sub_c.glob("tile=*")):
        tile = tile_dir.name
        cr = _read(tile_dir, "crossings.parquet")
        fe = _read(tile_dir, "features.parquet")
        cells_path = sub_f / tile / "cells.parquet"
        if not fe or not cells_path.exists():
            continue
        emitted = _emitted_brefs_by_cell(
            pq.ParquetFile(cells_path).read().to_pylist(), bref_decode, sem_id_to_tag
        )
        bcd = _by_cell_dir(emitted)
        fids_by_cell: dict[tuple[int, int], set[str]] = {}
        fid_class: dict[str, str] = {}
        for r in fe:
            fids_by_cell.setdefault((r["cell_i"], r["cell_j"]), set()).add(r["source_feature_id"])
            fid_class[r["source_feature_id"]] = str(r["class_raw"])
        for r in cr:
            if r["edge_extent_length_m"] != 0.0:
                continue
            fid = r["source_feature_id"]
            if fid_class.get(fid) in nonroad:
                continue
            li, lj, ax = r["lower_cell_i"], r["lower_cell_j"], r["axis"]
            cA, cB = ((li, lj), (li + 1, lj)) if ax == 0 else ((li, lj), (li, lj + 1))
            in_a = fid in fids_by_cell.get(cA, set())
            in_b = fid in fids_by_cell.get(cB, set())
            if in_a == in_b:
                continue  # real cross (both) or anomaly (neither) — not a touch-as-cross
            dirA, dirB = ("E", "W") if ax == 0 else ("S", "N")
            emitA = bool(bcd.get((cA, dirA)))
            emitB = bool(bcd.get((cB, dirB)))
            if emitA and emitB:
                key = "both_emit(sym)"
            elif emitA or emitB:
                key = "one_emit(asym)"
            else:
                key = "neither_emit(sym)"
            cnt[key] += 1
            if len(examples) < 6:
                examples.append(
                    f"  {tile} edge {cA}.{dirA}/{cB}.{dirB} road_in={'A' if in_a else 'B'} "
                    f"emitA={emitA} emitB={emitB} -> {key}"
                )
    print(f"\n===== TOUCH-DETAIL {city} =====")
    print("  ", cnt)
    for e in examples:
        print(e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="2026-04-15.0")
    ap.add_argument("--city")
    ap.add_argument("--max-examples", type=int, default=8)
    ap.add_argument("--audit-contract", action="store_true")
    ap.add_argument("--source-trace", action="store_true")
    ap.add_argument("--touch-census", action="store_true")
    ap.add_argument("--touch-detail", action="store_true")
    ap.add_argument(
        "--explain",
        nargs=5,
        metavar=("TILE", "CI", "CJ", "DIR", "_"),
        help="TILE CI CJ DIR (pass any 5th token, ignored)",
    )
    args = ap.parse_args()
    if args.explain:
        tile, ci, cj, direction, _ = args.explain
        explain_edge(args.release, args.city, tile, int(ci), int(cj), direction)
    elif args.touch_census:
        touch_census(args.release)
    elif args.touch_detail:
        touch_detail(args.release, args.city)
    elif args.source_trace:
        source_trace(args.release, args.city)
    elif args.audit_contract:
        audit_contract(args.release, args.city)
    else:
        scan_city(args.release, args.city, args.max_examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

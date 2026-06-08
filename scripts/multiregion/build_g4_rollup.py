#!/usr/bin/env python
"""Phase-G G4: full-corpus roll-up + THREE-PART DoD gate (canary + batch-2).

Run ON Leonardo when the batch-2 corpus has processed. Reads BOTH canary_v1.yaml
and batch2_v1.yaml (the full ~40-city diversity corpus), assembles the §6 roll-up,
and applies the DoD gate the PI locked (2026-06-04). A threshold alone is a trap, so
the gate is structural + per-city, never just the sum:

  (a) total_validated_tokens >= 550_000_000   (measured DIRECTLY; NOT tiles x 29,150;
      550M = PI-ratified v1 floor, reset from the stale 600M pre-measurement heuristic)
  (b) PER-CITY FLOOR: every validated city contributes a non-trivial tile/token
      count. A city that "passes" with a near-empty box (empty-sea/rural / silent
      clip; per-city counts are known-soft, fallback bbox / known_issues #15) is a
      SILENT DUD and must surface, not hide under the sum. Floor ~ umea canary low
      (36 tiles / ~0.8M tokens).
  (c) axis-coverage matrix green (rollup.assert_ready_for_next_batch — morphology/
      density/geography, 0 uncovered).
  (d) SHA COHERENCE: every city counted as validated must be at the current sub-F
      DERIVATION sha. The pipeline treats on-disk markers as ground truth, so the merge
      gate must NOT declare "corpus complete" on a stale `_PHASE1_VALIDATED` (a city
      blessed pre-fix at 1.1 and never re-derived). A city counts only if marker AND
      sha-current; any marker-but-stale city is a version_skew and fails the gate. This
      is the marker-trust safeguard — added 2026-06-07 after a driver wrote a false DONE.

AND: groups=0 is necessary, NOT sufficient. Pair each city's validator verdict with
a rough-numbers sanity glance — flag cities whose tiles/tokens are wildly off their
morphology/density peers (too clean, too empty, outlier), even if groups=0.

Reports the PER-CITY TABLE (not just the total) + the four-part verdict + flags.
Read-only over data/; writes the report YAML.
"""

from __future__ import annotations

import glob
import statistics
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from cfm.data.multiregion import rollup, selection  # noqa: E402
from cfm.data.sub_f.versions import SUB_F_DERIVATION_VERSION  # noqa: E402

RELEASE = "2026-04-15.0"
# DoD floor (measured directly). RESET 600M -> 550M (2026-06-06, PI-ratified v1 floor):
# 600M was the optimistic tail of the 30M-params x r=20 heuristic computed pre-measurement;
# the real EU yield came in at ~24k tok/tile (not the assumed 56k), and the PI ratified
# "550M + full coverage" as the v1 floor. This corrects a STALE constant to match the
# ratified DoD — NOT a soft override. 550M is a hard line: gate-a is True at >=550M and
# False below it (a below-550M shipped total is a real miss -> add cities before merge).
# (This is the CORPUS-completion / merge gate; the r=20 TRAIN-split floor is a separate,
# post-merge eval-set-gen concern. See reports/2026-06-06-eval-set-gen-scoping.md.)
TARGET_TOKENS = 550_000_000
FLOOR_TILES = 36  # umea canary low — below this with a generous box ⇒ suspect dud
FLOOR_TOKENS = 800_000  # umea canary low
PROC = _REPO / "data" / "processed"

# Cities NOT in the shipped corpus (B1-simple, 2026-06-06) — excluded from the DoD
# gate (full reasons in the close-out + known_issues). They may still appear in the
# per-city table tagged EXCLUDED, but do not count against gate-b / coverage.
# Only the genuinely-not-in-corpus cities remain excluded: dropped (pathological wall,
# never extracted -> no sub_f) or unprocessed. The six #19 inflation/degraded exclusions
# (rotterdam, warsaw, amsterdam, almere, a_coruna, lodz) were RECOVERED by the de-densify
# fix (#19) + guarded re-derive (2026-06-07): all re-derived under DERIVATION 1.2 and
# validated; rotterdam/warsaw additionally passed the path-length spot-check and were
# PI-re-admitted (#20 overturned). They now COUNT in the DoD. See known_issues #19/#20.
EXCLUDED = {
    "paris": "dropped: FR already covered, pathological ~30h wall, never extracted",
    "lyon": "dropped: FR already covered, pathological ~56h wall, never extracted",
    "madrid": "dropped: ES already covered, pathological ~104h wall, never extracted",
    "rome": "excluded: not extracted (IT already covered, ~26h+ wall, coverage-redundant)",
    "welwyn": "excluded: unprocessed; GB + modernist-sprawl/sparse already covered",
}


def _city_token_stats(region: str) -> tuple[int, int]:
    """(tile_count, token_count) from sub_f cells (read each file directly — avoid
    pyarrow Hive 'tile=' column inference)."""
    files = sorted(glob.glob(str(PROC / "sub_f" / RELEASE / region / "tile=*" / "cells.parquet")))
    n_tok = 0
    for f in files:
        t = pq.ParquetFile(f).read(columns=["token_sequence"])
        n_tok += len(t.column("token_sequence").combine_chunks().flatten())
    return len(files), n_tok


def _validated(region: str) -> bool:
    return (PROC / "sub_g" / RELEASE / region / "_PHASE1_VALIDATED").exists()


def _derivation_version(region: str) -> str | None:
    """Sub-F DERIVATION axis from the region manifest (the post-fix sha discriminator:
    '1.2' = re-derived under the #19 de-densify fix, '1.1' = stale pre-fix)."""
    m = PROC / "sub_f" / RELEASE / region / "manifest.yaml"
    if not m.exists():
        return None
    return str(yaml.safe_load(m.read_text()).get("sub_f_derivation_version"))


def _groups(region: str) -> int | None:
    p = PROC / "sub_g" / RELEASE / region / "quarantine_report.yaml"
    if not p.exists():
        return None
    g = yaml.safe_load(p.read_text()).get("groups")
    return len(g) if isinstance(g, list) else None


def main() -> int:
    canary = selection.load_canary_manifest(_REPO / "configs" / "multiregion" / "canary_v1.yaml")
    batch2 = selection.load_canary_manifest(_REPO / "configs" / "multiregion" / "batch2_v1.yaml")
    addcities_path = _REPO / "configs" / "multiregion" / "addcities_v1.yaml"
    addcities = selection.load_canary_manifest(addcities_path) if addcities_path.exists() else []
    cities = canary + batch2 + addcities

    rows: list[dict] = []
    records: list[rollup.CityRecord] = []
    for c in cities:
        name = c["name"]
        tiles, tokens = _city_token_stats(name)
        marker = _validated(name)
        dv = _derivation_version(name)
        at_sha = dv == SUB_F_DERIVATION_VERSION
        # A city counts as validated for the DoD ONLY if sub_g blessed it AND its sub_f is
        # at the current DERIVATION sha. A stale marker (blessed pre-fix at 1.1, never
        # re-derived) is NOT a clean corpus member — counting it would be a version skew.
        validated = marker and at_sha
        rows.append(
            {
                "name": name,
                "morphology": c["morphology"],
                "density": c["density"],
                "geography": c["geography"],
                "crs": c["projected_crs"],
                "tiles": tiles,
                "tokens": tokens,
                "groups": _groups(name),
                "validated": validated,
                "marker": marker,
                "derivation_version": dv,
                "at_sha": at_sha,
                "tok_per_tile": round(tokens / tiles, 1) if tiles else None,
            }
        )
        records.append(
            rollup.CityRecord(
                name=name,
                morphology=c["morphology"],
                density=c["density"],
                geography=c["geography"],
                region_crs=c["projected_crs"],
                tile_count=tiles,
                fetch_seconds=0.0,
                stage_shas={},
                release=RELEASE,
                validation_status=rollup.VALIDATED if validated else rollup.FAILED,
                token_count=tokens,
            )
        )
    r = rollup.RollUp(cities=records)

    # ---- (a) token DoD ----
    total_tokens = rollup.total_validated_tokens(r)
    gate_a = total_tokens >= TARGET_TOKENS

    # ---- (b) per-city floor (validated cities only) ----
    below_floor = [
        x["name"]
        for x in rows
        if x["validated"] and (x["tiles"] < FLOOR_TILES or x["tokens"] < FLOOR_TOKENS)
    ]
    not_validated = [x["name"] for x in rows if not x["validated"] and x["name"] not in EXCLUDED]
    gate_b = not below_floor and not not_validated

    # ---- (c) axis coverage ----
    uncovered = rollup.uncovered_axis_labels(r)
    gate_c = not uncovered

    # ---- (d) SHA COHERENCE — all-cities-at-target-sha, the marker-trust safeguard ----
    # The whole pipeline treats on-disk markers as ground truth, so the merge gate must
    # not declare "corpus complete" on a stale _PHASE1_VALIDATED. version_skew = cities
    # blessed by a marker but whose sub_f is NOT at the current DERIVATION sha (e.g. a
    # 1.1 cache that was never re-derived under the #19 fix). Any skew fails the gate.
    version_skew = [
        x["name"] for x in rows if x["marker"] and not x["at_sha"] and x["name"] not in EXCLUDED
    ]
    gate_d = not version_skew

    # ---- groups=0-not-sufficient: peer-median outlier sanity ----
    sanity_flags: list[str] = []
    by_cell: dict[tuple, list[dict]] = {}
    for x in rows:
        if x["validated"]:
            by_cell.setdefault((x["morphology"], x["density"]), []).append(x)
    for x in rows:
        if not x["validated"]:
            continue
        peers = [p for p in by_cell[(x["morphology"], x["density"])] if p["name"] != x["name"]]
        if len(peers) >= 2:
            med = statistics.median([p["tiles"] for p in peers])
            if med > 0 and (x["tiles"] < 0.4 * med or x["tiles"] > 2.5 * med):
                sanity_flags.append(
                    f"{x['name']}: {x['tiles']} tiles vs {x['morphology']}/{x['density']} "
                    f"peer-median {med:.0f} (outlier — review even though groups={x['groups']})"
                )
        if x["density"] == "dense-core" and x["tiles"] < FLOOR_TILES * 2:
            sanity_flags.append(f"{x['name']}: dense-core but only {x['tiles']} tiles (suspect)")
        if x["groups"]:  # groups > 0 (None and 0 are falsy) — validated but NOT clean
            sanity_flags.append(f"{x['name']}: groups={x['groups']} (NOT clean)")

    dod_pass = gate_a and gate_b and gate_c and gate_d

    # ---- report ----
    print("=== G4 PER-CITY TABLE (full corpus) ===")
    for x in sorted(rows, key=lambda z: (z["morphology"], z["density"], -z["tiles"])):
        flag = ""
        if x["name"] in EXCLUDED:
            flag = " <<EXCLUDED (not in shipped corpus)"
        elif not x["validated"]:
            flag = " <<NOT-VALIDATED"
        elif x["tiles"] < FLOOR_TILES or x["tokens"] < FLOOR_TOKENS:
            flag = " <<BELOW-FLOOR"
        print(
            f"  {x['name']:<16} {x['morphology']:<16} {x['density']:<10} {x['crs']:<11} "
            f"tiles={x['tiles']:>4} tokens={x['tokens']:>9} groups={x['groups']}{flag}"
        )
    n_val = sum(1 for x in rows if x["validated"])
    print(
        f"\n  cities={len(rows)} validated={n_val} tiles={rollup.total_validated_tiles(r)} "
        f"tokens={total_tokens:,}"
    )
    print(f"\n=== FOUR-PART DoD GATE (target sha = DERIVATION {SUB_F_DERIVATION_VERSION}) ===")
    print(
        f"  (a) tokens >= {TARGET_TOKENS // 1_000_000}M:        {gate_a}  "
        f"(raw total {total_tokens:,} / floor {TARGET_TOKENS:,})"
    )
    print(f"  (b) per-city floor:        {gate_b}  below_floor={below_floor}")
    print(f"      not_validated={not_validated}")
    print(f"  (c) axis-coverage green:   {gate_c}  uncovered={uncovered}")
    print(f"  (d) sha-coherence:         {gate_d}  version_skew={version_skew}")
    print(f"  ==> DoD PASS: {dod_pass}")
    print("\n=== groups=0-NOT-SUFFICIENT sanity flags ===")
    print(
        "\n".join(f"  ⚠ {s}" for s in sanity_flags) or "  (none — counts in-ballpark for all peers)"
    )
    print("\n=== EXCLUDED from shipped corpus (NOT counted in the DoD) ===")
    for name, reason in EXCLUDED.items():
        print(f"  {name}: {reason}")

    out = _REPO / "reports" / "2026-06-05-phase-2-g4-corpus-dod.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(
            {
                "per_city": rows,
                "totals": {
                    "cities": len(rows),
                    "validated": n_val,
                    "validated_tiles": rollup.total_validated_tiles(r),
                    "validated_tokens": total_tokens,
                },
                "dod_gate": {
                    "a_tokens_ge_floor": gate_a,
                    "b_per_city_floor": gate_b,
                    "c_axis_coverage": gate_c,
                    "d_sha_coherence": gate_d,
                    "target_derivation_sha": SUB_F_DERIVATION_VERSION,
                    "PASS": dod_pass,
                    "below_floor": below_floor,
                    "not_validated": not_validated,
                    "uncovered_axes": uncovered,
                    "version_skew": version_skew,
                },
                "sanity_flags": sanity_flags,
                "excluded_from_shipped": EXCLUDED,
            },
            sort_keys=False,
            allow_unicode=True,
        )
    )
    print(f"\nwrote {out}")
    return 0 if dod_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

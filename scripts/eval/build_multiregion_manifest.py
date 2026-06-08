#!/usr/bin/env python3
"""scripts/eval/build_multiregion_manifest.py — assemble + freeze the 4-city EU holdout.

Builds the multi-region whole-city holdout manifest (spec §2.1) for the four held-out
EU cities (glasgow, eisenhuttenstadt, munich, krakow), runs the §2.2 build-time
correctness-by-construction assertions, and — only when ``--lock`` is passed — freezes
it write-once to the ``multiregion/`` subdir alongside an ``_EVAL_SET_LOCKED`` marker.

WRITE-ONCE / POINT-OF-NO-RETURN: freezing is opt-in (``--lock``). Without it this is a
DRY RUN that builds, asserts, and prints the summary but writes NOTHING. The freeze
invalidates every later eval number if wrong, so it is never the default.

PATHS (spec §2.3): writes ONLY to ``multiregion_holdout_manifest_path(release)`` and
``multiregion_eval_set_locked_marker(release)`` — the ``eval_set/<release>/multiregion/``
subdir. It NEVER touches the Singapore set (``holdout_manifest_path`` /
``eval_set_locked_marker``), which is already frozen at ``eval_set/<release>/``.

Run on Leonardo against the real sub-D corpus (the controller runs it); there is no
corpus locally. Dry run:

    uv run python scripts/eval/build_multiregion_manifest.py \
        --release 2026-04-15.0 \
        --g4 reports/2026-06-05-phase-2-g4-corpus-dod.yaml \
        --usable-n reports/2026-06-08-usable-n.yaml

then, deliberately, the freeze:

    uv run python scripts/eval/build_multiregion_manifest.py ... --lock
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# iCloud-safe sys.path inject — mirrors scripts/eval/measure_usable_tiles.py
# (parents[2] = repo root; underscore-prefixed .pth files are hidden in the
# iCloud-synced .venv so the editable install can't be relied on).
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from cfm.eval.holdout.manifest import (  # noqa: E402
    build_holdout_manifest_multiregion,
    freeze_holdout_manifest,
    manifest_sha256,
)
from cfm.eval.holdout.paths import (  # noqa: E402
    epsg_label_for_region,
    multiregion_eval_set_locked_marker,
    multiregion_holdout_manifest_path,
    sub_d_region_dir,
    tile_dirname,
)

logger = logging.getLogger("build_multiregion_manifest")

#: The four held-out EU cities (spec §2.1; G4 evaluation split, sorted at freeze).
DEFAULT_HELD_OUT: tuple[str, ...] = ("glasgow", "eisenhuttenstadt", "munich", "krakow")
DEFAULT_RELEASE: str = "2026-04-15.0"
DEFAULT_DERIVATION_VERSION: str = "1.2"


def assemble_regions_payload(
    *,
    g4_cities: list[dict],
    usable_n: dict,
    per_city_tiles: dict[str, list[dict]],
) -> tuple[dict, dict]:
    """PURE assembly of the per-region payload + the corpus tile counts (spec §2.1).

    Given the parsed G4 city-block list, the parsed usable-n ``cities`` dict, and the
    per-city enumerated tiles (each tile a dict with tile_i/tile_j/provenance_sha256/
    macro_vocab_sha256), return:

    - ``regions_payload``: ``{city: {morphology, density, geography, crs, tokens,
      n_usable_tiles, tiles}}`` — stratification labels + tokens from G4, n_usable_tiles
      from usable-n, tiles passed straight through.
    - ``corpus_tile_counts``: ``{city: G4_tiles}`` — the frozen-corpus tile count §2.2(b)
      checks the enumeration against (NOT usable-n, NOT the enumeration length).

    Held-out cities are exactly the keys of ``per_city_tiles``. Does NO file I/O.

    Raises:
        KeyError: a held-out city missing from the G4 blocks or from usable-n
            (a mis-spelled / unprocessed city — fail loud, never silently drop).
    """
    g4_by_name = {c["name"]: c for c in g4_cities}
    regions_payload: dict[str, dict] = {}
    corpus_tile_counts: dict[str, int] = {}
    for city, tiles in per_city_tiles.items():
        g4 = g4_by_name[city]  # KeyError if city absent from G4 (mis-spelled/unprocessed)
        usable = usable_n[city]  # KeyError if city absent from usable-n census
        regions_payload[city] = {
            "morphology": g4["morphology"],
            "density": g4["density"],
            "geography": g4["geography"],
            "crs": g4["crs"],
            "tokens": int(g4["tokens"]),
            "n_usable_tiles": int(usable["n_usable_tiles"]),
            "tiles": tiles,
        }
        corpus_tile_counts[city] = int(g4["tiles"])
    return regions_payload, corpus_tile_counts


def _train_cities_and_tokens(g4_cities: list[dict], held_out: list[str]) -> tuple[set[str], int]:
    """Derive the train city set + train_tokens from VALIDATED G4 rows only (Fix I1).

    The G4 yaml carries ``validated: true|false`` per city. Unvalidated cities are
    listed under top-level ``excluded_from_shipped`` and currently carry ``tokens: 0`` —
    so summing over "all names" is right BY ACCIDENT. A future re-gen could stamp tokens
    on an unvalidated city and silently inflate the IRREVERSIBLE ``train_tokens``.

    So: ``train_cities = {validated G4 names} - held_out`` (a missing ``validated`` key is
    treated as NOT validated — never silently included), and ``train_tokens`` sums G4
    tokens over exactly that set. Also asserts every held-out city is itself validated.

    Raises:
        SystemExit: a held-out city is not ``validated: true`` in G4 (never freeze an
            eval number against an unvalidated city's tokens).
    """
    g4_validated = {c["name"] for c in g4_cities if c.get("validated") is True}
    missing = set(held_out) - g4_validated
    if missing:
        raise SystemExit(f"held-out cities not validated in G4: {sorted(missing)}")
    train_cities = g4_validated - set(held_out)
    train_tokens = sum(int(c["tokens"]) for c in g4_cities if c["name"] in train_cities)
    return train_cities, train_tokens


def _assert_usable_n_census_ok(usable_n: dict, held_out: list[str]) -> None:
    """Fail loud if the usable-n census is degraded for any held-out city (Fix M1).

    The usable-n yaml carries ``status`` and ``n_unreadable`` per city so a degraded
    census is detectable. A degraded census must NOT silently freeze a wrong
    ``n_usable_tiles``: for every held-out city assert ``status == "ok"`` and
    ``n_unreadable == 0``.

    Raises:
        SystemExit: a held-out city's census is degraded (status != "ok" or
            n_unreadable > 0), or the city is absent from the census entirely.
    """
    for city in held_out:
        if city not in usable_n:
            raise SystemExit(f"usable-n census missing held-out city: {city!r}")
        entry = usable_n[city]
        status = entry.get("status")
        n_unreadable = int(entry.get("n_unreadable", 0))
        if status != "ok" or n_unreadable != 0:
            raise SystemExit(
                f"usable-n census degraded for {city!r}: "
                f"status={status!r} n_unreadable={n_unreadable} "
                "(a degraded census must not silently freeze a wrong n_usable_tiles)"
            )


def _enumerate_city_tiles(release: str, city: str) -> list[dict]:
    """Read sub-D manifest tiles + per-tile provenance.yaml macro_vocab_sha256 for a city.

    Mirrors ``cfm.eval.holdout.pipeline._load_inventory`` + the per-tile provenance read
    (pipeline.py:184-206): the sub-D ``manifest.yaml`` ``tiles:`` list gives tile_i/
    tile_j/provenance_sha256 per tile; each tile's ``provenance.yaml`` supplies
    ``inputs.macro_vocab_sha256``. Tile dir name is the per-region CRS-labelled
    ``tile=EPSG<zone>_i{i}_j{j}``.
    """
    region_dir = sub_d_region_dir(release, city)
    md = yaml.safe_load((region_dir / "manifest.yaml").read_text(encoding="utf-8"))
    epsg_label = epsg_label_for_region(city)
    tiles: list[dict] = []
    for entry in md["tiles"]:
        ti, tj = int(entry["tile_i"]), int(entry["tile_j"])
        tile_dir = region_dir / tile_dirname(ti, tj, epsg_label)
        prov = yaml.safe_load((tile_dir / "provenance.yaml").read_text(encoding="utf-8"))
        tiles.append(
            {
                "tile_i": ti,
                "tile_j": tj,
                "provenance_sha256": entry["provenance_sha256"],
                "macro_vocab_sha256": prov["inputs"]["macro_vocab_sha256"],
            }
        )
    return tiles


def _verify_frozen_end_state(
    *,
    path: Path,
    held_out: list[str],
    corpus_tile_counts: dict[str, int],
) -> None:
    """Re-read the just-frozen manifest from disk and assert the end-state (spec §2.3).

    Trusts NOTHING from in-memory state: a false DONE is worse than a false RC. Recompute
    ``manifest_sha256`` over the loaded dict and assert it matches the stored field; then
    assert the structural invariants (sorted 4 held-out cities, whole_city declaration,
    n_tiles == G4 count, and a usable count that is present AND in ``[0, n_tiles]`` — a
    usable count exceeding the tile count is impossible and fails loud, Fix M4).
    """
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    recomputed = manifest_sha256(loaded)
    stored = loaded["manifest_sha256"]
    assert recomputed == stored, (
        f"manifest_sha256 MISMATCH after freeze: recomputed {recomputed} != stored {stored}"
    )
    assert loaded["held_out_cities"] == sorted(held_out), (
        f"held_out_cities {loaded['held_out_cities']} != sorted {sorted(held_out)}"
    )
    for city in held_out:
        region = loaded["regions"][city]
        assert region["holdout_kind"] == "whole_city", (
            f"{city}: holdout_kind {region['holdout_kind']!r} != 'whole_city'"
        )
        n_tiles = region["n_tiles"]
        assert n_tiles == corpus_tile_counts[city], (
            f"{city}: n_tiles {n_tiles} != G4 corpus count {corpus_tile_counts[city]}"
        )
        n_usable = region["n_usable_tiles"]
        assert n_usable is not None, f"{city}: n_usable_tiles is None"
        assert 0 <= n_usable <= n_tiles, (
            f"{city}: n_usable_tiles {n_usable} not in [0, n_tiles={n_tiles}] "
            "(a usable count exceeding the tile count is impossible)"
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default=DEFAULT_RELEASE, help="sub-D release id")
    parser.add_argument("--g4", required=True, type=Path, help="path to the G4 corpus-DoD YAML")
    parser.add_argument(
        "--usable-n", required=True, type=Path, help="path to the usable-n census YAML"
    )
    parser.add_argument(
        "--held-out",
        default=",".join(DEFAULT_HELD_OUT),
        help="comma-separated held-out city names",
    )
    parser.add_argument(
        "--derivation-version",
        default=DEFAULT_DERIVATION_VERSION,
        help="corpus derivation version this eval set pins to",
    )
    parser.add_argument(
        "--lock",
        action="store_true",
        help="FREEZE the manifest write-once (point of no return). Absent = DRY RUN.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(sys.argv[1:] if argv is None else argv)
    release: str = args.release
    held_out = [c.strip() for c in args.held_out.split(",") if c.strip()]

    # 1. parse G4 + usable-n YAML.
    g4_doc = yaml.safe_load(args.g4.read_text(encoding="utf-8"))
    g4_cities: list[dict] = g4_doc["per_city"]
    usable_doc = yaml.safe_load(args.usable_n.read_text(encoding="utf-8"))
    usable_n: dict = usable_doc["cities"]

    # 1b. Fix M1: a degraded usable-n census (status != "ok" / n_unreadable > 0) for any
    # held-out city must NOT silently freeze a wrong n_usable_tiles — fail loud first.
    _assert_usable_n_census_ok(usable_n, held_out)

    # 2. enumerate each held-out city's tiles (sub-D manifest + per-tile provenance).
    per_city_tiles = {city: _enumerate_city_tiles(release, city) for city in held_out}
    for city, tiles in per_city_tiles.items():
        logger.info("%s: enumerated %d tiles from sub-D", city, len(tiles))

    # 3. assemble the per-region payload + corpus tile counts (PURE).
    regions_payload, corpus_tile_counts = assemble_regions_payload(
        g4_cities=g4_cities,
        usable_n=usable_n,
        per_city_tiles=per_city_tiles,
    )

    # 4. Fix I1: train_cities + train_tokens derive from VALIDATED G4 rows only (asserts
    # the held-out cities are themselves validated; unvalidated cities — even if a re-gen
    # stamps nonzero tokens on them — never inflate the IRREVERSIBLE train_tokens).
    train_cities, train_tokens = _train_cities_and_tokens(g4_cities, held_out)

    # 5. build the manifest — runs §2.2 (a) disjoint + (b) tile-count assertions (RAISE).
    #    The §2.2(a) disjoint check now operates over the intended (validated) train set.
    man = build_holdout_manifest_multiregion(
        regions_payload,
        corpus_release=release,
        derivation_version=args.derivation_version,
        train_cities=train_cities,
        corpus_tile_counts=corpus_tile_counts,
    )

    # 6. stamp train_tokens (summed over the validated train cities in step 4).
    man["totals"]["train_tokens"] = train_tokens

    held_out_tokens = man["totals"]["held_out_tokens"]

    # Summary (always logged, lock or dry run).
    for city in man["held_out_cities"]:
        region = man["regions"][city]
        logger.info(
            "  %s: n_tiles=%d n_usable=%s tokens=%d",
            city,
            region["n_tiles"],
            region["n_usable_tiles"],
            region["tokens"],
        )
    logger.info(
        "held_out_cities=%s held_out_tokens=%d train_tokens=%d",
        man["held_out_cities"],
        held_out_tokens,
        train_tokens,
    )

    if not args.lock:
        logger.info("DRY RUN (no --lock): built + asserted manifest; wrote NOTHING.")
        return 0

    # 7. FREEZE write-once to the multiregion/ subdir (NEVER the SG paths).
    #    Order (Fix I2 + M2): (a) freeze the manifest write-once, (b) VERIFY the manifest
    #    end-state from disk (RAISES on any mismatch), (c) only THEN write the marker —
    #    itself write-once — LAST, (d) re-read the marker from disk and verify its
    #    irreversible operator-facing numbers. A verify failure leaves NO marker, so a
    #    false "DONE" can never poison a future session.
    man_path = multiregion_holdout_manifest_path(release)
    freeze_holdout_manifest(man, man_path)
    logger.info("FROZE multi-region holdout manifest -> %s", man_path)

    # (b) VERIFIED MANIFEST END-STATE: re-read from disk, recompute sha, assert invariants.
    #     RAISES before any marker is written.
    _verify_frozen_end_state(
        path=man_path,
        held_out=held_out,
        corpus_tile_counts=corpus_tile_counts,
    )
    logger.info(
        "VERIFIED manifest end-state: sha-recompute matches, %d held-out cities, all "
        "whole_city, n_tiles==G4, 0<=n_usable<=n_tiles.",
        len(held_out),
    )

    # (c) write the marker LAST, write-once (mirror freeze_holdout_manifest's posture).
    marker = multiregion_eval_set_locked_marker(release)
    if marker.exists():
        raise FileExistsError(
            f"lock marker already present at {marker}; it is written once and never "
            "regenerated. Delete deliberately only to re-lock the eval set."
        )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        yaml.safe_dump(
            {
                "release": release,
                "derivation_version": args.derivation_version,
                "held_out_cities": man["held_out_cities"],
                "held_out_tokens": held_out_tokens,
                "train_tokens": train_tokens,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    logger.info("wrote lock marker -> %s", marker)

    # (d) VERIFIED MARKER END-STATE: re-read the marker FROM DISK and assert its
    #     irreversible operator-facing numbers match the manifest (not trusted from memory).
    marker_loaded = yaml.safe_load(marker.read_text(encoding="utf-8"))
    assert marker_loaded["held_out_cities"] == man["held_out_cities"], (
        f"marker held_out_cities {marker_loaded['held_out_cities']} != "
        f"manifest {man['held_out_cities']}"
    )
    assert marker_loaded["held_out_tokens"] == man["totals"]["held_out_tokens"], (
        f"marker held_out_tokens {marker_loaded['held_out_tokens']} != "
        f"manifest {man['totals']['held_out_tokens']}"
    )
    assert marker_loaded["train_tokens"] == man["totals"]["train_tokens"], (
        f"marker train_tokens {marker_loaded['train_tokens']} != "
        f"manifest {man['totals']['train_tokens']}"
    )
    logger.info(
        "VERIFIED marker end-state: held_out_cities/tokens + train_tokens match. LOCK COMPLETE."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

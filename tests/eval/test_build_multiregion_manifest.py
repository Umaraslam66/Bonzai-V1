"""Unit tests for the PURE assembly function of the multi-region build driver (T6).

These exercise ``assemble_regions_payload`` with synthetic dicts only — NO file I/O,
NO corpus. The real provenance/manifest reads happen in ``main`` and run on Leonardo;
here we prove the pure G4 + usable-n + per-city-tiles → (regions_payload,
corpus_tile_counts) mapping is correct, and that the assembled payload feeds
``build_holdout_manifest_multiregion`` to a valid schema-2.0 manifest that passes the
§2.2 assertions.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from cfm.eval.holdout.manifest import build_holdout_manifest_multiregion

# Load the script module by path (scripts/ is not an importable package).
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "build_multiregion_manifest.py"
_spec = importlib.util.spec_from_file_location("build_multiregion_manifest", _SCRIPT)
assert _spec is not None and _spec.loader is not None
build_mod = importlib.util.module_from_spec(_spec)
sys.modules["build_multiregion_manifest"] = build_mod
_spec.loader.exec_module(build_mod)

assemble_regions_payload = build_mod.assemble_regions_payload
_train_cities_and_tokens = build_mod._train_cities_and_tokens
_assert_usable_n_census_ok = build_mod._assert_usable_n_census_ok


def _g4_blocks() -> list[dict]:
    # 2 held-out (krakow, munich) + 1 train (hamburg). All validated.
    return [
        dict(
            name="krakow",
            morphology="medieval-organic",
            density="moderate",
            geography="PL",
            crs="EPSG:25834",
            tiles=2,
            tokens=17118283,
            validated=True,
        ),
        dict(
            name="munich",
            morphology="mixed",
            density="moderate",
            geography="DE",
            crs="EPSG:25832",
            tiles=1,
            tokens=10060491,
            validated=True,
        ),
        dict(
            name="hamburg",
            morphology="mixed",
            density="dense-core",
            geography="DE",
            crs="EPSG:25832",
            tiles=3,
            tokens=28620067,
            validated=True,
        ),
    ]


def _usable_n() -> dict:
    return {
        "krakow": {"n_tiles": 2, "n_usable_tiles": 2, "n_unreadable": 0, "status": "ok"},
        "munich": {"n_tiles": 1, "n_usable_tiles": 1, "n_unreadable": 0, "status": "ok"},
        "hamburg": {"n_tiles": 3, "n_usable_tiles": 3, "n_unreadable": 0, "status": "ok"},
    }


def _per_city_tiles() -> dict:
    return {
        "krakow": [
            dict(tile_i=0, tile_j=0, provenance_sha256="ka", macro_vocab_sha256="v"),
            dict(tile_i=0, tile_j=1, provenance_sha256="kb", macro_vocab_sha256="v"),
        ],
        "munich": [
            dict(tile_i=3, tile_j=4, provenance_sha256="ma", macro_vocab_sha256="v"),
        ],
    }


def test_assemble_maps_g4_usable_n_and_tiles_per_region():
    payload, counts = assemble_regions_payload(
        g4_cities=_g4_blocks(),
        usable_n=_usable_n(),
        per_city_tiles=_per_city_tiles(),
    )
    # Only the cities enumerated in per_city_tiles become held-out regions.
    assert set(payload) == {"krakow", "munich"}
    assert set(counts) == {"krakow", "munich"}

    kr = payload["krakow"]
    assert kr["morphology"] == "medieval-organic"  # from G4
    assert kr["density"] == "moderate"  # from G4
    assert kr["geography"] == "PL"  # from G4
    assert kr["crs"] == "EPSG:25834"  # from G4
    assert kr["tokens"] == 17118283  # from G4
    assert kr["n_usable_tiles"] == 2  # from usable-n
    assert kr["tiles"] == _per_city_tiles()["krakow"]  # passed through

    mu = payload["munich"]
    assert mu["morphology"] == "mixed"
    assert mu["n_usable_tiles"] == 1
    assert mu["tiles"] == _per_city_tiles()["munich"]

    # corpus_tile_counts come from the G4 tile counts (NOT usable-n, NOT enumeration).
    assert counts == {"krakow": 2, "munich": 1}


def test_assembled_payload_feeds_builder_to_schema_2_0():
    payload, counts = assemble_regions_payload(
        g4_cities=_g4_blocks(),
        usable_n=_usable_n(),
        per_city_tiles=_per_city_tiles(),
    )
    man = build_holdout_manifest_multiregion(
        payload,
        corpus_release="2026-04-15.0",
        derivation_version="1.2",
        train_cities={"hamburg"},
        corpus_tile_counts=counts,
    )
    assert man["manifest_schema_version"] == "2.0"
    assert man["held_out_cities"] == ["krakow", "munich"]  # sorted
    for city in ("krakow", "munich"):
        assert man["regions"][city]["holdout_kind"] == "whole_city"
        assert man["regions"][city]["n_usable_tiles"] is not None
    # §2.2 assertions passed (no raise); held-out tokens summed.
    assert man["totals"]["held_out_tokens"] == 17118283 + 10060491


def test_assemble_raises_on_unknown_city_in_per_city_tiles():
    # A per-city-tiles key with no matching G4 block is a build error (mis-spelled city).
    with pytest.raises(KeyError):
        assemble_regions_payload(
            g4_cities=_g4_blocks(),
            usable_n=_usable_n(),
            per_city_tiles={"nowhere": []},
        )


# --- Fix I1: train_cities derive from VALIDATED G4 rows only --------------------------


def test_train_cities_exclude_unvalidated_ghostcity_with_nonzero_tokens():
    # I1 REGRESSION: an unvalidated city carrying NONZERO tokens (a future re-gen could
    # stamp tokens on it) must NOT inflate train_cities OR the IRREVERSIBLE train_tokens.
    # This fixture FAILS against the old "all G4 names minus held_out" logic.
    g4 = [
        *_g4_blocks(),
        dict(
            name="ghostcity",
            morphology="mixed",
            density="moderate",
            geography="DE",
            crs="EPSG:25832",
            tiles=5,
            tokens=999,  # nonzero — the trap
            validated=False,
        ),
    ]
    held_out = ["krakow", "munich"]
    train_cities, train_tokens = _train_cities_and_tokens(g4, held_out)
    # ghostcity is excluded; only the validated, non-held-out city (hamburg) is a train city.
    assert train_cities == {"hamburg"}
    assert "ghostcity" not in train_cities
    # The nonzero ghostcity tokens do NOT inflate the sum — train_tokens == hamburg only.
    assert train_tokens == 28620067


def test_train_cities_held_out_must_be_validated_else_raises():
    # I1: a held-out city that is NOT validated in G4 must raise (never freeze an eval
    # number against an unvalidated city's tokens).
    g4 = _g4_blocks()
    # Flip munich (a held-out city) to validated: False.
    for c in g4:
        if c["name"] == "munich":
            c["validated"] = False
    with pytest.raises(SystemExit, match="not validated"):
        _train_cities_and_tokens(g4, ["krakow", "munich"])


def test_train_cities_treats_missing_validated_key_as_unvalidated():
    # Defensive: a G4 row lacking the `validated` key (legacy/partial yaml) is NOT
    # treated as validated — it is excluded from the train set, never silently included.
    g4 = _g4_blocks()
    for c in g4:
        if c["name"] == "hamburg":
            del c["validated"]  # missing key
    train_cities, train_tokens = _train_cities_and_tokens(g4, ["krakow", "munich"])
    assert train_cities == set()  # hamburg dropped (not validated is True)
    assert train_tokens == 0


# --- Fix M1: respect the usable-n census status / n_unreadable -------------------------


def test_usable_n_census_guard_raises_on_error_status():
    # M1: a held-out city with status != "ok" must raise (degraded census must not
    # silently freeze a wrong n_usable_tiles).
    usable = _usable_n()
    usable["krakow"]["status"] = "error"
    with pytest.raises(SystemExit, match="census"):
        _assert_usable_n_census_ok(usable, ["krakow", "munich"])


def test_usable_n_census_guard_raises_on_nonzero_unreadable():
    # M1: a held-out city with n_unreadable > 0 must raise even if status reads "ok".
    usable = _usable_n()
    usable["munich"]["n_unreadable"] = 1
    with pytest.raises(SystemExit, match="census"):
        _assert_usable_n_census_ok(usable, ["krakow", "munich"])


def test_usable_n_census_guard_passes_clean_census():
    # M1: a clean census ("ok", n_unreadable == 0) for every held-out city does NOT raise.
    _assert_usable_n_census_ok(_usable_n(), ["krakow", "munich"])

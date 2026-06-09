"""Tests for the multi-region build driver (T6).

Two test classes:

1. **Pure unit tests** (``TestAssembly``) — exercise ``assemble_regions_payload`` with
   synthetic dicts only, NO file I/O, NO corpus. Prove the G4 + usable-n + per-city-tiles
   → (regions_payload, corpus_tile_counts) mapping is correct, and that the assembled
   payload feeds ``build_holdout_manifest_multiregion`` to a valid schema-2.0 manifest
   that passes the §2.2 assertions.

2. **Integration tests** (``TestFreezeOrchestration``) — exercise the full ``main``
   --lock path with monkeypatched I/O (no real corpus). These tests specifically verify
   the write-once / freeze→verify→marker ordering / no-false-DONE safety properties that
   are otherwise visible only by code-reading.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

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


# ---------------------------------------------------------------------------
# Shared fixtures for integration tests
# ---------------------------------------------------------------------------

#: Tile counts per held-out city — kept small so the fixture stays fast.
_HELD_OUT_TILE_COUNTS: dict[str, int] = {
    "glasgow": 2,
    "eisenhuttenstadt": 1,
    "munich": 3,
    "krakow": 2,
}

_HELD_OUT_TOKENS: dict[str, int] = {
    "glasgow": 5_000_000,
    "eisenhuttenstadt": 2_000_000,
    "munich": 8_000_000,
    "krakow": 4_000_000,
}

_TRAIN_CITY = "hamburg"
_TRAIN_TOKENS = 12_000_000

_RELEASE = "2026-04-15.0"


def _make_g4_yaml(tmp_path: Path) -> Path:
    """Write a synthetic G4 corpus-DoD YAML to ``tmp_path``.

    Contains all 4 held-out cities (validated=True) plus one validated train city
    (hamburg). The ``tiles`` field for each held-out city matches ``_HELD_OUT_TILE_COUNTS``
    so the §2.2(b) enumeration check passes.
    """
    per_city = []
    for city in _HELD_OUT_TILE_COUNTS:
        per_city.append(
            {
                "name": city,
                "morphology": "mixed",
                "density": "moderate",
                "geography": "EU",
                "crs": "EPSG:25832",
                "tiles": _HELD_OUT_TILE_COUNTS[city],
                "tokens": _HELD_OUT_TOKENS[city],
                "validated": True,
            }
        )
    per_city.append(
        {
            "name": _TRAIN_CITY,
            "morphology": "mixed",
            "density": "dense-core",
            "geography": "DE",
            "crs": "EPSG:25832",
            "tiles": 5,
            "tokens": _TRAIN_TOKENS,
            "validated": True,
        }
    )
    path = tmp_path / "g4.yaml"
    path.write_text(yaml.safe_dump({"per_city": per_city}, sort_keys=False), encoding="utf-8")
    return path


def _make_usable_n_yaml(tmp_path: Path) -> Path:
    """Write a synthetic usable-n census YAML to ``tmp_path``.

    Each held-out city has ``n_tiles`` == ``_HELD_OUT_TILE_COUNTS[city]`` (matching G4),
    ``n_usable_tiles <= n_tiles``, ``n_unreadable: 0``, and ``status: ok`` — all guards
    pass cleanly.
    """
    cities: dict[str, Any] = {}
    for city, n in _HELD_OUT_TILE_COUNTS.items():
        cities[city] = {
            "n_tiles": n,
            "n_usable_tiles": n,
            "n_unreadable": 0,
            "status": "ok",
        }
    # Also include the train city in the census (the guard only checks held-out cities).
    cities[_TRAIN_CITY] = {
        "n_tiles": 5,
        "n_usable_tiles": 5,
        "n_unreadable": 0,
        "status": "ok",
    }
    path = tmp_path / "usable_n.yaml"
    path.write_text(yaml.safe_dump({"cities": cities}, sort_keys=False), encoding="utf-8")
    return path


def _synthetic_tile_enumerator(release: str, city: str) -> list[dict]:
    """Replacement for ``_enumerate_city_tiles`` — returns synthetic tile dicts.

    The number of tiles matches ``_HELD_OUT_TILE_COUNTS[city]`` so the §2.2(b) assertion
    inside ``build_holdout_manifest_multiregion`` passes. No disk I/O.
    """
    n = _HELD_OUT_TILE_COUNTS[city]
    return [
        {
            "tile_i": i,
            "tile_j": 0,
            "provenance_sha256": f"prov_{city}_{i:04x}",
            "macro_vocab_sha256": f"vocab_{city}_{i:04x}",
        }
        for i in range(n)
    ]


def _make_argv(
    g4_path: Path,
    usable_n_path: Path,
    *,
    lock: bool = False,
    release: str = _RELEASE,
) -> list[str]:
    """Build a ``main()`` argv list for the standard 4-city held-out run."""
    argv = [
        "--release",
        release,
        "--g4",
        str(g4_path),
        "--usable-n",
        str(usable_n_path),
    ]
    if lock:
        argv.append("--lock")
    return argv


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


# ===========================================================================
# Integration tests: write-once freeze orchestration (no real corpus)
# ===========================================================================


class TestFreezeOrchestration:
    """Integration tests for the --lock path in ``main``.

    All disk I/O against the real corpus is replaced by monkeypatches on the
    already-loaded ``build_mod`` module object:

    - ``build_mod._enumerate_city_tiles`` → ``_synthetic_tile_enumerator``
      (returns synthetic tile dicts; no sub-D manifest or provenance.yaml needed)
    - ``build_mod.multiregion_holdout_manifest_path`` → lambda returning a path under
      ``tmp_path`` (keeps the freeze write away from any real eval_set/ tree)
    - ``build_mod.multiregion_eval_set_locked_marker`` → lambda returning a path under
      ``tmp_path``

    The three safety properties under test:

    1. **Happy-path lock** — ``main([... --lock ...])`` writes a valid schema-2.0
       manifest and a marker whose operator-facing numbers match the manifest.
    2. **Write-once** — a second ``main([... --lock ...])`` on the same paths raises
       ``FileExistsError`` (the manifest's ``freeze_holdout_manifest`` guard fires first,
       before the marker guard, so either guard suffices; the manifest guard fires here).
    3. **No-false-DONE** — if ``_verify_frozen_end_state`` raises (monkeypatched), the
       marker is NOT written (the ordering guarantee: marker is written only AFTER a
       passing verify).
    """

    def _patch_module(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> tuple[Path, Path]:
        """Apply the three standard module-level monkeypatches.

        Returns (manifest_path, marker_path) — paths under tmp_path.
        """
        manifest_path = tmp_path / "holdout_manifest.yaml"
        marker_path = tmp_path / "_EVAL_SET_LOCKED"

        monkeypatch.setattr(build_mod, "_enumerate_city_tiles", _synthetic_tile_enumerator)
        monkeypatch.setattr(
            build_mod,
            "multiregion_holdout_manifest_path",
            lambda release: manifest_path,
        )
        monkeypatch.setattr(
            build_mod,
            "multiregion_eval_set_locked_marker",
            lambda release: marker_path,
        )
        return manifest_path, marker_path

    def test_lock_writes_valid_manifest_and_matching_marker(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Happy-path: --lock produces a schema-2.0 manifest and a consistent marker.

        Asserts:
        - manifest exists at the monkeypatched path with manifest_schema_version == "2.0"
        - marker exists
        - marker's held_out_cities / held_out_tokens / train_tokens match the manifest
        """
        manifest_path, marker_path = self._patch_module(monkeypatch, tmp_path)
        g4_path = _make_g4_yaml(tmp_path)
        usable_n_path = _make_usable_n_yaml(tmp_path)

        rc = build_mod.main(_make_argv(g4_path, usable_n_path, lock=True))
        assert rc == 0, "main() should return 0 on a successful lock"

        # Manifest must exist and carry schema 2.0.
        assert manifest_path.exists(), "manifest file must be written by --lock"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        assert manifest["manifest_schema_version"] == "2.0"

        # Marker must exist.
        assert marker_path.exists(), "lock marker must be written by --lock"
        marker = yaml.safe_load(marker_path.read_text(encoding="utf-8"))

        # Marker's operator-facing numbers must match the manifest.
        assert marker["held_out_cities"] == manifest["held_out_cities"], (
            "marker held_out_cities must match manifest"
        )
        assert marker["held_out_tokens"] == manifest["totals"]["held_out_tokens"], (
            "marker held_out_tokens must match manifest totals"
        )
        assert marker["train_tokens"] == manifest["totals"]["train_tokens"], (
            "marker train_tokens must match manifest totals"
        )

        # Sanity-check the held-out cities are the sorted default 4.
        assert manifest["held_out_cities"] == sorted(_HELD_OUT_TILE_COUNTS.keys())

    def test_write_once_second_lock_raises_file_exists_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Write-once safety: a second --lock run on the same paths must raise FileExistsError.

        The first run writes the manifest (via ``freeze_holdout_manifest``). The second run
        hits the same path and ``freeze_holdout_manifest`` refuses to overwrite — the
        FileExistsError propagates out of ``main`` without being caught, proving the
        write-once guard is active and not swallowed.
        """
        # Paths are shared state — the patch wires them in; we don't inspect them here.
        _manifest_path, _marker_path = self._patch_module(monkeypatch, tmp_path)
        g4_path = _make_g4_yaml(tmp_path)
        usable_n_path = _make_usable_n_yaml(tmp_path)

        argv = _make_argv(g4_path, usable_n_path, lock=True)

        # First run must succeed.
        rc = build_mod.main(argv)
        assert rc == 0

        # Second run on the same paths: the manifest already exists → FileExistsError.
        with pytest.raises(FileExistsError):
            build_mod.main(argv)

    def test_no_false_done_marker_absent_when_verify_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """No-false-DONE: if _verify_frozen_end_state raises, the marker is NOT written.

        The freeze→verify→marker ordering guarantee (Fix I2 + M2): the marker is written
        ONLY after a passing verify. If verify raises, the marker must never exist so a
        future session cannot be poisoned by a false "DONE".

        Injection: monkeypatch ``_verify_frozen_end_state`` on the module to raise an
        AssertionError unconditionally, simulating a post-freeze integrity failure.
        The manifest IS written (freeze already happened); the marker must NOT be.
        """
        manifest_path, marker_path = self._patch_module(monkeypatch, tmp_path)
        g4_path = _make_g4_yaml(tmp_path)
        usable_n_path = _make_usable_n_yaml(tmp_path)

        # Force _verify_frozen_end_state to fail AFTER the manifest is frozen.
        def _always_fail(**kwargs: object) -> None:
            raise AssertionError("injected verify failure: simulating end-state mismatch")

        monkeypatch.setattr(build_mod, "_verify_frozen_end_state", _always_fail)

        # main() should propagate the AssertionError (not swallow it).
        with pytest.raises(AssertionError, match="injected verify failure"):
            build_mod.main(_make_argv(g4_path, usable_n_path, lock=True))

        # The manifest was written by freeze_holdout_manifest before verify ran.
        assert manifest_path.exists(), (
            "manifest is written by freeze before verify — expected on disk even on verify failure"
        )

        # THE SAFETY PROPERTY: the marker must NOT exist because verify raised before
        # the marker-write code was reached.
        assert not marker_path.exists(), (
            "marker must NOT exist after a verify failure: "
            "writing the marker before verify passes is the false-DONE anti-pattern"
        )

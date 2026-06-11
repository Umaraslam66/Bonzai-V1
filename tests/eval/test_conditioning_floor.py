"""Conditioning-floor machinery teeth (readiness-closure Task 25 step 1; spec §8).

The floor artifact is the measurement instrument for the re-scoped Phase-2 eval.
TWO BH FAMILIES (PI call 2026-06-11): family 1 = D-D held-out pairwise (the
stage-1 determinism anchor; halts + delta ladder live here), family 2 = D-T
cross with its OWN BH (the Lane-M strata family); no T-T pair ever. Per
held-out city D and (metric, stratum), ``floor_heldout`` = strict min over D's
family-1 pairs and ``floor_all`` = strict min over family-1 u family-2 pairs
(PI knob 1; medians as context only). Lane S scores excess over ``floor_all``;
Lane M is the memorization discriminator over CROSS-selected strata.

Every tooth here is synthetic (no corpus locally): the fixtures are uniform
integer grids whose pairwise KS values are EXACT by construction and re-asserted
in-test before they drive any verdict (the house fixture self-check pattern).

Named teeth covered (dispatch list):
  1. write-once + sha + refusal regimes (tamper / marker / overwrite / version-skew)
  2. Lane-S refusal BEFORE any KS against an absent/tampered artifact
  3. regurgitator MUST FAIL Lane M / oracle MUST PASS both lanes
  4. collapse/explosion halts fire ONLY on their regime; healthy freezes
  5. no-leakage pin: strata selection takes ONLY the real-real pair table
  6. floor strictness: min over other cities, median as context

Task-25 spec-review pins (on top of the dispatch list):
  R1. Lane-M exact tie => FAIL with margin 0.0 (kills a `<` -> `<=` mutation)
  R2. non-degenerate floor fixture: the table's global-min pair EXCLUDES D
  R3. producer halts when a held-out city has zero qualifying pairs (no floor)

Task-25 quality-review pins:
  Q1. pair-table parity vs conditioning_discrimination_verdict (external truth)
  Q3. VerifiedFloorArtifact proof token unforgeable + payload deep-copy isolation
  Q6. Lane S/M min_n defaults from the artifact's frozen methodology (warn on
      explicit mismatch, never silent)
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

import pytest
import yaml

import cfm.eval.conditioning_floor as cf
from cfm.eval.conditioning_discrimination import (
    benjamini_hochberg,
    conditioning_discrimination_verdict,
)
from cfm.eval.conditioning_floor import (
    FLOOR_ARTIFACT_LOCK_NAME,
    FloorArtifactError,
    FloorCollapseError,
    FloorExplosionError,
    VerifiedFloorArtifact,
    build_floor_artifact_payload,
    compute_cross_pair_table,
    compute_floors,
    compute_pair_table,
    cross_pair_table_from_payload,
    discriminating_strata_from_artifact,
    floor_artifact_sha256,
    freeze_floor_artifact,
    lane_m_verdict,
    lane_s_excess,
    load_verified_floor,
    pair_table_from_payload,
    select_discriminating_strata,
)
from cfm.eval.realism import ks_distance

# --------------------------------------------------------------------------- #
# Synthetic fixtures: uniform integer grids -> EXACT pairwise KS by construction
# --------------------------------------------------------------------------- #

_SA = ("R", "S1", 1, "inland")  # (zoning, skeleton, density, coastal)
_SB = ("C", "S2", 2, "coastal")
_M = "building_area_m2"


def _grid(shift: int, n: int = 100) -> list[float]:
    """n distinct values shift..shift+n-1: KS(grid(a), grid(b)) == |a-b|/n exactly."""
    return [float(i + shift) for i in range(n)]


def _features_one_stratum(
    shifts: dict[str, int], stratum: tuple = _SA, metric: str = _M
) -> dict[tuple[str, tuple, str], list[float]]:
    return {(city, stratum, metric): _grid(s) for city, s in shifts.items()}


#: Floor-strictness fixture (tooth 6): KS(D,T1)=0.2, KS(D,T2)=0.4, KS(T1,T2)=0.2.
_STRICT_SHIFTS = {"d_city": 0, "t1_city": 20, "t2_city": 40}

#: Non-degenerate floor fixture (Task-25 spec review #2): the SMALLEST pair in
#: the table EXCLUDES D — KS(T1,T2)=0.1 < KS(D,T1)=0.3 < KS(D,T2)=0.4 — so a
#: global-min-over-the-table implementation (0.1) is distinguishable from the
#: correct min-over-D's-own-pairs (0.3). NOTE: the review's example KS(D,T2)=0.5
#: is unreachable here — KS is a metric, so KS(D,T2) <= KS(D,T1) + KS(T1,T2)
#: = 0.4; 0.4 preserves the property under test.
_NONDEGEN_SHIFTS = {"d_city": 0, "t1_city": 30, "t2_city": 40}

#: Regurgitator/oracle fixture (tooth 3): KS(D,T1)=0.3, KS(D,T2)=0.5, KS(T1,T2)=0.2;
#: all three pairs BH-significant at alpha=0.05 with n=100 (screened: p_bh max 0.031).
_LANE_SHIFTS = {"d_city": 0, "t1_city": 30, "t2_city": 50}

#: Collapse fixture: pairwise KS 0.01/0.02/0.01 -> median 0.01 < 0.049.
_COLLAPSE_SHIFTS = {"d_city": 0, "t1_city": 1, "t2_city": 2}

#: Explosion fixture: disjoint supports -> every KS == 1.0 > 0.5.
_EXPLOSION_SHIFTS = {"d_city": 0, "t1_city": 1000, "t2_city": 2000}


def test_fixture_self_check_grid_ks_values_are_exact() -> None:
    """The house fixture self-check: the KS regime is asserted BEFORE any tooth
    leans on it — a fixture drift turns this red, not a tooth silently vacuous."""
    assert ks_distance(_grid(0), _grid(20)) == pytest.approx(0.2)
    assert ks_distance(_grid(0), _grid(30)) == pytest.approx(0.3)
    assert ks_distance(_grid(0), _grid(40)) == pytest.approx(0.4)
    assert ks_distance(_grid(0), _grid(50)) == pytest.approx(0.5)
    assert ks_distance(_grid(0), _grid(1)) == pytest.approx(0.01)
    assert ks_distance(_grid(0), _grid(1000)) == 1.0
    assert ks_distance(_grid(0), _grid(0)) == 0.0
    assert ks_distance(_grid(30), _grid(40)) == pytest.approx(0.1)  # _NONDEGEN_SHIFTS T1-T2


# --------------------------------------------------------------------------- #
# Pair table
# --------------------------------------------------------------------------- #


def test_pair_table_has_every_unordered_pair_with_bh_adjusted_p() -> None:
    table = compute_pair_table(_features_one_stratum(_LANE_SHIFTS), min_n=50, alpha=0.05)
    assert len(table.pairs) == 3
    by_pair = {(p.city_a, p.city_b): p for p in table.pairs}
    assert set(by_pair) == {
        ("d_city", "t1_city"),
        ("d_city", "t2_city"),
        ("t1_city", "t2_city"),
    }
    assert by_pair[("d_city", "t1_city")].ks == pytest.approx(0.3)
    assert by_pair[("d_city", "t2_city")].ks == pytest.approx(0.5)
    assert by_pair[("t1_city", "t2_city")].ks == pytest.approx(0.2)
    for p in table.pairs:
        assert 0.0 <= p.p_bh <= 1.0
        assert p.p_bh >= p.p_raw  # BH never makes a p smaller
        assert p.n_a == p.n_b == 100
    # screened: all three pairs BH-significant at alpha=0.05
    assert all(p.p_bh < 0.05 for p in table.pairs)


def test_pair_table_qualify_rule_is_min_n_same_as_verdict_fn() -> None:
    """A city below min_n is excluded from pairing (the Task-22 qualify rule)."""
    features = _features_one_stratum(_STRICT_SHIFTS)
    features[("thin_city", _SA, _M)] = _grid(0, n=49)  # 49 < min_n=50
    table = compute_pair_table(features, min_n=50, alpha=0.05)
    cities = {c for p in table.pairs for c in (p.city_a, p.city_b)}
    assert "thin_city" not in cities
    assert table.n_excluded_thin == 1
    assert len(table.pairs) == 3


def test_pair_table_parity_with_the_verdict_fn() -> None:
    """External-source-of-truth pin (Task-25 quality review #1):
    ``compute_pair_table`` re-implements the qualify -> group -> pair ->
    global-BH steps of ``conditioning_discrimination_verdict``; run BOTH on one
    fixture and assert the (metric, stratum, city_a, city_b, ks, p_raw, p_bh)
    multisets match EXACTLY (same fns on same data => bit-equal; no tolerance).
    The fixture exercises every step: two strata (group), a thin city below
    the qualify boundary AND a city exactly AT it (a `>=` -> `>` drift on
    either side flips the at-boundary city and breaks parity; n=49-only would
    be regime-blind to it), a single-city stratum (pairing skip), and >1 pair
    per BH input (the global correction). ``effect_size_floor`` only sets the
    verdict fn's ``significant`` flag, which is not a compared pair field."""
    features = _features_one_stratum(_LANE_SHIFTS, stratum=_SA)
    features.update(_features_one_stratum({"d_city": 0, "t1_city": 40, "t2_city": 80}, stratum=_SB))
    features[("thin_city", _SA, _M)] = _grid(0, n=49)  # qualify: strictly below min_n
    features[("edge_city", _SB, _M)] = _grid(120, n=50)  # qualify: EXACTLY min_n
    features[("d_city", ("R", "S9", 9, "inland"), _M)] = _grid(0)  # lone-city stratum

    table = compute_pair_table(features, min_n=50, alpha=0.05)
    verdict = conditioning_discrimination_verdict(
        features, min_n=50, alpha=0.05, effect_size_floor=0.15
    )

    def _pair_multiset(pairs) -> list[tuple]:
        return sorted(
            (p.metric, p.stratum, p.city_a, p.city_b, p.ks, p.p_raw, p.p_bh) for p in pairs
        )

    # regime check: both strata paired AND the at-boundary city is IN the table
    assert len(table.pairs) == 9  # 3 in _SA + C(4,2)=6 in _SB
    assert any("edge_city" in (p.city_a, p.city_b) for p in table.pairs)
    assert _pair_multiset(table.pairs) == _pair_multiset(verdict.pairs)
    # the qualify/group bookkeeping agrees too
    assert table.n_excluded_thin == verdict.n_excluded_thin
    assert table.n_strata_too_few_cities == verdict.n_strata_too_few_cities


def test_pair_table_counts_strata_with_too_few_cities() -> None:
    features = _features_one_stratum(_STRICT_SHIFTS)
    features[("d_city", _SB, _M)] = _grid(0)  # only one qualifying city in _SB
    table = compute_pair_table(features, min_n=50, alpha=0.05)
    assert table.n_strata_too_few_cities == 1
    assert len(table.pairs) == 3  # _SB contributes no pair


# --------------------------------------------------------------------------- #
# Tooth 6: floor strictness — STRICT min over other cities + median context
# --------------------------------------------------------------------------- #


def test_floor_is_strict_min_over_other_cities_with_median_context() -> None:
    """No cross table: floor_heldout carries the strict-min rule and floor_all
    equals it exactly (reverse-lock: `floor`/`floor_median_context` renamed to
    the two-variant fields under the two-family design)."""
    table = compute_pair_table(_features_one_stratum(_STRICT_SHIFTS), min_n=50, alpha=0.05)
    floors = compute_floors(table, ["d_city"])
    entry = floors["d_city"][(_M, _SA)]
    assert entry.floor_heldout == pytest.approx(0.2)  # min(0.2, 0.4) — PI knob 1, STRICT
    assert entry.floor_heldout_median_context == pytest.approx(0.3)  # median(0.2, 0.4)
    assert entry.n_heldout_pairs == 2
    assert entry.floor_all == entry.floor_heldout  # no cross pairs: EXACT equality
    assert entry.n_cross_pairs == 0


def test_floors_for_multiple_held_out_cities_use_each_citys_own_pairs() -> None:
    table = compute_pair_table(_features_one_stratum(_STRICT_SHIFTS), min_n=50, alpha=0.05)
    floors = compute_floors(table, ["d_city", "t1_city"])
    # t1's pairs: KS(t1,d)=0.2, KS(t1,t2)=0.2 -> floor 0.2, context 0.2
    entry = floors["t1_city"][(_M, _SA)]
    assert entry.floor_heldout == pytest.approx(0.2)
    assert entry.floor_heldout_median_context == pytest.approx(0.2)


def test_floor_ignores_a_smaller_pair_that_excludes_d() -> None:
    """Non-degenerate strictness pin (Task-25 spec review #2): in _STRICT_SHIFTS
    the table's global min equals D's own min, so a global-min-over-the-table
    implementation would pass it. Here the table's smallest KS (0.1, the T1-T2
    pair) EXCLUDES D: floor_D must be 0.3 — D's own strict min — never 0.1."""
    table = compute_pair_table(_features_one_stratum(_NONDEGEN_SHIFTS), min_n=50, alpha=0.05)
    # regime check: the global table min is the D-excluding T1-T2 pair
    global_min_pair = min(table.pairs, key=lambda p: p.ks)
    assert (global_min_pair.city_a, global_min_pair.city_b) == ("t1_city", "t2_city")
    assert global_min_pair.ks == pytest.approx(0.1)

    floors = compute_floors(table, ["d_city"])
    entry = floors["d_city"][(_M, _SA)]
    assert entry.floor_heldout == pytest.approx(0.3)  # min over D's OWN pairs (0.3, 0.4)
    assert entry.floor_heldout != pytest.approx(0.1)  # the D-excluding 0.1 is ignored
    assert entry.floor_heldout_median_context == pytest.approx(0.35)  # median(0.3, 0.4)
    assert entry.n_heldout_pairs == 2


# --------------------------------------------------------------------------- #
# Tooth 4: collapse / explosion halts — each fires ONLY on its regime
# --------------------------------------------------------------------------- #


def _payload_kwargs() -> dict:
    """REVERSE-LOCK (two-family design): all three fixture cities are HELD OUT
    with no training cities, so family 1 reproduces the old joint table exactly
    (the old kwargs put t1/t2 in train_cities, which under the two-family
    builder would leave the lone held-out city with zero family-1 pairs)."""
    return {
        "release": "2026-04-15.0",
        "held_out_cities": ["d_city", "t1_city", "t2_city"],
        "train_cities": [],
        "min_n": 50,
        "alpha": 0.05,
        "delta": 0.15,
    }


def test_collapse_fixture_raises_floor_collapse_error_only() -> None:
    with pytest.raises(FloorCollapseError):
        build_floor_artifact_payload(_features_one_stratum(_COLLAPSE_SHIFTS), **_payload_kwargs())


def test_explosion_fixture_raises_floor_explosion_error_only() -> None:
    with pytest.raises(FloorExplosionError):
        build_floor_artifact_payload(_features_one_stratum(_EXPLOSION_SHIFTS), **_payload_kwargs())


def test_halt_types_are_distinct_and_neither_catches_the_other() -> None:
    """THE GUARD MUST DISTINGUISH REGIMES: collapse never raises Explosion and
    vice versa — proven by asserting the OTHER type does not fire."""
    assert not issubclass(FloorCollapseError, FloorExplosionError)
    assert not issubclass(FloorExplosionError, FloorCollapseError)
    with pytest.raises(FloorCollapseError) as ci:
        build_floor_artifact_payload(_features_one_stratum(_COLLAPSE_SHIFTS), **_payload_kwargs())
    assert not isinstance(ci.value, FloorExplosionError)
    with pytest.raises(FloorExplosionError) as ei:
        build_floor_artifact_payload(_features_one_stratum(_EXPLOSION_SHIFTS), **_payload_kwargs())
    assert not isinstance(ei.value, FloorCollapseError)


def test_halts_run_on_family1_not_the_cross_family(tmp_path: Path) -> None:
    """The halts live on FAMILY 1 (the floor-measurement core): collapse-close
    TRAINING cities cannot fire the collapse halt when the held-out family is
    healthy — and the same shifts as held-out cities DO fire it (regime pair)."""
    healthy_held = {"d_city": 0, "h_city": 20}
    collapse_train = {"t1_city": 1, "t2_city": 2}  # KS 0.01 from each other
    features = _features_one_stratum({**healthy_held, **collapse_train})
    payload = build_floor_artifact_payload(
        features,
        release="2026-04-15.0",
        held_out_cities=["d_city", "h_city"],
        train_cities=["t1_city", "t2_city"],
        min_n=50,
        alpha=0.05,
        delta=0.15,
    )
    freeze_floor_artifact(payload, tmp_path / "floor.yaml")  # no halt fired
    with pytest.raises(FloorCollapseError):  # same regime AS held-out cities: fires
        build_floor_artifact_payload(_features_one_stratum(_COLLAPSE_SHIFTS), **_payload_kwargs())


def test_healthy_fixture_neither_halts_and_freezes_verified(tmp_path: Path) -> None:
    """The recon regime (median KS ~0.2-0.3): no halt, artifact freezes, verifies."""
    payload = build_floor_artifact_payload(_features_one_stratum(_LANE_SHIFTS), **_payload_kwargs())
    path = tmp_path / "conditioning-floor.yaml"
    freeze_floor_artifact(payload, path)
    verified = load_verified_floor(path)
    assert isinstance(verified, VerifiedFloorArtifact)
    assert verified.payload["held_out_cities"] == ["d_city", "t1_city", "t2_city"]
    assert (tmp_path / FLOOR_ARTIFACT_LOCK_NAME).exists()


def test_unsupported_zero_qualifying_pairs_is_loud() -> None:
    """UNSUPPORTED is loud, never a silent empty artifact (mirror of the Task-22
    UNSUPPORTED contract): zero qualifying pairs refuses to build a payload."""
    thin = {("d_city", _SA, _M): _grid(0, n=10)}
    with pytest.raises(ValueError, match="UNSUPPORTED"):
        build_floor_artifact_payload(thin, **_payload_kwargs())


def test_missing_held_out_city_halts_the_producer_naming_the_city() -> None:
    """Task-25 spec review #3: a held-out city with zero qualifying pairs must
    halt the producer BEFORE any artifact content exists — a silently absent
    floor shrinks the worst-case max domain weeks later at Lane-S consumption
    against a write-once artifact (the aggregate-hides-subsets class)."""
    features = _features_one_stratum(_STRICT_SHIFTS)
    features[("ghost_city", _SA, _M)] = _grid(0, n=10)  # thin: zero qualifying pairs
    # regime check: the family-1 table itself is HEALTHY (3 pairs among the others)
    table = compute_pair_table(features, min_n=50, alpha=0.05)
    assert len(table.pairs) == 3
    kwargs = _payload_kwargs()
    kwargs["held_out_cities"] = ["d_city", "t1_city", "t2_city", "ghost_city"]
    with pytest.raises(ValueError, match="ghost_city") as excinfo:
        build_floor_artifact_payload(features, **kwargs)
    assert "UNSUPPORTED" in str(excinfo.value)
    assert "d_city" not in str(excinfo.value)  # names ONLY the missing city
    # the healthy fixture (every held-out city floored) is unaffected
    healthy = build_floor_artifact_payload(
        _features_one_stratum(_STRICT_SHIFTS), **_payload_kwargs()
    )
    assert {rec["city"] for rec in healthy["floors"]} == {"d_city", "t1_city", "t2_city"}


# --------------------------------------------------------------------------- #
# Tooth 1: write-once + sha + refusal regimes (the 24a registry grammar)
# --------------------------------------------------------------------------- #


def _frozen_payload() -> dict:
    """The canonical two-family payload: held-out {d,h}, train {t1,t2} (the
    _TWOFAM fixture from the two-family section below) — so the artifact-based
    teeth all exercise a payload with BOTH families populated."""
    return build_floor_artifact_payload(
        _features_one_stratum(_TWOFAM_SHIFTS),
        release="2026-04-15.0",
        held_out_cities=_TWOFAM_HELD,
        train_cities=_TWOFAM_TRAIN,
        min_n=50,
        alpha=0.05,
        delta=0.15,
    )


def _frozen_artifact(tmp_path: Path) -> Path:
    path = tmp_path / "conditioning-floor.yaml"
    freeze_floor_artifact(_frozen_payload(), path)
    return path


def test_freeze_is_write_once(tmp_path: Path) -> None:
    path = _frozen_artifact(tmp_path)
    with pytest.raises(FileExistsError):
        freeze_floor_artifact(_frozen_payload(), path)


def test_content_tamper_with_stale_sha_is_refused(tmp_path: Path) -> None:
    path = _frozen_artifact(tmp_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["floors"][0]["floor_all"] = 0.0001  # a friendlier floor; sha left stale
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(FloorArtifactError, match="sha"):
        load_verified_floor(path)


def test_missing_lock_marker_is_refused(tmp_path: Path) -> None:
    path = _frozen_artifact(tmp_path)
    (tmp_path / FLOOR_ARTIFACT_LOCK_NAME).unlink()
    with pytest.raises(FloorArtifactError, match=FLOOR_ARTIFACT_LOCK_NAME):
        load_verified_floor(path)


def test_missing_sha_field_is_refused_fail_closed(tmp_path: Path) -> None:
    path = _frozen_artifact(tmp_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    del data["floor_sha256"]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(FloorArtifactError, match="floor_sha256"):
        load_verified_floor(path)


def test_schema_version_skew_is_refused_with_restamped_sha(tmp_path: Path) -> None:
    """The 24a pattern: re-stamp the sha after the version edit so ONLY the
    version check can fire — never a sha artifact masquerading as the guard."""
    path = _frozen_artifact(tmp_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["floor_schema_version"] = "0.0"
    data["floor_sha256"] = floor_artifact_sha256(data)  # sha valid; only version skewed
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(FloorArtifactError, match="floor_schema_version"):
        load_verified_floor(path)


def test_absent_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FloorArtifactError, match="exist"):
        load_verified_floor(tmp_path / "never-written.yaml")


def test_malformed_yaml_is_refused_inside_the_taxonomy(tmp_path: Path) -> None:
    path = tmp_path / "conditioning-floor.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    (tmp_path / FLOOR_ARTIFACT_LOCK_NAME).touch()  # marker present: reach the parse seam
    with pytest.raises(FloorArtifactError, match="malformed"):
        load_verified_floor(path)


def test_verified_floor_artifact_cannot_be_forged_by_direct_construction(tmp_path: Path) -> None:
    """Task-25 quality review #3: the proof is a module-private sentinel only
    load_verified_floor passes — VerifiedFloorArtifact(path, payload) (no token)
    and a guessed token both refuse; the dataclass is no longer forgeable."""
    path = _frozen_artifact(tmp_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    with pytest.raises(TypeError):  # _token is keyword-only and REQUIRED
        VerifiedFloorArtifact(path, payload)
    with pytest.raises(FloorArtifactError, match="token"):
        VerifiedFloorArtifact(path=path, payload=payload, _token=object())  # guessed token


def test_mutating_a_loaded_payload_never_reaches_a_reload(tmp_path: Path) -> None:
    """Task-25 quality review #3: a load's payload is an isolated deep copy —
    mutating it post-verify cannot poison any other consumer; a re-load
    re-reads and re-verifies the sealed file from disk."""
    path = _frozen_artifact(tmp_path)
    first = load_verified_floor(path)
    pristine = first.payload["floors"][0]["floor_all"]
    first.payload["floors"][0]["floor_all"] = -1.0  # in-place mutation of ONE load's copy
    second = load_verified_floor(path)
    assert second.payload["floors"][0]["floor_all"] == pristine
    assert second.payload["floors"][0]["floor_all"] != -1.0


# --------------------------------------------------------------------------- #
# Tooth 2: Lane-S refusal BEFORE any KS is computed
# --------------------------------------------------------------------------- #


def _lane_features(shift: int) -> dict[tuple[str, tuple], list[float]]:
    return {(_M, _SA): _grid(shift)}


def test_lane_s_refuses_tampered_artifact_before_any_ks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _frozen_artifact(tmp_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["floors"][0]["floor_all"] = 0.9999
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    calls: list[object] = []

    def _recording_ks(a: list[float], b: list[float]) -> float:
        calls.append((a, b))
        raise AssertionError("ks_distance must never run against an unverified artifact")

    monkeypatch.setattr(cf, "ks_distance", _recording_ks)
    with pytest.raises(FloorArtifactError, match="sha"):
        lane_s_excess(_lane_features(0), _lane_features(0), path, city="d_city", min_n=50)
    assert calls == []  # the refusal fired BEFORE any KS


def test_lane_s_refuses_absent_artifact(tmp_path: Path) -> None:
    with pytest.raises(FloorArtifactError, match="exist"):
        lane_s_excess(
            _lane_features(0),
            _lane_features(0),
            tmp_path / "never-written.yaml",
            city="d_city",
            min_n=50,
        )


def test_lane_s_refuses_a_raw_dict_that_skipped_verification(tmp_path: Path) -> None:
    """Only a Path (verified inside) or a VerifiedFloorArtifact (the load result
    type) can reach scoring — a bare payload dict cannot impersonate one."""
    path = _frozen_artifact(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    with pytest.raises(FloorArtifactError, match="VerifiedFloorArtifact"):
        lane_s_excess(_lane_features(0), _lane_features(0), raw, city="d_city", min_n=50)


# --------------------------------------------------------------------------- #
# Tooth 3: regurgitator MUST FAIL Lane M / oracle MUST PASS both lanes
# --------------------------------------------------------------------------- #


def _twofam_cross_strata() -> dict[tuple[str, str], tuple[tuple[str, tuple], ...]]:
    """Discriminating strata from the CROSS family (the production path)."""
    cross = compute_cross_pair_table(
        _features_one_stratum(_TWOFAM_SHIFTS),
        held_out=_TWOFAM_HELD,
        train=_TWOFAM_TRAIN,
        min_n=50,
        alpha=0.05,
    )
    return select_discriminating_strata(cross, delta=0.15)


def test_regurgitator_fails_lane_m_against_the_city_it_copied() -> None:
    """gen := real_T1 samples. KS(gen, D)=0.3 (large, in-test asserted) while
    KS(gen, T1)=0 -> gen matches T1 strictly better than D -> Lane M FAILS."""
    gen = _lane_features(30)  # T1's real samples, verbatim
    real_d = _lane_features(0)
    real_t1 = _lane_features(30)
    # fixture self-check: the regime is asserted before the verdict is read
    assert ks_distance(gen[(_M, _SA)], real_d[(_M, _SA)]) == pytest.approx(0.3)
    assert ks_distance(gen[(_M, _SA)], real_t1[(_M, _SA)]) == 0.0

    strata = _twofam_cross_strata()[("d_city", "t1_city")]
    assert strata == ((_M, _SA),)  # the fixture's single stratum is selected

    result = lane_m_verdict(gen, real_d, real_t1, strata, min_n=50)
    assert result.verdict == "FAIL"
    assert result.median_ks_gen_d == pytest.approx(0.3)
    assert result.median_ks_gen_t == 0.0
    assert result.margin == pytest.approx(-0.3)  # negative: closer to T than to D


def test_oracle_passes_lane_m_against_every_training_city() -> None:
    """gen := real_D samples -> KS(gen,D)=0 strictly below KS(gen,T) for both T."""
    gen = _lane_features(0)
    real_d = _lane_features(0)
    strata_sel = _twofam_cross_strata()
    for t_city, t_shift in (("t1_city", 30), ("t2_city", 50)):
        real_t = _lane_features(t_shift)
        expected_ks = t_shift / 100.0
        assert ks_distance(gen[(_M, _SA)], real_t[(_M, _SA)]) == pytest.approx(expected_ks)
        strata = strata_sel[tuple(sorted(("d_city", t_city)))]
        result = lane_m_verdict(gen, real_d, real_t, strata, min_n=50)
        assert result.verdict == "PASS"
        assert result.median_ks_gen_d == 0.0
        assert result.margin == pytest.approx(expected_ks)


def test_lane_m_exact_tie_is_fail_with_zero_margin() -> None:
    """Boundary pin for the discriminator's STRICT `<` (Task-25 spec review #1):
    gen EQUIDISTANT from D and T must FAIL with margin exactly 0.0 — a `<` ->
    `<=` mutation flips this verdict to PASS and turns the test red. The tie
    regime is satisfiability-screened in-test through the real ks_distance:
    gen=_grid(15) sits exactly 0.15 from both real_D=_grid(0) and
    real_T=_grid(30), and the two KS values are asserted bit-equal."""
    gen = _lane_features(15)
    real_d = _lane_features(0)
    real_t = _lane_features(30)
    ks_d = ks_distance(gen[(_M, _SA)], real_d[(_M, _SA)])
    ks_t = ks_distance(gen[(_M, _SA)], real_t[(_M, _SA)])
    assert ks_d == pytest.approx(0.15)
    assert ks_d == ks_t  # EXACT float equality: the tie regime really holds

    result = lane_m_verdict(gen, real_d, real_t, [(_M, _SA)], min_n=50)
    assert result.median_ks_gen_d == result.median_ks_gen_t  # equal medians
    assert result.verdict == "FAIL"  # strict `<`: a tie is NOT a pass
    assert result.margin == 0.0


def test_oracle_lane_s_excess_is_exactly_zero(tmp_path: Path) -> None:
    """gen := real_D -> KS(gen, real_D)=0 <= floor -> every excess is 0.0."""
    path = _frozen_artifact(tmp_path)
    result = lane_s_excess(_lane_features(0), _lane_features(0), path, city="d_city", min_n=50)
    assert result.n_qualifying == 1
    assert result.per_stratum_excess[(_M, _SA)] == 0.0
    assert result.median_excess == 0.0
    assert result.p90_excess == 0.0


def test_lane_s_excess_is_ks_minus_floor_clamped_at_zero(tmp_path: Path) -> None:
    """Two strata with floors 0.2 (SA) and 0.4 (SB); gen drifts 0.3 in SA
    (excess 0.1) and matches D in SB (excess 0): median/p90 are aggregates
    over per-stratum excess (PI knob 3)."""
    features = _features_one_stratum(_STRICT_SHIFTS, stratum=_SA)
    features.update(_features_one_stratum({"d_city": 0, "t1_city": 40, "t2_city": 80}, stratum=_SB))
    payload = build_floor_artifact_payload(features, **_payload_kwargs())
    path = tmp_path / "floor.yaml"
    freeze_floor_artifact(payload, path)

    gen = {(_M, _SA): _grid(30), (_M, _SB): _grid(0)}
    real_d = {(_M, _SA): _grid(0), (_M, _SB): _grid(0)}
    result = lane_s_excess(gen, real_d, path, city="d_city", min_n=50)
    assert result.per_stratum_excess[(_M, _SA)] == pytest.approx(0.1)  # 0.3 - 0.2
    assert result.per_stratum_excess[(_M, _SB)] == 0.0  # max(0, 0 - 0.4)
    assert result.median_excess == pytest.approx(0.05)
    assert result.p90_excess == pytest.approx(0.09)  # linear-interp p90 of [0, 0.1]
    assert result.n_qualifying == 2


def test_lane_m_with_zero_scoreable_strata_is_loud() -> None:
    thin = {(_M, _SA): _grid(0, n=10)}  # below min_n: nothing scoreable
    with pytest.raises(ValueError, match="zero"):
        lane_m_verdict(thin, thin, thin, [(_M, _SA)], min_n=50)


# --------------------------------------------------------------------------- #
# Task-25 quality review #6: min_n defaults from the artifact's frozen methodology
# --------------------------------------------------------------------------- #


def _frozen_at_min_n_60(
    tmp_path: Path,
    shifts: dict[str, int],
    held_out: list[str] | None = None,
    train: list[str] | None = None,
) -> Path:
    kwargs = _payload_kwargs()
    kwargs["min_n"] = 60
    if held_out is not None:
        kwargs["held_out_cities"] = held_out
    if train is not None:
        kwargs["train_cities"] = train
    payload = build_floor_artifact_payload(_features_one_stratum(shifts), **kwargs)
    path = tmp_path / "floor.yaml"
    freeze_floor_artifact(payload, path)
    return path


def test_lane_s_min_n_defaults_from_the_artifact_methodology(tmp_path: Path) -> None:
    """Artifact frozen at min_n=60; gen/real carry 55 samples. Without an
    explicit min_n, Lane S must score at the ARTIFACT's qualify rule (55 < 60
    -> thin -> loud UNSUPPORTED) — under the old hardcoded 50 it would have
    silently scored at a different rule than the frozen floors."""
    path = _frozen_at_min_n_60(tmp_path, _STRICT_SHIFTS)
    gen = {(_M, _SA): _grid(30, n=55)}
    real = {(_M, _SA): _grid(0, n=55)}
    with pytest.raises(ValueError, match="min_n=60"):
        lane_s_excess(gen, real, path, city="d_city")  # no explicit min_n


def test_lane_s_explicit_min_n_mismatch_warns_never_silent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An explicit min_n that disagrees with the artifact's methodology is
    honored but WARNED (never a silent rescore); a matching explicit value
    stays quiet."""
    path = _frozen_at_min_n_60(tmp_path, _STRICT_SHIFTS)
    gen = {(_M, _SA): _grid(30, n=55)}
    real = {(_M, _SA): _grid(0, n=55)}
    with caplog.at_level(logging.WARNING, logger="cfm.eval.conditioning_floor"):
        result = lane_s_excess(gen, real, path, city="d_city", min_n=50)
    assert result.n_qualifying == 1  # honored: scored at the explicit 50
    assert any(
        "min_n=50" in r.getMessage() and "min_n=60" in r.getMessage() for r in caplog.records
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="cfm.eval.conditioning_floor"):
        lane_s_excess({(_M, _SA): _grid(30)}, {(_M, _SA): _grid(0)}, path, city="d_city", min_n=60)
    assert not caplog.records  # explicit == frozen: no warning


def test_lane_m_min_n_defaults_from_the_artifact_when_given(tmp_path: Path) -> None:
    """Lane M with the optional artifact keyword sources min_n from the frozen
    methodology (55 < 60 -> loud); a bare call keeps the legacy default 50; a
    raw payload dict cannot impersonate the artifact keyword."""
    path = _frozen_at_min_n_60(tmp_path, _TWOFAM_SHIFTS, held_out=_TWOFAM_HELD, train=_TWOFAM_TRAIN)
    verified = load_verified_floor(path)
    strata = discriminating_strata_from_artifact(verified, "d_city", "t1_city")
    feats55 = {(_M, _SA): _grid(0, n=55)}
    with pytest.raises(ValueError, match="min_n=60"):
        lane_m_verdict(feats55, feats55, feats55, strata, artifact=verified)
    result = lane_m_verdict(feats55, feats55, feats55, strata)  # bare: legacy 50
    assert result.n_strata_scored == 1
    with pytest.raises(FloorArtifactError, match="VerifiedFloorArtifact"):
        lane_m_verdict(feats55, feats55, feats55, strata, artifact=verified.payload)


# --------------------------------------------------------------------------- #
# Tooth 5: no-leakage pin — strata selection reads ONLY the real-real pair table
# --------------------------------------------------------------------------- #


def test_no_leakage_select_discriminating_strata_signature_pin() -> None:
    """BY SIGNATURE: the only inputs are the real-real pair table and the delta
    knob — generated data has no parameter to arrive through."""
    sig = inspect.signature(select_discriminating_strata)
    assert list(sig.parameters) == ["pair_table", "delta"]
    assert sig.parameters["delta"].kind is inspect.Parameter.KEYWORD_ONLY


def test_no_leakage_frozen_strata_equal_recomputation_from_real_table(tmp_path: Path) -> None:
    """Behavioral pin: the artifact's FROZEN strata selection equals a fresh
    recomputation from its own real-real CROSS table (the production input
    under the two-family design) — together with the signature pin above, this
    is the whole no-leakage tooth. (The former with/without-gen-data re-derive
    was trimmed, Task-25 quality review #7: a pure function of the table
    trivially repeats itself, proving nothing.)"""
    path = _frozen_artifact(tmp_path)
    verified = load_verified_floor(path)
    table = cross_pair_table_from_payload(verified.payload)
    assert table.pairs  # regime: the cross family is populated, not vacuous
    recomputed = select_discriminating_strata(table, delta=0.15)
    frozen = {
        (rec["city_a"], rec["city_b"]): tuple(
            (s["metric"], tuple(s["stratum"])) for s in rec["strata"]
        )
        for rec in verified.payload["discriminating_strata"]
    }
    assert frozen == recomputed


def test_discriminating_strata_from_artifact_requires_the_verified_type(
    tmp_path: Path,
) -> None:
    """The Lane-M path derives strata EXCLUSIVELY from the verified artifact:
    a raw payload dict (anything that skipped load_verified_floor) is refused."""
    path = _frozen_artifact(tmp_path)
    verified = load_verified_floor(path)
    strata = discriminating_strata_from_artifact(verified, "d_city", "t1_city")
    assert strata == ((_M, _SA),)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    with pytest.raises(FloorArtifactError, match="VerifiedFloorArtifact"):
        discriminating_strata_from_artifact(raw, "d_city", "t1_city")
    # a pair ABSENT from the table is loud, never a silent empty selection
    with pytest.raises(FloorArtifactError, match="no pair"):
        discriminating_strata_from_artifact(verified, "d_city", "nowhere_city")


def test_select_requires_bh_significance_not_just_ks_magnitude() -> None:
    """delta alone is not enough: a pair must ALSO be BH-significant. With tiny
    n the same KS=0.3 is not significant (noise floor ~0.86 at n=10). Built on
    the CROSS table — the production input to the selection."""
    features = {
        ("d_city", _SA, _M): _grid(0, n=10),
        ("t1_city", _SA, _M): _grid(3, n=10),  # KS = 0.3 but n=10
    }
    table = compute_cross_pair_table(
        features, held_out=["d_city"], train=["t1_city"], min_n=5, alpha=0.05
    )
    assert len(table.pairs) == 1
    assert table.pairs[0].ks == pytest.approx(0.3)
    assert table.pairs[0].p_bh >= 0.05  # not significant at this n
    assert select_discriminating_strata(table, delta=0.15)[("d_city", "t1_city")] == ()


# --------------------------------------------------------------------------- #
# Payload round-trip + ladder
# --------------------------------------------------------------------------- #


def test_pair_table_round_trips_through_the_payload(tmp_path: Path) -> None:
    """Family-1 round-trip: with every fixture city held out, the builder's
    family-1 table equals a direct compute_pair_table on the same features."""
    table = compute_pair_table(_features_one_stratum(_LANE_SHIFTS), min_n=50, alpha=0.05)
    payload = build_floor_artifact_payload(_features_one_stratum(_LANE_SHIFTS), **_payload_kwargs())
    rebuilt = pair_table_from_payload(payload)
    assert rebuilt.pairs == table.pairs
    assert rebuilt.min_n == table.min_n
    assert rebuilt.alpha == table.alpha


def test_delta_ladder_counts_bh_significant_pairs_per_anchor() -> None:
    payload = build_floor_artifact_payload(_features_one_stratum(_LANE_SHIFTS), **_payload_kwargs())
    ladder = {row["delta"]: row["n_pairs"] for row in payload["delta_ladder"]}
    assert ladder[0.15] == 3  # 0.2, 0.3, 0.5 all significant and >= 0.15
    assert ladder[0.25] == 2  # 0.3, 0.5
    assert ladder[0.35] == 1  # 0.5
    assert ladder[0.5] == 1


def test_payload_methodology_records_the_knobs_and_no_raw_samples() -> None:
    """The artifact stores KS TABLES, never raw samples (the orchestrator-fixed
    size decision): no list of 100 floats appears anywhere in the payload."""
    payload = build_floor_artifact_payload(_features_one_stratum(_LANE_SHIFTS), **_payload_kwargs())
    meth = payload["methodology"]
    assert meth["min_n"] == 50
    assert meth["alpha"] == 0.05
    assert meth["delta"] == 0.15
    assert meth["floor_rule"] == "strict_min_over_other_cities"
    assert payload["release"] == "2026-04-15.0"

    def _no_long_float_lists(node: object) -> None:
        if isinstance(node, dict):
            for v in node.values():
                _no_long_float_lists(v)
        elif isinstance(node, list):
            assert not (len(node) >= 50 and all(isinstance(x, float) for x in node))
            for v in node:
                _no_long_float_lists(v)

    _no_long_float_lists(payload)


# --------------------------------------------------------------------------- #
# Two-BH-family teeth (PI call 2026-06-11; supersedes the single joint family)
#
# Family 1 (D-D, held-out pairwise) = the stage-1 computation + determinism
# anchor; family 2 (D-T cross) = its own BH, the Lane-M strata family; no T-T
# pair exists anywhere; Lane S scores floor_all, floor_heldout is context.
# --------------------------------------------------------------------------- #

#: Canonical two-family fixture: 2 held-out + 2 training cities, one stratum.
#: family 1: KS(d,h)=0.2. cross: KS(d,t1)=0.3, KS(d,t2)=0.5, KS(h,t1)=0.1,
#: KS(h,t2)=0.3 (all exact; screened — cross own-BH: (h,t1) p_bh 0.677 NOT
#: significant, the other three are).
_TWOFAM_SHIFTS = {"d_city": 0, "h_city": 20, "t1_city": 30, "t2_city": 50}
_TWOFAM_HELD = ["d_city", "h_city"]
_TWOFAM_TRAIN = ["t1_city", "t2_city"]

#: Tooth-(d)/(e) fixture: the TRAINING city is D's closest real city.
#: family 1: KS(d,dfar)=0.4; cross: KS(d,tnear)=0.1, KS(dfar,tnear)=0.3.
#: Screened: the KS triangle inequality holds (0.4 <= 0.1 + 0.3, tight).
_TIGHTEN_SHIFTS = {"d_city": 0, "dfar_city": 40, "tnear_city": 10}
_TIGHTEN_HELD = ["d_city", "dfar_city"]
_TIGHTEN_TRAIN = ["tnear_city"]


def _build_payload(
    features: dict[tuple[str, tuple, str], list[float]],
    held_out: list[str],
    train: list[str],
) -> dict:
    return build_floor_artifact_payload(
        features,
        release="2026-04-15.0",
        held_out_cities=held_out,
        train_cities=train,
        min_n=50,
        alpha=0.05,
        delta=0.15,
    )


def test_twofam_fixture_self_check_grid_ks_values_are_exact() -> None:
    """House self-check for the two-family fixtures (asserted BEFORE any tooth)."""
    assert ks_distance(_grid(0), _grid(20)) == pytest.approx(0.2)  # (d,h)
    assert ks_distance(_grid(20), _grid(30)) == pytest.approx(0.1)  # (h,t1)
    assert ks_distance(_grid(20), _grid(50)) == pytest.approx(0.3)  # (h,t2)
    assert ks_distance(_grid(0), _grid(40)) == pytest.approx(0.4)  # (d,dfar)
    assert ks_distance(_grid(0), _grid(10)) == pytest.approx(0.1)  # (d,tnear)
    assert ks_distance(_grid(40), _grid(10)) == pytest.approx(0.3)  # (dfar,tnear)


def test_cross_pair_table_is_dt_only_with_its_own_bh() -> None:
    """Family 2: exactly the (d, t) pairs, each held-out x train per stratum;
    BH is over the cross family's OWN p_raws — provably NOT the joint-family
    adjustment (the two disagree on this fixture)."""
    features = _features_one_stratum(_TWOFAM_SHIFTS)
    cross = compute_cross_pair_table(
        features, held_out=_TWOFAM_HELD, train=_TWOFAM_TRAIN, min_n=50, alpha=0.05
    )
    by_pair = {(p.city_a, p.city_b): p for p in cross.pairs}
    assert set(by_pair) == {
        ("d_city", "t1_city"),
        ("d_city", "t2_city"),
        ("h_city", "t1_city"),
        ("h_city", "t2_city"),
    }
    assert by_pair[("d_city", "t1_city")].ks == pytest.approx(0.3)
    assert by_pair[("d_city", "t2_city")].ks == pytest.approx(0.5)
    assert by_pair[("h_city", "t1_city")].ks == pytest.approx(0.1)
    assert by_pair[("h_city", "t2_city")].ks == pytest.approx(0.3)
    # own BH: bit-equal to the BH helper over ONLY the cross p_raws, in order
    assert [p.p_bh for p in cross.pairs] == benjamini_hochberg([p.p_raw for p in cross.pairs])
    # ... and NOT the joint-family adjustment (regime check: they differ here)
    joint = compute_pair_table(features, min_n=50, alpha=0.05)
    joint_dt1 = next(p for p in joint.pairs if (p.city_a, p.city_b) == ("d_city", "t1_city"))
    assert joint_dt1.p_bh != by_pair[("d_city", "t1_city")].p_bh


def test_cross_pair_table_refuses_a_city_in_both_lists() -> None:
    """A city in BOTH held_out and train breaks the D-vs-T boundary — loud."""
    features = _features_one_stratum(_TWOFAM_SHIFTS)
    with pytest.raises(ValueError, match="d_city"):
        compute_cross_pair_table(
            features, held_out=["d_city"], train=["d_city", "t1_city"], min_n=50, alpha=0.05
        )


def test_tooth_a_family1_is_bit_equal_with_or_without_training_cities() -> None:
    """PI tooth (a), family-1 invariance: building the payload from
    {held-out + training} features yields family-1 pairs / p_bh / ladder /
    floor_heldout BIT-EQUAL to building from held-out-only features — family 1
    IS the stage-1 computation, so the stage-1 <-> stage-2 determinism check
    holds as a statistics fact. The OLD joint-BH path fails this: joint p_bh
    for the (d,h) pair is 0.0377 vs family-1's 0.0314 (asserted below — the
    regime the tooth distinguishes)."""
    features_all = _features_one_stratum(_TWOFAM_SHIFTS)
    held_set = set(_TWOFAM_HELD)
    features_held = {key: vals for key, vals in features_all.items() if key[0] in held_set}

    with_train = _build_payload(features_all, _TWOFAM_HELD, _TWOFAM_TRAIN)
    without_train = _build_payload(features_held, _TWOFAM_HELD, [])

    assert with_train["pairs"] == without_train["pairs"]  # bit-equal records (incl. p_bh)
    assert with_train["pair_table_median_ks"] == without_train["pair_table_median_ks"]
    assert with_train["delta_ladder"] == without_train["delta_ladder"]

    def _heldout_floor_fields(payload: dict) -> list[tuple]:
        return [
            (
                rec["city"],
                rec["metric"],
                tuple(rec["stratum"]),
                rec["floor_heldout"],
                rec["floor_heldout_median_context"],
            )
            for rec in payload["floors"]
        ]

    assert _heldout_floor_fields(with_train) == _heldout_floor_fields(without_train)

    # RED-against-the-old-path: the joint family (all cities into ONE BH) shifts
    # the held-out pair's p_bh — family size changes the BH adjustment.
    joint = compute_pair_table(features_all, min_n=50, alpha=0.05)
    fam1 = pair_table_from_payload(with_train)
    joint_dh = next(p for p in joint.pairs if (p.city_a, p.city_b) == ("d_city", "h_city"))
    fam1_dh = next(p for p in fam1.pairs if (p.city_a, p.city_b) == ("d_city", "h_city"))
    assert joint_dh.p_bh != fam1_dh.p_bh


def test_tooth_b_families_are_bh_independent() -> None:
    """PI tooth (b): removing a cross pair (drop a training city) leaves
    family-1 p_bh records bit-equal; adding a held-out-only stratum (new
    family-1 pair, zero new cross pairs) leaves family-2 records bit-equal.
    Each direction carries a regime check that the OTHER family really moved."""
    features = _features_one_stratum(_TWOFAM_SHIFTS)
    base = _build_payload(features, _TWOFAM_HELD, _TWOFAM_TRAIN)

    # direction 1: drop t2_city -> fewer cross pairs; family 1 untouched
    no_t2_features = {key: vals for key, vals in features.items() if key[0] != "t2_city"}
    no_t2 = _build_payload(no_t2_features, _TWOFAM_HELD, ["t1_city"])
    assert no_t2["pairs"] == base["pairs"]  # family-1 bit-equal
    assert no_t2["cross_pairs"] != base["cross_pairs"]  # regime: cross really moved
    # the surviving (d,t1) cross pair's p_bh CHANGED (its own BH family shrank)
    rec_base = next(
        r for r in base["cross_pairs"] if (r["city_a"], r["city_b"]) == ("d_city", "t1_city")
    )
    rec_no_t2 = next(
        r for r in no_t2["cross_pairs"] if (r["city_a"], r["city_b"]) == ("d_city", "t1_city")
    )
    assert rec_base["p_bh"] != rec_no_t2["p_bh"]

    # direction 2: a held-out-only stratum adds a family-1 pair, no cross pair
    widened = dict(features)
    widened.update(
        _features_one_stratum({"d_city": 0, "h_city": 30}, stratum=_SB)
    )  # t-cities absent in _SB -> no new cross pair
    wide = _build_payload(widened, _TWOFAM_HELD, _TWOFAM_TRAIN)
    assert wide["cross_pairs"] == base["cross_pairs"]  # family-2 bit-equal
    assert wide["pairs"] != base["pairs"]  # regime: family 1 really moved


def test_tooth_c_no_tt_pair_in_either_family_and_the_tooth_bites() -> None:
    """PI tooth (c): with 2 held-out + 2 training cities, NO (t, t) pair exists
    in either table — family 1 is D-D only, family 2 is exactly one-D-one-T.
    Teeth proof: the deliberately mis-scoped OLD call (all cities into
    compute_pair_table) contains a T-T pair and FAILS the same assertion."""
    features = _features_one_stratum(_TWOFAM_SHIFTS)
    payload = _build_payload(features, _TWOFAM_HELD, _TWOFAM_TRAIN)
    held, train = set(_TWOFAM_HELD), set(_TWOFAM_TRAIN)

    def _assert_no_tt(records: list[dict]) -> None:
        for rec in records:
            assert not (rec["city_a"] in train and rec["city_b"] in train), (
                f"T-T pair leaked: {rec['city_a']}, {rec['city_b']}"
            )

    _assert_no_tt(payload["pairs"])
    _assert_no_tt(payload["cross_pairs"])
    for rec in payload["pairs"]:  # family 1: BOTH cities held-out
        assert rec["city_a"] in held and rec["city_b"] in held
    for rec in payload["cross_pairs"]:  # family 2: exactly one of each
        cities = {rec["city_a"], rec["city_b"]}
        assert len(cities & held) == 1
        assert len(cities & train) == 1

    # the tooth BITES: the mis-scoped joint call yields a (t1,t2) pair and the
    # very assertion above goes red on it
    joint = compute_pair_table(features, min_n=50, alpha=0.05)
    joint_records = [{"city_a": p.city_a, "city_b": p.city_b} for p in joint.pairs]
    assert ("t1_city", "t2_city") in {(r["city_a"], r["city_b"]) for r in joint_records}
    with pytest.raises(AssertionError, match="T-T pair leaked"):
        _assert_no_tt(joint_records)


def test_tooth_d_training_city_closest_tightens_floor_all_exactly() -> None:
    """PI tooth (d): a training city closer than every held-out city tightens
    floor_all BELOW floor_heldout — exact values, both variants + contexts."""
    features = _features_one_stratum(_TIGHTEN_SHIFTS)
    payload = _build_payload(features, _TIGHTEN_HELD, _TIGHTEN_TRAIN)
    rows = {rec["city"]: rec for rec in payload["floors"]}

    d = rows["d_city"]
    assert d["floor_heldout"] == pytest.approx(0.4)  # min over family-1 pairs: {0.4}
    assert d["floor_all"] == pytest.approx(0.1)  # min over {0.4} u {0.1}
    assert d["floor_all"] < d["floor_heldout"]
    assert d["floor_heldout_median_context"] == pytest.approx(0.4)
    assert d["floor_all_median_context"] == pytest.approx(0.25)  # median(0.4, 0.1)

    dfar = rows["dfar_city"]
    assert dfar["floor_heldout"] == pytest.approx(0.4)
    assert dfar["floor_all"] == pytest.approx(0.3)  # min over {0.4} u {0.3}


def test_tooth_d_no_qualifying_cross_pair_means_exact_equality() -> None:
    """PI tooth (d), second leg: training city present but THIN (never
    qualifies) -> zero cross pairs -> floor_all == floor_heldout EXACTLY
    (bit-equal, not approx), and the cross side of the payload is empty."""
    features = _features_one_stratum({"d_city": 0, "dfar_city": 40})
    features[("tnear_city", _SA, _M)] = _grid(10, n=10)  # thin: below min_n=50
    payload = _build_payload(features, _TIGHTEN_HELD, _TIGHTEN_TRAIN)
    assert payload["cross_pairs"] == []
    assert payload["cross_median_ks"] is None
    for rec in payload["floors"]:
        assert rec["floor_all"] == rec["floor_heldout"]  # EXACT equality
        assert rec["floor_all_median_context"] == rec["floor_heldout_median_context"]
        assert rec["n_cross_pairs"] == 0


def test_tooth_d_floor_all_never_exceeds_floor_heldout() -> None:
    """Invariant: floor_all is a min over a SUPERSET of floor_heldout's pairs,
    so floor_all <= floor_heldout on every row of every payload."""
    for shifts, held, train in (
        (_TWOFAM_SHIFTS, _TWOFAM_HELD, _TWOFAM_TRAIN),
        (_TIGHTEN_SHIFTS, _TIGHTEN_HELD, _TIGHTEN_TRAIN),
    ):
        payload = _build_payload(_features_one_stratum(shifts), held, train)
        assert payload["floors"]  # regime: rows exist
        for rec in payload["floors"]:
            assert rec["floor_all"] <= rec["floor_heldout"]


def test_tooth_e_lane_s_consumes_floor_all_not_floor_heldout(tmp_path: Path) -> None:
    """PI tooth (e), scored-bar pin: on a fixture where the variants DIFFER
    (floor_all 0.1, floor_heldout 0.4) and KS(gen, real_D) = 0.3, the excess
    must match floor_all's arithmetic (0.3 - 0.1 = 0.2) — floor_heldout's
    arithmetic would clamp to 0.0 and is asserted away."""
    features = _features_one_stratum(_TIGHTEN_SHIFTS)
    payload = _build_payload(features, _TIGHTEN_HELD, _TIGHTEN_TRAIN)
    path = tmp_path / "floor.yaml"
    freeze_floor_artifact(payload, path)

    gen = {(_M, _SA): _grid(30)}
    real_d = {(_M, _SA): _grid(0)}
    assert ks_distance(gen[(_M, _SA)], real_d[(_M, _SA)]) == pytest.approx(0.3)  # regime

    result = lane_s_excess(gen, real_d, path, city="d_city", min_n=50)
    assert result.per_stratum_excess[(_M, _SA)] == pytest.approx(0.2)  # 0.3 - floor_all
    assert result.per_stratum_excess[(_M, _SA)] != pytest.approx(0.0)  # NOT floor_heldout's
    assert result.median_excess == pytest.approx(0.2)


def test_discriminating_strata_select_on_the_cross_family() -> None:
    """Design point 5: the strata selection runs on the CROSS table — per
    (D, T), KS >= delta AND cross-family-BH-significant. (h,t1) has KS 0.1
    (below delta AND not significant at cross-BH) -> empty selection."""
    features = _features_one_stratum(_TWOFAM_SHIFTS)
    cross = compute_cross_pair_table(
        features, held_out=_TWOFAM_HELD, train=_TWOFAM_TRAIN, min_n=50, alpha=0.05
    )
    sel = select_discriminating_strata(cross, delta=0.15)
    assert sel[("d_city", "t1_city")] == ((_M, _SA),)
    assert sel[("d_city", "t2_city")] == ((_M, _SA),)
    assert sel[("h_city", "t2_city")] == ((_M, _SA),)
    assert sel[("h_city", "t1_city")] == ()  # KS 0.1: below delta, not significant


def test_two_family_payload_shape_and_methodology_lineage() -> None:
    """Schema 2.0 shape: pairs + cross_pairs (same record grammar) + two-variant
    floors + cross_median_ks; methodology records floor_scored=floor_all with
    floor_heldout as context + the PI rationale. A held-out-only build carries
    an EMPTY cross side (cross_pairs [], cross_median_ks None, no strata)."""
    features = _features_one_stratum(_TWOFAM_SHIFTS)
    payload = _build_payload(features, _TWOFAM_HELD, _TWOFAM_TRAIN)
    assert payload["floor_schema_version"] == "2.0"
    meth = payload["methodology"]
    assert meth["floor_scored"] == "floor_all"
    assert meth["floor_context"] == "floor_heldout"
    assert "closest real city" in meth["floor_scored_rationale"]
    assert payload["cross_median_ks"] == pytest.approx(0.3)  # median(0.3, 0.5, 0.1, 0.3)
    # same record grammar across the two families
    assert set(payload["pairs"][0]) == set(payload["cross_pairs"][0])

    held_only = _build_payload(
        {key: vals for key, vals in features.items() if key[0] in set(_TWOFAM_HELD)},
        _TWOFAM_HELD,
        [],
    )
    assert held_only["cross_pairs"] == []
    assert held_only["cross_median_ks"] is None
    assert held_only["discriminating_strata"] == []


def test_cross_pair_table_round_trips_through_the_payload() -> None:
    features = _features_one_stratum(_TWOFAM_SHIFTS)
    payload = _build_payload(features, _TWOFAM_HELD, _TWOFAM_TRAIN)
    cross = compute_cross_pair_table(
        features, held_out=_TWOFAM_HELD, train=_TWOFAM_TRAIN, min_n=50, alpha=0.05
    )
    rebuilt = cross_pair_table_from_payload(payload)
    assert rebuilt.pairs == cross.pairs
    assert rebuilt.min_n == cross.min_n
    assert rebuilt.alpha == cross.alpha

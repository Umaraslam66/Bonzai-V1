"""Conditioning-floor machinery teeth (readiness-closure Task 25 step 1; spec §8).

The floor artifact is the measurement instrument for the re-scoped Phase-2 eval:
per held-out city D and (metric, stratum), ``floor_D = min over other real cities
T of KS(real_D, real_T)`` (STRICT min, PI knob 1; median-over-T as context only).
Lane S scores generated output as excess-over-floor; Lane M is the memorization
discriminator over REAL-data-selected discriminating strata.

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
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

import cfm.eval.conditioning_floor as cf
from cfm.eval.conditioning_floor import (
    FLOOR_ARTIFACT_LOCK_NAME,
    FloorArtifactError,
    FloorCollapseError,
    FloorExplosionError,
    VerifiedFloorArtifact,
    build_floor_artifact_payload,
    compute_floors,
    compute_pair_table,
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
    table = compute_pair_table(_features_one_stratum(_STRICT_SHIFTS), min_n=50, alpha=0.05)
    floors = compute_floors(table, ["d_city"])
    entry = floors["d_city"][(_M, _SA)]
    assert entry.floor == pytest.approx(0.2)  # min(0.2, 0.4) — PI knob 1, STRICT
    assert entry.floor_median_context == pytest.approx(0.3)  # median(0.2, 0.4)
    assert entry.n_other_cities == 2


def test_floors_for_multiple_held_out_cities_use_each_citys_own_pairs() -> None:
    table = compute_pair_table(_features_one_stratum(_STRICT_SHIFTS), min_n=50, alpha=0.05)
    floors = compute_floors(table, ["d_city", "t1_city"])
    # t1's pairs: KS(t1,d)=0.2, KS(t1,t2)=0.2 -> floor 0.2, context 0.2
    entry = floors["t1_city"][(_M, _SA)]
    assert entry.floor == pytest.approx(0.2)
    assert entry.floor_median_context == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# Tooth 4: collapse / explosion halts — each fires ONLY on its regime
# --------------------------------------------------------------------------- #


def _payload_kwargs() -> dict:
    return {
        "release": "2026-04-15.0",
        "held_out_cities": ["d_city"],
        "train_cities": ["t1_city", "t2_city"],
        "delta": 0.15,
    }


def test_collapse_fixture_raises_floor_collapse_error_only() -> None:
    table = compute_pair_table(_features_one_stratum(_COLLAPSE_SHIFTS), min_n=50, alpha=0.05)
    with pytest.raises(FloorCollapseError):
        build_floor_artifact_payload(table, **_payload_kwargs())


def test_explosion_fixture_raises_floor_explosion_error_only() -> None:
    table = compute_pair_table(_features_one_stratum(_EXPLOSION_SHIFTS), min_n=50, alpha=0.05)
    with pytest.raises(FloorExplosionError):
        build_floor_artifact_payload(table, **_payload_kwargs())


def test_halt_types_are_distinct_and_neither_catches_the_other() -> None:
    """THE GUARD MUST DISTINGUISH REGIMES: collapse never raises Explosion and
    vice versa — proven by asserting the OTHER type does not fire."""
    collapse_table = compute_pair_table(
        _features_one_stratum(_COLLAPSE_SHIFTS), min_n=50, alpha=0.05
    )
    explosion_table = compute_pair_table(
        _features_one_stratum(_EXPLOSION_SHIFTS), min_n=50, alpha=0.05
    )
    assert not issubclass(FloorCollapseError, FloorExplosionError)
    assert not issubclass(FloorExplosionError, FloorCollapseError)
    with pytest.raises(FloorCollapseError) as ci:
        build_floor_artifact_payload(collapse_table, **_payload_kwargs())
    assert not isinstance(ci.value, FloorExplosionError)
    with pytest.raises(FloorExplosionError) as ei:
        build_floor_artifact_payload(explosion_table, **_payload_kwargs())
    assert not isinstance(ei.value, FloorCollapseError)


def test_healthy_fixture_neither_halts_and_freezes_verified(tmp_path: Path) -> None:
    """The recon regime (median KS ~0.2-0.3): no halt, artifact freezes, verifies."""
    table = compute_pair_table(_features_one_stratum(_LANE_SHIFTS), min_n=50, alpha=0.05)
    payload = build_floor_artifact_payload(table, **_payload_kwargs())
    path = tmp_path / "conditioning-floor.yaml"
    freeze_floor_artifact(payload, path)
    verified = load_verified_floor(path)
    assert isinstance(verified, VerifiedFloorArtifact)
    assert verified.payload["held_out_cities"] == ["d_city"]
    assert (tmp_path / FLOOR_ARTIFACT_LOCK_NAME).exists()


def test_unsupported_zero_qualifying_pairs_is_loud() -> None:
    """UNSUPPORTED is loud, never a silent empty artifact (mirror of the Task-22
    UNSUPPORTED contract): zero qualifying pairs refuses to build a payload."""
    thin = {("d_city", _SA, _M): _grid(0, n=10)}
    table = compute_pair_table(thin, min_n=50, alpha=0.05)
    with pytest.raises(ValueError, match="UNSUPPORTED"):
        build_floor_artifact_payload(table, **_payload_kwargs())


# --------------------------------------------------------------------------- #
# Tooth 1: write-once + sha + refusal regimes (the 24a registry grammar)
# --------------------------------------------------------------------------- #


def _frozen_artifact(tmp_path: Path) -> Path:
    table = compute_pair_table(_features_one_stratum(_LANE_SHIFTS), min_n=50, alpha=0.05)
    payload = build_floor_artifact_payload(table, **_payload_kwargs())
    path = tmp_path / "conditioning-floor.yaml"
    freeze_floor_artifact(payload, path)
    return path


def test_freeze_is_write_once(tmp_path: Path) -> None:
    path = _frozen_artifact(tmp_path)
    table = compute_pair_table(_features_one_stratum(_LANE_SHIFTS), min_n=50, alpha=0.05)
    payload = build_floor_artifact_payload(table, **_payload_kwargs())
    with pytest.raises(FileExistsError):
        freeze_floor_artifact(payload, path)


def test_content_tamper_with_stale_sha_is_refused(tmp_path: Path) -> None:
    path = _frozen_artifact(tmp_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["floors"][0]["floor"] = 0.0001  # a friendlier floor; sha left stale
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
    data["floors"][0]["floor"] = 0.9999
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


def test_regurgitator_fails_lane_m_against_the_city_it_copied() -> None:
    """gen := real_T1 samples. KS(gen, D)=0.3 (large, in-test asserted) while
    KS(gen, T1)=0 -> gen matches T1 strictly better than D -> Lane M FAILS."""
    gen = _lane_features(30)  # T1's real samples, verbatim
    real_d = _lane_features(0)
    real_t1 = _lane_features(30)
    # fixture self-check: the regime is asserted before the verdict is read
    assert ks_distance(gen[(_M, _SA)], real_d[(_M, _SA)]) == pytest.approx(0.3)
    assert ks_distance(gen[(_M, _SA)], real_t1[(_M, _SA)]) == 0.0

    table = compute_pair_table(_features_one_stratum(_LANE_SHIFTS), min_n=50, alpha=0.05)
    strata = select_discriminating_strata(table, delta=0.15)[("d_city", "t1_city")]
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
    table = compute_pair_table(_features_one_stratum(_LANE_SHIFTS), min_n=50, alpha=0.05)
    strata_sel = select_discriminating_strata(table, delta=0.15)
    for t_city, t_shift in (("t1_city", 30), ("t2_city", 50)):
        real_t = _lane_features(t_shift)
        expected_ks = t_shift / 100.0
        assert ks_distance(gen[(_M, _SA)], real_t[(_M, _SA)]) == pytest.approx(expected_ks)
        strata = strata_sel[tuple(sorted(("d_city", t_city)))]
        result = lane_m_verdict(gen, real_d, real_t, strata, min_n=50)
        assert result.verdict == "PASS"
        assert result.median_ks_gen_d == 0.0
        assert result.margin == pytest.approx(expected_ks)


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
    table = compute_pair_table(features, min_n=50, alpha=0.05)
    payload = build_floor_artifact_payload(table, **_payload_kwargs())
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
# Tooth 5: no-leakage pin — strata selection reads ONLY the real-real pair table
# --------------------------------------------------------------------------- #


def test_no_leakage_select_discriminating_strata_signature_pin() -> None:
    """BY SIGNATURE: the only inputs are the real-real pair table and the delta
    knob — generated data has no parameter to arrive through."""
    sig = inspect.signature(select_discriminating_strata)
    assert list(sig.parameters) == ["pair_table", "delta"]
    assert sig.parameters["delta"].kind is inspect.Parameter.KEYWORD_ONLY


def test_no_leakage_strata_are_bit_equal_with_and_without_gen_data(tmp_path: Path) -> None:
    """Behavioral pin: the strata derived from a frozen artifact's pair table are
    bit-equal whether or not generated data exists anywhere in the process."""
    path = _frozen_artifact(tmp_path)
    verified = load_verified_floor(path)
    table = pair_table_from_payload(verified.payload)
    before = select_discriminating_strata(table, delta=0.15)
    _gen_data_exists_now = {(_M, _SA): _grid(7), (_M, _SB): _grid(13)}
    assert _gen_data_exists_now  # gen data is alive in the process during the re-derive
    after = select_discriminating_strata(table, delta=0.15)
    assert before == after
    # and the artifact's own frozen selection equals the recomputation
    frozen = {
        (rec["city_a"], rec["city_b"]): tuple(
            (s["metric"], tuple(s["stratum"])) for s in rec["strata"]
        )
        for rec in verified.payload["discriminating_strata"]
    }
    assert frozen == before


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
    n the same KS=0.3 is not significant (noise floor ~0.86 at n=10)."""
    features = {
        ("d_city", _SA, _M): _grid(0, n=10),
        ("t1_city", _SA, _M): _grid(3, n=10),  # KS = 0.3 but n=10
    }
    table = compute_pair_table(features, min_n=5, alpha=0.05)
    assert len(table.pairs) == 1
    assert table.pairs[0].ks == pytest.approx(0.3)
    assert table.pairs[0].p_bh >= 0.05  # not significant at this n
    assert select_discriminating_strata(table, delta=0.15)[("d_city", "t1_city")] == ()


# --------------------------------------------------------------------------- #
# Payload round-trip + ladder
# --------------------------------------------------------------------------- #


def test_pair_table_round_trips_through_the_payload(tmp_path: Path) -> None:
    table = compute_pair_table(_features_one_stratum(_LANE_SHIFTS), min_n=50, alpha=0.05)
    payload = build_floor_artifact_payload(table, **_payload_kwargs())
    rebuilt = pair_table_from_payload(payload)
    assert rebuilt.pairs == table.pairs
    assert rebuilt.min_n == table.min_n
    assert rebuilt.alpha == table.alpha


def test_delta_ladder_counts_bh_significant_pairs_per_anchor() -> None:
    table = compute_pair_table(_features_one_stratum(_LANE_SHIFTS), min_n=50, alpha=0.05)
    payload = build_floor_artifact_payload(table, **_payload_kwargs())
    ladder = {row["delta"]: row["n_pairs"] for row in payload["delta_ladder"]}
    assert ladder[0.15] == 3  # 0.2, 0.3, 0.5 all significant and >= 0.15
    assert ladder[0.25] == 2  # 0.3, 0.5
    assert ladder[0.35] == 1  # 0.5
    assert ladder[0.5] == 1


def test_payload_methodology_records_the_knobs_and_no_raw_samples() -> None:
    """The artifact stores KS TABLES, never raw samples (the orchestrator-fixed
    size decision): no list of 100 floats appears anywhere in the payload."""
    table = compute_pair_table(_features_one_stratum(_LANE_SHIFTS), min_n=50, alpha=0.05)
    payload = build_floor_artifact_payload(table, **_payload_kwargs())
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

"""Bake-off decision layer teeth (readiness-closure Task 26; spec §8 + A-3;
delta-spec §2 Rule 2 + §4 T5-as-discharged).

The decision quantity is per-city Lane-S EXCESS-OVER-FLOOR (median + p90 per
city; worst-case across held-out cities binds; binding city reported), and
``pick_winner`` carries the PI-mandated ordering: MEMORIZATION GATES CROWNING
REGARDLESS OF FIDELITY — a candidate with the best excess that fails Lane M is
refused by name, never crowned and never silently demoted.

Named teeth (dispatch step-1 items, (a)-(c) + (h)):
  (a) decide() asserts the decision path == the persisted basis (YAML in,
      DecisionBasis enum compared — Rule 2's basis discipline).
  (b) held-out-city completeness: set(real_by_city) == frozenset(manifest
      held_out_cities), STRICT manifest read (correction #12) — else loud.
  (c) pick_winner requires >= 2 backbones AND per-fit structural_check_ok
      (garbage non-monotone fit -> no crowning; single entry never auto-wins).
  (h) Lane-M must-fire pair AT THE pick_winner SEAM (regurgitator refused
      naming memorization even at best excess; oracle crowned), red-on-
      divergence via the inverted-pairing mutation; floor-sha refusal at
      decide() BEFORE any KS; no-leakage pin (strata reach Lane M only via the
      verified-artifact accessor, by signature + identity); excess-over-floor
      quantity pinned to hand-computed values; binding-city power gate
      demotion reported, not gated (delta-spec §4 rule 1).

Fixtures are the house uniform-grid grammar (KS exact by construction) and
were SCREENED for satisfiability (#15) against the existing floor machinery
before these teeth were written: floors d=0.2 / h=0.1 (h tightened by its
cross pair — floor_all, the scored bar, differs from floor_heldout there);
memorizer best-by-excess yet Lane-M FAIL on (d,t1)+(d,t2); honest/oracle PASS;
mutation-crowning gap 0.3 > power floor 0.1358; demotion fixture falls d->h.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

import cfm.eval.bakeoff_decision as bd
import cfm.eval.conditioning_floor as cf
from cfm.eval.bakeoff_decision import (
    BackboneEval,
    CandidateScore,
    MemorizationCheck,
    MemorizationRefusal,
    decide,
    decision_record,
    memorization_check,
    pick_winner,
    read_held_out_cities,
    read_persisted_basis,
)
from cfm.eval.conditioning_floor import (
    FloorArtifactError,
    build_floor_artifact_payload,
    freeze_floor_artifact,
    lane_s_excess,
    load_verified_floor,
)
from cfm.eval.feature_resolution import single_region_floor_gap
from cfm.eval.ladder import DecisionBasis
from cfm.eval.realism import ks_distance

# --------------------------------------------------------------------------- #
# Fixtures: uniform integer grids -> EXACT pairwise KS (screened, see module doc)
# --------------------------------------------------------------------------- #

_SA = ("R", "S1", 1, "inland")
_M = "building_area_m2"
_SCALE = 30_000_000
_HELD = ["d_city", "h_city"]
_TRAIN_CITIES = ["t1_city", "t2_city"]

#: city shifts: family-1 KS(d,h)=0.2; cross KS(d,t1)=0.3, KS(d,t2)=0.5,
#: KS(h,t1)=0.1 (below delta -> NO strata), KS(h,t2)=0.3 (all n=100).
_SHIFTS = {"d_city": 0, "h_city": 20, "t1_city": 30, "t2_city": 50}


def _grid(shift: int, n: int = 100) -> list[float]:
    return [float(i + shift) for i in range(n)]


def _feat(shift: int) -> dict[tuple[str, tuple], list[float]]:
    return {(_M, _SA): _grid(shift)}


_REAL = {"d_city": _feat(0), "h_city": _feat(20)}
_TRAIN = {"t1_city": _feat(30), "t2_city": _feat(50)}

#: gen := t1's REAL samples for d (the regurgitator) — BEST excess (0.1 at d).
_GEN_MEMORIZER = {"d_city": _feat(30), "h_city": _feat(20)}
#: plausible-but-mediocre: far from d AND from every training city (Lane-M PASS).
_GEN_HONEST = {"d_city": _feat(-60), "h_city": _feat(10)}
#: gen := D's own held-out samples (the oracle) — PASSES both lanes.
_GEN_ORACLE = {"d_city": _feat(0), "h_city": _feat(20)}

_N_REF = {"d_city": 100, "h_city": 100}


def _frozen_floor(tmp_path: Path) -> Path:
    features = {(c, _SA, _M): _grid(s) for c, s in _SHIFTS.items()}
    payload = build_floor_artifact_payload(
        features, release="test", held_out_cities=_HELD, train_cities=_TRAIN_CITIES
    )
    path = tmp_path / "conditioning-floor.yaml"
    freeze_floor_artifact(payload, path)
    return path


def _manifest(tmp_path: Path, cities: list[str] | None = None) -> Path:
    path = tmp_path / "holdout_manifest.yaml"
    path.write_text(yaml.safe_dump({"held_out_cities": cities or list(_HELD)}), encoding="utf-8")
    return path


def _basis(tmp_path: Path, value: str = "fixed_scale_plus_s13") -> Path:
    path = tmp_path / "decision_basis.yaml"
    path.write_text(yaml.safe_dump({"decision_basis": value}), encoding="utf-8")
    return path


def _candidate(name: str, gen: dict, verified, *, structural: bool = True) -> CandidateScore:
    """A CandidateScore via the REAL scoring path (lane_s_excess + memorization_check),
    so the pick_winner teeth exercise the full features -> lanes -> crowning seam."""
    lane_s = {c: lane_s_excess(gen[c], _REAL[c], verified, city=c) for c in _REAL}
    memo = memorization_check(gen, _REAL, _TRAIN, verified)
    return CandidateScore(
        backbone=name, lane_s_by_city=lane_s, structural_check_ok=structural, memorization=memo
    )


def test_fixture_self_check_ks_and_floors_are_the_screened_regime(tmp_path: Path) -> None:
    """Re-assert the screened regime BEFORE any tooth leans on it."""
    assert ks_distance(_grid(0), _grid(30)) == pytest.approx(0.3)
    assert ks_distance(_grid(0), _grid(-60)) == pytest.approx(0.6)
    assert ks_distance(_grid(20), _grid(10)) == pytest.approx(0.1)
    verified = load_verified_floor(_frozen_floor(tmp_path))
    floors = {r["city"]: r["floor_all"] for r in verified.payload["floors"]}
    assert floors["d_city"] == pytest.approx(0.2)
    assert floors["h_city"] == pytest.approx(0.1)  # tightened by the (h,t1) cross pair
    strata_n = {
        (r["city_a"], r["city_b"]): len(r["strata"])
        for r in verified.payload["discriminating_strata"]
    }
    assert strata_n == {
        ("d_city", "t1_city"): 1,
        ("d_city", "t2_city"): 1,
        ("h_city", "t1_city"): 0,  # KS 0.1 < delta: NO discriminator for this pair
        ("h_city", "t2_city"): 1,
    }
    assert single_region_floor_gap(n_reference_features=100) == pytest.approx(0.1358)


# --------------------------------------------------------------------------- #
# (h) Lane-M must-fire pair AT THE pick_winner SEAM
# --------------------------------------------------------------------------- #


def test_regurgitator_with_best_excess_is_refused_naming_memorization(tmp_path: Path) -> None:
    """THE TOOTH: gen := training city t1's real samples beats the floor (best
    excess: 0.1 at the binding city vs honest's 0.4) yet pick_winner REFUSES to
    crown it — by the NAMED MemorizationRefusal, never a generic check."""
    verified = load_verified_floor(_frozen_floor(tmp_path))
    memorizer = _candidate("memorizer", _GEN_MEMORIZER, verified)
    honest = _candidate("honest", _GEN_HONEST, verified)

    # regime check: the memorizer IS best by fidelity (its excess is lowest at
    # the binding city) — fidelity alone would crown it.
    assert memorizer.lane_s_by_city["d_city"].median_excess == pytest.approx(0.1)
    assert honest.lane_s_by_city["d_city"].median_excess == pytest.approx(0.4)
    assert memorizer.memorization.ok is False
    assert ("d_city", "t1_city") in memorizer.memorization.failing_pairs

    with pytest.raises(MemorizationRefusal, match="memoriz") as exc:
        pick_winner([memorizer, honest], n_reference_by_city=_N_REF)
    assert "memorizer" in str(exc.value)  # the refusal names the candidate


def test_inverted_pairing_mutation_would_crown_the_memorizer(tmp_path: Path) -> None:
    """RED-ON-DIVERGENCE: the SAME candidates with the memorization verdict
    forged to ok=True (simulating 'memorization not consulted') ARE crowned
    memorizer-first — proving the Lane-M pairing is the ONLY thing standing
    between the regurgitator and the crown (a gate must distinguish regimes)."""
    verified = load_verified_floor(_frozen_floor(tmp_path))
    memorizer = _candidate("memorizer", _GEN_MEMORIZER, verified)
    honest = _candidate("honest", _GEN_HONEST, verified)
    forged = CandidateScore(
        backbone=memorizer.backbone,
        lane_s_by_city=memorizer.lane_s_by_city,
        structural_check_ok=True,
        memorization=MemorizationCheck(ok=True, verdicts={}, failing_pairs=(), n_pairs_no_strata=0),
    )
    decision = pick_winner([forged, honest], n_reference_by_city=_N_REF)
    assert decision.winner == "memorizer"  # the un-gated outcome: fidelity crowns it
    assert decision.binding.binding_city == "d_city"


def test_oracle_is_crowned_normally(tmp_path: Path) -> None:
    """The must-fire pair's other half: gen := D's own held-out samples passes
    BOTH lanes and is crowned (excess 0 everywhere; Lane M PASS on every pair)."""
    verified = load_verified_floor(_frozen_floor(tmp_path))
    oracle = _candidate("oracle", _GEN_ORACLE, verified)
    honest = _candidate("honest", _GEN_HONEST, verified)
    assert oracle.memorization.ok is True
    decision = pick_winner([oracle, honest], n_reference_by_city=_N_REF)
    assert decision.winner == "oracle"
    assert decision.binding.binding_city == "d_city"  # worst-case city reported
    assert decision.binding.gap == pytest.approx(0.4)


def test_memorization_is_checked_before_structural(tmp_path: Path) -> None:
    """PI ordering pin: a candidate failing BOTH gates is refused by the
    memorization name, not the structural one — memorization gates crowning
    regardless of anything else about the fit."""
    verified = load_verified_floor(_frozen_floor(tmp_path))
    memorizer = _candidate("memorizer", _GEN_MEMORIZER, verified, structural=False)
    honest = _candidate("honest", _GEN_HONEST, verified)
    with pytest.raises(MemorizationRefusal, match="memoriz"):
        pick_winner([memorizer, honest], n_reference_by_city=_N_REF)


# --------------------------------------------------------------------------- #
# (c) pick_winner: >=2 backbones + per-fit structural_check_ok pairing
# --------------------------------------------------------------------------- #


def test_single_backbone_never_auto_wins(tmp_path: Path) -> None:
    verified = load_verified_floor(_frozen_floor(tmp_path))
    oracle = _candidate("oracle", _GEN_ORACLE, verified)
    with pytest.raises(ValueError, match="2 backbones"):
        pick_winner([oracle], n_reference_by_city=_N_REF)


def test_garbage_fit_blocks_crowning_naming_structural(tmp_path: Path) -> None:
    """The §2 pairing: a non-monotone/garbage fit (structural_check_ok=False)
    cannot crown a winner even when its excess is best."""
    verified = load_verified_floor(_frozen_floor(tmp_path))
    oracle = _candidate("oracle", _GEN_ORACLE, verified, structural=False)
    honest = _candidate("honest", _GEN_HONEST, verified)
    with pytest.raises(ValueError, match="structural_check_ok") as exc:
        pick_winner([oracle, honest], n_reference_by_city=_N_REF)
    assert "oracle" in str(exc.value)


# --------------------------------------------------------------------------- #
# binding-city power gate (delta-spec §4 rule 1): demotion reported, not gated
# --------------------------------------------------------------------------- #


def test_underpowered_binding_city_demotes_to_next_worst_resolved(tmp_path: Path) -> None:
    """d binds first (worst mean excess) but its winner-vs-runner-up gap 0.1 <
    its own floor 0.1358 -> demoted (reported); the decision falls to h."""
    verified = load_verified_floor(_frozen_floor(tmp_path))
    # screened: c1 d-excess 0.3 (KS 0.5 via grid(-50)), c2 d-excess 0.4 (grid(-60));
    # h: c1 excess 0 (grid 20), c2 excess 0.4 (KS 0.5 via grid(-30), floor 0.1).
    c1 = _candidate("c1", {"d_city": _feat(-50), "h_city": _feat(20)}, verified)
    c2 = _candidate("c2", {"d_city": _feat(-60), "h_city": _feat(-30)}, verified)
    assert c1.lane_s_by_city["d_city"].median_excess == pytest.approx(0.3)
    assert c2.lane_s_by_city["d_city"].median_excess == pytest.approx(0.4)
    decision = pick_winner([c1, c2], n_reference_by_city=_N_REF)
    assert decision.binding.binding_city == "h_city"
    assert decision.binding.demoted_from == ("d_city",)  # reported, never silent
    assert decision.winner == "c1"


# --------------------------------------------------------------------------- #
# memorization_check: all-training-cities completeness + no vacuous verdicts
# --------------------------------------------------------------------------- #


def test_memorization_check_requires_every_training_city(tmp_path: Path) -> None:
    """PI knob 2 is ALL training cities, not top-k: a missing training city's
    real samples make the all-38 sweep unrunnable -> loud, naming the city."""
    verified = load_verified_floor(_frozen_floor(tmp_path))
    with pytest.raises(ValueError, match="t2_city"):
        memorization_check(_GEN_ORACLE, _REAL, {"t1_city": _feat(30)}, verified)


def test_memorization_check_refuses_unverified_artifact(tmp_path: Path) -> None:
    verified = load_verified_floor(_frozen_floor(tmp_path))
    with pytest.raises(FloorArtifactError, match="VerifiedFloorArtifact"):
        memorization_check(_GEN_ORACLE, _REAL, _TRAIN, verified.payload)


def test_memorization_check_with_zero_discriminating_pairs_is_loud(tmp_path: Path) -> None:
    """A training city indistinct from every held-out city yields NO strata
    anywhere -> a vacuous all-PASS would guard nothing; refuse instead."""
    features = {(c, _SA, _M): _grid(s) for c, s in [("d_city", 0), ("h_city", 20), ("t", 10)]}
    payload = build_floor_artifact_payload(
        features, release="test", held_out_cities=_HELD, train_cities=["t"]
    )
    path = tmp_path / "floor-nostrata.yaml"
    freeze_floor_artifact(payload, path)
    verified = load_verified_floor(path)
    with pytest.raises(ValueError, match="vacuous"):
        memorization_check(_GEN_ORACLE, _REAL, {"t": _feat(10)}, verified)


# --------------------------------------------------------------------------- #
# (h) no-leakage pin: strata reach Lane M ONLY via the verified-artifact accessor
# --------------------------------------------------------------------------- #


def test_no_leakage_memorization_check_signature_and_accessor_identity() -> None:
    """BY SIGNATURE: memorization_check has no strata/pair-table parameter —
    the discriminating strata can only arrive through the verified artifact,
    via the ONE accessor (identity-pinned), so generated data has no path into
    the selection (the same pin grammar as the floor-instrument tests)."""
    sig = inspect.signature(memorization_check)
    assert list(sig.parameters) == [
        "gen_by_city",
        "real_by_city",
        "real_train_by_city",
        "artifact",
        "min_n",
    ]
    assert sig.parameters["min_n"].kind is inspect.Parameter.KEYWORD_ONLY
    assert not any("strata" in name or "pair" in name for name in sig.parameters)
    assert bd.discriminating_strata_from_artifact is cf.discriminating_strata_from_artifact


# --------------------------------------------------------------------------- #
# (a) decide(): decision path == persisted basis (YAML in, enum compared)
# --------------------------------------------------------------------------- #


def _evals() -> list[BackboneEval]:
    return [
        BackboneEval(
            "oracle", structural_check_ok=True, gen_by_city_by_scale={_SCALE: _GEN_ORACLE}
        ),
        BackboneEval(
            "honest", structural_check_ok=True, gen_by_city_by_scale={_SCALE: _GEN_HONEST}
        ),
    ]


def test_decide_matches_persisted_basis_and_crowns(tmp_path: Path) -> None:
    decision = decide(
        _evals(),
        _REAL,
        _TRAIN,
        artifact=_frozen_floor(tmp_path),
        holdout_manifest=_manifest(tmp_path),
        persisted_basis=_basis(tmp_path, "fixed_scale_plus_s13"),
    )
    assert decision.winner == "oracle"
    assert decision.basis is DecisionBasis.FIXED_SCALE_PLUS_S13  # enum, not string
    assert decision.top_scale_params == _SCALE
    assert decision.binding.binding_city == "d_city"  # binding city reported


def test_decide_refuses_a_basis_mismatch(tmp_path: Path) -> None:
    """1 feasible point is FIXED_SCALE_PLUS_S13; a persisted 'scaling_curve'
    basis means the run and its recorded decision rule diverged -> loud."""
    with pytest.raises(ValueError, match="scaling_curve") as exc:
        decide(
            _evals(),
            _REAL,
            _TRAIN,
            artifact=_frozen_floor(tmp_path),
            holdout_manifest=_manifest(tmp_path),
            persisted_basis=_basis(tmp_path, "scaling_curve"),
        )
    assert "fixed_scale_plus_s13" in str(exc.value)  # names BOTH sides


def test_persisted_basis_reader_is_strict() -> None:
    with pytest.raises(ValueError, match="decision_basis"):
        read_persisted_basis({})
    with pytest.raises(ValueError, match="not_a_basis"):
        read_persisted_basis({"decision_basis": "not_a_basis"})
    assert read_persisted_basis({"decision_basis": "scaling_curve"}) is (
        DecisionBasis.SCALING_CURVE
    )


def test_decide_requires_a_shared_scale_set(tmp_path: Path) -> None:
    """Backbones evaluated at DIFFERENT scale sets cannot share one basis count."""
    evals = _evals()
    evals[1] = BackboneEval(
        "honest",
        structural_check_ok=True,
        gen_by_city_by_scale={_SCALE: _GEN_HONEST, 100_000_000: _GEN_HONEST},
    )
    with pytest.raises(ValueError, match="scale"):
        decide(
            evals,
            _REAL,
            _TRAIN,
            artifact=_frozen_floor(tmp_path),
            holdout_manifest=_manifest(tmp_path),
            persisted_basis=_basis(tmp_path),
        )


def test_decide_with_zero_feasible_points_is_loud(tmp_path: Path) -> None:
    evals = [
        BackboneEval("a", structural_check_ok=True, gen_by_city_by_scale={}),
        BackboneEval("b", structural_check_ok=True, gen_by_city_by_scale={}),
    ]
    with pytest.raises(ValueError, match="feasible"):
        decide(
            evals,
            _REAL,
            _TRAIN,
            artifact=_frozen_floor(tmp_path),
            holdout_manifest=_manifest(tmp_path),
            persisted_basis=_basis(tmp_path, "escalate_more_data"),
        )


# --------------------------------------------------------------------------- #
# (b) 4-city completeness against the manifest (STRICT read — correction #12)
# --------------------------------------------------------------------------- #


def test_decide_refuses_missing_held_out_city(tmp_path: Path) -> None:
    real_missing = {"d_city": _feat(0)}  # h_city absent
    with pytest.raises(ValueError, match="h_city"):
        decide(
            _evals(),
            real_missing,
            _TRAIN,
            artifact=_frozen_floor(tmp_path),
            holdout_manifest=_manifest(tmp_path),
            persisted_basis=_basis(tmp_path),
        )


def test_decide_refuses_extra_city_not_in_manifest(tmp_path: Path) -> None:
    real_extra = {**_REAL, "x_city": _feat(5)}
    with pytest.raises(ValueError, match="x_city"):
        decide(
            _evals(),
            real_extra,
            _TRAIN,
            artifact=_frozen_floor(tmp_path),
            holdout_manifest=_manifest(tmp_path),
            persisted_basis=_basis(tmp_path),
        )


def test_decide_refuses_gen_city_set_mismatch(tmp_path: Path) -> None:
    evals = _evals()
    evals[0] = BackboneEval(
        "oracle",
        structural_check_ok=True,
        gen_by_city_by_scale={_SCALE: {"d_city": _feat(0)}},  # h_city missing from gen
    )
    with pytest.raises(ValueError, match="oracle"):
        decide(
            evals,
            _REAL,
            _TRAIN,
            artifact=_frozen_floor(tmp_path),
            holdout_manifest=_manifest(tmp_path),
            persisted_basis=_basis(tmp_path),
        )


def test_manifest_read_is_strict_no_get_fallback() -> None:
    """Correction #12: a manifest without the key must raise, never read as []."""
    with pytest.raises(ValueError, match="held_out_cities"):
        read_held_out_cities({"tiles": []})
    assert read_held_out_cities({"held_out_cities": ["a", "b"]}) == frozenset({"a", "b"})


# --------------------------------------------------------------------------- #
# (h) floor-sha refusal at the decision layer, BEFORE any scoring
# --------------------------------------------------------------------------- #


def test_decide_refuses_tampered_floor_before_any_ks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _frozen_floor(tmp_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["floors"][0]["floor_all"] = 0.9999  # friendlier floor; sha left stale
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    calls: list[object] = []

    def _recording_ks(a: list[float], b: list[float]) -> float:
        calls.append((len(a), len(b)))
        return 0.0

    monkeypatch.setattr(cf, "ks_distance", _recording_ks)
    with pytest.raises(FloorArtifactError, match="sha"):
        decide(
            _evals(),
            _REAL,
            _TRAIN,
            artifact=path,
            holdout_manifest=_manifest(tmp_path),
            persisted_basis=_basis(tmp_path),
        )
    assert calls == []  # refused BEFORE any KS was computed


def test_decide_refuses_absent_floor_artifact(tmp_path: Path) -> None:
    with pytest.raises(FloorArtifactError, match="does not exist"):
        decide(
            _evals(),
            _REAL,
            _TRAIN,
            artifact=tmp_path / "nowhere.yaml",
            holdout_manifest=_manifest(tmp_path),
            persisted_basis=_basis(tmp_path),
        )


# --------------------------------------------------------------------------- #
# (h) the decision quantity IS excess-over-floor (median + p90, hand-computed)
# --------------------------------------------------------------------------- #


def test_decision_quantity_is_excess_over_floor_all(tmp_path: Path) -> None:
    """honest at d: KS(grid(-60), grid(0)) = 0.6, floor_all(d) = 0.2 ->
    excess 0.4; at h: KS 0.1 - floor 0.1 -> 0.0. Oracle: clamped 0 everywhere
    (max(0, 0 - floor)). One stratum -> median == p90 == the value."""
    decision = decide(
        _evals(),
        _REAL,
        _TRAIN,
        artifact=_frozen_floor(tmp_path),
        holdout_manifest=_manifest(tmp_path),
        persisted_basis=_basis(tmp_path),
    )
    by_name = {c.backbone: c for c in decision.candidates}
    honest_d = by_name["honest"].lane_s_by_city["d_city"]
    assert honest_d.median_excess == pytest.approx(0.4)
    assert honest_d.p90_excess == pytest.approx(0.4)
    assert by_name["honest"].lane_s_by_city["h_city"].median_excess == pytest.approx(0.0)
    assert by_name["oracle"].lane_s_by_city["d_city"].median_excess == pytest.approx(0.0)
    assert decision.quantity == "lane_s_median_excess_over_floor_all"
    # the decision carries the floor it was judged against (lineage, not vibes)
    verified = load_verified_floor(decision.floor_artifact_path)
    assert decision.floor_sha256 == verified.payload["floor_sha256"]


# --------------------------------------------------------------------------- #
# decision_record: the persisted decision is plain-typed and complete
# --------------------------------------------------------------------------- #


def test_decision_record_is_yaml_safe_and_names_the_teeth(tmp_path: Path) -> None:
    decision = decide(
        _evals(),
        _REAL,
        _TRAIN,
        artifact=_frozen_floor(tmp_path),
        holdout_manifest=_manifest(tmp_path),
        persisted_basis=_basis(tmp_path),
    )
    record = decision_record(decision)
    dumped = yaml.safe_dump(record)  # plain types only — round-trips through YAML
    loaded = yaml.safe_load(dumped)
    assert loaded["winner"] == "oracle"
    assert loaded["decision_basis"] == "fixed_scale_plus_s13"
    assert loaded["binding_city"] == "d_city"
    assert loaded["quantity"] == "lane_s_median_excess_over_floor_all"
    assert loaded["floor_sha256"] == decision.floor_sha256
    oracle = next(c for c in loaded["candidates"] if c["backbone"] == "oracle")
    assert oracle["memorization_check_ok"] is True
    assert oracle["structural_check_ok"] is True
    assert oracle["lane_s"]["d_city"]["median_excess"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# scripts/run_bakeoff_decision.py: thin CLI over decide() (end-to-end smoke)
# --------------------------------------------------------------------------- #


def _feature_records(features: dict[tuple[str, tuple], list[float]]) -> list[dict]:
    return [{"metric": m, "stratum": list(s), "samples": vals} for (m, s), vals in features.items()]


def _script_inputs(tmp_path: Path) -> dict[str, Path]:
    eval_results = tmp_path / "eval_results.yaml"
    eval_results.write_text(
        yaml.safe_dump(
            {
                "backbones": [
                    {
                        "backbone": name,
                        "structural_check_ok": True,
                        "scales": [
                            {
                                "scale_params": _SCALE,
                                "gen_by_city": {c: _feature_records(f) for c, f in gen.items()},
                            }
                        ],
                    }
                    for name, gen in [("oracle", _GEN_ORACLE), ("honest", _GEN_HONEST)]
                ]
            }
        ),
        encoding="utf-8",
    )
    real_features = tmp_path / "real_features.yaml"
    real_features.write_text(
        yaml.safe_dump(
            {
                "real_by_city": {c: _feature_records(f) for c, f in _REAL.items()},
                "real_train_by_city": {c: _feature_records(f) for c, f in _TRAIN.items()},
            }
        ),
        encoding="utf-8",
    )
    return {
        "eval_results": eval_results,
        "real_features": real_features,
        "floor": _frozen_floor(tmp_path),
        "manifest": _manifest(tmp_path),
        "basis": _basis(tmp_path),
        "out": tmp_path / "decision.yaml",
    }


def test_script_end_to_end_writes_the_decision_record(tmp_path: Path) -> None:
    import scripts.run_bakeoff_decision as rbd

    p = _script_inputs(tmp_path)
    rbd.main(
        [
            "--eval-results",
            str(p["eval_results"]),
            "--real-features",
            str(p["real_features"]),
            "--floor-artifact",
            str(p["floor"]),
            "--holdout-manifest",
            str(p["manifest"]),
            "--persisted-basis",
            str(p["basis"]),
            "--out",
            str(p["out"]),
        ]
    )
    record = yaml.safe_load(p["out"].read_text(encoding="utf-8"))
    assert record["winner"] == "oracle"
    assert record["binding_city"] == "d_city"
    assert record["quantity"] == "lane_s_median_excess_over_floor_all"


def test_script_inherits_the_floor_refusal(tmp_path: Path) -> None:
    """The CLI hands decide() the PATH, not a pre-parsed payload — a tampered
    artifact refuses at the script layer too (no decision file written)."""
    import scripts.run_bakeoff_decision as rbd

    p = _script_inputs(tmp_path)
    data = yaml.safe_load(p["floor"].read_text(encoding="utf-8"))
    data["floors"][0]["floor_all"] = 0.9999
    p["floor"].write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(FloorArtifactError, match="sha"):
        rbd.main(
            [
                "--eval-results",
                str(p["eval_results"]),
                "--real-features",
                str(p["real_features"]),
                "--floor-artifact",
                str(p["floor"]),
                "--holdout-manifest",
                str(p["manifest"]),
                "--persisted-basis",
                str(p["basis"]),
                "--out",
                str(p["out"]),
            ]
        )
    assert not p["out"].exists()

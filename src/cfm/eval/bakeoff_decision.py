"""Bake-off decision layer (readiness-closure Task 26; spec §8 + A-3;
delta-spec §2 Rule 2 + §4 T5-as-discharged).

What decides, in one breath: per (metric, stratum) and held-out city D the
quantity is the Lane-S EXCESS-OVER-FLOOR ``max(0, KS(gen, real_D) -
floor_all_D)`` against the sha-verified conditioning-floor artifact;
per-city aggregate is the median (+ p90 reported) over strata; cross-city
aggregation is WORST-CASE with the #21 binding-city power gate
(``city_aggregate.binding_city_verdict`` — one source, not a re-derivation);
and CROWNING is gated by the Lane-M memorization discriminator over ALL
training cities (PI knob 2) BEFORE any fidelity comparison — a regurgitator
passes realism by construction and must be refused BY NAME, never crowned and
never silently demoted.

Rule-2 basis discipline (delta-spec §2): the decision path actually taken
(a step function of the number of feasible scale points) is compared — as a
``DecisionBasis`` ENUM, with the persisted record parsed from YAML — against
the basis the diagnostic persisted. A mismatch means the run and its recorded
decision rule diverged; loud, never reconciled silently.

Refusal teeth (each is a test):
  * floor artifact verifies (sha + lock + schema) BEFORE any KS is computed
    (mirrors ``lane_s_excess``'s Task-20 reader-side discipline);
  * STRICT ``held_out_cities`` manifest read (correction #12: never
    ``.get(..., [])``) + set-equality completeness on the real AND generated
    city sets;
  * ``pick_winner`` requires >= 2 backbones (kills the live single-entry
    auto-win) AND per-fit ``structural_check_ok`` AND ``memorization_check_ok``
    — with memorization checked FIRST (the PI-mandated ordering);
  * the all-training-cities Lane-M sweep refuses a missing training city and
    refuses a vacuous (zero-discriminating-pairs) verdict;
  * NO-LEAKAGE: discriminating strata reach Lane M ONLY via
    ``discriminating_strata_from_artifact`` on the proof-carrying
    ``VerifiedFloorArtifact`` — ``memorization_check`` has no strata/pair-table
    parameter by signature, so generated data has no path into the selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from cfm.eval.city_aggregate import BindingVerdict, PerCityKS, binding_city_verdict
from cfm.eval.conditioning_floor import (
    FloorArtifactError,
    LaneMResult,
    LaneSResult,
    VerifiedFloorArtifact,
    discriminating_strata_from_artifact,
    lane_m_verdict,
    lane_s_excess,
    load_verified_floor,
)
from cfm.eval.ladder import DecisionBasis, decision_basis

if TYPE_CHECKING:
    from collections.abc import Sequence

#: (metric, stratum) -> feature samples, the one feature-dict grammar shared
#: with the conditioning-floor lanes. A stratum is the mixed string/int tuple
#: the floor artifact freezes (e.g. ("R", "S1", 1, "inland")).
GenFeatures = dict[tuple[str, tuple[str | int, ...]], list[float]]

#: What the per-city scalar IS (stamped into every decision record so a reader
#: can never mistake it for a bare KS). DECISION: the BINDING scalar is the
#: per-city MEDIAN excess (PI knob 3 lists median first; p90 rides along in
#: every record as the tail diagnostic). Revisit if a real run shows a
#: candidate winning on median while its p90 explodes.
DECISION_QUANTITY = "lane_s_median_excess_over_floor_all"


class MemorizationRefusal(RuntimeError):
    """The would-be decision involves a candidate that FAILED the Lane-M
    memorization discriminator — no crowning (hard halt), by name."""


# --------------------------------------------------------------------------- #
# STRICT readers (YAML in, typed values out; never .get-with-default)
# --------------------------------------------------------------------------- #


def _load_mapping(source: dict | str | Path, *, what: str) -> dict:
    if isinstance(source, dict):
        return source
    data = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{what} {source}: expected a YAML mapping, got {type(data).__name__}")
    return data


def read_held_out_cities(manifest: dict | str | Path) -> frozenset[str]:
    """STRICT ``held_out_cities`` read (correction #12): a manifest missing the
    key raises — never a silent ``.get(..., [])`` that evaluates zero cities."""
    data = _load_mapping(manifest, what="holdout manifest")
    if "held_out_cities" not in data:
        raise ValueError(
            "holdout manifest has no 'held_out_cities' key; refusing to derive "
            "the held-out city set from a default (a .get fallback would make "
            "the completeness check vacuously pass on zero cities)."
        )
    value = data["held_out_cities"]
    if not isinstance(value, list):
        raise ValueError(
            f"holdout manifest 'held_out_cities' must be a list, got "
            f"{type(value).__name__} ({value!r}) — a YAML scalar would frozenset "
            "into its CHARACTERS, a silently wrong city set."
        )
    return frozenset(value)


def read_persisted_basis(basis: dict | str | Path) -> DecisionBasis:
    """The PERSISTED decision basis (written by the Task-1 diagnostic), parsed
    from YAML into the ``DecisionBasis`` enum — the comparison in ``decide`` is
    enum-to-enum, never string-to-string."""
    data = _load_mapping(basis, what="persisted basis")
    if "decision_basis" not in data:
        raise ValueError(
            "persisted basis record has no 'decision_basis' key; refusing — the "
            "Rule-2 path==basis assertion needs the recorded basis, not a guess."
        )
    raw = data["decision_basis"]
    try:
        return DecisionBasis(raw)
    except ValueError as exc:
        valid = [b.value for b in DecisionBasis]
        raise ValueError(
            f"persisted decision_basis {raw!r} is not a DecisionBasis value "
            f"(expected one of {valid})."
        ) from exc


# --------------------------------------------------------------------------- #
# Lane-M all-training-cities sweep (memorization_check_ok, PI knob 2: ALL of them)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MemorizationCheck:
    """One candidate's all-training-cities Lane-M sweep. ``ok`` iff EVERY
    (held-out D, training T) pair with discriminating strata returned PASS."""

    ok: bool
    verdicts: dict[tuple[str, str], LaneMResult]  # keyed (D, T)
    failing_pairs: tuple[tuple[str, str], ...]
    n_pairs_no_strata: int  # pairs with no discriminator (D, T not measured-distinct)


def memorization_check(
    gen_by_city: dict[str, GenFeatures],
    real_by_city: dict[str, GenFeatures],
    real_train_by_city: dict[str, GenFeatures],
    artifact: VerifiedFloorArtifact,
    *,
    min_n: int | None = None,
) -> MemorizationCheck:
    """Lane-M sweep for ONE candidate over ALL the artifact's training cities
    (PI knob 2: all of them, never top-k): per (D, T) with discriminating
    strata, PASS iff ``median KS(gen, real_D) < median KS(gen, real_T)``.

    NO-LEAKAGE BY SIGNATURE: there is no strata/pair-table parameter — the
    discriminating strata derive EXCLUSIVELY from the verified artifact via
    ``discriminating_strata_from_artifact`` (real-data-selected, frozen,
    sha-verified), so generated data has no path into the selection.

    Refusals:
      * a non-``VerifiedFloorArtifact`` (anything that skipped
        ``load_verified_floor``) — before any work;
      * ``real_train_by_city`` not covering EXACTLY the artifact's
        ``train_cities`` (a missing city makes the all-cities sweep a top-k
        sweep silently; an extra city has no frozen strata to score);
      * zero (D, T) pairs with discriminating strata anywhere — an all-PASS
        over nothing is vacuous and guards nothing.

    PRECONDITION (enforced by ``decide`` step 2 on the orchestrated path):
    ``gen_by_city`` and ``real_by_city`` cover the artifact's
    ``held_out_cities`` — they are bare-indexed per held-out D below, so a
    direct caller handing a shrunk dict gets a KeyError, not a named refusal.
    """
    if not isinstance(artifact, VerifiedFloorArtifact):
        raise FloorArtifactError(
            "memorization_check requires a VerifiedFloorArtifact (the "
            "load_verified_floor result); refusing an unverified "
            f"{type(artifact).__name__}."
        )
    train_cities = list(artifact.payload["train_cities"])
    held_out = list(artifact.payload["held_out_cities"])
    missing = sorted(set(train_cities) - set(real_train_by_city))
    extra = sorted(set(real_train_by_city) - set(train_cities))
    if missing or extra:
        raise ValueError(
            "memorization_check: real_train_by_city must cover EXACTLY the "
            f"artifact's train_cities (missing: {missing}, extra: {extra}) — "
            "the Lane-M sweep is ALL training cities, never a silent subset."
        )

    verdicts: dict[tuple[str, str], LaneMResult] = {}
    failing: list[tuple[str, str]] = []
    n_no_strata = 0
    for d_city in sorted(held_out):
        for t_city in sorted(train_cities):
            strata = discriminating_strata_from_artifact(artifact, d_city, t_city)
            if not strata:
                n_no_strata += 1  # D and T not measured-distinct: no discriminator
                continue
            verdict = lane_m_verdict(
                gen_by_city[d_city],
                real_by_city[d_city],
                real_train_by_city[t_city],
                strata,
                min_n=min_n,
                artifact=artifact,
            )
            verdicts[(d_city, t_city)] = verdict
            if verdict.verdict != "PASS":
                failing.append((d_city, t_city))
    if not verdicts:
        raise ValueError(
            "memorization_check: zero (D, T) pairs carry discriminating strata "
            f"({n_no_strata} pairs skipped) — a vacuous all-PASS memorization "
            "verdict guards nothing; refusing."
        )
    return MemorizationCheck(
        ok=not failing,
        verdicts=verdicts,
        failing_pairs=tuple(failing),
        n_pairs_no_strata=n_no_strata,
    )


# --------------------------------------------------------------------------- #
# pick_winner: the crowning seam (memorization FIRST, then structural, then excess)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CandidateScore:
    """One backbone at the decision-bearing scale: Lane-S per held-out city +
    the two paired Gate facts (fit structure, Lane-M memorization)."""

    backbone: str
    lane_s_by_city: dict[str, LaneSResult]
    structural_check_ok: bool
    memorization: MemorizationCheck


@dataclass(frozen=True)
class WinnerDecision:
    winner: str
    binding: BindingVerdict
    quantity: str = DECISION_QUANTITY


def pick_winner(
    candidates: Sequence[CandidateScore],
    *,
    n_reference_by_city: dict[str, int],
) -> WinnerDecision:
    """Crown the backbone with the best (lowest) per-city median excess at the
    binding (worst) held-out city, under the #21 power gate.

    ORDER OF GATES (PI-mandated): memorization FIRST — a Lane-M failure refuses
    CROWNING regardless of fidelity, via the NAMED ``MemorizationRefusal``
    (a regurgitator passes realism by construction; its excess is meaningless).
    Then per-fit ``structural_check_ok`` (a garbage non-monotone fit cannot
    crown a winner). Then the worst-case excess comparison with the binding-
    city power gate (``binding_city_verdict``: an under-powered binding city is
    DEMOTED and reported in ``binding.demoted_from``, never silently gated —
    delta-spec §4 rule 1).

    DECISION: BOTH gates are required of EVERY candidate, not just the
    would-be winner — a memorizer or a garbage fit anywhere in the pool means
    the comparison itself is compromised (training leakage / broken fit), an
    integrity event the PI reviews, not a candidate to route around.

    Requires >= 2 candidates: a single entry never auto-wins (the live
    single-entry auto-win in ``curve.pick_winner`` is exactly what this layer
    retires for crowning purposes).
    """
    if len(candidates) < 2:
        raise ValueError(
            f"pick_winner: got {len(candidates)} candidate(s); a bake-off decision "
            "needs >= 2 backbones — a single entry never auto-wins."
        )
    names = [c.backbone for c in candidates]
    if len(set(names)) != len(names):
        raise ValueError(f"pick_winner: duplicate backbone names: {names}")

    memorizers = [c.backbone for c in candidates if not c.memorization.ok]
    if memorizers:
        details = "; ".join(
            f"{c.backbone}: failing (D, T) pairs {list(c.memorization.failing_pairs)}"
            for c in candidates
            if not c.memorization.ok
        )
        raise MemorizationRefusal(
            f"memorization gate: candidate(s) {memorizers} FAILED the Lane-M "
            f"discriminator ({details}) — a memorizer beats the floor by "
            "construction, so fidelity cannot crown it; no winner is declared "
            "(hard halt, PI reviews)."
        )

    bad_fits = [c.backbone for c in candidates if not c.structural_check_ok]
    if bad_fits:
        raise ValueError(
            f"pick_winner: candidate(s) {bad_fits} fail structural_check_ok "
            "(garbage / non-monotone fit) — a broken fit cannot crown a winner; "
            "route to the §13 branch upstream, never crown over it."
        )

    per_backbone = {
        c.backbone: [
            PerCityKS(
                city=city,
                ks=c.lane_s_by_city[city].median_excess,  # the DECISION_QUANTITY scalar
                n_features=n_reference_by_city[city],
            )
            for city in sorted(c.lane_s_by_city)
        ]
        for c in candidates
    }
    binding = binding_city_verdict(per_backbone)
    return WinnerDecision(winner=binding.winner, binding=binding)


# --------------------------------------------------------------------------- #
# decide(): the per-(backbone, scale) orchestration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BackboneEval:
    """Per-backbone eval results: generated features per held-out city, per
    feasible scale (params). ``structural_check_ok`` is the §2 paired fit fact
    (``curve.structural_check_ok`` over this backbone's scale points)."""

    backbone: str
    structural_check_ok: bool
    gen_by_city_by_scale: dict[int, dict[str, GenFeatures]]


@dataclass(frozen=True)
class BakeoffDecision:
    winner: str
    basis: DecisionBasis
    top_scale_params: int
    binding: BindingVerdict
    candidates: tuple[CandidateScore, ...]
    n_reference_by_city: dict[str, int]
    floor_artifact_path: str
    floor_sha256: str
    quantity: str = DECISION_QUANTITY


def decide(
    evals: Sequence[BackboneEval],
    real_by_city: dict[str, GenFeatures],
    real_train_by_city: dict[str, GenFeatures],
    *,
    artifact: str | Path | VerifiedFloorArtifact,
    holdout_manifest: dict | str | Path,
    persisted_basis: dict | str | Path,
    min_n: int | None = None,
) -> BakeoffDecision:
    """The Task-12-runner decision: verified floor in, crowned winner (or a
    named refusal) out.

    Sequencing (each step is a tooth):
      1. FLOOR-SHA REFUSAL FIRST: the artifact path is verified via
         ``load_verified_floor`` before ANY scoring — tampered/missing/unsealed
         artifact => no decision (mirrors lane_s_excess's reader discipline).
      2. STRICT held-out completeness: ``set(real_by_city)`` must equal the
         manifest's ``held_out_cities`` (correction #12), and every backbone's
         generated city set at every scale must too — loud, naming the
         offender, never a silently-shrunk worst-case domain. The ARTIFACT's
         frozen ``held_out_cities`` must equal the manifest set too (Task-26
         spec review #4): floors frozen for a different city set are lineage
         skew — a NAMED refusal here, never a downstream silent pass (an
         artifact-extra city's floors are simply never consumed) or KeyError.
      3. Rule-2 basis assertion: the basis implied by the number of feasible
         scale points actually evaluated is compared (as enums) against the
         persisted basis record. All backbones must share ONE scale set.
      4. Lane-S excess + the Lane-M sweep are scored at the TOP feasible scale
         (the decision-bearing point under FIXED_SCALE_PLUS_S13; under
         SCALING_CURVE the curve/CI live upstream in ``curve.py`` and arrive
         here as ``structural_check_ok``).
      5. ``pick_winner`` (memorization -> structural -> power-gated worst-case).

    DECISION: ``n_reference_by_city`` for the #21 power gate is the TOTAL real
    feature count over the city's FLOORED strata ONLY (the artifact's floor
    rows for that city) — the city's reference population, backbone-independent.
    Unfloored strata are excluded (Task-26 spec review #3): counting them
    would inflate n and SHRINK the power floor — a silently more permissive
    gate. Revisit if per-stratum thinness ever diverges materially between
    candidates.
    """
    # 1. verified floor BEFORE any scoring
    if isinstance(artifact, (str, Path)):
        verified = load_verified_floor(Path(artifact))
    elif isinstance(artifact, VerifiedFloorArtifact):
        verified = artifact
    else:
        raise FloorArtifactError(
            "decide requires the floor-artifact PATH or a VerifiedFloorArtifact "
            f"(the load_verified_floor result); refusing an unverified "
            f"{type(artifact).__name__}."
        )

    # 2. STRICT completeness against the manifest
    held = read_held_out_cities(holdout_manifest)
    artifact_held = set(verified.payload["held_out_cities"])
    if artifact_held != held:
        a_missing = sorted(held - artifact_held)
        a_extra = sorted(artifact_held - held)
        raise ValueError(
            "decide: floor artifact held_out_cities do not match the manifest's "
            f"(artifact-extra: {a_extra}, artifact-missing: {a_missing}) — the "
            "floors were frozen for a DIFFERENT held-out city set; scoring "
            "against them is silent lineage skew, refusing."
        )
    if set(real_by_city) != held:
        missing = sorted(held - set(real_by_city))
        extra = sorted(set(real_by_city) - held)
        raise ValueError(
            "decide: real_by_city does not match the manifest's held_out_cities "
            f"(missing: {missing}, extra: {extra}) — a shrunk or padded city set "
            "silently changes the worst-case max domain; refusing."
        )
    if not evals:
        raise ValueError("decide: no backbone evals supplied")
    for ev in evals:
        for scale, gen_by_city in ev.gen_by_city_by_scale.items():
            if set(gen_by_city) != held:
                raise ValueError(
                    f"decide: backbone {ev.backbone!r} at scale {scale} covers "
                    f"cities {sorted(gen_by_city)}, expected {sorted(held)} — "
                    "every backbone must generate for every held-out city."
                )

    # 3. one shared scale set; basis(path) == basis(persisted), enum-compared
    scale_sets = {ev.backbone: tuple(sorted(ev.gen_by_city_by_scale)) for ev in evals}
    first = next(iter(scale_sets.values()))
    mismatched = {b: s for b, s in scale_sets.items() if s != first}
    if mismatched:
        raise ValueError(
            f"decide: backbones evaluated at DIFFERENT scale sets ({scale_sets}) "
            "— one basis count requires one shared feasible ladder; refusing."
        )
    if not first:
        raise ValueError(
            "decide: zero feasible scale points — the empty-ladder case is "
            "ESCALATE_MORE_DATA (delta-spec §2 Rule 1); there is no decision to "
            "make here, only more data to get."
        )
    actual = decision_basis(len(first))
    persisted = read_persisted_basis(persisted_basis)
    if persisted is not actual:
        raise ValueError(
            f"decide: decision path is {actual.value!r} ({len(first)} feasible "
            f"point(s)) but the persisted basis record says {persisted.value!r} "
            "— the run and its recorded decision rule diverged (Rule-2 basis "
            "discipline); refusing."
        )

    # 4. score the decision-bearing point (top feasible scale)
    top_scale = max(first)
    candidates = tuple(
        CandidateScore(
            backbone=ev.backbone,
            lane_s_by_city={
                city: lane_s_excess(
                    ev.gen_by_city_by_scale[top_scale][city],
                    real_by_city[city],
                    verified,
                    city=city,
                    min_n=min_n,
                )
                for city in sorted(held)
            },
            structural_check_ok=ev.structural_check_ok,
            memorization=memorization_check(
                ev.gen_by_city_by_scale[top_scale],
                real_by_city,
                real_train_by_city,
                verified,
                min_n=min_n,
            ),
        )
        for ev in evals
    )
    floored_strata_by_city: dict[str, set[tuple[str, tuple]]] = {}
    for rec in verified.payload["floors"]:
        floored_strata_by_city.setdefault(rec["city"], set()).add(
            (rec["metric"], tuple(rec["stratum"]))
        )
    # PRECONDITION for the bare floored_strata_by_city[city] index: step 4's
    # lane_s_excess already REFUSED any held-out city the artifact never floored
    # ("holds no floors for city"), so every city in `held` has floor rows here.
    n_reference_by_city = {
        city: sum(
            len(samples)
            for key, samples in real_by_city[city].items()
            if key in floored_strata_by_city[city]
        )
        for city in sorted(held)
    }

    # 5. crown (or refuse by name)
    winner = pick_winner(candidates, n_reference_by_city=n_reference_by_city)
    return BakeoffDecision(
        winner=winner.winner,
        basis=actual,
        top_scale_params=top_scale,
        binding=winner.binding,
        candidates=candidates,
        n_reference_by_city=n_reference_by_city,
        floor_artifact_path=str(verified.path),
        floor_sha256=str(verified.payload["floor_sha256"]),
    )


# --------------------------------------------------------------------------- #
# decision_record: the YAML-safe persisted decision (plain types only)
# --------------------------------------------------------------------------- #


def _lane_s_record(result: LaneSResult) -> dict:
    return {
        "median_excess": float(result.median_excess),
        "p90_excess": float(result.p90_excess),
        "n_qualifying": result.n_qualifying,
        "n_skipped_thin": result.n_skipped_thin,
    }


def decision_record(decision: BakeoffDecision) -> dict:
    """The decision as a YAML-safe mapping — what the runner persists beside
    the run report (every field plain-typed; enums by value)."""
    return {
        "winner": decision.winner,
        "decision_basis": decision.basis.value,
        "quantity": decision.quantity,
        "top_scale_params": decision.top_scale_params,
        "binding_city": decision.binding.binding_city,
        "binding_runner_up": decision.binding.runner_up,
        "binding_gap": float(decision.binding.gap),
        "binding_city_floor": float(decision.binding.city_floor),
        "demoted_cities": list(decision.binding.demoted_from),
        "n_reference_by_city": dict(decision.n_reference_by_city),
        "floor_artifact_path": decision.floor_artifact_path,
        "floor_sha256": decision.floor_sha256,
        "candidates": [
            {
                "backbone": c.backbone,
                "structural_check_ok": c.structural_check_ok,
                "memorization_check_ok": c.memorization.ok,
                "memorization_failing_pairs": [list(p) for p in c.memorization.failing_pairs],
                "memorization_n_pairs_no_strata": c.memorization.n_pairs_no_strata,
                "memorization_margins": {
                    f"{d}|{t}": float(v.margin)
                    for (d, t), v in sorted(c.memorization.verdicts.items())
                },
                "lane_s": {city: _lane_s_record(r) for city, r in sorted(c.lane_s_by_city.items())},
            }
            for c in decision.candidates
        ],
    }

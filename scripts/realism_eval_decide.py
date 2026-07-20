"""Scored realism-eval decision runner (Task 6) — LOCAL, GPU-free.

The scientific verdict of the PI-approved scored realism eval. It reads the 6 sealed
gen artifacts (2 backbones x 3 seeds), the checkpoint-independent ``real-features.yaml``,
the sealed Lane-S manifest and the frozen conditioning-floor artifact PATH, then computes
the crowned winner (or the named ``NO_DECISIVE_WINNER``) under the locked two-floor rule
with its seed-noise input populated.

ORCHESTRATOR DECISION (canon-verified 2026-07-20): the crown path does NOT call
``bakeoff_decision.decide()`` — decide() builds ``PerCityKS`` with ``seed_sem=0`` (its
scalar is a single fixed-scale excess) and would silently DROP the locked seed-noise floor
(GROUND_TRUTH §4). Instead this runner orchestrates the locked primitives directly:
per (backbone, city) mean of the 3 seeds' ``median_excess`` = ``ks``, per-seed std-error =
``seed_sem``, then ``city_aggregate.binding_city_verdict`` is the ONLY crown path
(``scoring.aggregate_seed_verdict``). decide()'s guard teeth are replicated as EXPLICIT
checks here: floor-sha verification (``load_verified_floor`` on the PATH), STRICT held-out
city-set equality across manifest / floor artifact / real features / every checkpoint's
gen, and MEMORIZATION FIRST.

Hard sequencing (never reordered):
  1. Verified floor (sha/lock) BEFORE any scoring.
  2. ``assert_city_sets`` — fail loud on any held-out mismatch.
  3. MEMORIZATION per (backbone, seed) FIRST — ANY ``ok=False`` writes ``memorization.yaml``
     and raises ``MemorizationHalt``; NO coverage/excess/verdict runs past it.
  4. ``verify_gen_coverage`` per (backbone, seed) — ceiling-bound shorts are recorded;
     any other short raises ``SamplerCoverageError`` (fail loud, never caught).
  5. ``lane_s_excess`` per (backbone, seed, city) with the floor PATH (sha self-verify).
  6. ``aggregate_seed_verdict`` -> ``BindingVerdict`` | ``NoDecisiveWinner`` (verbatim).
  7. write-once ``decision.yaml`` + prose ``summary.md`` under ``--out-dir``.

``NO_DECISIVE_WINNER`` is a valid, publishable verdict — written verbatim with per-city
(gap, resolution_floor, seed_noise_floor); never softened, never retried.

TORCH DISCIPLINE: every arg-parse / serialization / aggregation surface is torch-free and
unit-tested. The heavy ``read_gen_artifact -> gen_features_by_city`` decode (which reads
tile labels off disk) is the ops path inside ``main``; the tested core
(``score_and_decide``) takes already-extracted ``GenFeatures`` dicts, so importing this
module never pulls torch.

Run (ops; local / a CPU node — no GPU):
    python -m scripts.realism_eval_decide \\
        --gen-artifact <bb1-seed0.json> ... (x6) \\
        --real-features reports/realism_eval/real-features.yaml \\
        --floor-artifact <conditioning-floor.yaml> \\
        --manifest <sealed_lane_s_manifest.yaml> \\
        --out-dir reports/realism_eval/decision/
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import yaml

# iCloud-Drive-safe path inject (same pattern as scripts/run_bakeoff_decision.py).
# Repo-root goes on sys.path too so ``python scripts/realism_eval_decide.py`` (direct
# invocation) can import the ``scripts`` package (``from scripts.run_bakeoff_decision``),
# not only ``python -m scripts.realism_eval_decide``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# bakeoff_decision holds the STRICT per-candidate Lane-M sweep (memorization_check) and the
# exact real-features loader the decision runner uses — reused so the schema/teeth are ONE
# source, never a drifting copy. Both torch-free.
from cfm.eval.bakeoff_decision import memorization_check
from cfm.eval.city_aggregate import BindingVerdict, NoDecisiveWinner
from cfm.eval.conditioning_floor import (
    LaneSResult,
    VerifiedFloorArtifact,
    lane_s_excess,
    load_verified_floor,
)
from cfm.eval.lane_s_sampler import (
    CoverageReport,
    load_verified_manifest,
    verify_gen_coverage,
)
from cfm.eval.realism_driver import scoring
from cfm.eval.realism_driver.conditioning import load_verified_manifest_or_raise
from cfm.eval.realism_driver.scoring import DryRunReport, GenFeatures, MemorizationHalt
from scripts.run_bakeoff_decision import _load_real

logger = logging.getLogger(__name__)

#: Spec tags stamped into the emitted artifacts (lineage; bump on layout change).
DECISION_SPEC = "realism-eval-decision-v1"
GEN_FEATURES_SPEC = "realism-eval-gen-features-v1"

#: Overture release the gen artifacts + tile labels belong to (must match the gen run).
DEFAULT_RELEASE = "2026-04-15.0"

DECISION_FILENAME = "decision.yaml"
MEMORIZATION_FILENAME = "memorization.yaml"
SUMMARY_FILENAME = "summary.md"

#: rank-0 stdout sentinel — printed ONLY after decision.yaml is written (no marker without
#: end-state write).
SENTINEL = "REALISM_EVAL_DECISION_DONE"

#: DISTINCT dry-run sentinel — a dry run can NEVER print ``SENTINEL`` (no decision was made),
#: so a dry run is never mistaken for a scored decision downstream.
DRY_RUN_SENTINEL = "REALISM_EVAL_DRY_RUN_OK"

#: The LOCKED run shape (GROUND_TRUTH §4): exactly 2 backbones x exactly 3 seeds each.
#: REVIEW FIX I-1 (2026-07-20): enforced at the CLI boundary with NO override flag —
#: a partial (fewer-artifact) run would score with seed_sem degraded and silently
#: weaken the seed-noise floor; a lost checkpoint goes back to the PI, not to a flag.
EXPECTED_N_BACKBONES = 2
EXPECTED_SEEDS_PER_BACKBONE = 3


def assert_locked_run_shape(
    gen_by_ckpt: dict[tuple[str, int], dict[str, GenFeatures]],
) -> None:
    """Hard-require the locked scored-run shape: exactly ``EXPECTED_N_BACKBONES`` backbones
    with exactly ``EXPECTED_SEEDS_PER_BACKBONE`` seeds EACH. Raises ``SystemExit`` naming
    what was found vs expected. No escape hatch by design (review fix I-1)."""
    seeds_by_backbone: dict[str, list[int]] = {}
    for bb, seed in sorted(gen_by_ckpt):
        seeds_by_backbone.setdefault(bb, []).append(seed)
    found = {bb: sorted(s) for bb, s in seeds_by_backbone.items()}
    bad_counts = {bb: len(s) for bb, s in found.items() if len(s) != EXPECTED_SEEDS_PER_BACKBONE}
    if len(found) != EXPECTED_N_BACKBONES or bad_counts:
        raise SystemExit(
            f"realism_eval_decide: locked run shape is {EXPECTED_N_BACKBONES} backbones x "
            f"{EXPECTED_SEEDS_PER_BACKBONE} seeds each (GROUND_TRUTH §4); found "
            f"{len(found)} backbone(s) with seeds {found} "
            f"(wrong seed counts: {bad_counts or 'none'}). A partial run would degrade the "
            "seed-noise floor — refusing; there is no override flag. A missing checkpoint "
            "is a PI decision, not a CLI option."
        )


# --------------------------------------------------------------------------- #
# Write-once helpers (mirror the eval-set / gen-artifact discipline)
# --------------------------------------------------------------------------- #


def _write_once_yaml(path: Path, payload: dict) -> None:
    """Write ``payload`` as write-once YAML (atomic tmp + rename); refuse an existing file."""
    if path.exists():
        raise FileExistsError(
            f"{path} already exists; it is write-once — delete deliberately only to re-decide."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    logger.info("wrote %s", path)


def _write_once_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"{path} already exists; it is write-once.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# YAML-safe record builders (plain types only; verdicts by value)
# --------------------------------------------------------------------------- #


def verdict_record(verdict: BindingVerdict | NoDecisiveWinner) -> dict:
    """The verdict as a YAML-safe mapping. ``NO_DECISIVE_WINNER`` is written VERBATIM with
    per-(demoted) city ``(gap, resolution_floor, seed_noise_floor)`` — never softened."""
    if isinstance(verdict, BindingVerdict):
        return {
            "verdict": "DECISIVE",
            "winner": verdict.winner,
            "binding_city": verdict.binding_city,
            "runner_up": verdict.runner_up,
            "gap": float(verdict.gap),
            "city_floor": float(verdict.city_floor),
            "resolution_floor": float(verdict.resolution_floor),
            "seed_noise_floor": float(verdict.seed_noise_floor),
            "demoted_cities": list(verdict.demoted_from),
        }
    if isinstance(verdict, NoDecisiveWinner):
        return {
            "verdict": "NO_DECISIVE_WINNER",
            "basis": verdict.basis.value,
            "demoted_cities": list(verdict.demoted),
            "per_city": {
                city: {
                    "gap": float(verdict.gap[city]),
                    "resolution_floor": float(verdict.resolution_floor[city]),
                    "seed_noise_floor": float(verdict.seed_noise_floor[city]),
                }
                for city in verdict.demoted
            },
        }
    raise TypeError(f"verdict_record: unexpected verdict type {type(verdict).__name__}")


def _lane_s_table(
    lane_s_by_ckpt: dict[tuple[str, int], dict[str, LaneSResult]],
) -> list[dict]:
    """The per-(backbone, seed, city) median_excess table (the raw scored quantity)."""
    return [
        {
            "backbone": bb,
            "seed": seed,
            "city": city,
            "median_excess": float(r.median_excess),
            "p90_excess": float(r.p90_excess),
            "n_qualifying": r.n_qualifying,
            "n_skipped_thin": r.n_skipped_thin,
        }
        for (bb, seed), by_city in sorted(lane_s_by_ckpt.items())
        for city, r in sorted(by_city.items())
    ]


def _per_city_aggregation(
    lane_s_by_ckpt: dict[tuple[str, int], dict[str, LaneSResult]],
    n_reference_by_city: dict[str, int],
) -> list[dict]:
    """Per (backbone, city): the seed-aggregated ``ks`` (mean) and ``seed_sem`` (the
    seed-noise-floor input) — so a reader sees exactly what fed ``binding_city_verdict``."""
    per_backbone = scoring.seed_aggregated_per_backbone(
        lane_s_by_ckpt, n_reference_by_city=n_reference_by_city
    )
    return [
        {
            "backbone": bb,
            "city": pck.city,
            "ks": float(pck.ks),
            "seed_sem": float(pck.seed_sem),
            "n_features": pck.n_features,
        }
        for bb in sorted(per_backbone)
        for pck in per_backbone[bb]
    ]


def _memorization_record(
    memo_by_ckpt: dict[tuple[str, int], object],
) -> list[dict]:
    return [
        {
            "backbone": bb,
            "seed": seed,
            "ok": bool(m.ok),
            "failing_pairs": [list(p) for p in m.failing_pairs],
            "n_pairs_no_strata": m.n_pairs_no_strata,
        }
        for (bb, seed), m in sorted(memo_by_ckpt.items())
    ]


def _coverage_record(cov_by_ckpt: dict[tuple[str, int], CoverageReport]) -> list[dict]:
    return [
        {
            "backbone": bb,
            "seed": seed,
            "n_ok": len(c.ok),
            "ceiling_bound_excluded": [
                [city, metric, list(stratum)]
                for (city, metric, stratum) in c.ceiling_bound_excluded
            ],
        }
        for (bb, seed), c in sorted(cov_by_ckpt.items())
    ]


# --------------------------------------------------------------------------- #
# Per-checkpoint scored-lane steps (each testable in isolation)
# --------------------------------------------------------------------------- #


def check_memorization_all(
    gen_by_ckpt: dict[tuple[str, int], dict[str, GenFeatures]],
    real_by_city: dict[str, GenFeatures],
    real_train_by_city: dict[str, GenFeatures],
    verified: VerifiedFloorArtifact,
    *,
    min_n: int | None = None,
) -> dict[tuple[str, int], object]:
    """Lane-M memorization sweep per (backbone, seed) — RUN FIRST, before any coverage or
    excess. Returns ``{(backbone, seed): MemorizationCheck}``. The caller inspects ``.ok``
    and halts on any failure; this function itself never scores fidelity."""
    return {
        key: memorization_check(
            gen_by_ckpt[key], real_by_city, real_train_by_city, verified, min_n=min_n
        )
        for key in sorted(gen_by_ckpt)
    }


def check_coverage(
    gen_by_ckpt: dict[tuple[str, int], dict[str, GenFeatures]],
    manifest: dict,
    *,
    min_n: int | None = None,
) -> dict[tuple[str, int], CoverageReport]:
    """``verify_gen_coverage`` per (backbone, seed). A ceiling-bound short lands in
    ``ceiling_bound_excluded``; any OTHER short raises ``SamplerCoverageError`` — which is
    allowed to PROPAGATE (fail loud, never caught-and-continued: spec Gate 5 / §9)."""
    return {
        key: verify_gen_coverage(gen_by_ckpt[key], manifest, min_n=min_n)
        for key in sorted(gen_by_ckpt)
    }


def score_lane_s(
    gen_by_ckpt: dict[tuple[str, int], dict[str, GenFeatures]],
    real_by_city: dict[str, GenFeatures],
    artifact: str | Path | VerifiedFloorArtifact,
    held: frozenset[str],
    *,
    min_n: int | None = None,
) -> dict[tuple[str, int], dict[str, LaneSResult]]:
    """``lane_s_excess`` per (backbone, seed, city) against the floor (PATH or verified) so
    the sha self-verification runs; collects ``median_excess`` via the ``LaneSResult``."""
    return {
        key: {
            city: lane_s_excess(
                gen_by_ckpt[key][city], real_by_city[city], artifact, city=city, min_n=min_n
            )
            for city in sorted(held)
        }
        for key in sorted(gen_by_ckpt)
    }


# --------------------------------------------------------------------------- #
# The scored-lane core (torch-free; the tested seam)
# --------------------------------------------------------------------------- #


def score_and_decide(
    *,
    gen_by_ckpt: dict[tuple[str, int], dict[str, GenFeatures]],
    real_by_city: dict[str, GenFeatures],
    real_train_by_city: dict[str, GenFeatures],
    verified: VerifiedFloorArtifact,
    manifest: dict,
    out_dir: Path,
    config: dict,
    min_n: int | None = None,
) -> BindingVerdict | NoDecisiveWinner:
    """The full scored-lane pipeline over already-extracted ``GenFeatures`` (steps 2-7).

    Memorization is provably FIRST: no ``lane_s_excess`` runs unless every checkpoint's
    Lane-M sweep passes (a failure raises ``MemorizationHalt`` after writing
    ``memorization.yaml``). Coverage exclusions are recorded; a non-ceiling short propagates
    as ``SamplerCoverageError``. The verdict is written VERBATIM (including
    ``NO_DECISIVE_WINNER``) to a write-once ``decision.yaml`` plus a prose ``summary.md``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: STRICT held-out city-set equality (decide()'s Tooth-2, explicit).
    held = scoring.assert_city_sets(manifest, verified, real_by_city, gen_by_ckpt)

    # Step 3: dump each checkpoint's gen features (audit / reproducibility input).
    for (bb, seed), gen in sorted(gen_by_ckpt.items()):
        _write_once_yaml(
            out_dir / f"gen-features-{bb}-seed{seed}.yaml",
            {
                "spec": GEN_FEATURES_SPEC,
                "backbone": bb,
                "seed": seed,
                "gen_by_city": scoring.city_features_to_records(gen),
            },
        )

    # Step 4: MEMORIZATION FIRST — any failure halts BEFORE any scoring.
    memo_by_ckpt = check_memorization_all(
        gen_by_ckpt, real_by_city, real_train_by_city, verified, min_n=min_n
    )
    failing = {k: m for k, m in memo_by_ckpt.items() if not m.ok}
    if failing:
        _write_once_yaml(
            out_dir / MEMORIZATION_FILENAME,
            {
                "spec": DECISION_SPEC,
                "config": config,
                "halt": "MEMORIZATION_HALT",
                "memorization": _memorization_record(memo_by_ckpt),
            },
        )
        detail = "; ".join(
            f"({bb}, seed {seed}): failing pairs {[list(p) for p in m.failing_pairs]}"
            for (bb, seed), m in sorted(failing.items())
        )
        logger.error("MEMORIZATION HALT — %s", detail)
        raise MemorizationHalt(
            f"realism-eval decision HALTED: {len(failing)} checkpoint(s) FAILED the Lane-M "
            f"memorization discriminator ({detail}) — a regurgitator passes realism by "
            "construction, so no fidelity scoring runs. See "
            f"{out_dir / MEMORIZATION_FILENAME} (PI reviews)."
        )

    # Step 5: coverage per checkpoint (ceiling-bound exclusions recorded; other shorts raise).
    cov_by_ckpt = check_coverage(gen_by_ckpt, manifest, min_n=min_n)

    # Step 6: Lane-S excess per (backbone, seed, city) against the sha-verified floor.
    lane_s_by_ckpt = score_lane_s(gen_by_ckpt, real_by_city, verified, held, min_n=min_n)

    # Step 7: the ONLY crown path — seed-aggregated two-floor verdict.
    n_ref = scoring.n_reference_by_city(verified, real_by_city)
    verdict = scoring.aggregate_seed_verdict(lane_s_by_ckpt, n_reference_by_city=n_ref)

    decision = {
        "spec": DECISION_SPEC,
        "config": config,
        "n_reference_by_city": dict(sorted(n_ref.items())),
        "lane_s_median_excess": _lane_s_table(lane_s_by_ckpt),
        "per_city_aggregation": _per_city_aggregation(lane_s_by_ckpt, n_ref),
        "verdict": verdict_record(verdict),
        "coverage": _coverage_record(cov_by_ckpt),
        "memorization": _memorization_record(memo_by_ckpt),
    }
    _write_once_yaml(out_dir / DECISION_FILENAME, decision)
    _write_once_text(out_dir / SUMMARY_FILENAME, _summary_prose(verdict, config))
    logger.info("realism-eval decision written -> %s", out_dir / DECISION_FILENAME)
    return verdict


def _summary_prose(verdict: BindingVerdict | NoDecisiveWinner, config: dict) -> str:
    """A short human-readable verdict summary (never the sole authority — decision.yaml is)."""
    lines = ["# Scored realism-eval decision", ""]
    commit = config.get("commit")
    lines.append(f"- floor_sha256: `{config.get('floor_sha256')}`")
    if commit:
        lines.append(f"- commit: `{commit}`")
    lines.append("")
    if isinstance(verdict, BindingVerdict):
        lines += [
            f"**DECISIVE** — winner **{verdict.winner}** (runner-up {verdict.runner_up}).",
            "",
            f"Binding held-out city **{verdict.binding_city}**: gap {verdict.gap:.4f} cleared "
            f"the effective floor {verdict.city_floor:.4f} "
            f"(resolution {verdict.resolution_floor:.4f}, "
            f"seed-noise {verdict.seed_noise_floor:.4f}).",
        ]
        if verdict.demoted_from:
            lines.append(f"Demoted (under-powered) cities: {list(verdict.demoted_from)}.")
    else:
        lines += [
            "**NO_DECISIVE_WINNER** — no held-out city separated the backbones beyond BOTH the "
            "resolution floor AND the seed-noise floor.",
            "",
            "This is a valid, publishable verdict: route to the spec §13 simplest-backbone "
            "tie-break, never improvise. Per-city (gap, resolution_floor, seed_noise_floor):",
            "",
        ]
        for city in verdict.demoted:
            lines.append(
                f"- {city}: gap {verdict.gap[city]:.4f}, "
                f"resolution {verdict.resolution_floor[city]:.4f}, "
                f"seed-noise {verdict.seed_noise_floor[city]:.4f}"
            )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI (heavy disk/decode path — ops only)
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    """The CLI. Torch-free so it is unit-testable without a GPU."""
    ap = argparse.ArgumentParser(
        description="Scored realism-eval decision runner (Task 6; local, GPU-free).",
        epilog=(
            "OPS NOTE (M-3): after a memorization halt, re-running into the same --out-dir "
            "fails on the write-once gen-features dumps FIRST — deliberate (write-once "
            "discipline); use a fresh --out-dir for a re-decision."
        ),
    )
    ap.add_argument(
        "--gen-artifact",
        dest="gen_artifacts",
        action="append",
        help="a write-once gen artifact JSON (repeat once per backbone x seed; the 6 scored "
        "ckpts). Under --dry-run supply EXACTLY ONE (a single checkpoint slice).",
    )
    ap.add_argument(
        "--real-features", required=True, help="checkpoint-independent real-features YAML"
    )
    ap.add_argument(
        "--floor-artifact",
        required=True,
        help="frozen conditioning-floor YAML PATH (sha/lock verified here BEFORE any scoring)",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help="sealed Lane-S sampler manifest (verified read); REQUIRED for a scored decision, "
        "OPTIONAL under --dry-run (a synthetic 1-per-stratum manifest is built from the gen).",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="output directory (write-once decision.yaml); REQUIRED for a scored decision, "
        "IGNORED under --dry-run (a dry run writes nothing).",
    )
    ap.add_argument("--release", default=DEFAULT_RELEASE, help="Overture release for tile labels")
    ap.add_argument(
        "--min-n",
        type=int,
        default=None,
        help="override qualify min_n (default: the floor's frozen methodology.min_n)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="LOCAL scoring dry-run over ONE artifact: decode -> gen features -> coverage -> "
        "Lane-S excess, then STOP. NO seed aggregation, NO verdict, writes NOTHING — a dry run "
        "is structurally incapable of a crown.",
    )
    ap.add_argument(
        "--synthetic-stratum",
        default=None,
        help="--dry-run only: 'zoning,skeleton,density,coastal' (density an int bucket) — assign "
        "every decoded cell to ONE synthetic stratum WITHOUT reading tile labels off disk (for a "
        "stand-in artifact whose tiles are not on the local disk, e.g. the heldout-cache slice). "
        "Omit to use the real gen_features_by_city (needs local tile labels).",
    )
    ap.add_argument(
        "--verify-tokens",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="--dry-run only: re-decode each cell's tokens and assert they reproduce the stored "
        "aligned (blocks, geoms) bit-identically (default ON in --dry-run).",
    )
    return ap


def _git_commit() -> str | None:
    """Best-effort repo commit hash for the decision record (None if unavailable)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def _load_gen_artifacts(
    paths: list[str], *, release: str
) -> dict[tuple[str, int], dict[str, GenFeatures]]:
    """Read each gen artifact, decode -> ``gen_features_by_city`` (the heavy, disk-reading,
    torch-free step), key by ``(backbone, seed)`` from the artifact meta. A duplicate
    (backbone, seed) is a hard error (a re-submitted ckpt would silently overwrite)."""
    from cfm.eval.gen_realism import gen_features_by_city  # torch-free; reads tile labels
    from cfm.eval.realism_driver.driver import read_gen_artifact

    out: dict[tuple[str, int], dict[str, GenFeatures]] = {}
    for p in paths:
        meta, records = read_gen_artifact(Path(p))
        if "backbone" not in meta or "seed" not in meta:
            raise SystemExit(
                f"gen artifact {p} meta lacks 'backbone'/'seed' — cannot key the checkpoint."
            )
        key = (str(meta["backbone"]), int(meta["seed"]))
        if key in out:
            raise SystemExit(f"gen artifact {p}: duplicate (backbone, seed) {key}; refusing.")
        decoded = scoring.decoded_cells_from_artifact(meta, records, release=release)
        out[key] = gen_features_by_city(decoded, release=release)
        logger.info("loaded gen artifact %s -> %s (%d cells)", p, key, len(records))
    return out


# --------------------------------------------------------------------------- #
# Task 7a — LOCAL scoring dry-run CLI (no GPU, no checkpoint, no verdict)
# --------------------------------------------------------------------------- #


def _parse_synthetic_stratum(spec: str) -> tuple:
    """Parse ``'zoning,skeleton,density,coastal'`` into the floor's 4-tuple (density -> int)."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise SystemExit(
            f"--synthetic-stratum must be 'zoning,skeleton,density,coastal' (4 comma-separated "
            f"fields); got {spec!r}."
        )
    zoning, skeleton, density, coastal = parts
    try:
        density_i = int(density)
    except ValueError:
        raise SystemExit(
            f"--synthetic-stratum density (3rd field) must be an int bucket; got {density!r}."
        ) from None
    return (zoning, skeleton, density_i, coastal)


def _log_dry_run(report: DryRunReport) -> None:
    """Human-readable dry-run summary (logged; the sentinel is the only stdout marker)."""
    logger.info(
        "DRY-RUN summary: %d cells (%d self-terminated), verify_tokens=%s; cities=%s",
        report.n_cells,
        report.n_self_terminated,
        report.verify_tokens,
        report.cities,
    )
    logger.info(
        "DRY-RUN coverage: %d ok, %d ceiling-bound-excluded",
        len(report.coverage.ok),
        len(report.coverage.ceiling_bound_excluded),
    )
    for city in report.cities:
        keys = report.gen_stratum_keys.get(city, [])
        r = report.lane_s_by_city.get(city)
        if r is None:
            logger.info(
                "DRY-RUN %s: %d gen stratum keys; Lane-S NOT scored (reported)", city, len(keys)
            )
        else:
            logger.info(
                "DRY-RUN %s: %d gen stratum keys; Lane-S median_excess=%.4f (n_qualifying=%d, "
                "n_skipped_thin=%d)",
                city,
                len(keys),
                r.median_excess,
                r.n_qualifying,
                r.n_skipped_thin,
            )


def run_dry_run(args: argparse.Namespace) -> DryRunReport:
    """Wire and run the Task-7a LOCAL scoring dry-run over EXACTLY ONE gen artifact, print the
    dry-run sentinel, and return the report. Never asserts the locked 2x3 run shape, never
    scores a verdict, never writes anything."""
    from functools import partial

    from cfm.eval.gen_realism import gen_features_by_city  # torch-free; reads tile labels
    from cfm.eval.realism_driver.driver import read_gen_artifact

    n_arts = len(args.gen_artifacts or [])
    if n_arts != 1:
        raise SystemExit(
            f"--dry-run scores EXACTLY ONE --gen-artifact (a single checkpoint slice); got "
            f"{n_arts}. The locked 2x3 scored-run shape is a PRODUCTION concern, not a dry run."
        )
    verify_tokens = True if args.verify_tokens is None else bool(args.verify_tokens)

    verified = load_verified_floor(Path(args.floor_artifact))
    real_by_city, _real_train = _load_real(Path(args.real_features))
    manifest = load_verified_manifest(Path(args.manifest)) if args.manifest else None
    meta, records = read_gen_artifact(Path(args.gen_artifacts[0]))

    if args.synthetic_stratum:
        stratum = _parse_synthetic_stratum(args.synthetic_stratum)
        gen_features_fn = partial(scoring.single_stratum_gen_features, stratum=stratum)
        logger.info("DRY-RUN: single synthetic stratum %s (no disk tile-label read)", stratum)
    else:
        gen_features_fn = gen_features_by_city

    report = scoring.dry_run_score(
        meta=meta,
        records=records,
        real_by_city=real_by_city,
        verified=verified,
        release=args.release,
        gen_features_fn=gen_features_fn,
        manifest=manifest,
        min_n=args.min_n,
        verify_tokens=verify_tokens,
    )
    _log_dry_run(report)
    print(DRY_RUN_SENTINEL, flush=True)
    return report


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_arg_parser().parse_args(argv)

    if args.dry_run:
        run_dry_run(args)
        return

    # A scored decision hard-requires the artifacts the parser leaves optional for --dry-run.
    missing = [
        name
        for name, value in (
            ("--gen-artifact", args.gen_artifacts),
            ("--manifest", args.manifest),
            ("--out-dir", args.out_dir),
        )
        if not value
    ]
    if missing:
        raise SystemExit(
            f"realism_eval_decide: {', '.join(missing)} required for a scored decision run "
            "(only --dry-run may omit them)."
        )

    # Step 1: verified floor (sha/lock) BEFORE any scoring — decide()'s Tooth-1, explicit.
    verified = load_verified_floor(Path(args.floor_artifact))
    # The PINNED loader on the production scored path: the manifest must be the locked
    # Lane-S lineage (floor/census sha, 5705 cells, 146 strata) — a differently-sealed
    # manifest cannot slip past decide. The dry-run path deliberately uses the bare
    # load_verified_manifest (a dry run may score a stand-in slice).
    manifest = load_verified_manifest_or_raise(Path(args.manifest))
    real_by_city, real_train_by_city = _load_real(Path(args.real_features))
    gen_by_ckpt = _load_gen_artifacts(args.gen_artifacts, release=args.release)
    assert_locked_run_shape(gen_by_ckpt)  # review fix I-1: 2 backbones x 3 seeds, no override

    config = {
        "spec": DECISION_SPEC,
        "release": args.release,
        "commit": _git_commit(),
        "floor_artifact_path": str(args.floor_artifact),
        "floor_sha256": str(verified.payload["floor_sha256"]),
        "manifest_path": str(args.manifest),
        "manifest_sampler_sha256": manifest.get("sampler_sha256"),
        "manifest_floor_sha256": manifest.get("floor_sha256"),
        "manifest_census_sha256": manifest.get("census_sha256"),
        "real_features_path": str(args.real_features),
        "gen_artifact_paths": sorted(str(p) for p in args.gen_artifacts),
        "checkpoints": sorted(f"{bb}-seed{seed}" for (bb, seed) in gen_by_ckpt),
        "min_n": args.min_n,
    }

    verdict = score_and_decide(
        gen_by_ckpt=gen_by_ckpt,
        real_by_city=real_by_city,
        real_train_by_city=real_train_by_city,
        verified=verified,
        manifest=manifest,
        out_dir=Path(args.out_dir),
        config=config,
        min_n=args.min_n,
    )
    record = verdict_record(verdict)
    logger.info("realism-eval verdict: %s", record["verdict"])
    print(SENTINEL, flush=True)


if __name__ == "__main__":
    main()

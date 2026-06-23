"""Saturation metric (spec §3): loss-vs-steps plateau classification, read-only.

Answers "was training still descending at the final step (would more steps help)?" by
fitting a line to the final-step window and asking whether the slope is significantly
negative GIVEN the window's own noise. The plateau threshold is DERIVED from that noise
(2x the OLS slope standard error), NOT a magic absolute number (spec D5) — so the same
modest slope reads DESCENDING under low noise and PLATEAUED under high noise.

Pure / deterministic / torch-free. Units: loss in nats/token; slope & threshold in
nats/token per 1000 steps.
"""

from __future__ import annotations

import csv as _csv
from dataclasses import dataclass
from pathlib import Path

import yaml

#: slope must exceed this many standard errors (in magnitude) to be called non-flat.
_SIGNIFICANCE_SIGMA = 2.0

#: the bake-off run signature (spec D4): only a run matching ALL of these is a bake-off run;
#: the version dirs also hold old singapore d256 smokes that share backbone+seed.
_BAKEOFF_SIGNATURE = {
    "d_model": 512,
    "train_set": "eu-train-union",
    "conditioning_scheme": "value-char-v1",
}


def read_loss_series(csv_path: Path) -> tuple[list[int], list[float]]:
    """Read (step, train_loss) from a Lightning metrics.csv, skipping blank-loss rows.

    Columns: ``epoch,step,train_loss,val_loss``. train_loss is logged on most steps;
    val_loss is sparse. We track train_loss (the dense series).
    """
    steps: list[int] = []
    losses: list[float] = []
    with open(csv_path, newline="") as fh:
        for row in _csv.DictReader(fh):
            tl = (row.get("train_loss") or "").strip()
            st = (row.get("step") or "").strip()
            if not tl or not st:
                continue
            steps.append(int(st))
            losses.append(float(tl))
    return steps, losses


def resolve_bakeoff_run(logs_dir: Path, *, backbone: str, seed: int) -> Path:
    """Find the single ``version_N`` dir whose hparams match (backbone, seed) AND the
    bake-off signature. FAIL LOUD on ambiguity or no-match (D4: do not guess)."""
    logs_dir = Path(logs_dir)
    matches: list[Path] = []
    for vdir in sorted(logs_dir.glob("version_*")):
        hp_path = vdir / "hparams.yaml"
        if not hp_path.exists():
            continue
        hp = yaml.safe_load(hp_path.read_text()) or {}
        if str(hp.get("backbone")) != backbone or int(hp.get("seed", -1)) != int(seed):
            continue
        if all(hp.get(k) == v for k, v in _BAKEOFF_SIGNATURE.items()):
            matches.append(vdir)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"no matching bake-off run for backbone={backbone} seed={seed} "
            f"(signature {_BAKEOFF_SIGNATURE}) under {logs_dir} — refusing to guess (D4)"
        )
    raise ValueError(
        f"ambiguous: {len(matches)} version dirs match backbone={backbone} seed={seed} "
        f"({[m.name for m in matches]}) — refusing to guess; use a committed lookup table (D4)"
    )


@dataclass(frozen=True)
class SaturationResult:
    final_step: int
    final_loss: float
    loss_at_80pct: float
    loss_at_90pct: float
    final_window_slope: float  # nats/token per 1000 steps (signed; <0 = descending)
    final_window_noise: float  # residual std (nats/token) in the final window
    plateau_threshold: float  # nats/token per 1000 steps; 2*SE(slope), noise-derived
    n_window_points: int
    classification: str  # "DESCENDING" | "PLATEAUED"


def _loss_at_fraction(steps: list[int], losses: list[float], frac: float) -> float:
    """Loss at the logged step closest to frac * final_step."""
    target = frac * steps[-1]
    best = min(range(len(steps)), key=lambda i: abs(steps[i] - target))
    return losses[best]


def classify_saturation(
    steps: list[int], losses: list[float], *, window_steps: int = 10000
) -> SaturationResult:
    """Classify the final-window loss trend as DESCENDING or PLATEAUED.

    ``window_steps`` is the width (in training steps) of the final window used for the
    fit; if fewer than 3 points fall in it, the window is widened to the last 3 points.
    """
    if len(steps) < 3:
        raise ValueError("need >=3 logged points to classify saturation")
    pairs = sorted(zip(steps, losses, strict=True))
    steps = [int(s) for s, _ in pairs]
    losses = [float(v) for _, v in pairs]

    final_step = steps[-1]
    cutoff = final_step - window_steps
    idx = [i for i in range(len(steps)) if steps[i] >= cutoff]
    if len(idx) < 3:
        idx = list(range(len(steps) - 3, len(steps)))
    xs = [steps[i] for i in idx]
    ys = [losses[i] for i in idx]
    n = len(xs)

    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx  # loss per step
    intercept = ybar - slope * xbar
    sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys, strict=True))
    resid_std = (sse / (n - 2)) ** 0.5 if n > 2 else 0.0
    se_slope = resid_std / (sxx**0.5) if sxx > 0 else 0.0
    threshold = _SIGNIFICANCE_SIGMA * se_slope  # per step

    classification = "DESCENDING" if slope < -threshold else "PLATEAUED"

    return SaturationResult(
        final_step=final_step,
        final_loss=losses[-1],
        loss_at_80pct=_loss_at_fraction(steps, losses, 0.80),
        loss_at_90pct=_loss_at_fraction(steps, losses, 0.90),
        final_window_slope=slope * 1000.0,
        final_window_noise=resid_std,
        plateau_threshold=threshold * 1000.0,
        n_window_points=n,
        classification=classification,
    )

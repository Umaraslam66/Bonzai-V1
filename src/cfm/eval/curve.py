"""Scaling-curve fit + extrapolation + §13 structural check + tie-break (bake-off Task 13).

Each backbone gives points ``(measured_node_h, KS_realism)`` over {30M,100M,300M,1B}
(KS lower = better). We fit a power law ``KS = a * C^(-b)`` (log-log least squares),
bootstrap over the points for a confidence interval, extrapolate to the production
compute budget, and pick the winner -- UNLESS the top-two extrapolated CIs overlap
("doesn't separate"), in which case the pre-committed §13 tie-break is the simplest
backbone, transformer-ar. A crowning also requires the §2 structural check: a valid,
monotone-improving fit (a garbage / non-monotonic fit cannot crown a winner).

The bootstrap RNG is seeded so the decision is reproducible (config + commit + data
fully determine the experiment).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

#: Pre-committed §13 tie-break: when the curves don't separate, pick the SIMPLEST backbone
#: (no mamba-ssm kernels, no diffusion loss/gen complexity; CLAUDE.md default-to-simplicity).
#: Decided NOW so "we picked it because nothing separated" can't masquerade as "it won".
TIEBREAK_BACKBONE = "transformer-ar"

_MIN_SLOPE = 1e-3  # require a strictly-improving fit (KS falls as compute rises)
_BOOTSTRAP_SEED = 7


@dataclass(frozen=True)
class ScalingFit:
    intercept: float  # of log(KS) on log(C)
    slope: float  # negative for an improving curve; b = -slope
    points: tuple[tuple[float, float], ...]
    boot_params: tuple[tuple[float, float], ...]  # bootstrap (intercept, slope) samples

    @property
    def b(self) -> float:
        return -self.slope


def _linfit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx if sxx else 0.0
    return my - slope * mx, slope


def fit_scaling_curve(points: list[tuple[float, float]], *, n_bootstrap: int = 200) -> ScalingFit:
    """Fit ``log(KS) = intercept + slope*log(C)`` and bootstrap the params over the points."""
    if len(points) < 2:
        raise ValueError("need >=2 (compute, KS) points to fit a curve")
    xs = [math.log(c) for c, _ in points]
    ys = [math.log(ks) for _, ks in points]
    intercept, slope = _linfit(xs, ys)

    rng = random.Random(_BOOTSTRAP_SEED)
    n = len(points)
    boot: list[tuple[float, float]] = []
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        bxs = [xs[i] for i in idx]
        bys = [ys[i] for i in idx]
        if len({i for i in idx}) < 2:  # degenerate resample -> reuse the full-data fit
            boot.append((intercept, slope))
        else:
            boot.append(_linfit(bxs, bys))
    return ScalingFit(intercept, slope, tuple(points), tuple(boot))


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = q * (len(sorted_vals) - 1)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def extrapolate(fit: ScalingFit, *, target_node_h: float) -> tuple[float, float]:
    """5th-95th percentile CI of the predicted KS at ``target_node_h`` from the bootstrap."""
    lc = math.log(target_node_h)
    preds = sorted(math.exp(a + s * lc) for a, s in fit.boot_params)
    return _percentile(preds, 0.05), _percentile(preds, 0.95)


def structural_check_ok(fit: ScalingFit) -> bool:
    """§2 paired structural check: the fit must be a VALID, monotone-improving curve.

    A power law with ``b > 0`` (KS falls as compute rises) is monotone-improving; a
    non-improving / noisy fit (b <= 0) cannot crown a winner -> route to the §13 branch.
    """
    return len(fit.points) >= 3 and math.isfinite(fit.b) and fit.b > _MIN_SLOPE


def pick_winner(extrapolated_cis: dict[str, tuple[float, float]]) -> str:
    """The backbone with the best (lowest) extrapolated KS, IF its CI does not overlap the
    runner-up's; otherwise the pre-committed §13 tie-break (``TIEBREAK_BACKBONE``).

    "Doesn't separate" = the top-two CIs overlap (reuses the resolution-seam logic at the
    extrapolated point). Deciding the tie-break in advance prevents a no-separation outcome
    from masquerading as a genuine win.
    """
    if len(extrapolated_cis) < 2:
        return next(iter(extrapolated_cis))
    ranked = sorted(extrapolated_cis.items(), key=lambda kv: (kv[1][0] + kv[1][1]) / 2)
    (best_name, (_best_lo, best_hi)), (_, (run_lo, _run_hi)) = ranked[0], ranked[1]
    overlaps = best_hi >= run_lo  # lower=better: best's upper bound reaches the runner-up's lower
    return best_name if not overlaps else TIEBREAK_BACKBONE

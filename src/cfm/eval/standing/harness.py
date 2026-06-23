"""Per-checkpoint orchestration (spec §5): the three echo-immune metrics -> JSON + table.

eval_checkpoint(ckpt) runs:
  metric 1 (perplexity-gap, GPU): macro-only PRIMARY + full secondary on held-out cells.
  metric 2 (saturation, CPU): loss-vs-steps plateau classification from the run's metrics.csv.
  metric 3 (geometry-validity, GPU): generate the fixed probe set -> echo-immune structural metrics.

Per-checkpoint only; the 6-way seed-noise comparison is the aggregator's job.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from cfm.data.training.build_shards import character_stats_for_cell  # noqa: F401 (probe reuse)
from cfm.eval.standing.geometry_validity import geometry_validity_report
from cfm.eval.standing.heldout_cells import load_heldout_cells, read_heldout_cache
from cfm.eval.standing.nll import GapCell, compute_gap, effective_macro_shuffle_fraction
from cfm.eval.standing.saturation import classify_saturation, read_loss_series, resolve_bakeoff_run
from cfm.inference.generate import generate_cell_tokens
from cfm.models.backbone import build_backbone
from cfm.training.config import ScaffoldConfig

REPO = Path(__file__).resolve().parents[4]
# ASCII region keys as they appear on disk / in the manifest (NOT the GROUND_TRUTH "ü" spelling).
HELD_OUT_CITIES = ["glasgow", "eisenhuttenstadt", "munich", "krakow"]

#: spec §2 (ii): the macro-only gap is UNRELIABLE if fewer than this fraction of cells got a
#: genuinely different-macro donor (a near-zero gap must be distinguishable from a no-op shuffle).
#: With the macro-deranged donor this is 1.0 by construction; the floor guards regressions.
MACRO_SHUFFLE_FLOOR = 0.95
_PROBE = REPO / "scripts" / "_eyeball_gen_probe.py"


def _load_probe_module() -> Any:
    spec = importlib.util.spec_from_file_location("_eyeball_gen_probe", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model(ckpt_path: str, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Load a bake-off checkpoint into its backbone (backbone read from hyper_parameters)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ScaffoldConfig(**ck["hyper_parameters"])
    model = build_backbone(cfg.backbone, cfg)
    sd = {k[len("model.") :]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"state_dict mismatch missing={list(missing)} unexpected={list(unexpected)}"
        )
    model.eval().to(device)
    return model, {
        "backbone": cfg.backbone,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "seed": cfg.seed,
        "global_step": int(ck.get("global_step", -1)),
        "train_set": cfg.train_set,
        "conditioning_scheme": cfg.conditioning_scheme,
        "conditioning_ablation": cfg.conditioning_ablation,
    }


def run_perplexity_gap(
    model: torch.nn.Module,
    *,
    release: str,
    n_per_city: int | None,
    device: torch.device,
    sample_seed: int = 1234,
    shuffle_seed: int = 5678,
    max_tiles_per_city: int | None = None,
    heldout_cache: Path | None = None,
) -> dict[str, Any]:
    # The held-out cells are checkpoint-independent: read the cache when present (built once
    # for the full run); a byte-identical read-back is verified before the run (cached≡uncached).
    if heldout_cache is not None and Path(heldout_cache).exists():
        cells = read_heldout_cache(heldout_cache)
    else:
        cells = load_heldout_cells(
            release,
            HELD_OUT_CITIES,
            n_per_city=n_per_city,
            sample_seed=sample_seed,
            shuffle_seed=shuffle_seed,
            max_tiles_per_city=max_tiles_per_city,
        )
    dev = str(device)
    macro = [
        GapCell(c.region, c.body_tokens, c.own_prefix, c.donor_prefix, c.own_char, c.own_char)
        for c in cells
    ]
    full = [
        GapCell(c.region, c.body_tokens, c.own_prefix, c.donor_prefix, c.own_char, c.donor_char)
        for c in cells
    ]
    eff_fraction = effective_macro_shuffle_fraction(macro)
    macro_reliable = eff_fraction >= MACRO_SHUFFLE_FLOOR
    gap_macro = compute_gap(model, macro, device=dev)
    gap_full = compute_gap(model, full, device=dev)
    return {
        "n_cells": len(cells),
        "cities": sorted({c.region for c in cells}),
        "macro_shuffle_effective_fraction": eff_fraction,  # REQUIRED field (spec §2 (ii))
        "macro_shuffle_floor": MACRO_SHUFFLE_FLOOR,
        "macro_only_reliable": macro_reliable,
        "macro_only_primary": asdict(gap_macro),
        "full_secondary": asdict(gap_full),
    }


def run_saturation(logs_dir: Path, *, backbone: str, seed: int) -> dict[str, Any]:
    run_dir = resolve_bakeoff_run(Path(logs_dir), backbone=backbone, seed=seed)
    steps, losses = read_loss_series(run_dir / "metrics.csv")
    return {"version_dir": run_dir.name, **asdict(classify_saturation(steps, losses))}


def run_geometry_validity(
    model: torch.nn.Module,
    meta: dict,
    *,
    device: torch.device,
    out_dir: Path,
    max_new: int = 1536,
    cells_per_context: int = 7,
) -> dict[str, Any]:
    probe = _load_probe_module()
    records = []
    for ctx in probe.CONTEXTS:
        prefix, char_stats = probe.build_prefix(ctx)
        for i in range(cells_per_context):
            gen_seed = 1000 + i
            toks = generate_cell_tokens(
                model, prefix=prefix, max_new=max_new, seed=gen_seed, char_stats=char_stats
            )
            hit_cap = len(toks) >= max_new
            records.append(
                {
                    "context": ctx["name"],
                    "stratum": [
                        ctx["zoning"],
                        ctx["skeleton"],
                        ctx["cell_density"],
                        ctx["coastal"],
                    ],
                    "pop_density": ctx["pop_density"],
                    "cell_index": i,
                    "gen_seed": gen_seed,
                    "char_stats": char_stats,
                    "tokens": toks,
                    "n_tokens": len(toks),
                    "hit_cap": hit_cap,
                    "self_terminated": (260 in toks) and not hit_cap,
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_path = out_dir / "probe_gen_tokens.json"
    gen_path.write_text(json.dumps({"meta": meta, "max_new": max_new, "records": records}))
    rep = geometry_validity_report(gen_path)
    return {ctx: asdict(cg) for ctx, cg in rep.per_context.items()}


def eval_checkpoint(
    ckpt_path: str,
    *,
    release: str,
    logs_dir: Path,
    out_dir: Path,
    n_per_city: int | None = 2000,
    max_new: int = 1536,
    cells_per_context: int = 7,
    device: str | None = None,
    max_tiles_per_city: int | None = None,
    heldout_cache: Path | None = None,
) -> dict[str, Any]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, meta = load_model(ckpt_path, dev)
    ckpt_id = f"{meta['backbone']}-seed{meta['seed']}"
    result = {
        "ckpt_id": ckpt_id,
        "ckpt_path": str(ckpt_path),
        "meta": meta,
        "perplexity_gap": run_perplexity_gap(
            model,
            release=release,
            n_per_city=n_per_city,
            device=dev,
            max_tiles_per_city=max_tiles_per_city,
            heldout_cache=heldout_cache,
        ),
        "saturation": run_saturation(logs_dir, backbone=meta["backbone"], seed=meta["seed"]),
        "geometry_validity": run_geometry_validity(
            model,
            meta,
            device=dev,
            out_dir=out_dir / ckpt_id,
            max_new=max_new,
            cells_per_context=cells_per_context,
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{ckpt_id}.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (out_dir / f"{ckpt_id}.md").write_text(render_table(result))
    return result


def render_table(r: dict[str, Any]) -> str:
    m = r["meta"]
    pg = r["perplexity_gap"]
    sat = r["saturation"]
    macro = pg["macro_only_primary"]
    full = pg["full_secondary"]
    eff = pg["macro_shuffle_effective_fraction"]
    reliable = pg["macro_only_reliable"]
    macro_line = (
        f"- **MACRO-ONLY (primary)**: gap = **{macro['gap_nats_per_token']:+.4f}** "
        f"(frac_positive {macro['fraction_positive']:.2f}, "
        f"sign-test sig={macro['sign_test_significant_at_p']})"
        if reliable
        else f"- **MACRO-ONLY (primary): UNRELIABLE** — effective shuffle {eff:.0%} < floor "
        f"{pg['macro_shuffle_floor']:.0%}; gap ({macro['gap_nats_per_token']:+.4f}) not a clean "
        f"number"
    )
    lines = [
        f"# Standing eval — {r['ckpt_id']}",
        "",
        f"- model: {m['backbone']} d{m['d_model']}/{m['n_layers']}L step {m['global_step']} "
        f"({m['conditioning_scheme']}/{m['conditioning_ablation']})",
        "",
        "## (1) Perplexity-gap (nats/token; gap = NLL_shuffled - NLL_matched)",
        f"- n_cells={pg['n_cells']} cities={pg['cities']} "
        f"effective-macro-shuffle={eff:.0%} (floor {pg['macro_shuffle_floor']:.0%})",
        macro_line,
        f"- full (secondary): gap = {full['gap_nats_per_token']:+.4f} "
        f"(frac_positive {full['fraction_positive']:.2f})",
        "",
        "## (2) Saturation (would more training help?)",
        f"- **{sat['classification']}** — final_step={sat['final_step']} "
        f"final_loss={sat['final_loss']:.4f}",
        f"- final-window slope={sat['final_window_slope']:+.5f} nats/tok per 1k steps; "
        f"noise-derived threshold=±{sat['plateau_threshold']:.5f} "
        f"(noise={sat['final_window_noise']:.4f})",
        "",
        "## (3) Geometry-validity (echo-immune; per context)",
        "| context | self-term | decode | closure med | <5% | comp/seg | dangling |",
        "|---|---|---|---|---|---|---|",
    ]
    for ctx, g in r["geometry_validity"].items():
        lines.append(
            f"| {ctx} | {g['self_term_frac']:.0%} | {g['decode_frac']:.0%} | "
            f"{g['closure_gap_median']:.3f} | {g['closure_within_5pct']:.0%} | "
            f"{g['median_components_per_segment']:.2f} | {g['dangling_endpoint_frac']:.2f} |"
        )
    lines += [
        "",
        "_NOT a crown: no decide/floor/echo. Macro-only gap is the ranking number; "
        "seed-noise across the 3 seeds is computed by the aggregator._",
    ]
    return "\n".join(lines) + "\n"


def aggregate(per_ckpt: list[dict[str, Any]]) -> str:
    """6-way transformer-vs-mamba table with seed-noise (std across seeds) on the macro-only gap.

    Checkpoints whose macro-only gap is UNRELIABLE (effective shuffle below floor) are excluded
    from the ranking and listed — a vacuous-shuffle number never enters the mean."""
    by_bb: dict[str, list[float]] = {}
    unreliable: list[str] = []
    for r in per_ckpt:
        if not r["perplexity_gap"].get("macro_only_reliable", True):
            unreliable.append(r["ckpt_id"])
            continue
        by_bb.setdefault(r["meta"]["backbone"], []).append(
            r["perplexity_gap"]["macro_only_primary"]["gap_nats_per_token"]
        )
    lines = [
        "# Standing eval — aggregate (macro-only perplexity-gap)",
        "",
    ]
    if unreliable:
        lines.append(f"> EXCLUDED as UNRELIABLE (vacuous shuffle): {unreliable}")
        lines.append("")
    lines += [
        "| backbone | n_seeds | mean gap | seed-noise (std) |",
        "|---|---|---|---|",
    ]
    means = {}
    for bb, gaps in sorted(by_bb.items()):
        mean = statistics.mean(gaps)
        std = statistics.pstdev(gaps) if len(gaps) > 1 else float("nan")
        means[bb] = (mean, std)
        lines.append(f"| {bb} | {len(gaps)} | {mean:+.4f} | {std:.4f} |")
    if len(means) == 2:
        (_b1, (m1, s1)), (_b2, (m2, s2)) = means.items()
        diff = abs(m1 - m2)
        noise = max(s1, s2)
        verdict = (
            "NO_DECISIVE"
            if diff < noise
            else f"gap difference ({diff:.4f}) exceeds seed-noise ({noise:.4f})"
        )
        lines += ["", f"**Ranking read:** {verdict} _(descriptive; not a decide() verdict)_"]
    return "\n".join(lines) + "\n"

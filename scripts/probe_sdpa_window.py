"""SDPA flash-attention / memory probe at the LOCKED window (13,312).

Execution obligation #2 of the token-budget coupled decision
(reports/2026-06-11-token-budget-coupled-decision-memo.md): the 13,312 lock is
CONTINGENT on PyTorch's flash/memory-efficient SDPA kernel actually engaging for
MicroAR's attention call (explicit causal mask + is_causal=True) at T=13,322 on
A100 bf16. If the math path engages instead, the score matrix alone is
~batch*heads*T^2 and the window is not runnable as-is -> HALT, decision returns
to Umar.

NON-SCORED kernel/memory probe. No training, no eval, no checkpoints. Single GPU.

Evidence design (a gate must distinguish regimes -- never infer "flash" from the
absence of an error):
  1. DEFAULT-context fwd+bwd at the worst real batch (8 x 13,322, toy scale):
     the production question. Peak memory is the regime discriminator.
  2. Forced FLASH_ATTENTION-only / EFFICIENT_ATTENTION-only contexts: in a forced
     context, SDPA raises if that backend cannot serve the call -- success is
     positive evidence the backend CAN engage. (A model path that never reaches
     SDPA would also "succeed", so this is never read alone -- see 3+4.)
  3. Forced MATH-only at batch 1 (forward): the memory CONTRAST control. Math at
     b1 materialises heads*T^2 scores (~2.8 GiB bf16) vs flash-class ~0; if the
     "flash" variants show math-sized peaks, flash did NOT actually engage.
  4. Profiler kernel names on a default-context forward: direct evidence (look
     for flash/fmha/mem_eff/cutlass kernels vs gemm+softmax pairs).
  5. Math-only at batch 8 (forward): documents the OOM cliff the lock must avoid
     (expected ~45 GiB fp32 / ~22.7 GiB bf16 scores; OOM is an EXPECTED outcome
     here and does not fail the probe).
  6. DEFAULT-context fwd+bwd at 300M-class shape (d=1024, 24L, 16H, batch 2):
     the PRD top-of-ladder scale at the locked window (scaleup sbatch's batch).

Verdict rule (written into the YAML):
  PASS iff variant 1 ok with peak < 10 GiB (toy flash-class) AND variant 6 ok
  with peak < 35 GiB (300M flash-class: the legitimate flash-path base at d=1024
  b2 -- fp32 weights + grads + activations -- is ~20 GiB; a math-path attention
  would add >= ~22 GiB of b2*16-head*T^2 scores, so 35 GiB separates regimes
  with margin on both sides). Anything else -> HALT (bring to Umar; no
  workaround, no silent smaller window).

Run 45898653 (first attempt) measured everything then crashed serialising
torch.__version__ (TorchVersion, a str subclass yaml.safe_dump refuses); it also
exposed that a single flat 10 GiB threshold would false-HALT the 300M shape.
Both fixed here; this version produces the recorded artifact.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfm.models.backbone import build_backbone
from cfm.training.config import ScaffoldConfig
from cfm.training.env_lock import LOCKED_TORCH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("probe_sdpa_window")

LOCKED_WINDOW = 13_312  # the locked DEFAULT_MAX_CELL_TOKENS / max_len (decision memo)
PREFIX = 10  # 9 conditioning id positions + 1 continuous character position
T = LOCKED_WINDOW + PREFIX  # full worst-case sequence the model must hold
TOY_FLASH_CLASS_GIB = 10.0  # toy shape: flash-class is ~6 GiB; math-class is ~45 GiB
BIG_FLASH_CLASS_GIB = 35.0  # 300M shape: flash-class base ~20 GiB; math adds >=~22 GiB
OUT_PATH = Path("reports/2026-06-11-sdpa-window-probe.yaml")


def _make_batch(n_subf_vocab: int, batch: int, device: str) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(7)
    return {
        "ids": torch.randint(1, n_subf_vocab, (batch, T), generator=g).to(device),
        "prefix_len": torch.full((batch,), PREFIX, dtype=torch.long, device=device),
        "seq_len": torch.full((batch,), T, dtype=torch.long, device=device),
        "char_stats": torch.randn(batch, 7, generator=g).to(device),
    }


def _make_model(d_model: int, n_layers: int, n_heads: int, device: str) -> torch.nn.Module:
    cfg = ScaffoldConfig(
        region="singapore",  # region REQUIRED (no default); SDPA-window probe is region-agnostic
        max_len=LOCKED_WINDOW,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
    )
    return build_backbone("transformer-ar", cfg).to(device)


def _run_variant(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    label: str,
    backends: list[SDPBackend] | None,
    backward: bool,
) -> dict:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    res: dict = {"label": label, "ok": False, "backward": backward, "error": None}
    try:
        ctx = sdpa_kernel(backends) if backends is not None else nullcontext()
        with ctx, torch.autocast("cuda", torch.bfloat16):
            out = model.training_loss(
                batch["ids"],
                prefix_len=batch["prefix_len"],
                seq_len=batch["seq_len"],
                char_stats=batch["char_stats"],
            )
            if backward:
                out.loss.backward()
        torch.cuda.synchronize()
        res["ok"] = True
        res["loss"] = float(out.loss.detach())
        del out
    except RuntimeError as e:  # torch.cuda.OutOfMemoryError subclasses RuntimeError
        res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    res["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(
        "variant %-28s ok=%-5s peak=%.3f GiB err=%s",
        label,
        res["ok"],
        res["peak_gib"],
        res["error"],
    )
    return res


def _profile_default_forward(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> list[str]:
    """Direct kernel-name evidence for the DEFAULT-context attention path."""
    from torch.profiler import ProfilerActivity, profile

    gc.collect()
    torch.cuda.empty_cache()
    try:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                model(batch["ids"], char_stats=batch["char_stats"])
            torch.cuda.synchronize()
    except RuntimeError as e:  # OOM on a math-fallback default path: evidence, not a crash
        gc.collect()
        torch.cuda.empty_cache()
        return [f"PROFILER_FAILED {type(e).__name__}: {str(e)[:200]}"]
    rows = sorted(
        (e for e in prof.key_averages() if e.device_time_total > 0),
        key=lambda e: e.device_time_total,
        reverse=True,
    )
    return [r.key[:120] for r in rows[:15]]


def main() -> int:
    assert torch.cuda.is_available(), "probe requires a GPU"
    device = "cuda:0"
    torch.manual_seed(7)
    script = Path(__file__).resolve()

    env = {
        "torch": torch.__version__,
        "locked_torch": LOCKED_TORCH,
        "torch_matches_lock": torch.__version__ == LOCKED_TORCH,
        "gpu": torch.cuda.get_device_name(0),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "total_mem_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
    }
    logger.info("env: %s", env)

    toy = _make_model(256, 6, 8, device)
    n_subf = toy.cfg.n_subf_vocab
    b8 = _make_batch(n_subf, 8, device)
    b1 = _make_batch(n_subf, 1, device)

    # warmup (cublas init etc.) at a short, harmless shape -- not measured
    warm = {k: (v[:, :512] if v.dim() == 2 and v.shape[1] == T else v) for k, v in b1.items()}
    warm["seq_len"] = torch.full((1,), 512, dtype=torch.long, device=device)
    with torch.autocast("cuda", torch.bfloat16):
        toy.training_loss(
            warm["ids"],
            prefix_len=warm["prefix_len"],
            seq_len=warm["seq_len"],
            char_stats=warm["char_stats"],
        ).loss.backward()
    toy.zero_grad(set_to_none=True)

    variants = [
        _run_variant(toy, b8, label="toy_default_b8_fwdbwd", backends=None, backward=True),
        _run_variant(
            toy,
            b8,
            label="toy_flash_only_b8_fwdbwd",
            backends=[SDPBackend.FLASH_ATTENTION],
            backward=True,
        ),
        _run_variant(
            toy,
            b8,
            label="toy_efficient_only_b8_fwdbwd",
            backends=[SDPBackend.EFFICIENT_ATTENTION],
            backward=True,
        ),
        _run_variant(
            toy,
            b1,
            label="toy_math_only_b1_fwd_CONTRAST",
            backends=[SDPBackend.MATH],
            backward=False,
        ),
        _run_variant(
            toy, b8, label="toy_math_only_b8_fwd_CLIFF", backends=[SDPBackend.MATH], backward=False
        ),
    ]
    kernels = _profile_default_forward(toy, b8)
    logger.info("default-context top CUDA kernels: %s", kernels)

    del toy, b8
    gc.collect()
    torch.cuda.empty_cache()

    big = _make_model(1024, 24, 16, device)
    params_m = round(sum(p.numel() for p in big.parameters()) / 1e6, 1)
    b2 = _make_batch(n_subf, 2, device)
    variants.append(
        _run_variant(big, b2, label="300m_default_b2_fwdbwd", backends=None, backward=True)
    )

    by = {v["label"]: v for v in variants}
    toy_ok = (
        by["toy_default_b8_fwdbwd"]["ok"]
        and by["toy_default_b8_fwdbwd"]["peak_gib"] < TOY_FLASH_CLASS_GIB
    )
    big_ok = (
        by["300m_default_b2_fwdbwd"]["ok"]
        and by["300m_default_b2_fwdbwd"]["peak_gib"] < BIG_FLASH_CLASS_GIB
    )
    # NOTE: "cutlass" deliberately excluded -- plain gemms use CUTLASS too; a hit
    # there is not attention-path evidence (verify the KIND of yes).
    kernel_evidence = [
        k for k in kernels if any(s in k.lower() for s in ("flash", "fmha", "mem_eff", "attention"))
    ]
    verdict = "PASS" if (toy_ok and big_ok) else "HALT"

    payload = {
        "probe": "sdpa-window-13312",
        "decision_memo": "reports/2026-06-11-token-budget-coupled-decision-memo.md",
        "locked_window": LOCKED_WINDOW,
        "sequence_length_probed": T,
        "flash_class_threshold_gib": {"toy": TOY_FLASH_CLASS_GIB, "300m": BIG_FLASH_CLASS_GIB},
        "env": env,
        "model_300m_params_m": params_m,
        "variants": variants,
        "default_context_top_kernels": kernels,
        "attention_kernel_evidence": kernel_evidence,
        "verdict": verdict,
        "verdict_rule": (
            "PASS iff default-context fwd+bwd succeeds at toy b8 with peak < "
            f"{TOY_FLASH_CLASS_GIB} GiB AND at 300M b2 with peak < {BIG_FLASH_CLASS_GIB} GiB "
            "(per-shape flash-class bounds; math-path attention exceeds both by >=2x); "
            "the math-only b1 contrast variant must show a visibly larger peak for the "
            "discrimination to hold"
        ),
        "scope": "kernel engagement + worst-batch memory ONLY; non-scored; no training",
    }
    # json round-trip with default=str: yaml.safe_dump refuses str/float SUBCLASSES
    # (TorchVersion killed run 45898653); normalise everything to plain types first.
    payload = json.loads(json.dumps(payload, default=str))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(yaml.safe_dump(payload, sort_keys=False))
    print(yaml.safe_dump(payload, sort_keys=False))
    logger.info("verdict=%s written=%s", verdict, OUT_PATH)
    return 0 if verdict == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())

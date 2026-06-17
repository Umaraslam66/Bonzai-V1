"""mamba-ssm verify-before-lock — GPU NUMERICS half (Phase-2 bake-off Task 5 / step b).

The CPU half (W4) proved the locked stack builds + imports. THIS is the actual
verify-before-lock verdict: do the fused CUDA kernels (``selective_scan_fn``,
``causal_conv1d_fn``) compute CORRECTLY on an A100 under the EXACT locked stack,
forward AND backward? "It ran" is not the verdict — fused-kernel numerics can be
silently wrong, which would corrupt a whole bake-off backbone.

CORRECTNESS CRITERION. A fused kernel in fp32/bf16 cannot be bit-identical to a
PyTorch reference: both accumulate float error over the (2048-step) scan in
different orders, so an absolute tolerance is the wrong instrument. The reference
ops (``selective_scan_ref``, ``causal_conv1d_ref``) have internal fp32 assumptions
and do NOT run in fp64 — so fp32 is the ground truth, exactly as in mamba-ssm's own
test suite. Two regimes:

- **bf16 (the TRAINING dtype — the verdict that matters): the ratio test.**
    truth = reference in fp32; naive = reference in bf16; fused = CUDA kernel in bf16.
    PASS iff  ||fused-truth||inf <= K*||naive-truth||inf + floor  (fused no less
    accurate than the naive same-dtype reference). A correct kernel passes; a wrong
    one (error orders of magnitude larger than naive) fails hard. The fused kernel
    accumulates internally in fp32 so it is typically BETTER than the bf16 naive.

- **fp32: a direct generous relative tolerance** (rtol/atol 2e-2). No higher-
    precision reference exists for the ratio test; this is a sanity bound (a wrong
    fp32 kernel would be orders of magnitude off, not ~1%).

Also asserts torch is STILL 2.5.1+cu121 at runtime (the W4 torch_matches_lock tooth).
Run under the locked probe venv with the three candidate-pin preconditions — see
scripts/probe_mamba_gpu_half.sh.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import torch

LOCKED_TORCH = "2.5.1+cu121"
K_SLACK = 3.0          # bf16: fused error may be up to K x the naive reference error vs fp32 truth
BF16_FLOOR = 1e-2      # absorbs near-zero naive_err in the bf16 ratio test
FP32_RTOL, FP32_ATOL = 2e-2, 2e-2  # fp32 direct sanity bound (2048-step accumulation)

# Representative mamba-hybrid shape at the scaffold scale (ScaffoldConfig d_model=256,
# mamba expand=2 -> d_inner=512; d_state=16 standard; seqlen 2048 a real cell length;
# batch 2 — small, cheap dbg shape that still exercises the scan/conv kernels fully).
BATCH, D_INNER, SEQLEN, D_STATE, D_CONV = 2, 512, 2048, 16, 4
D_MODEL = 256
DEV = "cuda"
SEED_IN, SEED_G = 0, 100  # identical across truth/naive/fused so only the kernel-vs-ref differs


def _version(dist: str) -> str:
    try:
        import importlib.metadata as md

        return md.version(dist)
    except Exception as e:  # noqa: BLE001
        return f"ABSENT ({type(e).__name__})"


def _grad(t: torch.Tensor):
    return t.grad.detach().double() if (t is not None and t.grad is not None) else None


def _err(a, truth) -> float | None:
    if a is None or truth is None:
        return None
    return (a.double() - truth).abs().max().item()


def _verdict(truth, naive, fused, *, ratio: bool, floor: float) -> dict:
    """One tensor (forward output or a gradient): ratio test (bf16) or relative tol (fp32)."""
    if truth is None and naive is None and fused is None:
        return {"ok": True, "note": "no grad on any path (input not differentiated)"}
    fe, ne = _err(fused, truth), _err(naive, truth)
    rec = {"fused_err": fe, "naive_err": ne, "truth_absmax": (truth.abs().max().item() if truth is not None else None)}
    if fe is None:
        rec["ok"] = False
        rec["note"] = "fused grad missing"
        return rec
    if ratio:
        rec["ok"] = bool(fe <= K_SLACK * (ne or 0.0) + floor)
    else:  # fp32 direct relative tolerance vs the (fp32) truth
        rec["ok"] = bool(torch.allclose(fused.double(), truth.double(), rtol=FP32_RTOL, atol=FP32_ATOL))
    return rec


# ---- selective_scan -----------------------------------------------------------------
def _scan_run(io_dtype: torch.dtype, use_fused: bool):
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

    fn = selective_scan_fn if use_fused else selective_scan_ref
    torch.manual_seed(SEED_IN)  # identical values across runs; only dtype / fn differs
    u = torch.randn(BATCH, D_INNER, SEQLEN, device=DEV, dtype=io_dtype, requires_grad=True)
    delta = torch.rand(BATCH, D_INNER, SEQLEN, device=DEV, dtype=io_dtype, requires_grad=True)
    A = torch.empty(D_INNER, D_STATE, device=DEV, dtype=torch.float32).uniform_(-1.0, 0.0).requires_grad_(True)
    Bm = torch.randn(BATCH, D_STATE, SEQLEN, device=DEV, dtype=io_dtype, requires_grad=True)
    Cm = torch.randn(BATCH, D_STATE, SEQLEN, device=DEV, dtype=io_dtype, requires_grad=True)
    Dm = torch.randn(D_INNER, device=DEV, dtype=torch.float32, requires_grad=True)
    z = torch.randn(BATCH, D_INNER, SEQLEN, device=DEV, dtype=io_dtype, requires_grad=True)
    db = torch.randn(D_INNER, device=DEV, dtype=torch.float32, requires_grad=True)
    leaves = {"u": u, "delta": delta, "A": A, "B": Bm, "C": Cm, "D": Dm, "z": z, "delta_bias": db}
    out = fn(u, delta, A, Bm, Cm, Dm, z=z, delta_bias=db, delta_softplus=True)
    torch.manual_seed(SEED_G)
    out.backward(torch.randn(BATCH, D_INNER, SEQLEN, device=DEV, dtype=out.dtype))
    return out.detach(), {k: _grad(v) for k, v in leaves.items()}


def _conv_run(io_dtype: torch.dtype, use_fused: bool):
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref

    fn = causal_conv1d_fn if use_fused else causal_conv1d_ref
    torch.manual_seed(SEED_IN)
    x = torch.randn(BATCH, D_INNER, SEQLEN, device=DEV, dtype=io_dtype, requires_grad=True)
    w = torch.randn(D_INNER, D_CONV, device=DEV, dtype=io_dtype, requires_grad=True)
    b = torch.randn(D_INNER, device=DEV, dtype=io_dtype, requires_grad=True)
    out = fn(x, w, b, activation="silu")
    torch.manual_seed(SEED_G)
    out.backward(torch.randn(BATCH, D_INNER, SEQLEN, device=DEV, dtype=out.dtype))
    return out.detach(), {"x": _grad(x), "weight": _grad(w), "bias": _grad(b)}


def _kernel_check(runner, grad_names: tuple[str, ...], io_dtype: torch.dtype, *, ratio: bool, floor: float) -> dict:
    t_out, t_grad = runner(torch.float32, False)  # truth: reference in fp32
    n_out, n_grad = runner(io_dtype, False)  # naive: reference in io_dtype
    f_out, f_grad = runner(io_dtype, True)  # fused: CUDA kernel in io_dtype
    out = {"io_dtype": str(io_dtype), "mode": "ratio(K=3,fp32-truth)" if ratio else "relative-tol",
           "forward": _verdict(t_out, n_out, f_out, ratio=ratio, floor=floor)}
    grads, all_ok = {}, True
    for name in grad_names:
        v = _verdict(t_grad[name], n_grad[name], f_grad[name], ratio=ratio, floor=floor)
        grads[name] = v
        all_ok = all_ok and v["ok"]
    out["backward"] = grads
    out["ok"] = bool(out["forward"]["ok"] and all_ok)
    return out


_SCAN_GRADS = ("u", "delta", "A", "B", "C", "D", "z", "delta_bias")
_CONV_GRADS = ("x", "weight", "bias")


def mamba_module_check(dtype: torch.dtype) -> dict:
    """The full Mamba block at the scaffold width: fwd+bwd run, all grads finite — the
    block the bake-off would actually train, end to end on A100."""
    from mamba_ssm.modules.mamba_simple import Mamba

    torch.manual_seed(2)
    m = Mamba(d_model=D_MODEL, d_state=D_STATE, d_conv=D_CONV, expand=2).to(DEV).to(dtype)
    x = torch.randn(BATCH, SEQLEN, D_MODEL, device=DEV, dtype=dtype, requires_grad=True)
    y = m(x)
    loss = y.float().pow(2).mean()
    loss.backward()
    grads_finite = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in m.parameters())
    x_grad_finite = x.grad is not None and torch.isfinite(x.grad).all().item()
    return {
        "dtype": str(dtype), "n_params": int(sum(p.numel() for p in m.parameters())),
        "forward_shape": list(y.shape), "loss": float(loss.item()),
        "param_grads_finite": bool(grads_finite), "input_grad_finite": bool(x_grad_finite),
        "ok": bool(grads_finite and x_grad_finite and torch.isfinite(loss).item()),
    }


def _guarded(fn, *args, **kw) -> dict:
    try:
        return fn(*args, **kw)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1800:]}


def main() -> None:
    report_path = sys.argv[1] if len(sys.argv) > 1 else "mamba-gpu-half-probe.json"
    rep: dict = {
        "probe": "mamba-gpu-half verify-before-lock (step b)",
        "scope": "fused CUDA kernel numerics vs fp32 reference, fwd+bwd, A100",
        "criterion": "bf16: fused_err <= 3*naive_err+floor vs fp32 truth; fp32: relative tol 2e-2",
        "locked_torch_required": LOCKED_TORCH,
        "torch_version": torch.__version__,
        "torch_matches_lock": torch.__version__ == LOCKED_TORCH,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": {p: _version(p) for p in ("torch", "triton", "mamba-ssm", "causal-conv1d")},
        "shape": {"batch": BATCH, "d_inner": D_INNER, "seqlen": SEQLEN, "d_state": D_STATE,
                  "d_conv": D_CONV, "d_model": D_MODEL},
    }

    rep["selective_scan_bf16"] = _guarded(_kernel_check, _scan_run, _SCAN_GRADS, torch.bfloat16, ratio=True, floor=BF16_FLOOR)
    rep["selective_scan_fp32"] = _guarded(_kernel_check, _scan_run, _SCAN_GRADS, torch.float32, ratio=False, floor=0.0)
    rep["causal_conv1d_bf16"] = _guarded(_kernel_check, _conv_run, _CONV_GRADS, torch.bfloat16, ratio=True, floor=BF16_FLOOR)
    rep["causal_conv1d_fp32"] = _guarded(_kernel_check, _conv_run, _CONV_GRADS, torch.float32, ratio=False, floor=0.0)
    rep["mamba_module_bf16"] = _guarded(mamba_module_check, torch.bfloat16)

    rep["verdict_PASS"] = bool(
        rep["torch_matches_lock"]
        and rep["cuda_available"]
        and rep["selective_scan_bf16"].get("ok")
        and rep["selective_scan_fp32"].get("ok")
        and rep["causal_conv1d_bf16"].get("ok")
        and rep["causal_conv1d_fp32"].get("ok")
        and rep["mamba_module_bf16"].get("ok")
    )

    Path(report_path).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()

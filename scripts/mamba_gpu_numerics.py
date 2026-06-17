"""mamba-ssm verify-before-lock — GPU NUMERICS half (Phase-2 bake-off Task 5 / step b).

The CPU half (W4) proved the locked stack builds + imports. THIS is the actual
verify-before-lock verdict: do the fused CUDA kernels (``selective_scan_fn``,
``causal_conv1d_fn``) compute CORRECTLY on an A100 under the EXACT locked stack,
forward AND backward? "It ran" is not the verdict — fused-kernel numerics can be
silently wrong, which would corrupt a whole bake-off backbone.

CORRECTNESS CRITERION — the fp64 ground-truth ratio test (mamba-ssm's own test
methodology). A fused kernel run in fp32/bf16 cannot be bit-identical to a PyTorch
reference: both accumulate float error over the (here 2048-step) scan, just in
different orders. So an absolute tolerance is the WRONG instrument. Instead, for
each output/gradient:

  truth  = pure-PyTorch reference in **fp64**            (high-precision ground truth)
  naive  = pure-PyTorch reference in the **test dtype**  (the honest same-dtype baseline)
  fused  = the **CUDA fused kernel** in the test dtype

  PASS iff  ||fused - truth||_inf  <=  K * ||naive - truth||_inf  + floor

i.e. the fused kernel is NO LESS ACCURATE than the reference math against the fp64
truth. A correct kernel passes at any dtype/length; a genuinely wrong kernel has an
error orders of magnitude larger than naive and fails hard. (K=3 slack absorbs the
benign reduction-order difference; the fused kernel often does BETTER than naive in
bf16 because it accumulates internally in fp32.)

References (the package's own): ``selective_scan_ref`` (mamba_ssm.ops.selective_scan_interface),
``causal_conv1d_ref`` (causal_conv1d.causal_conv1d_interface). Also asserts torch is
STILL 2.5.1+cu121 at runtime (the W4 torch_matches_lock tooth). Run under the locked
probe venv with the three candidate-pin preconditions — see scripts/probe_mamba_gpu_half.sh.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import torch

LOCKED_TORCH = "2.5.1+cu121"
K_SLACK = 3.0  # fused error may be up to K x the naive reference error vs fp64 truth

# Representative mamba-hybrid shape at the scaffold scale (ScaffoldConfig d_model=256,
# mamba expand=2 -> d_inner=512; d_state=16 standard; seqlen 2048 a real cell length;
# batch 2 — small, cheap dbg shape that still exercises the scan/conv kernels fully).
BATCH, D_INNER, SEQLEN, D_STATE, D_CONV = 2, 512, 2048, 16, 4
D_MODEL = 256
DEV = "cuda"


def _version(dist: str) -> str:
    try:
        import importlib.metadata as md

        return md.version(dist)
    except Exception as e:  # noqa: BLE001
        return f"ABSENT ({type(e).__name__})"


def _err_vs(a: torch.Tensor, truth: torch.Tensor) -> float:
    return (a.double() - truth).abs().max().item()


def _ratio_ok(fused_err: float, naive_err: float, floor: float) -> bool:
    return fused_err <= K_SLACK * naive_err + floor


def _compare(label: str, truth, naive, fused, floor: float) -> dict:
    """Per-tensor fp64-ground-truth ratio verdict, carrying the raw magnitudes."""
    fe, ne = _err_vs(fused, truth), _err_vs(naive, truth)
    return {label: {"ok": _ratio_ok(fe, ne, floor), "fused_err": fe, "naive_err": ne,
                    "truth_absmax": truth.abs().max().item()}}


# ---- selective_scan: fp64 truth vs naive(dtype) vs fused(dtype) ----------------------
def _scan_run(param_dtype: torch.dtype, use_fused: bool, base: dict, g_up: torch.Tensor):
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

    fn = selective_scan_fn if use_fused else selective_scan_ref
    # A/D/delta_bias stay fp32 in real mamba usage (S4D-real params) unless fp64 truth.
    pf = torch.float64 if param_dtype == torch.float64 else torch.float32
    leaf = {
        "u": base["u"].to(param_dtype), "delta": base["delta"].to(param_dtype),
        "A": base["A"].to(pf), "B": base["B"].to(param_dtype), "C": base["C"].to(param_dtype),
        "D": base["D"].to(pf), "z": base["z"].to(param_dtype), "delta_bias": base["delta_bias"].to(pf),
    }
    for t in leaf.values():
        t.requires_grad_(True)
    out = fn(leaf["u"], leaf["delta"], leaf["A"], leaf["B"], leaf["C"], leaf["D"],
             z=leaf["z"], delta_bias=leaf["delta_bias"], delta_softplus=True)
    out.backward(g_up.to(out.dtype))
    return out.detach(), {k: v.grad.detach() for k, v in leaf.items()}


def selective_scan_check(dtype: torch.dtype, floor: float) -> dict:
    torch.manual_seed(0)
    base = {
        "u": torch.randn(BATCH, D_INNER, SEQLEN, device=DEV, dtype=torch.float64),
        "delta": torch.rand(BATCH, D_INNER, SEQLEN, device=DEV, dtype=torch.float64),
        "A": -torch.rand(D_INNER, D_STATE, device=DEV, dtype=torch.float64),  # S4D-real: A<0
        "B": torch.randn(BATCH, D_STATE, SEQLEN, device=DEV, dtype=torch.float64),
        "C": torch.randn(BATCH, D_STATE, SEQLEN, device=DEV, dtype=torch.float64),
        "D": torch.randn(D_INNER, device=DEV, dtype=torch.float64),
        "z": torch.randn(BATCH, D_INNER, SEQLEN, device=DEV, dtype=torch.float64),
        "delta_bias": torch.randn(D_INNER, device=DEV, dtype=torch.float64),
    }
    g_up = torch.randn(BATCH, D_INNER, SEQLEN, device=DEV, dtype=torch.float64)

    t_out, t_grad = _scan_run(torch.float64, False, base, g_up)  # truth
    n_out, n_grad = _scan_run(dtype, False, base, g_up)  # naive (ref @ dtype)
    f_out, f_grad = _scan_run(dtype, True, base, g_up)  # fused kernel @ dtype

    out = {"dtype": str(dtype), "K_slack": K_SLACK, "floor": floor}
    out.update(_compare("forward", t_out, n_out, f_out, floor))
    grads, grad_all_ok = {}, True
    for name in ("u", "delta", "A", "B", "C", "D", "z", "delta_bias"):
        c = _compare(name, t_grad[name], n_grad[name], f_grad[name], floor)
        grads.update(c)
        grad_all_ok = grad_all_ok and c[name]["ok"]
    out["backward"] = grads
    out["ok"] = bool(out["forward"]["ok"] and grad_all_ok)
    return out


# ---- causal_conv1d: fp64 truth vs naive(dtype) vs fused(dtype) ------------------------
def _conv_run(dtype: torch.dtype, use_fused: bool, base: dict, g_up: torch.Tensor):
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref

    fn = causal_conv1d_fn if use_fused else causal_conv1d_ref
    leaf = {k: base[k].to(dtype).requires_grad_(True) for k in ("x", "w", "b")}
    out = fn(leaf["x"], leaf["w"], leaf["b"], activation="silu")
    out.backward(g_up.to(out.dtype))
    return out.detach(), {k: v.grad.detach() for k, v in leaf.items()}


def causal_conv1d_check(dtype: torch.dtype, floor: float) -> dict:
    torch.manual_seed(1)
    base = {
        "x": torch.randn(BATCH, D_INNER, SEQLEN, device=DEV, dtype=torch.float64),
        "w": torch.randn(D_INNER, D_CONV, device=DEV, dtype=torch.float64),
        "b": torch.randn(D_INNER, device=DEV, dtype=torch.float64),
    }
    g_up = torch.randn(BATCH, D_INNER, SEQLEN, device=DEV, dtype=torch.float64)
    t_out, t_grad = _conv_run(torch.float64, False, base, g_up)
    n_out, n_grad = _conv_run(dtype, False, base, g_up)
    f_out, f_grad = _conv_run(dtype, True, base, g_up)

    out = {"dtype": str(dtype), "K_slack": K_SLACK, "floor": floor}
    out.update(_compare("forward", t_out, n_out, f_out, floor))
    grads, grad_all_ok = {}, True
    for name, leaf in (("x", "x"), ("weight", "w"), ("bias", "b")):
        c = _compare(name, t_grad[leaf], n_grad[leaf], f_grad[leaf], floor)
        grads.update(c)
        grad_all_ok = grad_all_ok and c[name]["ok"]
    out["backward"] = grads
    out["ok"] = bool(out["forward"]["ok"] and grad_all_ok)
    return out


def mamba_module_check(dtype: torch.dtype) -> dict:
    """The full Mamba block at the scaffold width: forward + backward run, all grads
    finite — the block the bake-off would actually train, end to end on A100."""
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


def _guarded(fn, *args) -> dict:
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1800:]}


def main() -> None:
    report_path = sys.argv[1] if len(sys.argv) > 1 else "mamba-gpu-half-probe.json"
    rep: dict = {
        "probe": "mamba-gpu-half verify-before-lock (step b)",
        "scope": "fused CUDA kernel numerics vs fp64 ground-truth (ratio test), fwd+bwd, A100",
        "criterion": "fused_err <= K*naive_err + floor vs fp64 reference truth (K=3)",
        "locked_torch_required": LOCKED_TORCH,
        "torch_version": torch.__version__,
        "torch_matches_lock": torch.__version__ == LOCKED_TORCH,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": {p: _version(p) for p in ("torch", "triton", "mamba-ssm", "causal-conv1d")},
        "shape": {"batch": BATCH, "d_inner": D_INNER, "seqlen": SEQLEN, "d_state": D_STATE,
                  "d_conv": D_CONV, "d_model": D_MODEL},
    }

    # fp32: tight floor; bf16 (the training dtype): coarser floor (bf16 ~3 sig digits).
    rep["selective_scan_fp32"] = _guarded(selective_scan_check, torch.float32, 1e-4)
    rep["selective_scan_bf16"] = _guarded(selective_scan_check, torch.bfloat16, 5e-2)
    rep["causal_conv1d_fp32"] = _guarded(causal_conv1d_check, torch.float32, 1e-4)
    rep["causal_conv1d_bf16"] = _guarded(causal_conv1d_check, torch.bfloat16, 5e-2)
    rep["mamba_module_bf16"] = _guarded(mamba_module_check, torch.bfloat16)

    rep["verdict_PASS"] = bool(
        rep["torch_matches_lock"]
        and rep["cuda_available"]
        and rep["selective_scan_fp32"].get("ok")
        and rep["selective_scan_bf16"].get("ok")
        and rep["causal_conv1d_fp32"].get("ok")
        and rep["causal_conv1d_bf16"].get("ok")
        and rep["mamba_module_bf16"].get("ok")
    )

    Path(report_path).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()

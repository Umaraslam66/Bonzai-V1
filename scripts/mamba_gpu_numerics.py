"""mamba-ssm verify-before-lock — GPU NUMERICS half (Phase-2 bake-off Task 5 / step b).

The CPU half (W4) proved the locked stack builds + imports. THIS is the actual
verify-before-lock verdict: do the fused CUDA kernels (`selective_scan_fn`,
`causal_conv1d_fn`) compute the SAME thing as mamba-ssm's pure-PyTorch reference
ops (`selective_scan_ref`, `causal_conv1d_ref`) on an A100, forward AND backward,
to a stated tolerance? "It ran without crashing" is NOT the verdict — fused-kernel
numerics can be silently wrong, which would corrupt a whole bake-off backbone.

References (the package's own, what mamba-ssm's test suite compares against):
- `mamba_ssm.ops.selective_scan_interface.selective_scan_ref`  (pure PyTorch)
- `causal_conv1d.causal_conv1d_ref`                            (pure PyTorch)

Tolerances:
- fp32 (correctness): rtol=1e-3, atol=1e-4 (mamba's own test convention).
- bf16 (the TRAINING dtype, bfloat16 on A100): rtol=3e-2, atol=3e-2 — bf16 has ~3
  decimal digits; this confirms the training-dtype path is sane, not bit-exact.

Also asserts torch is STILL 2.5.1+cu121 at runtime (the W4 torch_matches_lock
tooth — proves the GPU env did not drift the version).

Each check runs independently and records its own result (one failure cannot hide
another); the JSON report is the product. Run under the locked probe venv with the
three candidate-pin preconditions (gcc/12.2.0 toolchain, no-install, gcc-12
libstdc++ preloaded) — see scripts/probe_mamba_gpu_half.sh.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import torch

LOCKED_TORCH = "2.5.1+cu121"

# Representative mamba-hybrid shape at the scaffold scale (ScaffoldConfig d_model=256,
# mamba expand=2 -> d_inner=512; d_state=16 standard; seqlen 2048 a real cell length;
# batch 2 — small, cheap dbg shape that still exercises the scan/conv kernels fully).
BATCH, D_INNER, SEQLEN, D_STATE, D_CONV = 2, 512, 2048, 16, 4
D_MODEL = 256


def _version(dist: str) -> str:
    try:
        import importlib.metadata as md

        return md.version(dist)
    except Exception as e:  # noqa: BLE001
        return f"ABSENT ({type(e).__name__})"


def _allclose(a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float) -> tuple[bool, float]:
    a32, b32 = a.float(), b.float()
    max_abs = (a32 - b32).abs().max().item()
    return bool(torch.allclose(a32, b32, rtol=rtol, atol=atol)), max_abs


def selective_scan_check(dtype: torch.dtype, rtol: float, atol: float) -> dict:
    """Fused CUDA selective_scan_fn vs pure-PyTorch selective_scan_ref: fwd out +
    every input gradient, within tolerance."""
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

    torch.manual_seed(0)
    dev = "cuda"
    # Build inputs ONCE, then two independent leaf copies so fused/ref grads compare.
    u0 = torch.randn(BATCH, D_INNER, SEQLEN, device=dev, dtype=dtype)
    delta0 = torch.rand(BATCH, D_INNER, SEQLEN, device=dev, dtype=dtype)  # >0; softplus'd anyway
    A0 = -torch.rand(D_INNER, D_STATE, device=dev, dtype=torch.float32)  # S4D-real: A<0, fp32
    B0 = torch.randn(BATCH, D_STATE, SEQLEN, device=dev, dtype=dtype)  # variable B (ngroups=1)
    C0 = torch.randn(BATCH, D_STATE, SEQLEN, device=dev, dtype=dtype)
    D0 = torch.randn(D_INNER, device=dev, dtype=torch.float32)
    z0 = torch.randn(BATCH, D_INNER, SEQLEN, device=dev, dtype=dtype)
    db0 = torch.randn(D_INNER, device=dev, dtype=torch.float32)
    names = ["u", "delta", "A", "B", "C", "D", "z", "delta_bias"]

    def leaves() -> list[torch.Tensor]:
        return [t.detach().clone().requires_grad_(True) for t in (u0, delta0, A0, B0, C0, D0, z0, db0)]

    fused = leaves()
    ref = leaves()
    out_f = selective_scan_fn(
        fused[0], fused[1], fused[2], fused[3], fused[4],
        fused[5], z=fused[6], delta_bias=fused[7], delta_softplus=True,
    )
    out_r = selective_scan_ref(
        ref[0], ref[1], ref[2], ref[3], ref[4],
        ref[5], z=ref[6], delta_bias=ref[7], delta_softplus=True,
    )
    fwd_ok, fwd_max = _allclose(out_f, out_r, rtol, atol)

    g = torch.randn_like(out_f)
    out_f.backward(g)
    out_r.backward(g.clone())
    grads: dict[str, dict] = {}
    grad_all_ok = True
    for name, tf, tr in zip(names, fused, ref, strict=True):
        if tf.grad is None or tr.grad is None:
            grads[name] = {"ok": tf.grad is None and tr.grad is None, "note": "no grad both" if tf.grad is None and tr.grad is None else "grad MISSING on one side"}
            grad_all_ok = grad_all_ok and grads[name]["ok"]
            continue
        ok, mx = _allclose(tf.grad, tr.grad, rtol, atol)
        grads[name] = {"ok": ok, "max_abs_err": mx}
        grad_all_ok = grad_all_ok and ok

    return {
        "dtype": str(dtype), "rtol": rtol, "atol": atol,
        "forward": {"ok": fwd_ok, "max_abs_err": fwd_max},
        "backward": grads, "backward_all_ok": grad_all_ok,
        "ok": fwd_ok and grad_all_ok,
    }


def causal_conv1d_check(dtype: torch.dtype, rtol: float, atol: float) -> dict:
    """Fused causal_conv1d_fn vs pure-PyTorch causal_conv1d_ref (the conv1d half of the
    mamba block): fwd + grads within tolerance."""
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_ref

    torch.manual_seed(1)
    dev = "cuda"
    x0 = torch.randn(BATCH, D_INNER, SEQLEN, device=dev, dtype=dtype)
    w0 = torch.randn(D_INNER, D_CONV, device=dev, dtype=dtype)
    b0 = torch.randn(D_INNER, device=dev, dtype=dtype)

    def leaves() -> list[torch.Tensor]:
        return [t.detach().clone().requires_grad_(True) for t in (x0, w0, b0)]

    fused = leaves()
    ref = leaves()
    out_f = causal_conv1d_fn(fused[0], fused[1], fused[2], activation="silu")
    out_r = causal_conv1d_ref(ref[0], ref[1], ref[2], activation="silu")
    fwd_ok, fwd_max = _allclose(out_f, out_r, rtol, atol)

    g = torch.randn_like(out_f)
    out_f.backward(g)
    out_r.backward(g.clone())
    grads = {}
    grad_all_ok = True
    for name, tf, tr in zip(["x", "weight", "bias"], fused, ref, strict=True):
        ok, mx = _allclose(tf.grad, tr.grad, rtol, atol)
        grads[name] = {"ok": ok, "max_abs_err": mx}
        grad_all_ok = grad_all_ok and ok

    return {
        "dtype": str(dtype), "rtol": rtol, "atol": atol,
        "forward": {"ok": fwd_ok, "max_abs_err": fwd_max},
        "backward": grads, "backward_all_ok": grad_all_ok,
        "ok": fwd_ok and grad_all_ok,
    }


def mamba_module_check(dtype: torch.dtype) -> dict:
    """The full Mamba block on GPU at the scaffold width: forward + backward run, all
    grads finite. The production fused path (use_fast_path default) — proves the block
    the bake-off would train actually executes end to end on A100."""
    from mamba_ssm.modules.mamba_simple import Mamba

    torch.manual_seed(2)
    dev = "cuda"
    m = Mamba(d_model=D_MODEL, d_state=D_STATE, d_conv=D_CONV, expand=2).to(dev).to(dtype)
    x = torch.randn(BATCH, SEQLEN, D_MODEL, device=dev, dtype=dtype, requires_grad=True)
    y = m(x)
    loss = y.float().pow(2).mean()
    loss.backward()
    n_params = sum(p.numel() for p in m.parameters())
    grads_finite = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in m.parameters())
    x_grad_finite = x.grad is not None and torch.isfinite(x.grad).all().item()
    return {
        "dtype": str(dtype), "n_params": int(n_params),
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
        "scope": "fused CUDA kernel numerics vs pure-PyTorch reference, fwd+bwd, on A100",
        "locked_torch_required": LOCKED_TORCH,
        "torch_version": torch.__version__,
        "torch_matches_lock": torch.__version__ == LOCKED_TORCH,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": {p: _version(p) for p in ("torch", "triton", "mamba-ssm", "causal-conv1d")},
        "shape": {"batch": BATCH, "d_inner": D_INNER, "seqlen": SEQLEN, "d_state": D_STATE,
                  "d_conv": D_CONV, "d_model": D_MODEL},
    }

    rep["selective_scan_fp32"] = _guarded(selective_scan_check, torch.float32, 1e-3, 1e-4)
    rep["selective_scan_bf16"] = _guarded(selective_scan_check, torch.bfloat16, 3e-2, 3e-2)
    rep["causal_conv1d_fp32"] = _guarded(causal_conv1d_check, torch.float32, 1e-3, 1e-4)
    rep["mamba_module_bf16"] = _guarded(mamba_module_check, torch.bfloat16)

    # VERDICT — teeth: torch held the lock, fp32 kernel numerics correct (fwd+bwd),
    # and the module trains. bf16 is informational (training-dtype sanity), not a hard
    # gate (bf16 noise is expected), but flagged if it blows past the loose tolerance.
    rep["verdict_PASS"] = bool(
        rep["torch_matches_lock"]
        and rep["cuda_available"]
        and rep["selective_scan_fp32"].get("ok")
        and rep["causal_conv1d_fp32"].get("ok")
        and rep["mamba_module_bf16"].get("ok")
    )
    rep["bf16_scan_within_loose_tol"] = rep["selective_scan_bf16"].get("ok")

    Path(report_path).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()

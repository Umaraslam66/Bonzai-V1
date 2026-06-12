#!/bin/bash
# W4 (GPU-wait CPU drain): mamba-lock CPU-VERIFIABLE HALF.
#
# The mamba-ssm verify-before-lock (bake-off design Task 2 / "mamba-lock") needs
# import + forward/backward on a GPU under the EXACT locked stack before the lock
# lands in env_lock.py. This probe does ONLY the CPU-verifiable half:
#   1. fresh ISOLATED venv (never the repo .venv — that is the bake-off
#      comparability lock; a guard below refuses to run inside it);
#   2. install torch==2.5.1+cu121 (the exact locked build) + companions;
#   3. pip-resolve causal-conv1d and mamba-ssm against that stack, recording
#      resolved versions and whether each came as a prebuilt CUDA wheel or an
#      nvcc source build (MAX_JOBS bounded for login-class memory limits);
#   4. import-test on the CPU node, recording the exact outcome (a clean import,
#      or the precise error classified for the GPU half to confirm);
#   5. write a candidate-pin report (JSON) into the repo's reports/ dir.
#
# PARKED (GPU half, the actual verify-before-lock verdict): kernel
# forward/backward numerics; the env_lock.py pin extension; building the
# mamba-hybrid backbone (stays behind BackboneNotYetBuilt until the verdict).
# If anything here forces a torch/CUDA change: that is a RE-LOCK-ALL design
# finding -> HALT and bring to Umar; never work around it.

set -uo pipefail  # NOT -e: each step records its rc; the report is the product

ENVDIR=/leonardo_work/AIFAC_P02_222/envs/mamba-cpu-probe
REPO=/leonardo_work/AIFAC_P02_222/Bonzai-OSM
REPORT=$REPO/reports/2026-06-12-mamba-cpu-half-probe.json
LOG_PREFIX="mamba-cpu-half"

# gcc/12.2.0: run 46014130 proved the default RHEL-8 gcc 8.5 CANNOT compile
# against torch 2.5.1 headers ("We need GCC 9 or later"); 12.2.0 is the newest
# module and the max host compiler CUDA 12.2 supports. NOT a locked-stack
# change (gcc is toolchain, not env_lock) — recorded in the report.
module load python/3.11.7 cuda/12.2 gcc/12.2.0
export CC=$(command -v gcc) CXX=$(command -v g++) CUDAHOSTCXX=$(command -v g++)
echo "toolchain: gcc=$($CC --version | head -1) nvcc-host=$CUDAHOSTCXX"

# --- guard: never the locked .venv ---------------------------------------------------
if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "$LOG_PREFIX: refusing to run inside an active venv ($VIRTUAL_ENV)"; exit 64
fi
case "$ENVDIR" in
  "$REPO"/*) echo "$LOG_PREFIX: ENVDIR inside the repo — refusing"; exit 64;;
esac

echo "$LOG_PREFIX: fresh venv at $ENVDIR"
rm -rf "$ENVDIR"
python -m venv "$ENVDIR"
source "$ENVDIR/bin/activate"
python -m pip install -q -U pip setuptools wheel packaging ninja || { echo "pip bootstrap FAILED"; exit 65; }

# --- step 1: the exact locked torch --------------------------------------------------
echo "$LOG_PREFIX: installing torch==2.5.1+cu121 (the locked build) + numpy"
pip install "torch==2.5.1+cu121" numpy --index-url https://download.pytorch.org/whl/cu121 \
  --extra-index-url https://pypi.org/simple 2>&1 | tail -3
TORCH_RC=$?

# --- step 2+3: companions, resolution + build mode recorded --------------------------
# CONSTRAINT TOOTH (run 46019067 lesson): unconstrained, pip UPGRADED torch to
# 2.12.0+cu130 to satisfy the LATEST companions (causal-conv1d 1.6.2 / mamba-ssm
# 2.3.2) -> ABI skew (undefined c10 symbol), caught by torch_matches_lock. The
# locked torch is a hard constraint: pip must BACKTRACK to companion versions
# that accept 2.5.1, or fail loudly — never move the lock.
CONSTRAINTS=/tmp/lock-constraints.$$.txt
printf 'torch==2.5.1+cu121\ntriton==3.1.0\n' > "$CONSTRAINTS"
export MAX_JOBS=2  # bound nvcc parallelism for login-class memory limits
echo "$LOG_PREFIX: installing causal-conv1d (constrained to the locked torch)"
pip install causal-conv1d -c "$CONSTRAINTS" --no-build-isolation \
  2>&1 | tee /tmp/cc1d.$$.log | tail -5
CC1D_RC=$?
echo "$LOG_PREFIX: installing mamba-ssm (constrained to the locked torch)"
pip install mamba-ssm -c "$CONSTRAINTS" --no-build-isolation \
  2>&1 | tee /tmp/mamba.$$.log | tail -5
MAMBA_RC=$?

# build-mode evidence: a prebuilt CUDA wheel shows a +cu..torch.. local tag or a
# direct GitHub-releases wheel URL in the pip log; an nvcc build shows
# "Building wheel" + nvcc invocations.
CC1D_MODE=$(grep -oE "Downloading .*causal_conv1d.*whl|Building wheel for causal[-_]conv1d" /tmp/cc1d.$$.log | head -1)
MAMBA_MODE=$(grep -oE "Downloading .*mamba_ssm.*whl|Building wheel for mamba[-_]ssm" /tmp/mamba.$$.log | head -1)
# prebuilt-wheel fetch evidence: the setup.py "guess wheel URL" + any 404 (run
# 46014130 saw a 404 -> source-build fallback; record the URL it tried)
WHEEL_FETCH=$(grep -hoE "https://github.com[^ '\"]*\.whl|HTTP Error 404[^\"]*" /tmp/cc1d.$$.log /tmp/mamba.$$.log | sort -u | head -4 | tr '\n' ' ')

# --- step 4: import test on this CPU node + step 5: the report -----------------------
# RUNTIME REQUIREMENT (run 46024139 lesson): the .so files are compiled with gcc
# 12.2 and need GLIBCXX_3.4.29, but the spack python's library chain resolves
# libstdc++.so.6 from the gcc-8.5 RUNTIME -> ImportError. The gcc-12 libstdc++
# must be visible at import (LD_PRELOAD here; production sbatch entry points for
# mamba runs carry the same requirement — part of the candidate pin).
GCC12_LIBSTDCXX="$(dirname "$($CC -print-file-name=libstdc++.so.6)")/libstdc++.so.6"
export MAMBA_IMPORT_PRELOAD="$GCC12_LIBSTDCXX"
python - "$REPORT" "$TORCH_RC" "$CC1D_RC" "$MAMBA_RC" "$CC1D_MODE" "$MAMBA_MODE" "${WHEEL_FETCH:-none}" "$($CC --version | head -1)" <<'PY'
import json, subprocess, sys, traceback
from pathlib import Path

report_path, torch_rc, cc1d_rc, mamba_rc, cc1d_mode, mamba_mode = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
    sys.argv[5], sys.argv[6],
)
wheel_fetch, gcc_version = sys.argv[7], sys.argv[8]

def ver(dist):
    try:
        import importlib.metadata as md
        return md.version(dist)
    except Exception as e:
        return f"ABSENT ({type(e).__name__})"

def try_import(stmt):
    """Import in a SUBPROCESS so one segfaulting .so cannot kill the report.
    LD_PRELOAD carries the gcc-12 libstdc++ (GLIBCXX_3.4.29 runtime requirement)."""
    import os
    env = dict(os.environ)
    preload = env.get("MAMBA_IMPORT_PRELOAD", "")
    if preload:
        env["LD_PRELOAD"] = preload
    r = subprocess.run([sys.executable, "-c", stmt], capture_output=True, text=True,
                       timeout=300, env=env)
    return {"ok": r.returncode == 0, "rc": r.returncode,
            "stdout": r.stdout.strip()[-400:], "stderr": r.stderr.strip()[-600:]}

nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
report = {
    "probe": "mamba-cpu-half (W4)",
    "scope": "CPU-verifiable half ONLY; GPU fwd/bwd = the actual verify-before-lock verdict, PARKED",
    "locked_torch_required": "2.5.1+cu121",
    "install_rcs": {"torch": torch_rc, "causal_conv1d": cc1d_rc, "mamba_ssm": mamba_rc},
    "build_mode": {"causal_conv1d": cc1d_mode or "UNKNOWN (see log)",
                   "mamba_ssm": mamba_mode or "UNKNOWN (see log)"},
    "wheel_fetch_evidence": wheel_fetch,
    "host_compiler": gcc_version,
    "runtime_requirement": {
        "glibcxx": "GLIBCXX_3.4.29 (gcc-12 libstdc++) must be visible at import",
        "preload_used": __import__("os").environ.get("MAMBA_IMPORT_PRELOAD", ""),
    },
    "resolved_versions": {
        "torch": ver("torch"), "triton": ver("triton"),
        "causal-conv1d": ver("causal-conv1d"), "mamba-ssm": ver("mamba-ssm"),
    },
    "nvcc": (nvcc.stdout.splitlines()[-1] if nvcc.returncode == 0 else "ABSENT"),
    "imports": {
        "torch_matches_lock": try_import(
            "import torch; assert torch.__version__=='2.5.1+cu121', torch.__version__; print(torch.__version__)"
        ),
        "causal_conv1d": try_import("import causal_conv1d; print(causal_conv1d.__version__)"),
        "mamba_ssm": try_import("import mamba_ssm; print(mamba_ssm.__version__)"),
        "mamba_module_cpu_construct": try_import(
            "from mamba_ssm.modules.mamba_simple import Mamba; import torch; "
            "m = Mamba(d_model=64); print('constructed', sum(p.numel() for p in m.parameters()))"
        ),
    },
}
ok = (
    report["imports"]["torch_matches_lock"]["ok"]
    and torch_rc == 0 and cc1d_rc == 0 and mamba_rc == 0
    and report["imports"]["mamba_ssm"]["ok"]
)
report["cpu_half_verdict"] = "RESOLVES" if ok else "BLOCKED (HALT -> Umar; possible re-lock-all)"
Path(report_path).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
PY
PY_RC=$?
echo "$LOG_PREFIX: report rc=$PY_RC -> $REPORT"
exit $PY_RC

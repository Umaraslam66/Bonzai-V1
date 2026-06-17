# Phase-2 Bake-off (2-backbone) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the 2-backbone (`transformer-ar` + `mamba-hybrid`) architecture bake-off — build the swappable-mixer scaffold and the mamba backbone, fix the toolchain, run the diagnostic to set the feasible matrix, then run the scored comparison to a verdict.

**Architecture:** Extract `MicroAR`'s shared scaffold (embedding / conditioning prefix / char-carrier / head / AR loss / AR generation) into a base class with a swappable `_mix` method; `transformer-ar` keeps its `TransformerEncoder` mixer (bit-identical), `mamba-hybrid` plugs in a 7:1 Jamba-style Mamba/transformer interleave at **param-matched** scales. Toolchain fix (gcc/12.2.0) unblocks the compiled eval path. The Task-1 diagnostic (on the existing transformer-ar) sets the feasible scale ladder + eval-budget projection, gating the scored runs.

**Tech Stack:** PyTorch 2.5.1+cu121, `mamba-ssm` 2.3.1 + `causal-conv1d` 1.6.2.post1 (locked, GPU-verified 2026-06-17), Lightning 2.6.5, pytest, Slurm on Leonardo (`AIFAC_P02_548`).

**Spec:** `docs/superpowers/specs/2026-06-17-phase-2-bakeoff-2backbone-delta-design.md` (delta on the locked `2026-06-02` + `2026-06-09` bake-off specs). **Protocol:** `docs/protocols/sub-project-planning-protocol-v3.md`.

**Branch:** `phase-2-bakeoff-2backbone` (already open; spec committed). Commit task-by-task; no merge/push/GPU-run/scored-run without Umar's per-step word + `--no-ff`.

**Dependency graph:** Phase 1 (sbatch fix) gates Phases 2 & 3. **Phase 2 (mamba build) and Phase 3 (diagnostic on transformer-ar) run in PARALLEL** — no dependency. Phase 4 (scored) needs Phase 2 + Phase 3, and is GATED on the diagnostic's feasible-ladder + budget output (a re-plan checkpoint, not pre-pinned here).

---

## Phase 1 — Shared sbatch toolchain fix (no GPU)

### Task 1: gcc/12.2.0 on the compiled scored path

**Files:**
- Modify: `scripts/bakeoff_run.sbatch` (the `module load` line, ~line 34)
- Test: `tests/training/test_cli_contract.py` (append a content-contract assertion)

- [ ] **Step 1: Write the failing content-contract test**

```python
def test_bakeoff_run_sbatch_loads_gcc12_and_preloads_libstdcxx():
    """The compiled scored path (torch.compile inductor CPU codegen) needs gcc/12.2.0,
    not the RHEL-8 gcc-8.5 that crashed eval at Step 18.5; and the mamba run needs the
    gcc-12 libstdc++ at import. Shared run sbatch (parameterized by --backbone), so one
    fix serves both backbones; the LD_PRELOAD is harmless for transformer-ar."""
    text = Path("scripts/bakeoff_run.sbatch").read_text()
    assert "module load python/3.11.7 cuda/12.2 gcc/12.2.0" in text
    assert "export CC=" in text and "CXX=" in text and "CUDAHOSTCXX=" in text
    assert "LD_PRELOAD" in text and "libstdc++.so.6" in text
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/training/test_cli_contract.py::test_bakeoff_run_sbatch_loads_gcc12_and_preloads_libstdcxx -q`
Expected: FAIL (sbatch loads only `cuda/12.2`).

- [ ] **Step 3: Edit the sbatch** — change the module-load line and add the toolchain exports right after it (mirroring `scripts/probe_mamba_gpu_half.sh` which is GPU-verified):

```bash
module load python/3.11.7 cuda/12.2 gcc/12.2.0
export CC=$(command -v gcc) CXX=$(command -v g++) CUDAHOSTCXX=$(command -v g++)
export LD_PRELOAD="$(dirname "$($CC -print-file-name=libstdc++.so.6)")/libstdc++.so.6"
```

- [ ] **Step 4: Run the full cli-contract file, verify green**

Run: `uv run pytest tests/training/test_cli_contract.py -q` (Mac note: if torch-absent collection errors, run on Leonardo `.venv`; the new assertion is pure text and runs anywhere the module imports).
Expected: PASS (incl. the existing sbatch-content assertions).

- [ ] **Step 5: Dry-verify on Leonardo (NO real submit) — Umar's word to touch the cluster**

Deploy the branch (git bundle), then:
Run: `ssh leonardo 'cd $REPO && sbatch --test-only scripts/bakeoff_run.sbatch'` with the run's env vars set.
Expected: `Job ... to start at ...` (account/QOS/partition valid), NOT an account/module error. (`--test-only` submits nothing, costs nothing.)

- [ ] **Step 6: Commit**

```bash
git add scripts/bakeoff_run.sbatch tests/training/test_cli_contract.py
git commit -m "fix(orchestration): gcc/12.2.0 + libstdc++ preload on the compiled scored path (18.5 + mamba precond)"
```

> Note: `bakeoff_diagnostic.sbatch` runs `--no-compile` and cannot hit the inductor crash; it gets the gcc load **precautionarily** in Phase 3 Task 8, not here.

---

## Phase 2 — Backbone scaffold refactor + mamba-hybrid build + smoke

### Task 2: Capture the behavior-preservation golden tensors (Gate-6, BEFORE refactor)

**Files:**
- Test: `tests/models/test_micro_ar_behavior_preserved.py` (create)
- Create: `tests/models/_golden/` (golden tensors committed as a fixture)

- [ ] **Step 1: Write a test that records pre-refactor MicroAR outputs as golden**

```python
# Captured against the CURRENT (pre-refactor) MicroAR. After the Task-3 refactor,
# refactored MicroAR must reproduce these BIT-IDENTICALLY (external-source-of-truth,
# protocol Gate-6 — assert against the old module's real output, not its description).
import torch
from cfm.models.micro_ar import MicroAR, MicroARConfig

def _fixed_model_and_input():
    torch.manual_seed(1234)
    cfg = MicroARConfig(d_model=32, n_layers=2, n_heads=2, n_subf_vocab=40, n_cond=16,
                        max_len=24, n_char_stats=7, char_position=9)
    m = MicroAR(cfg).eval()
    ids = torch.randint(0, 40 + 16, (2, 12))
    char = torch.randn(2, 7)
    prefix_len = torch.tensor([10, 10]); seq_len = torch.tensor([12, 11])
    return m, ids, char, prefix_len, seq_len

def test_micro_ar_forward_and_loss_match_golden():
    m, ids, char, pl, sl = _fixed_model_and_input()
    with torch.no_grad():
        logits = m(ids, char_stats=char)
        loss = m.training_loss(ids, prefix_len=pl, seq_len=sl, char_stats=char).loss
    golden = Path("tests/models/_golden/micro_ar_v1.pt")
    if not golden.exists():  # one-time capture, then committed + frozen
        torch.save({"logits": logits, "loss": loss}, golden)
    g = torch.load(golden)
    assert torch.equal(logits, g["logits"])
    assert torch.equal(loss, g["loss"])
```

- [ ] **Step 2: Run it once to capture + commit the golden**

Run: `uv run pytest tests/models/test_micro_ar_behavior_preserved.py -q` (on Leonardo `.venv` — needs torch). First run writes `micro_ar_v1.pt`; re-run asserts equality (PASS).

- [ ] **Step 3: Commit the golden + test**

```bash
git add tests/models/test_micro_ar_behavior_preserved.py tests/models/_golden/micro_ar_v1.pt
git commit -m "test(models): freeze pre-refactor MicroAR forward+loss as Gate-6 golden"
```

### Task 3: Extract the shared scaffold base (`_mix` seam), MicroAR delegates

**Files:**
- Create: `src/cfm/models/scaffold_backbone.py`
- Modify: `src/cfm/models/micro_ar.py` (MicroAR inherits the base; `_mix` = TransformerEncoder)
- Test: `tests/models/test_micro_ar_behavior_preserved.py` (already green target), `tests/models/test_micro_ar.py` (stays green)

- [ ] **Step 1: Write the base with the shared scaffold + abstract `_mix`**

```python
# src/cfm/models/scaffold_backbone.py
from __future__ import annotations
import torch
from torch import nn

_IGNORE = -100

class ScaffoldBackbone(nn.Module):
    """Shared bake-off scaffold: embedding (sub-F vocab + conditioning id-block),
    positions, the Task-24b char-carrier, the sub-F head, AR loss, AR generation.
    Subclasses implement ONE thing — `_mix` (the sequence-mixing layers). Everything
    else is shared BY IDENTITY so backbones differ only in the mixer (spec §3, §8)."""

    def __init__(self, *, d_model, n_subf_vocab, n_cond, max_len, n_char_stats=0,
                 char_position=None, dropout=0.0) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_subf_vocab + n_cond, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.char_proj = nn.Linear(n_char_stats, d_model) if n_char_stats > 0 else None
        self.head = nn.Linear(d_model, n_subf_vocab)
        self._n_char_stats, self._char_position = n_char_stats, char_position

    def _mix(self, x: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError  # subclass: the only divergence point

    def _input_embeddings(self, ids, char_stats):
        # EXACT lines lifted from MicroAR.forward (char-overwrite + additive position),
        # so refactored transformer-ar is bit-identical (Gate-6 golden test).
        ...  # (move the embed/char/pos block here verbatim)

    def forward(self, ids, char_stats=None):
        x = self._input_embeddings(ids, char_stats)
        t = ids.shape[1]
        causal = nn.Transformer.generate_square_subsequent_mask(t, device=ids.device)
        return self.head(self._mix(x, causal))

    def training_loss(self, ids, *, prefix_len, seq_len=None, char_stats=None):
        ...  # (move MicroAR.training_loss here verbatim — calls self(ids, char_stats=...))
```

- [ ] **Step 2: Make MicroAR a thin subclass**

```python
# src/cfm/models/micro_ar.py — MicroARConfig unchanged; MicroAR now:
class MicroAR(ScaffoldBackbone):
    def __init__(self, cfg: MicroARConfig) -> None:
        super().__init__(d_model=cfg.d_model, n_subf_vocab=cfg.n_subf_vocab,
                         n_cond=cfg.n_cond, max_len=cfg.max_len,
                         n_char_stats=cfg.n_char_stats, char_position=cfg.char_position,
                         dropout=cfg.dropout)
        self.cfg = cfg
        layer = nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=4*cfg.d_model, dropout=cfg.dropout, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)

    def _mix(self, x, causal):
        return self.blocks(x, mask=causal, is_causal=True)
```

- [ ] **Step 3: Run the golden + existing MicroAR tests, verify ALL green**

Run (Leonardo `.venv`): `python -m pytest tests/models/test_micro_ar_behavior_preserved.py tests/models/test_micro_ar.py tests/models/test_character_prefix.py tests/models/test_backbone_identity_lock.py -q`
Expected: PASS — refactored MicroAR is **bit-identical** to the golden (the order of module construction is preserved, so RNG-seeded init matches; if it doesn't, the refactor changed construction order — fix until `torch.equal` holds).

- [ ] **Step 4: Commit**

```bash
git add src/cfm/models/scaffold_backbone.py src/cfm/models/micro_ar.py
git commit -m "refactor(models): extract ScaffoldBackbone (_mix seam); MicroAR delegates, behavior-identical (W2-grade)"
```

### Task 4: MambaHybridConfig + the 7:1 interleave mixer

**Files:**
- Create: `src/cfm/models/mamba_hybrid.py`
- Test: `tests/models/test_mamba_hybrid.py` (create — runs only where mamba-ssm is installed; guard with `pytest.importorskip("mamba_ssm")`)

- [ ] **Step 1: Write the failing construction + forward-shape test**

```python
import pytest
pytest.importorskip("mamba_ssm")  # only the locked mamba env (probe venv / future repo env)
import torch
from cfm.models.mamba_hybrid import MambaHybrid, MambaHybridConfig

def test_mamba_hybrid_forward_shape_and_interleave():
    cfg = MambaHybridConfig(d_model=128, n_layers=8, n_heads=4, n_subf_vocab=40,
        n_cond=16, max_len=64, n_char_stats=7, char_position=9, d_state=16, d_conv=4, expand=2)
    m = MambaHybrid(cfg)
    # 7:1 over 8 layers => 7 mamba + 1 transformer; ≥1 transformer always
    assert m.n_transformer_layers >= 1
    assert m.n_mamba_layers + m.n_transformer_layers == cfg.n_layers
    ids = torch.randint(0, 56, (2, 20)); char = torch.randn(2, 7)
    out = m(ids, char_stats=char)
    assert out.shape == (2, 20, 40)  # head → sub-F range
```

- [ ] **Step 2: Run, verify fail** (`ModuleNotFoundError: cfm.models.mamba_hybrid`).

- [ ] **Step 3: Implement the interleave mixer**

```python
# src/cfm/models/mamba_hybrid.py
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from cfm.models.scaffold_backbone import ScaffoldBackbone

@dataclass(frozen=True)
class MambaHybridConfig:
    d_model: int; n_layers: int; n_heads: int; n_subf_vocab: int; n_cond: int; max_len: int
    n_char_stats: int = 0; char_position: int | None = None; dropout: float = 0.0
    d_state: int = 16; d_conv: int = 4; expand: int = 2
    transformer_every: int = 7  # 1 transformer per `transformer_every` mamba (Jamba 7:1)

def _interleave_positions(n_layers: int, every: int) -> list[bool]:
    """True = transformer layer. ≥1 transformer always; place a transformer every `every`
    layers (the last slot of each group)."""
    is_tf = [((i + 1) % (every + 1) == 0) for i in range(n_layers)]
    if not any(is_tf):
        is_tf[-1] = True  # small layer counts: never leave attention absent
    return is_tf

class MambaHybrid(ScaffoldBackbone):
    def __init__(self, cfg: MambaHybridConfig) -> None:
        super().__init__(d_model=cfg.d_model, n_subf_vocab=cfg.n_subf_vocab,
            n_cond=cfg.n_cond, max_len=cfg.max_len, n_char_stats=cfg.n_char_stats,
            char_position=cfg.char_position, dropout=cfg.dropout)
        self.cfg = cfg
        from mamba_ssm.modules.mamba_simple import Mamba
        layout = _interleave_positions(cfg.n_layers, cfg.transformer_every)
        self.n_transformer_layers = sum(layout); self.n_mamba_layers = len(layout) - self.n_transformer_layers
        self._is_tf = layout
        blocks = []
        for is_tf in layout:
            if is_tf:
                blocks.append(nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.n_heads,
                    dim_feedforward=4*cfg.d_model, dropout=cfg.dropout, batch_first=True, norm_first=True))
            else:
                blocks.append(Mamba(d_model=cfg.d_model, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand))
        self.blocks = nn.ModuleList(blocks)

    def _mix(self, x, causal):
        for blk, is_tf in zip(self.blocks, self._is_tf, strict=True):
            x = blk(x, src_mask=causal, is_causal=True) if is_tf else blk(x)  # Mamba is causal by construction
        return x
```

- [ ] **Step 4: Run the test on the mamba env, verify PASS**

Run (probe venv, with the 3 preconditions — see Task 7's sbatch for the exact module/preload): submit a tiny `--test-only`-style CPU/GPU check, OR run inside `scripts/probe_mamba_gpu_half.sh`'s env. Expected: PASS (constructs, 7+1 layout, forward shape `(2,20,40)`).

- [ ] **Step 5: Commit**

```bash
git add src/cfm/models/mamba_hybrid.py tests/models/test_mamba_hybrid.py
git commit -m "feat(models): MambaHybrid — 7:1 Jamba interleave on the shared scaffold"
```

### Task 5: The param-matched scale table — a VERIFIED gate (spec §3.3)

**Files:**
- Create: `src/cfm/models/bakeoff_scales.py` (the per-scale config builders for both backbones)
- Test: `tests/models/test_bakeoff_param_match.py`

- [ ] **Step 1: Write the failing param-match gate test**

```python
import pytest; pytest.importorskip("mamba_ssm")
from cfm.models.bakeoff_scales import build_pair_for_scale, BAKEOFF_SCALES  # {30M,100M,300M,1B}
from cfm.models.micro_ar import MicroAR
from cfm.models.mamba_hybrid import MambaHybrid

def _params(m): return sum(p.numel() for p in m.parameters())

@pytest.mark.parametrize("scale", BAKEOFF_SCALES)
def test_backbones_param_matched_within_tolerance(scale):
    """Architecture must be the ONLY variable: transformer-ar and mamba-hybrid must land
    at the SAME param count per scale (not the same depth) — equal-depth would be
    unequal-capacity (Jamba layers differ in per-layer params), confounding architecture
    with capacity (spec §3.3, §8). Count the ACTUAL built model, not the mapping."""
    tcfg, mcfg = build_pair_for_scale(scale)
    nt, nm = _params(MicroAR(tcfg)), _params(MambaHybrid(mcfg))
    rel = abs(nt - nm) / nt
    assert rel <= 0.02, f"{scale}: transformer {nt:,} vs mamba {nm:,} = {rel:.1%} > 2% tol"
```

- [ ] **Step 2: Run, verify fail** (`build_pair_for_scale` missing).

- [ ] **Step 3: Implement `build_pair_for_scale`** — for each scale {30M,100M,300M,1B} pick `d_model`/`n_layers`/`n_heads` for transformer-ar at the param target, then tune the mamba mixer's depth/width (`n_layers`, `transformer_every`, or `d_model`) so `MambaHybrid` lands within 2% of the transformer's actual count. Record the chosen per-scale `(d_model, n_layers, transformer_every)` for both as constants (the interleave table). Iterate the knobs until the test passes (this is the table's load-bearing work).

- [ ] **Step 4: Run, verify PASS for all four scales.** Expected: each pair within 2%.

- [ ] **Step 5: Commit**

```bash
git add src/cfm/models/bakeoff_scales.py tests/models/test_bakeoff_param_match.py
git commit -m "feat(models): param-matched per-scale backbone table (verified gate, ≤2% per scale)"
```

### Task 6: Wire mamba-hybrid into build_backbone + Gate-6 identity

**Files:**
- Modify: `src/cfm/models/backbone.py` (mamba-hybrid branch → build `MambaHybrid`)
- Test: `tests/models/test_backbone_identity_lock.py` (replace the build-gated assertion with a real-build + identity check, guarded by `importorskip`)

- [ ] **Step 1: Write the failing Gate-6 identity test** (runs on the mamba env):

```python
import pytest; pytest.importorskip("mamba_ssm")
def test_mamba_hybrid_shares_scaffold_by_identity():
    """Shared base = SAME objects ⇒ backbones can't drift (spec §3.4). is, not ==."""
    from cfm.models.backbone import build_backbone, shared_conditioning_builder, subf_vocab_size
    from cfm.data.training.conditioning import build_value_bearing_prefix
    m = build_backbone("mamba-hybrid", _tiny_cfg(backbone="mamba-hybrid"))
    assert m.head.out_features == subf_vocab_size() == 1508
    assert shared_conditioning_builder() is build_value_bearing_prefix
    # same forward contract as transformer-ar (embedding spans the conditioning id span)
    ...  # (mirror test_embedding_covers_the_value_bearing_conditioning_id_span)
```

- [ ] **Step 2: Run, verify fail** (`BackboneNotYetBuilt`).

- [ ] **Step 3: Build the mamba branch in `build_backbone`** — keep `assert_mamba_env_locked()` first, then construct `MambaHybrid(MambaHybridConfig(...))` from `cfg` (mirroring the transformer-ar branch's shared-field wiring + the mamba params from `cfg`). Remove the `BackboneNotYetBuilt` raise for mamba-hybrid only.

- [ ] **Step 4: Update the existing gate tests** — `test_mamba_hybrid_asserts_mamba_env_lock_then_gates` no longer raises `BackboneNotYetBuilt` on the mamba env; split: on the **mamba env** it BUILDS (identity test above); the **repo .venv** (no mamba) still raises `TrainingEnvMismatch` from the lock. Keep `discrete-diffusion → BackboneNotYetBuilt`.

- [ ] **Step 5: Run the model + env-lock suites on both envs, verify green.**

- [ ] **Step 6: Commit**

```bash
git add src/cfm/models/backbone.py tests/models/test_backbone_identity_lock.py
git commit -m "feat(models): build mamba-hybrid backbone; Gate-6 identity by shared-object reference"
```

### Task 7: Non-scored GPU smoke + quantified compile-stability check (spec §3.5, §5)

**Files:**
- Create: `scripts/mamba_smoke.sbatch` (4×A100, `boost_qos_dbg`, account 548, the 3 mamba preconditions, compiled)
- Create: `scripts/measure_compile_stability.py` (instrument `torch._dynamo` recompiles + compile-time fraction over the real cell-length distribution)
- Report: `reports/2026-06-17-mamba-smoke.md`

- [ ] **Step 1: Write the compile-stability instrument**

```python
# Counts torch._dynamo recompilation events + cumulative compile time over a window of
# REAL variable cell lengths; emits {recompiles, compile_overhead_frac, plateaued}.
import torch._dynamo as dyn  # dyn.utils.counters["stats"]["unique_graphs"] / recompiles
...  # train a compiled MambaHybrid for ≥200 steps on real shards; record per-step compile time
```

- [ ] **Step 2: Write the smoke sbatch** — mirror `scripts/probe_mamba_gpu_half.sh`'s 3 preconditions (`module load … gcc/12.2.0`, `CC/CXX`, `LD_PRELOAD` libstdc++) + the repo `.venv` won't have mamba, so the smoke uses the **mamba env** (extend the probe venv into a training-capable env, OR install mamba into a dedicated bake-off env — decide at execution; the env must carry the locked stack + mamba pins, `assert_mamba_env_locked` passes). Run: short `train_scaffold.py --backbone mamba-hybrid` (NO `--scored-run`) at the smallest scale, **compiled**, then the eval (generate→decode→score) + the compile-stability measure.

- [ ] **Step 3: [GATED — Umar's word] Submit the smoke; verify the three teeth**
  1. **Trains** — finite grads, loss decreasing.
  2. **Evals clean** — generate→decode→score completes (proves the gcc/compile eval-path fix on the mamba backbone, compiled).
  3. **Compile-stable** — recompiles plateau (≤ ~10, no new recompiles in the window's 2nd half) AND compile overhead < 10% of wall-clock. **If either fails → record the finding and set scored runs to `--no-compile`** (spec §5).

- [ ] **Step 4: Write `reports/2026-06-17-mamba-smoke.md`** — the three verdicts + the compile-stability numbers (recompiles, overhead frac) + the compile-on/off decision.

- [ ] **Step 5: Commit**

```bash
git add scripts/mamba_smoke.sbatch scripts/measure_compile_stability.py reports/2026-06-17-mamba-smoke.md
git commit -m "feat(bakeoff): mamba-hybrid non-scored GPU smoke + quantified compile-stability verdict"
```

---

## Phase 3 — Task-1 diagnostic (runs in PARALLEL with Phase 2; on transformer-ar)

### Task 8: Extend the diagnostic to capture per-token generation cost

**Files:**
- Modify: `scripts/bakeoff_diagnostic.sbatch` (add the gcc load precautionarily; it's `--no-compile`), and `scripts/train_scaffold.py` eval-cost reporting (emit per-token gen seconds)
- Test: extend the relevant cost-report test (e.g. `tests/training/test_*cost*` if present; else a focused unit test on the per-token metric)

- [ ] **Step 1: Write a failing test** that the diagnostic's cost report includes a `gen_seconds_per_token` (or equivalent) field — the §6 budget projection needs cost-at-13,312 extrapolated from the measured per-token cost, not the 2048 total alone.
- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** — in the post-train eval cost recording, divide generation wall-clock by tokens generated → `gen_seconds_per_token`; add `module load … gcc/12.2.0` to the diagnostic sbatch (precautionary; it's `--no-compile`).
- [ ] **Step 4: Run, verify PASS.**
- [ ] **Step 5: Commit** `feat(bakeoff): diagnostic captures per-token gen cost for the 13,312 eval-budget projection`.

### Task 9: [GATED — Umar's word] Run the diagnostic → feasible ladder + budget projection

- [ ] **Step 1: Submit `bakeoff_diagnostic.sbatch`** (transformer-ar, 90M, `--no-compile`) on Leonardo (account 548). It's the locked spec's HARD GATE before scored runs.
- [ ] **Step 2: Analyze the output** — emergence floor, geometry-`r`, per-scale eval cost, `gen_seconds_per_token`. Compute the **feasible scale ladder** (`feasible_ladder`, `r·N ≤ 623.9M·E`) and **project the 13,312 scored-eval budget** for 2 backbones × feasible scales × seed-repeats vs **5000 node-h**, with `n_cells` minimized to the must-rank gap.
- [ ] **Step 3: Write `reports/2026-06-17-bakeoff-diagnostic.md`** — the measured numbers, the feasible ladder, the decision basis (step-function), the revised eval-dominated budget, the seed count (from the measured per-run noise).
- [ ] **Step 4: PRESENT to Umar for approval BEFORE any scored run** (spec §6). This is the Phase-4 re-plan checkpoint.
- [ ] **Step 5: Commit** the report.

---

## Phase 4 — Scored runs → verdict (GATED; re-planned from Phase 3 output)

> **This phase is intentionally NOT pre-pinned.** The feasible matrix (which scales, how many seeds), the eval-dominated budget, and the compile on/off decision are all OUTPUTS of Phase 2's smoke (Task 7) and Phase 3's diagnostic (Task 9). Per protocol §10.1 (never size a commitment by a provisional parameter) and spec §6, Phase 4's detailed task list is written **after** Task 9, from its measured numbers, and re-reviewed before any scored GPU-hour.

### Task 10: [GATED — re-plan after Task 9, then Umar's word] Scored matrix → decision

- [ ] **Step 1: Re-plan** — from Task 9's feasible ladder + budget + seed count + Task 7's compile decision, enumerate the exact scored matrix (2 backbones × feasible scales × seeds) and confirm it fits 5000 node-h. Bring the matrix for approval.
- [ ] **Step 2: [GATED] Run the scored matrix** via `scripts/bakeoff_run.sbatch` per cell: `--scored-run` (asserts max_len == 13,312 ∧ eval_max_new ≥ 13,312) + `--shard-cache` + the gated config, against the frozen floor artifact (`reports/conditioning_floor/2026-04-15.0/`). USR1 verified-resubmit + end-state markers per cell.
- [ ] **Step 3: Run the decision** — `scripts/run_bakeoff_decision.py` over the per-cell reports + real-features + the frozen floor: `decide()` 5 teeth → `pick_winner` (memorization-first hard-halt → structural → power-gated worst-case + munich→manchester). 
- [ ] **Step 4: Write the verdict report** — the scaling curves (geometry-fidelity vs measured node-h), the winner (or the §13 tie-break / memorization-halt), the per-city worst-case, the binding-city power-gate result.
- [ ] **Step 5: Commit** the verdict; this closes the 2-backbone bake-off. (discrete-diffusion second wave re-opens as its own sub-project.)

---

## Self-review notes (coverage)

- Spec §2 scope → Phases 2/4 (2 backbones; diffusion absent = deferred).
- Spec §3.1/§3.2 extract-base + behavior-preservation → Tasks 2–3 (golden Gate-6).
- Spec §3.3 7:1 interleave + **param-matched verified gate** → Tasks 4–5.
- Spec §3.4 Gate-6 identity → Task 6.
- Spec §3.5 non-scored smoke → Task 7.
- Spec §4.1 sbatch gcc fix + `--test-only` → Task 1; mamba preconditions → Tasks 7.
- Spec §4.2 diagnostic + per-token cost → Tasks 8–9.
- Spec §5 compile-on smoke-gated quantified → Task 7.
- Spec §6 eval-dominated budget projection (pre-scored) → Task 9.
- Spec §9 seeds from diagnostic noise → Task 9.
- Spec §8 unchanged decision layer / scored mechanics → Task 10 (uses the locked code).

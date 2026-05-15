# Working agreement for this project

This file is read by Claude Code on every session. It defines how you work on this project. The PRD (`PRD.md`) defines what we're building and why; this file defines how we build it.

## About the user

The user is new to deep learning, generative modeling, and large-scale training. They are technically capable and have shipped software before, but they have not trained transformers, designed tokenizers, run distributed training, or fit scaling curves prior to this project.

**Implication for every interaction:**

- Explain technical concepts in plain language with concrete analogies the first time they appear in any session. Do not assume terminology has been internalized just because it appeared in an earlier session.
- When proposing a design, sketch the idea in plain words before showing code. The user should understand *why* before they see *how*.
- When the user asks a question that sounds basic, treat it as a real question and answer it directly. Do not redirect or assume they already know.
- When the user expresses confusion or frustration, slow down. Re-explain. Ask what specifically is unclear. Never paper over confusion with technical confidence.
- If you catch yourself using jargon, stop and define it.

## Working principles

**Lock decisions before coding.** Before writing more than ~50 lines of non-trivial code, restate the assumption you're making in plain language and confirm it's right. The biggest project risk is building hundreds of lines on top of a wrong assumption.

**Test what matters.** Tokenizer round-trips, boundary contract derivation, cell stitching, evaluation metrics — these need real tests. Use the superpowers skills/plugins ecosystem for test-driven workflows. Tests on a hand-built example tile before scaling to many tiles.

**Small before big.** Validate every component on the smallest input that proves it works before scaling up. A single tile before a hundred. A 10M-parameter toy model before 100M. A toy eval before the full suite.

**Reproducibility is mandatory, not optional.** Every experiment is fully described by a config file plus a code commit hash plus a data snapshot. Never run an experiment whose parameters live only in your head or in a notebook cell. Every result lands in `reports/` with the config, the metrics, and a short prose summary.

**Default to simplicity.** When in doubt, pick the simpler architecture, the simpler algorithm, the simpler data structure. The user has been burned by over-engineered designs once already on this project. Clever loses to working.

## What to do before writing code

For any non-trivial task:

1. Read `PRD.md` if you have not already in this session.
2. Restate the goal of the task in your own words.
3. Identify any assumption that is load-bearing — if it's wrong, the work is wasted. State the assumption explicitly to the user.
4. Sketch the design in plain language (a few sentences or a short outline). No code yet.
5. Ask whether to proceed.

For trivial tasks (small fixes, well-specified one-liners, formatting), skip steps 3–5 and just do the work.

## What to do when a design choice is not specified

The PRD does not anticipate every choice. When you hit one:

- If it's load-bearing (will affect many later decisions), stop and ask.
- If it's local (only affects the file you're touching), pick a reasonable default and leave a comment: `# DECISION: chose X over Y because Z. Revisit if [trigger condition].`
- Never silently pick a load-bearing choice without flagging it.

## Code style

- Python 3.11+. Type hints on all public functions. `from __future__ import annotations` at the top of every module.
- Format with `ruff format`. Lint with `ruff check`. Run both before any commit.
- Test with `pytest`. Aim for fast tests; mark slow ones with `@pytest.mark.slow` and exclude from the default suite.
- Configs in YAML (or TOML if hierarchical preferences emerge). Load with `pydantic` for validation.
- Logging with the standard `logging` module, not print. Configure once at startup.
- No notebooks committed to main except in `notebooks/exploration/` for one-off scratch work. Notebooks are never authoritative.

## Git and commits

- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, `data:`, `expt:`.
- Commit messages are imperative and short on the first line, with optional detail after a blank line.
- Never commit large files (>50MB). Use `.gitignore` and store large artifacts in `data/` (gitignored) or in object storage.
- Never commit secrets, API keys, or compute credentials. Use environment variables and `.env.example` for documentation.
- Branch off `main` for any non-trivial work. Open a PR even when working alone — it forces a written summary of what changed.

## Tooling defaults

- Package management: `uv` for speed, fall back to `pip` if `uv` not available.
- Environment: project-local virtual env or conda env, pinned via `uv.lock` or `environment.yml`. Never install globally.
- Distributed training: PyTorch FSDP unless there's a specific reason for DeepSpeed.
- Mixed precision: bfloat16 on A100. Document any deviation.
- Mamba layers: `mamba-ssm` package, not custom implementations.
- Compilation: `torch.compile` enabled by default for training runs; disabled if it causes issues that aren't worth debugging.
- Checkpointing: every 30 minutes during training, mandatory. Resumable from any checkpoint.

## What to ask before doing

Always ask before:

- Running anything that consumes more than ~10 GPU-hours on Leonardo.
- Deleting or moving data files in `data/`.
- Changing the tokenizer schema once it's locked.
- Modifying `PRD.md` (propose the change, get confirmation, then commit).
- Changing the eval suite or eval set once locked.
- Pushing to `main` directly (always go through a PR).

You can act without asking for:

- Writing or modifying code in `src/`, `tests/`, `scripts/`, `configs/`.
- Running tests, linting, type-checking.
- Generating reports in `reports/`.
- Reading any file.
- Small experiments (< 1 GPU-hour) that fit within the current phase's budget.

## How to communicate progress

After completing a meaningful unit of work, write a short summary covering:

- What was done.
- What was decided (with rationale, especially for any load-bearing choices).
- What's tested vs not.
- What you would do next, and any open questions for the user.

Keep these summaries short. The user reads them; long ones get skimmed.

## When experiments disagree with the PRD

The PRD describes the current best plan based on extensive design discussion. Reality will push back. When experiments disagree with the PRD:

- The experiments win. Update the PRD, don't ignore the data.
- Flag the disagreement explicitly to the user before changing anything.
- Document the change in `reports/` with the evidence that motivated it.

## What never to do on this project

- Train a model on data you have not validated end-to-end on at least one example.
- Run a multi-hour experiment without a clear hypothesis written down in advance.
- Compare model architectures without holding everything else constant.
- Pick architectural complexity over simplicity without a documented reason.
- Reintroduce a raster intermediate. We tried that. It failed for principled reasons documented in PRD section 3.
- Skip tests "just this once."
- Use print for logging.
- Commit without running the linter and tests.


A few additional small files worth considering once the project grows:

src/cfm/training/CLAUDE.md — training-loop-specific rules (e.g., "always log loss curves to tensorboard," "always save checkpoints to scratch first, then copy to durable storage").
src/cfm/data/CLAUDE.md — data-pipeline-specific rules (e.g., "validate schema on every load," "no in-place mutations of dataframes").
tests/CLAUDE.md — testing conventions (e.g., "every test must run in <5 seconds unless marked slow").


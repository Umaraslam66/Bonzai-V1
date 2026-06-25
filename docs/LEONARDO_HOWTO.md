# Using CINECA Leonardo — a practical how-to

A generic field guide for getting real work done on **Leonardo** (EuroHPC pre-exascale
system at CINECA). It covers access, filesystems, accounting, software, Slurm,
transfers, distributed training, and the non-obvious gotchas that cost people days.
**Limits and policies change** — always treat the official docs as truth:
<https://docs.hpc.cineca.it/hpc/leonardo.html> (scheduler, storage, and data-transfer
pages linked from there).

## The machine in one paragraph

Two main partitions. **Booster** = GPU nodes, each **4× NVIDIA A100 (64 GB)** + one
**32-core** Intel CPU, wired with **HDR InfiniBand**. **Data-Centric (DCGP)** = CPU-only
nodes. Scheduler is **Slurm**. There is **no inbound** network to compute nodes (it is
batch HPC — you cannot host a reachable web server) and compute nodes have **no outbound
internet** — do all downloads/installs from login nodes or pre-stage them.

## Access & authentication

```bash
# One-time 2FA enrolment + login token (smallstep CLI):
step ssh login '<your-email>' --provisioner cineca-hpc
# Then:
ssh <username>@login.leonardo.cineca.it
```

2FA is **mandatory**; tokens expire, so re-run `step ssh login` when SSH starts failing.
To avoid re-authenticating on every connection, use an **SSH ControlMaster** socket in
`~/.ssh/config` (one login, many multiplexed sessions). Note the socket **dies when your
laptop sleeps** — re-establish it after waking.

## Filesystems (Lustre)

| Path | Use | Caveats |
|---|---|---|
| `$HOME` | configs, small code | small quota; **do not** put data or venvs here |
| `$WORK` = `/leonardo_work/<account>` | project data, venvs, checkpoints | persistent, large, **not backed up** |
| `$SCRATCH` = `/leonardo_scratch/large/userexternal/<user>` | scratch I/O | huge, **auto-purged** — never archive here |

Lustre is fast for big sequential I/O but **slow for millions of tiny files** — tar or
shard small files, and stripe large ones. Measured healthy `$WORK` throughput is ~1.6 GB/s
write+fsync and ~5 GB/s read; use a `dd` test (below) to judge FS health, **not** Python
import speed.

## Accounting & allowances

Budgets are in **core-hours** (`saldo -b`, the "local h" column, with start/end dates).
Every job names a project: `#SBATCH --account=<account>`. Switch the active project with
`chprj`. **A Booster node is billed per node** (= 32 core-h per wall-hour) **regardless of
how many of its 4 GPUs you use**. So the unit math is:

```
1 node-hour = 4 GPU-hours = 32 core-hours
```

**Never submit a 1-GPU job to a GPU node** — you pay for all four. Default to 4-GPU jobs.

## Software & environments

```bash
module load python/3.11.7        # or `module avail` to list
python -m venv $WORK/proj/.venv  # build in $WORK, never $HOME
source $WORK/proj/.venv/bin/activate
pip install --upgrade pip
```

Load toolchains via `module load` (CUDA, gcc, NCCL). Build/install on a **login node**
(internet); compute nodes can't reach PyPI. **Rebuild venvs cleanly** — copied/stale envs
start hanging on imports. Compiled C++ extensions (anything needing a newer `libstdc++`
than the system one) may fail to import on compute nodes; fix by `LD_PRELOAD`-ing the
`libstdc++.so` from a recent **gcc module**. Verify the GPU stack **inside a Slurm job**,
not just on the login node:

```bash
python - <<'PY'
import torch; print(torch.__version__, torch.cuda.is_available())
PY
```

## Slurm

```bash
squeue -u $USER                 # your queue
sacct -j <jobid> -X --format=JobID,State,Elapsed,ExitCode
scancel <jobid>
sinfo                           # partitions + live limits (authoritative)
```

Common partitions/QOS: `boost_usr_prod` (GPU production), `dcgp_usr_prod` (CPU
production), `lrd_all_serial` (short, budget-free serial — use for long-but-light tasks
like cache builds), and a debug QOS (`--qos=boost_qos_dbg`, short walltime / few nodes,
fast to schedule). **Never run heavy work on login nodes — they are SIGKILLed.** Confirm
walltime caps with `sinfo`/docs (production is typically up to ~24 h), and **checkpoint
every ~30 min and resume** so a walltime cut or preemption never loses progress.

Minimal 4-GPU template:

```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=<account>
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # one task per GPU
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
mkdir -p logs && cd "$WORK/proj"
source .venv/bin/activate
srun python train.py
```

## Data transfer / rsync

Small repos/configs — `rsync` over SSH (multiplex via ControlMaster, wrap long runs in
`tmux`):

```bash
rsync -avzP ./ <username>@login.leonardo.cineca.it:/leonardo_work/<account>/proj/
```

Large datasets — use the **data movers** (`data.leonardo.cineca.it`, `dmover[1-4]…`) or
Globus, **not** the interactive login shells. If login nodes lack your git credentials, a
clean pattern is `git bundle` locally → `rsync` the bundle → `git clone/pull` it on
Leonardo.

## Distributed training

One rank **per GPU** (4/node) over NCCL on InfiniBand; launch with `srun` (or `torchrun`).
Multi-node = `--nodes=N --ntasks-per-node=4`. Use **bf16** on A100. Practical notes:
seed **before** model init; with frameworks like Lightning, `save_checkpoint` is a
**collective** call (all ranks must reach it); write checkpoints to **`$WORK`** (durable),
not scratch; bit-identical reproduction is unrealistic under NCCL — aim for functional
identity at float tolerance. Smoke-test a tiny `fast_dev_run`/single-step job before
committing a full run.

## Gotchas that waste days

- **Slow `torch` import (~60–80 s) and a hang at interpreter exit on the login node are
  NORMAL**, not a filesystem outage. Judge `$WORK` health with a `dd` write+fsync vs a
  `$SCRATCH` baseline, never by import speed. (To dodge the GPU-less teardown hang in a
  script, capture results then `os._exit(0)`.)
- **Scratch is purged** — anything you need next month lives in `$WORK`.
- **Login-node compute gets killed** — push it to `lrd_all_serial` or a batch job.
- **Confirm `torch.cuda.is_available()` inside Slurm**, not only on login.

## Pre-flight checklist

1. SSH + 2FA working. 2. `$WORK`/scratch paths exist, space checked (`du -sh`). 3. No stale
jobs (`squeue -u $USER`). 4. Imports succeed **in a Slurm job** with GPUs visible. 5. A tiny
smoke job passes before any full run. 6. Logs written immediately; checkpoints to `$WORK`.

# Leonardo Reference

This is the local restart note for using CINECA Leonardo after the Bonzai cleanup.
All old Bonzai experiment files were intentionally wiped from both Leonardo and
this local directory.

## Official Docs

- Leonardo cluster page: https://docs.hpc.cineca.it/hpc/leonardo.html
- General CINECA HPC documentation: https://www.hpc.cineca.it/user-support/documentation/
- Scheduler and Slurm guide: https://docs.hpc.cineca.it/hpc/hpc_scheduler.html
- File systems and data transfer: https://docs.hpc.cineca.it/hpc/hpc_data_storage.html
- CINECA support / help desk: https://www.hpc.cineca.it/user-support/

Use official docs as source of truth for limits, partitions, and access changes.
CINECA changes policies and outages can happen.

## Account And Paths

Known account context:

```bash
username=uaslam00
login_host=login.leonardo.cineca.it
project_account=AIFAC_P02_222
WORK=/leonardo_work/AIFAC_P02_222
CINECA_SCRATCH=/leonardo_scratch/large/userexternal/uaslam00
```

Current cleanup state as of 2026-05-13:

```text
/leonardo_work/AIFAC_P02_222                  empty
/leonardo_scratch/large/userexternal/uaslam00 empty
```

The root directories still exist. Their contents were deleted.

## SSH

Basic login:

```bash
ssh uaslam00@login.leonardo.cineca.it
```

Batch/non-interactive check:

```bash
ssh -o BatchMode=yes uaslam00@login.leonardo.cineca.it 'hostname; date; pwd'
```

Leonardo uses two-factor authentication. If SSH fails, first check whether your
CINECA certificate/session needs renewal. Official Leonardo docs list the login
endpoint and direct login node hostnames.

## Data Transfer

For small repo syncs, regular `rsync` to the login host is usually fine:

```bash
rsync -av ./ uaslam00@login.leonardo.cineca.it:/leonardo_work/AIFAC_P02_222/my_project/
```

For large datasets, use CINECA data mover guidance. Official docs list:

```text
data.leonardo.cineca.it
dmover1.leonardo.cineca.it
dmover2.leonardo.cineca.it
dmover3.leonardo.cineca.it
dmover4.leonardo.cineca.it
```

Important: data movers are for transfer commands, not normal interactive SSH
shell work.

## Storage Use

Use:

```text
$WORK             persistent project work area
$CINECA_SCRATCH   temporary large user scratch
$HOME             small user home, not for big data
```

Leonardo docs warn that `$HOME` backup may not be active. Scratch is temporary
and should not be treated as archive storage.

Check space:

```bash
ssh uaslam00@login.leonardo.cineca.it 'du -sh $WORK ${CINECA_SCRATCH:-/leonardo_scratch/large/userexternal/$USER} 2>/dev/null'
```

## Slurm Basics

Common commands:

```bash
squeue -u $USER
sacct -j <jobid> -X --format=JobID,State,Elapsed,ExitCode
scancel <jobid>
sinfo
```

Known partitions from the previous work:

```text
lrd_all_serial   CPU/serial jobs, budget-free, short wall time
boost_usr_prod   GPU production jobs
dcgp_usr_prod    CPU production jobs
```

Always confirm current limits in official docs before launching new work.

Minimal GPU job template:

```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=AIFAC_P02_222
#SBATCH --job-name=my-gpu-job
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs
cd "$WORK/my_project"
source .venv/bin/activate
python script.py
```

Submit:

```bash
sbatch job.sbatch
```

Watch:

```bash
squeue -u $USER
tail -f logs/my-gpu-job-<jobid>.out
```

## Python Environment Pattern

Use Python 3.11 on Leonardo:

```bash
module load python/3.11.7
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For PyTorch GPU work, install a known-good torch wheel version and test imports:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

Do not assume a copied old venv works. In the failed Bonzai run, the old venv
began hanging on imports. Prefer rebuilding venvs cleanly for the new approach.

## Safety Checklist Before Big Jobs

1. SSH works.
2. `$WORK` and scratch paths exist.
3. `squeue -u $USER` has no stale jobs.
4. Python imports succeed on the login node.
5. A tiny Slurm smoke job runs before any full training.
6. Logs are written immediately.
7. For GPU jobs, check `torch.cuda.is_available()` inside Slurm, not only on the
   login node.

## Cleanup Commands

List current contents:

```bash
ssh uaslam00@login.leonardo.cineca.it 'ls -la $WORK; ls -la ${CINECA_SCRATCH:-/leonardo_scratch/large/userexternal/$USER}'
```

Delete a project directory only when you are sure:

```bash
ssh uaslam00@login.leonardo.cineca.it 'rm -rf $WORK/my_project'
```

Avoid broad deletes unless you explicitly intend a full wipe.


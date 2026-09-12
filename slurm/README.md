# SLURM job scripts

**You do not need any of this.** `sparc-pretrain` and `sparc-eval` run anywhere;
this directory only adds job submission for SLURM clusters.

## Setup

```bash
cp slurm/cluster_env.sh.example slurm/cluster_env.sh
$EDITOR slurm/cluster_env.sh      # account, GPU type, dataset roots
```

`cluster_env.sh` is the only file with site-specific settings, and it is
gitignored. Nothing else here hardcodes an account, a partition, a GPU type or a
module name — those reach `sbatch` from that file via `submit_sweep.sh`.

## Files

| File | Role |
|---|---|
| `cluster_env.sh.example` | Site settings. Copy and edit. |
| `runtime_env.sh` | Sourced by every job: modules, venv, banner. |
| `stage_dataset.sh` | Copies the corpus to node-local storage. |
| `pretrain.sbatch` | One array task = one configuration's pretraining. |
| `evaluate.sbatch` | One array task = one downstream fine-tune. |
| `submit_sweep.sh` | Submits a sweep and chains evaluation to it. |

Two scripts cover every sweep and both tasks, because the configuration comes
from a manifest row rather than from arrays baked into the script.

## Why staging exists

Reading ~118k small image files per epoch directly off a shared parallel
filesystem was measured at **602 ms/batch, against 57 ms/batch** from node-local
NVMe. The first epoch of an unstaged run took roughly three hours.

Masks are extracted from a single tar rather than copied file by file, which also
sidesteps an inode problem: each mask set is 118,287 files, and filesystem quotas
are usually counted in files, not bytes. A tar is one file.

`stage_dataset.sh` refuses to continue unless the mask count matches the image
count. A precompute job once reported success while silently leaving 287 images
without masks, and it surfaced only when training hit a missing file hours later.

## Chaining

Evaluation depends on pretraining with `aftercorr`, which pairs array task *i* of
the evaluation with array task *i* of the pretraining. Plain `afterok` would let
one failed pretrain cancel the entire evaluation array.

## SEED is required

Every job script demands `SEED` and never defaults it. A forgotten export would
resolve to the bare seed-1 `run_id` and overwrite a finished replicate's outputs.

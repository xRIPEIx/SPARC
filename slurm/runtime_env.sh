#!/bin/bash
# Sourced by every job script. Loads site configuration and activates the
# environment. Never submitted directly.
set -euo pipefail

SPARC_REPO="${SPARC_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$SPARC_REPO"

if [[ ! -f slurm/cluster_env.sh ]]; then
    echo "ERROR: slurm/cluster_env.sh not found." >&2
    echo "       cp slurm/cluster_env.sh.example slurm/cluster_env.sh and edit it." >&2
    exit 1
fi
# shellcheck disable=SC1091
source slurm/cluster_env.sh

# Module systems are not universal; skip cleanly when absent.
#
# No `module purge` here, and certainly not `--force purge`: on Alliance
# clusters that also unloads the sticky StdEnv that every other module lives
# under, after which `python/3.11` is "unknown". A job that did this failed in
# its first second -- after five hours in the queue. Load on top of whatever
# the job inherited instead; list StdEnv in SPARC_MODULES if you need a
# specific one.
if command -v module >/dev/null 2>&1 && [[ ${#SPARC_MODULES[@]} -gt 0 ]]; then
    for m in "${SPARC_MODULES[@]}"; do module load "$m"; done
fi

if [[ -n "${SPARC_VENV:-}" && -f "$SPARC_VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$SPARC_VENV/bin/activate"
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

echo "=================================================================="
echo "host       : $(hostname)"
echo "job        : ${SLURM_JOB_ID:-none} ${SLURM_ARRAY_TASK_ID:+array task $SLURM_ARRAY_TASK_ID}"
echo "repo       : $SPARC_REPO"
echo "commit     : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "python     : $(command -v python)"
echo "torch      : $(python -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>/dev/null || echo n/a)"
echo "=================================================================="

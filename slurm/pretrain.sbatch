#!/bin/bash
#SBATCH --job-name=sparc-pretrain
#SBATCH --output=slurm_output/%x_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# Everything that varies by site -- account, partition, GPU type, CPUs, memory,
# time -- is passed on the sbatch command line by submit_sweep.sh from
# cluster_env.sh, deliberately NOT hardcoded here. That is what makes this file
# portable.
#
# One script for every sweep. The swept value comes from the manifest row, so
# pretraining and evaluation read the same row and cannot disagree about what a
# checkpoint was trained with.
set -euo pipefail
source slurm/runtime_env.sh
source slurm/stage_dataset.sh

MANIFEST="${MANIFEST:?Set MANIFEST to a sweep manifest CSV}"
# Required, never defaulted: a forgotten SEED would silently resolve to the
# bare seed-1 run_id and overwrite a finished replicate's outputs.
SEED="${SEED:?Set SEED (1-5)}"
INDEX="${SLURM_ARRAY_TASK_ID:?This script must be run as a job array}"

row() { sparc-sweep row "$MANIFEST" --array-index "$INDEX" --seed "$SEED" --field "$1"; }
RUN_ID=$(row run_id)
CONFIG=$(row pretrain_config)
OVERRIDES=$(row pretrain_overrides)

if [[ -z "$CONFIG" ]]; then
    echo "Row $INDEX/seed $SEED has no pretrain_config (e.g. a random-init baseline); nothing to do."
    exit 0
fi

echo "run_id=$RUN_ID config=$CONFIG"
echo "overrides=$OVERRIDES"

MASK_NAME="${MASK_NAME:-}"
stage_dataset "${SPARC_COCO_IMAGES:?}" "$MASK_NAME"

EXTRA=("data.images=$STAGED_IMAGE_DIR")
[[ -n "${STAGED_MASK_DIR:-}" ]] && EXTRA+=("data.masks=$STAGED_MASK_DIR")
[[ -n "${SLURM_CPUS_PER_TASK:-}" ]] && EXTRA+=("data.num_workers=$SLURM_CPUS_PER_TASK")

# shellcheck disable=SC2086
sparc-pretrain --config "$CONFIG" --set $OVERRIDES "${EXTRA[@]}"

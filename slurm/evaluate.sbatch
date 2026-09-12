#!/bin/bash
#SBATCH --job-name=sparc-eval
#SBATCH --output=slurm_output/%x_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#
# Fine-tunes one pretrained checkpoint on one downstream task. TASK selects
# which; the manifest row supplies the identity, so this is the same script for
# every sweep and both tasks.
set -euo pipefail
source slurm/runtime_env.sh

MANIFEST="${MANIFEST:?Set MANIFEST to a sweep manifest CSV}"
SEED="${SEED:?Set SEED (1-5)}"
TASK="${TASK:?Set TASK to segmentation or detection}"
INDEX="${SLURM_ARRAY_TASK_ID:?This script must be run as a job array}"
CONFIG="${DOWNSTREAM_CONFIG:?Set DOWNSTREAM_CONFIG to a downstream YAML}"

row() { sparc-sweep row "$MANIFEST" --array-index "$INDEX" --seed "$SEED" --field "$1"; }
RUN_ID=$(row run_id)
OVERRIDES=$(row downstream_overrides)
PRETRAIN_CONFIG=$(row pretrain_config)

EXTRA=()
if [[ -n "$PRETRAIN_CONFIG" ]]; then
    # Arms that pretrain load their own checkpoint. Arms that do not (random
    # init, supervised ImageNet) carry model.init in downstream_overrides
    # instead, so nothing is added here.
    CKPT="${SPARC_OUTPUT_ROOT:?}/checkpoints/${RUN_ID}/last.pth"
    [[ -f "$CKPT" ]] || { echo "ERROR: no checkpoint at $CKPT" >&2; exit 1; }
    EXTRA+=("model.init=ssl" "model.ssl_ckpt=$CKPT")
fi
[[ -n "${SLURM_CPUS_PER_TASK:-}" ]] && EXTRA+=("data.num_workers=$SLURM_CPUS_PER_TASK")

echo "run_id=$RUN_ID task=$TASK"
# shellcheck disable=SC2086
sparc-eval --config "$CONFIG" --set $OVERRIDES "${EXTRA[@]}"

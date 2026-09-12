#!/bin/bash
# Submit one seed of one sweep: pretraining, then both downstream tasks.
#
#   bash slurm/submit_sweep.sh --sweep sparc_lambda --seed 1
#   for s in 1 2 3 4 5; do bash slurm/submit_sweep.sh --sweep sparc_lambda --seed $s; done
#
# Evaluation is chained with --dependency=aftercorr, which pairs array task i of
# the evaluation with array task i of the pretraining. With plain afterok a
# single failed pretrain would cancel the entire evaluation array.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
[[ -f slurm/cluster_env.sh ]] || {
    echo "ERROR: cp slurm/cluster_env.sh.example slurm/cluster_env.sh and edit it." >&2
    exit 1
}
# shellcheck disable=SC1091
source slurm/cluster_env.sh

SWEEP="" ; SEED="" ; MASK_NAME="slic_n100" ; LIMIT="%4" ; TASKS="segmentation detection" ; ARRAY_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sweep) SWEEP="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --mask-name) MASK_NAME="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;       # e.g. %4 to cap concurrency
        --tasks) TASKS="$2"; shift 2 ;;
        --array) ARRAY_OVERRIDE="$2"; shift 2 ;;   # e.g. "3" to run one config
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$SWEEP" && -n "$SEED" ]] || { echo "Usage: --sweep NAME --seed N" >&2; exit 2; }

MANIFEST="experiments/${SWEEP}/manifest.csv"
[[ -f "$MANIFEST" ]] || {
    echo "No manifest at $MANIFEST. Generate it with:" >&2
    echo "  sparc-sweep expand configs/sweeps/${SWEEP}.yaml -o $MANIFEST" >&2
    exit 1
}

N=$(awk -F, 'NR>1{print $1}' "$MANIFEST" | sort -un | wc -l)
ARRAY="${ARRAY_OVERRIDE:-0-$((N - 1))${LIMIT}}"
mkdir -p slurm_output

COMMON=(--account="$SPARC_ACCOUNT" --cpus-per-task="$SPARC_CPUS" --mem="$SPARC_MEM")
[[ -n "${SPARC_PARTITION:-}" ]] && COMMON+=(--partition="$SPARC_PARTITION")
EXPORTS="ALL,MANIFEST=$MANIFEST,SEED=$SEED,MASK_NAME=$MASK_NAME,SPARC_REPO=$REPO"

echo "sweep=$SWEEP seed=$SEED array=$ARRAY ($N configurations)"

PRETRAIN_ID=$(sbatch --parsable "${COMMON[@]}" \
    --array="$ARRAY" --gpus="$SPARC_GPU" --time="$SPARC_PRETRAIN_TIME" \
    --job-name="${SWEEP}-pt" --export="$EXPORTS" slurm/pretrain.sbatch)
echo "  pretrain: $PRETRAIN_ID"

for TASK in $TASKS; do
    case "$TASK" in
        segmentation) TIME="$SPARC_SEG_TIME"; CONFIG="configs/downstream/voc_seg_fcn_r18.yaml" ;;
        detection)    TIME="$SPARC_DET_TIME"; CONFIG="configs/downstream/voc_det_frcnn_r18.yaml" ;;
        *) echo "Unknown task: $TASK" >&2; exit 2 ;;
    esac
    EVAL_ID=$(sbatch --parsable "${COMMON[@]}" \
        --array="$ARRAY" --gpus="$SPARC_GPU" --time="$TIME" \
        --job-name="${SWEEP}-${TASK:0:3}" \
        --dependency="aftercorr:$PRETRAIN_ID" \
        --export="$EXPORTS,TASK=$TASK,DOWNSTREAM_CONFIG=$CONFIG" \
        slurm/evaluate.sbatch)
    echo "  $TASK: $EVAL_ID (aftercorr:$PRETRAIN_ID)"
done

#!/bin/bash
# Stage the training corpus onto node-local storage.
#
# WHY: reading ~118k small image files per epoch directly off a shared parallel
# filesystem was measured at 602 ms/batch, against 57 ms/batch from node-local
# NVMe -- a >10x difference that dominated epoch time. The first epoch of an
# unstaged run took roughly three hours.
#
# Masks are extracted from a single tar rather than copied file by file. That
# also sidesteps an inode problem: each mask set is 118,287 files, and a shared
# filesystem quota is usually counted in files, not bytes. A tar is one file.
set -euo pipefail

stage_dataset() {
    local image_root="${1:?stage_dataset: image root required}"
    local mask_name="${2:-}"

    local dest="${SPARC_STAGE_DIR:?SPARC_STAGE_DIR not set}"
    mkdir -p "$dest/images"

    echo "[stage] copying images from $image_root"
    # Parallel copy: one process per file is far too slow at this file count.
    find "$image_root" -type f -print0 \
        | xargs -0 -P "${SLURM_CPUS_PER_TASK:-8}" -n 500 cp -t "$dest/images"
    export STAGED_IMAGE_DIR="$dest/images"
    local n_images
    n_images=$(find "$dest/images" -type f | wc -l)
    echo "[stage] staged $n_images images -> $STAGED_IMAGE_DIR"

    if [[ -z "$mask_name" ]]; then
        echo "[stage] no mask set requested (method needs none)"
        return 0
    fi

    local archive="${SPARC_MASK_ARCHIVE_ROOT:?}/${mask_name}.tar"
    [[ -f "$archive" ]] || { echo "[stage] ERROR: no archive at $archive" >&2; return 1; }

    mkdir -p "$dest/superpixel_masks"
    echo "[stage] extracting $archive"
    # Archives are built as `tar cf -C <src_root> <name>`, so entries are
    # already prefixed with <name>/. Extract one level above; no
    # --strip-components.
    tar -xf "$archive" -C "$dest/superpixel_masks"
    export STAGED_MASK_DIR="$dest/superpixel_masks/$mask_name"

    # Integrity gate. A precompute job once reported success while silently
    # leaving 287 images without masks; it surfaced only when training hit a
    # missing file hours later. Count before training, not during.
    local n_masks
    n_masks=$(find "$STAGED_MASK_DIR" -name '*.npy' | wc -l)
    if [[ "$n_masks" -ne "$n_images" ]]; then
        echo "[stage] ERROR: $n_masks masks for $n_images images -- the mask set is incomplete." >&2
        echo "[stage]        Regenerate with sparc-masks; it skips files that already exist." >&2
        return 1
    fi
    echo "[stage] staged $n_masks masks -> $STAGED_MASK_DIR"
}

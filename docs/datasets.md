# Datasets

Nothing is shipped. This page says where to put things and what they cost.

## Where paths are set

**One file:** [`configs/paths.yaml`](../configs/paths.yaml). Every other config
refers to `${paths.*}` and never to a literal path.

Two ways to point it at your data, both supported at once:

```bash
# 1. environment variables (natural on a cluster)
export SPARC_COCO_IMAGES=/data/coco/train2017
export SPARC_SUPERPIXEL_ROOT=/data/superpixel_masks
export SPARC_VOC_ROOT=/data/VOC                   # the directory CONTAINING VOCdevkit

# 2. a gitignored local file (natural on a workstation)
cp configs/paths.local.yaml.example configs/paths.local.yaml && $EDITOR configs/paths.local.yaml
```

## Pretraining: COCO train2017

118,287 images, ~19 GB. Any directory tree of images works — COCO is what the
paper uses. Download from https://cocodataset.org (`train2017.zip`; annotations
are not needed).

```
<coco_images>/
  000000000009.jpg
  ...
```

## Superpixel masks

One mask set per configuration, mirroring the image tree, one `.npy` per image:

```
<superpixel_root>/slic_n100/
  superpixel_mask_meta.json      # how the set was made
  000000000009.npy               # int label map, same H×W as the image
  ...
```

Generate with `sparc-masks --config configs/superpixel/slic_n100.yaml`
(≈2 CPU-hours for COCO across 32 shards; resumable). Masks regenerate
identically from the pinned scikit-image.

**Budget honestly — in files, not only bytes.** A mask set is **118,287 files**.
Shared filesystems usually cap the *number* of files per user, and that ceiling
is what bit this project, not disk space. Masks are stored `uint8` (≈35 GB per
set; the original `int32` files were four times that, three-quarters of it zero
bytes). For cluster jobs, keep each set as one tar and extract it onto
node-local storage at job start — `slurm/stage_dataset.sh` does this, and
verifies the mask count matches the image count before training starts.

## Downstream: PASCAL VOC2012

~2 GB. `voc_root` is the directory **containing** `VOCdevkit`; torchvision
appends `VOCdevkit/VOC2012` itself:

```
<voc_root>/VOCdevkit/VOC2012/
  JPEGImages/  Annotations/  SegmentationClass/  ImageSets/{Main,Segmentation}/
```

Segmentation trains on `train` (1,464 images) and validates on `val` (1,449).
Detection uses the `Main` splits. **These are different, only partially
overlapping image subsets** — some segmentation-annotated images carry no
bounding boxes — so never reuse a fixed sample list from one task for the other.

Optionally, SBD `train_noval` (larger segmentation training set):
`--set data.seg_train_source=sbd` with `sbd_root` pointing at it.

## Two protocol details the numbers depend on

- Difficult boxes are **included** in detection training and evaluation
  (`ignore_difficult: false`). The standard VOC protocol excludes them; the
  published numbers include them. Set it `true` for the conventional protocol.
- Evaluation images are resized so the shorter side is 520 and are not cropped.

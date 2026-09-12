# Reproducing the paper

Three tiers, cheapest first. Each states what it needs, what it costs, and what
number to expect. Nothing in tier 0 requires downloading a dataset.

Seed-to-seed s.d. over five seeds is **≈0.4 mIoU points** and **≈0.2 AP
points**, so a reproduction that lands within about twice that of the published
mean is consistent with it. Different GPUs and cuDNN versions, and the explicit
DataLoader seeding in this code, all shift the RNG stream; bit-identical numbers
across machines are not the bar.

## Tier 0a — see it work (5 minutes, CPU, no dataset)

```bash
pip install -e .
python scripts/download_checkpoints.py seg_head det_head
python scripts/demo_predict.py --task segmentation \
    --weights checkpoints/sparc_lambda_0p5_voc_segmentation_resnet18.pth \
    --images path/to/any/jpegs --out demo_out
```

One PNG per image with the predicted VOC classes overlaid. Measured: four
images in ~9 s on two CPU cores.

## Tier 0b — the headline table (≈2 GPU-hours, VOC2012 only)

Fine-tune the released encoders yourself. Needs VOC2012 (~2 GB), nothing else.

```bash
python scripts/download_checkpoints.py
export SPARC_VOC_ROOT=/path/containing/VOCdevkit
for enc in sparc_lambda_0p5 moco densecl_lambda_0p5; do
  sparc-eval --config configs/downstream/voc_seg_fcn_r18.yaml \
      --set model.ssl_ckpt=checkpoints/${enc}_resnet18_coco_ep100.pth experiment.config_id=$enc
done
```

Expected (seed 1, mIoU): SPARC λ=0.5 **39.2**, DenseCL **37.5**, MoCo v2 **33.9**
— see `experiments/results/aggregate/` for every seed. Segmentation takes
≈30 min per encoder on an A100 MIG slice; detection (`voc_det_frcnn_r18.yaml`)
≈1 h and gives AP 25.5 / 24.1 / 23.9.

## Tier 1 — one configuration end to end (≈15 GPU-hours)

Needs COCO train2017 (~19 GB) and one mask set.

```bash
export SPARC_COCO_IMAGES=/path/to/coco/train2017 SPARC_SUPERPIXEL_ROOT=/path/to/masks
sparc-masks --config configs/superpixel/slic_n100.yaml           # ≈2 CPU-h total; shard it
sparc-pretrain --config configs/pretrain/sparc_coco_r18.yaml       # ≈13 GPU-h, 100 epochs
sparc-eval --config configs/downstream/voc_seg_fcn_r18.yaml \
    --set model.ssl_ckpt=runs/checkpoints/sparc_lambda_0p5/last.pth experiment.config_id=sparc_lambda_0p5
sparc-eval --config configs/downstream/voc_det_frcnn_r18.yaml --set ...
```

Expected: mIoU 38.8 ± 0.4, AP 25.4 ± 0.2 (5-seed mean ± s.d.). The masks you
generate will be **identical** to the study's: the pinned scikit-image
regenerates 20 archived masks bit for bit.

## Tier 2 — the full study (≈70 pretrains + 140 fine-tunes)

Both λ sweeps at five seeds plus the baselines, from the same single mask set.
On a SLURM cluster:

```bash
cp slurm/cluster_env.sh.example slurm/cluster_env.sh && $EDITOR slurm/cluster_env.sh
for sweep in sparc_lambda densecl_lambda baselines; do
  for seed in 1 2 3 4 5; do bash slurm/submit_sweep.sh --sweep $sweep --seed $seed; done
done
sparc-sweep status experiments/sparc_lambda/manifest.csv --results-root "$SPARC_OUTPUT_ROOT/results"
```

Then aggregate and render everything:

```bash
for s in sparc_lambda densecl_lambda baselines; do for t in segmentation detection; do
  sparc-report aggregate --manifest experiments/$s/manifest.csv \
      --results-root "$SPARC_OUTPUT_ROOT/results" --task $t \
      -o experiments/results/aggregate/${s}_${t}_by_config.csv
done; done
make paper          # check + README table + figures + LaTeX
```

`sparc-report check` must pass: `sparc_lambda_0`, `densecl_lambda_0` and MoCo v2
are three routes to the same objective and agree within 2·SE on the published
data (largest gap 0.21 mIoU points against a 0.39-point bound).

## What the published numbers were produced with

- Single GPU (A100 3g.20gb MIG), AMP, batch 64, 100 epochs, Adam 1e-4.
- Global NT-Xent temperature **0.5** (lightly's default — the study never set it).
- Detection **includes** VOC "difficult" boxes in training and evaluation.
- Downstream lr 7.48e-5 / wd 1.20e-4, tuned once on segmentation and frozen for
  every arm and both tasks.
- scikit-image 0.26.0 masks, SLIC n=100, compactness 10, sigma 1.

All of these are the defaults in `configs/`. The code is a restructuring of the
implementation the study ran with; on a fixed batch the two agree bit for bit
in loss and gradients.

## What is not reproduced

Nothing here will reproduce a *resumed* run's RNG stream exactly, and the
superpixel-method and grid-size ablations from the original study are not part
of this repository.

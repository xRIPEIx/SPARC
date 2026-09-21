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
    --weights checkpoints/sparc_lambda_0p5_seed1_voc_segmentation_resnet18.pth \
    --images path/to/any/jpegs --out demo_out
```

One PNG per image with the predicted VOC classes overlaid. Measured: four
images in ~9 s on two CPU cores.

## Tier 0b — the headline table (≈2 GPU-hours, VOC2012 only)

Fine-tune the released encoders yourself. Needs VOC2012 (~2 GB), nothing else.

```bash
python scripts/download_checkpoints.py baselines sparc_lambda_0p5 densecl_lambda_0p5
export SPARC_VOC_ROOT=/path/containing/VOCdevkit
for enc in sparc_lambda_0p5 moco densecl_lambda_0p5; do
  sparc-eval --config configs/downstream/voc_seg_fcn_r18.yaml \
      --set model.ssl_ckpt=checkpoints/${enc}_seed1_resnet18_coco_ep100.pth experiment.config_id=$enc
done
```

Expected (seed 1, mIoU): SPARC λ=0.5 **39.2**, DenseCL **37.5**, MoCo v2 **33.9**
— see `experiments/results/aggregate/` for every seed. Segmentation takes
≈30 min per encoder on an A100 MIG slice; detection (`voc_det_frcnn_r18.yaml`)
≈1 h and gives AP 25.5 / 24.1 / 23.9.

## Tier 0c — the λ ablation from released weights (≈45 GPU-hours, VOC2012 only)

Every encoder in the README's λ table is released at all five seeds, so the
whole ablation reproduces from fine-tuning alone — no COCO, no masks, no
pretraining. Seed 1 of each is on the GitHub Release now; seeds 2–5 arrive with
the Zenodo record (see [`MODEL_ZOO.md`](../MODEL_ZOO.md)), and until then the
download below fetches the 14 seed-1 encoders and skips the rest, which still
gives one full curve per method. Each fine-tune is a separate `sparc-eval` run keyed by the same
`config_id` / `run_id` / seed the sweep manifests use, so `sparc-report`
aggregates the results exactly as it does for the archived study.

```bash
python scripts/download_checkpoints.py sparc_lambda densecl_lambda --seeds all   # 70 encoders, ~3.1 GB
export SPARC_VOC_ROOT=/path/containing/VOCdevkit
for cid in sparc_lambda_{0,0p1,0p3,0p5,0p7,0p9,1} densecl_lambda_{0,0p1,0p3,0p5,0p7,0p9,1}; do
  for seed in 1 2 3 4 5; do
    run=$cid; [ $seed -gt 1 ] && run=${cid}_seed$seed
    for cfg in voc_seg_fcn_r18 voc_det_frcnn_r18; do
      sparc-eval --config configs/downstream/$cfg.yaml \
          --set model.ssl_ckpt=checkpoints/${cid}_seed${seed}_resnet18_coco_ep100.pth \
                train.seed=$seed experiment.config_id=$cid experiment.run_id=$run
    done
  done
done
for s in sparc_lambda densecl_lambda; do for t in segmentation detection; do
  sparc-report aggregate --manifest experiments/$s/manifest.csv \
      --results-root runs/results --task $t -o my_aggregate/${s}_${t}_by_config.csv
done; done
```

Expected: the README λ table, row for row, within seed noise. Budget ≈30 min
per segmentation and ≈1 h per detection fine-tune on an A100 MIG slice; on a
SLURM cluster, `slurm/` runs the same loop as job arrays (see `docs/cluster.md`).

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

## Verification run

Before release, the reference configuration was pretrained once from scratch
with this repository's code and `slurm/` scripts (seed 1, 100 epochs, single
A100 MIG slice, 12 h) and fine-tuned on both tasks, then scored against the
five archived seeds of the same configuration:

| metric | this code | archived, 5 seeds | z |
|---|---|---|---|
| mIoU | 38.94 | 38.80 ± 0.39 | +0.4 |
| pixel acc. | 84.36 | 84.34 ± 0.16 | +0.2 |
| AP | 25.21 | 25.41 ± 0.16 | −1.3 |
| AP50 | 48.73 | 49.28 ± 0.24 | −2.3 |
| AP75 | 23.11 | 23.15 ± 0.30 | −0.1 |

Segmentation reproduces within a fraction of a standard deviation. Detection is
within two standard deviations on AP and AP75; AP50 is 2.3 s.d. below the
archived mean, 0.2 points under the lowest archived seed. Two things bear on
how to read that: the archived study's own repeated detection fine-tunes of a
single checkpoint differed by 0.1–0.3 AP points, so detection fine-tuning at
batch 2 for 10 epochs is itself noisy at this level; and a standard deviation
estimated from five values is uncertain enough that one metric in five landing
at 2.3 s.d. is not remarkable. The training loss at epoch 1 matched the
original implementation's to 3e-4 on real data, and the two implementations
agree bit-for-bit on a fixed batch.

The honest summary: the repository reproduces the published numbers within
seed noise, with detection AP50 at the edge of that bound. Expect the same when
you reproduce it.

For what it is worth on wall-clock: pretraining took 12 h 02 m, segmentation
45 min and detection 64 min, all with 16 CPU workers.

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

# SPARC

**S**uper**P**ixel-**A**ware **R**egion **C**ontrastive learning for self-supervised dense prediction.

> **Status: under construction.** Pretraining and downstream evaluation both
> work end to end. The sweep tooling, mask generation CLI, model zoo and full
> docs land in subsequent phases.

SPARC adds a region-level contrastive term to image-level self-supervised
learning. Superpixels define the regions, and are used to pool encoder features
*after* the backbone — they are never fed into the CNN as an extra input
channel, so a SPARC-pretrained backbone transfers with no mask dependency:

```
L = (1 - λ) · L_global + λ · L_region
```

At `λ = 0` this reduces exactly to the MoCo-global objective, which is what lets
the SPARC and DenseCL λ sweeps meet the MoCo v2 baseline at a single shared point.

## Where is…?

| I want to understand… | Go to |
|---|---|
| The region contrastive loss | [`losses/region_contrastive.py`](src/sparc/losses/region_contrastive.py) |
| Superpixel region pooling | [`models/region_pooling.py`](src/sparc/models/region_pooling.py) |
| How the pieces compose into a loss | [`methods/sparc.py`](src/sparc/methods/sparc.py) |
| The pretraining loop | [`engine/trainer.py`](src/sparc/engine/trainer.py) |
| Aligned image/mask augmentation | [`data/transforms.py`](src/sparc/data/transforms.py) |
| Segmentation fine-tuning | [`eval/segmentation.py`](src/sparc/eval/segmentation.py) |
| Detection fine-tuning | [`eval/detection.py`](src/sparc/eval/detection.py) |
| How mIoU and AP are computed | [`eval/metrics.py`](src/sparc/eval/metrics.py) |
| How superpixel masks are generated | [`data/superpixel/`](src/sparc/data/superpixel/) |
| Adding a backbone | [`models/backbones/`](src/sparc/models/backbones/) |
| Checkpoint → downstream backbone | [`engine/checkpoint.py`](src/sparc/engine/checkpoint.py) |
| **Where my datasets are** | [`configs/paths.yaml`](configs/paths.yaml) — the only file you must edit |

## Install

```bash
pip install -e ".[dev]"
pytest                       # fast, CPU-only, no datasets needed
```

## Configure

Dataset locations live in exactly one place. Either set environment variables:

```bash
export SPARC_COCO_IMAGES=/path/to/coco/train2017
export SPARC_SUPERPIXEL_ROOT=/path/to/superpixel_masks
```

or copy `configs/paths.local.yaml.example` to `configs/paths.local.yaml` (which
is gitignored) and edit it. Every other config refers to `${paths.*}` and never
to a literal path.

## Pretrain

```bash
sparc-pretrain --config configs/pretrain/sparc_coco_r18.yaml
```

Change anything from the command line; unknown keys are rejected rather than
silently ignored:

```bash
sparc-pretrain --config configs/pretrain/sparc_coco_r18.yaml \
               --set method.lambda_region=0.7 train.seed=3
```

Baselines need no superpixel masks:

```bash
sparc-pretrain --config configs/pretrain/moco_coco_r18.yaml
sparc-pretrain --config configs/pretrain/densecl_coco_r18.yaml
```

## Evaluate

Fine-tune a pretrained backbone on VOC and write a result row:

```bash
sparc-eval --config configs/downstream/voc_seg_fcn_r18.yaml \
           --set model.ssl_ckpt=runs/checkpoints/sparc_lambda_0p5/last.pth \
                 experiment.config_id=sparc_lambda_0p5

sparc-eval --config configs/downstream/voc_det_frcnn_r18.yaml \
           --set model.ssl_ckpt=runs/checkpoints/sparc_lambda_0p5/last.pth \
                 experiment.config_id=sparc_lambda_0p5
```

Baselines need no checkpoint — `--set model.init=random` or
`model.init=supervised_imagenet`.

Both tasks share one frozen protocol (learning rate, weight decay, schedule),
tuned once and then applied to every arm, so the comparison measures the
pretraining objective rather than how much hyper-parameter search each arm got.

Loading a checkpoint that does not fit the requested architecture is a hard
error, not a warning: a partial load would leave most of the network randomly
initialised and still report a perfectly plausible metric.

## Swap the backbone

A different architecture is a config change, not a code change:

```bash
sparc-pretrain --config configs/pretrain/sparc_coco_r18.yaml --set backbone.name=resnet50
sparc-pretrain --config configs/pretrain/sparc_coco_r18.yaml --set backbone.name=timm:convnext_tiny
```

The same override works for `sparc-eval`. Built in: `resnet18/34/50/101/152`.
Any [timm](https://github.com/huggingface/pytorch-image-models) model works via
the `timm:` prefix (`pip install 'sparc-ssl[timm]'`). Adding a new family means
writing one registry entry — the FCN and Faster R-CNN heads size themselves from
the per-stage channel counts it reports, so neither needs to know the
architecture.

## What this repository does and does not contain

Code, configs and experiment manifests.

**No datasets, no superpixel masks and no model weights.** Masks are generated
by `sparc-masks`; pretrained weights are published as GitHub Release assets.
This keeps a clone small, and is enforced by CI rather than only by
`.gitignore`.

## License

MIT — see [LICENSE](LICENSE).

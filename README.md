# SPARC

**S**uper**P**ixel-**A**ware **R**egion **C**ontrastive learning for self-supervised dense prediction.

> **Status: under construction.** Pretraining works end to end. Downstream
> evaluation, the sweep tooling, the model zoo and the full docs land in
> subsequent phases.

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

## Swap the backbone

A different architecture is a config change, not a code change:

```bash
sparc-pretrain --config configs/pretrain/sparc_coco_r18.yaml --set backbone.name=resnet50
sparc-pretrain --config configs/pretrain/sparc_coco_r18.yaml --set backbone.name=timm:convnext_tiny
```

Built in: `resnet18/34/50/101/152`. Any [timm](https://github.com/huggingface/pytorch-image-models)
model works via the `timm:` prefix (`pip install 'sparc-ssl[timm]'`). Adding a
new family means writing one registry entry — the segmentation and detection
heads build themselves from the channel counts it reports.

## What this repository does and does not contain

Code, configs and experiment manifests.

**No datasets, no superpixel masks and no model weights.** Masks are generated
by `sparc-masks`; pretrained weights are published as GitHub Release assets.
This keeps a clone small, and is enforced by CI rather than only by
`.gitignore`.

## License

MIT — see [LICENSE](LICENSE).

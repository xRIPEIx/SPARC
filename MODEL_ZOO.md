# Model zoo

All weights are **GitHub Release assets** for tag `v1.0.0`, not files in this
repository, so a clone stays small. Download with checksum verification:

```bash
python scripts/download_checkpoints.py            # everything, ~385 MB
python scripts/download_checkpoints.py seg_head   # just the five-minute demo
python scripts/download_checkpoints.py --list
```

## Pretrained encoders

ResNet-18, COCO train2017, 100 epochs, batch 64, single GPU, seed 1. Each file
holds the transferable backbone only (`encoder`, plain torchvision key names)
plus the resolved config it was trained with. Momentum encoder, optimiser state
and memory bank are not included, so these are ~45 MB rather than 181 MB.

| Name | Objective | VOC mIoU | VOC AP | Size |
|---|---|---|---|---|
| `sparc_lambda_0p5` | **SPARC, λ = 0.5 — the paper's reference** | 39.2 | 25.5 | 44.8 MB |
| `sparc_lambda_0` | SPARC at λ = 0, i.e. the MoCo-global objective (the shared endpoint) | 34.4 | 23.7 | 44.8 MB |
| `sparc_lambda_1` | SPARC region term only | 38.3 | 25.5 | 44.8 MB |
| `moco` | MoCo v2 baseline | 34.2 | 23.7 | 44.8 MB |
| `densecl_lambda_0p5` | DenseCL baseline (= DenseCL at its default λ) | 37.4 | 24.0 | 44.8 MB |

Metrics are the seed-1 result for the reference encoder and the 5-seed mean for
the others; see the README table for mean ± s.d. Seed-to-seed s.d. is ≈0.4 mIoU
points, so the three SPARC rows at λ ≥ 0.5 are not distinguishable at this
sample size.

Use one as the initialisation for downstream fine-tuning:

```bash
sparc-eval --config configs/downstream/voc_seg_fcn_r18.yaml \
           --set model.ssl_ckpt=checkpoints/sparc_lambda_0p5_resnet18_coco_ep100.pth \
                 experiment.config_id=sparc_lambda_0p5
```

Or load the backbone into your own code — the `encoder` entry is a plain
`torchvision.models.resnet18` state dict with `fc` removed:

```python
import torch, torchvision
net = torchvision.models.resnet18(); net.fc = torch.nn.Identity()
net.load_state_dict(torch.load("checkpoints/sparc_lambda_0p5_resnet18_coco_ep100.pth")["encoder"])
```

## Fine-tuned VOC2012 models

The reference encoder after downstream fine-tuning (seed 1). These are what
[`scripts/demo_predict.py`](scripts/demo_predict.py) runs, and they need no
dataset at all — point them at any JPEG.

| Name | Model | VOC2012 val | Size |
|---|---|---|---|
| `seg_head` | FCN, ResNet-18 | mIoU 39.2, pixel acc. 84.4 | 47.2 MB |
| `det_head` | Faster R-CNN + FPN, ResNet-18 | AP 25.5, AP50 49.5, AP75 23.4 | 113.6 MB |

```bash
python scripts/demo_predict.py --task segmentation \
    --weights checkpoints/sparc_lambda_0p5_voc_segmentation_resnet18.pth \
    --images path/to/photos --out demo_out
```

## Checksums

`SHA256SUMS` is attached to the release and embedded in the download script;
every download is verified before it is kept.

## Provenance

These are the exact weights behind the published numbers, extracted from the
original study's full training checkpoints by a one-time conversion that copied
the encoder tensors unchanged and rewrote only the metadata. Each was verified
to load into this repository's code with zero missing and zero unexpected
parameters before release.

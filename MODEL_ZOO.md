# Model zoo

No weights live in this repository, so a clone stays small. The released set
covers **every pretrained run behind the README results table**: SPARC and
DenseCL at each of the seven λ values, and MoCo v2, all at five seeds — 75
encoders — plus two fine-tuned heads. They are hosted in two places:

| Where | What | Status |
|---|---|---|
| GitHub Release `v1.0.0` | seed 1 of all 15 configurations + both heads (17 files, 0.83 GB) | available |
| Zenodo record (DOI) | all 77 files, including seeds 2–5 (3.5 GB) | **not yet published** — coming with the archived release |

Seed 1 is enough for the demo, the headline table and one full λ curve per
method. Seeds 2–5 exist only to reproduce the error bars, so they are kept with
the citable archive rather than on the release page. Download with checksum
verification:

```bash
python scripts/download_checkpoints.py                 # seed 1 of every encoder + both heads, ~0.8 GB
python scripts/download_checkpoints.py seg_head        # just the five-minute demo, 47 MB
python scripts/download_checkpoints.py sparc_lambda --seeds 1    # one sweep, seed 1
python scripts/download_checkpoints.py --seeds all     # every published seed (2–5 skipped until the Zenodo record exists)
python scripts/download_checkpoints.py --list          # every file, with its published status
```

## File naming

```
<config_id>_seed<N>_resnet18_coco_ep100.pth              pretrained encoder
sparc_lambda_0p5_seed1_voc_<task>_resnet18.pth           fine-tuned VOC2012 model
```

`config_id` is the run name used everywhere in this repository and in
`experiments/results/aggregate/`; `0p5` reads as 0.5. `seed<N>` is the
pretraining seed, 1–5. The README table is the mean ± s.d. over those five.

## Pretrained encoders

ResNet-18, COCO train2017, 100 epochs, batch 64, single GPU. Each file holds
the transferable backbone only (`encoder`, plain torchvision key names) plus the
resolved config it was trained with. Momentum encoder, optimiser state and
memory bank are not included, so these are ~45 MB rather than 181 MB.

VOC2012 numbers below are the 5-seed mean from `experiments/results/aggregate/`;
the s.d. is ≈0.4 mIoU and ≈0.2 AP points, so rows closer than ≈0.8 mIoU are not
distinguishable at this sample size.

| `config_id` | Objective | VOC mIoU | VOC AP |
|---|---|---|---|
| `moco` | MoCo v2 baseline | 34.2 | 23.7 |
| `sparc_lambda_0` | SPARC, λ = 0 (reduces to MoCo v2) | 34.4 | 23.7 |
| `sparc_lambda_0p1` | SPARC, λ = 0.1 | 36.7 | 24.5 |
| `sparc_lambda_0p3` | SPARC, λ = 0.3 | 38.4 | 25.1 |
| `sparc_lambda_0p5` | **SPARC, λ = 0.5 — the paper's reference** | **38.8** | **25.4** |
| `sparc_lambda_0p7` | SPARC, λ = 0.7 | 38.9 | 25.5 |
| `sparc_lambda_0p9` | SPARC, λ = 0.9 | 38.8 | 25.3 |
| `sparc_lambda_1` | SPARC, region term only | 38.3 | 25.5 |
| `densecl_lambda_0` | DenseCL, λ = 0 (reduces to MoCo v2) | 34.2 | 23.7 |
| `densecl_lambda_0p1` | DenseCL, λ = 0.1 | 36.5 | 24.0 |
| `densecl_lambda_0p3` | DenseCL, λ = 0.3 | 37.5 | 24.2 |
| `densecl_lambda_0p5` | DenseCL, λ = 0.5 — the DenseCL baseline | 37.4 | 24.0 |
| `densecl_lambda_0p7` | DenseCL, λ = 0.7 | 36.6 | 23.5 |
| `densecl_lambda_0p9` | DenseCL, λ = 0.9 | 33.7 | 21.7 |
| `densecl_lambda_1` | DenseCL, pixel term only | 27.6 | 18.4 |

The Random-init and Supervised-ImageNet rows of the README table need no
released weights; torchvision provides both initialisations.

Use one as the initialisation for downstream fine-tuning:

```bash
sparc-eval --config configs/downstream/voc_seg_fcn_r18.yaml \
           --set model.ssl_ckpt=checkpoints/sparc_lambda_0p5_seed1_resnet18_coco_ep100.pth \
                 experiment.config_id=sparc_lambda_0p5
```

Or load the backbone into your own code — the `encoder` entry is a plain
`torchvision.models.resnet18` state dict with `fc` removed:

```python
import torch, torchvision

net = torchvision.models.resnet18()
net.fc = torch.nn.Identity()
net.load_state_dict(
    torch.load("checkpoints/sparc_lambda_0p5_seed1_resnet18_coco_ep100.pth")["encoder"]
)
```

## Fine-tuned VOC2012 models

The reference encoder (`sparc_lambda_0p5`, seed 1) after downstream
fine-tuning. These are what [`scripts/demo_predict.py`](scripts/demo_predict.py)
runs, and they need no dataset at all — point them at any JPEG.

| Name | File | Model | VOC2012 val | Size |
|---|---|---|---|---|
| `seg_head` | `sparc_lambda_0p5_seed1_voc_segmentation_resnet18.pth` | FCN, ResNet-18 | mIoU 39.2, pixel acc. 84.4 | 47.2 MB |
| `det_head` | `sparc_lambda_0p5_seed1_voc_detection_resnet18.pth` | Faster R-CNN + FPN, ResNet-18 | AP 25.5, AP50 49.5, AP75 23.4 | 113.6 MB |

```bash
python scripts/demo_predict.py --task segmentation \
    --weights checkpoints/sparc_lambda_0p5_seed1_voc_segmentation_resnet18.pth \
    --images path/to/photos --out demo_out
```

## Checksums

`SHA256SUMS` and `assets.json` cover all 77 files and are attached to the
release; the same manifest is committed as
[`scripts/release_assets.json`](scripts/release_assets.json), with a
`published` flag per file, and every download is verified against it before it
is kept. The checksums are the same wherever a file is hosted, so a file from
the Zenodo record verifies against the manifest in this repository.

## Provenance

These are the exact weights behind the published numbers, extracted from the
original study's full training checkpoints by a one-time conversion that copied
the encoder tensors unchanged and rewrote only the metadata. Each was verified
to load into this repository's code with zero missing and zero unexpected
parameters before release.

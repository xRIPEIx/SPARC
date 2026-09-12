# The method, with pointers to the code

SPARC adds a **region-level** contrastive term to image-level self-supervised
learning, where the regions are superpixels.

## Objective

```
L = (1 − λ) · L_global + λ · L_region
```

**L_global** — [`methods/sparc.py`](../src/sparc/methods/sparc.py), via lightly's
`NTXentLoss`. NT-Xent between the pooled embedding of view *q* and the momentum
embedding of view *k*, against a memory bank of 4,096 past keys. Temperature
**0.5**.

**L_region** — [`losses/region_contrastive.py`](../src/sparc/losses/region_contrastive.py).
For region embeddings *z* from both views, a pair is **positive** when it comes
from the same image *and* the same original superpixel id; every other valid
region in the candidate view is a negative. Averaged symmetrically over both
directions. Temperature **0.2**. There is no region queue.

At **λ = 0** the objective is exactly MoCo v2. That is what lets the SPARC and
DenseCL λ sweeps meet the MoCo baseline at one shared point — a three-way
identity that `sparc-report check` verifies on the shipped results.

## Where the regions come from

Superpixels ([`data/superpixel/`](../src/sparc/data/superpixel/)), SLIC with 100
segments in the paper, precomputed once per image. The mask is **only ever used
to pool features after the encoder**. It is never an input channel, so a
SPARC-pretrained backbone sees ordinary RGB and transfers downstream with no
mask dependency whatsoever.

## Keeping image and mask aligned

[`data/transforms.py`](../src/sparc/data/transforms.py). Each view is a random
resized crop plus flip plus colour ops. The crop and flip parameters are drawn
once and applied to image (bicubic) and mask (nearest) alike; colour ops touch
the image only. A test checks this by building an image whose pixel values
encode its own labels and asserting they still agree after augmentation.

## Which regions are compared

[`models/region_pooling.py`](../src/sparc/models/region_pooling.py).

1. `match_region_ids`: the superpixel ids visible in **both** augmented views
   (the intersection — a region cropped out of one view cannot be a positive).
   Up to `max_regions = 16` are sampled per image. Label 0 is a real region.
2. `SuperpixelRegionPool`: the C×7×7 feature map is pooled over each selected
   region. Each region's mask is downsampled to the feature grid as **soft
   occupancy** — a cell a quarter covered contributes with weight 0.25 — and
   the pooled vector is the occupancy-weighted mean. A test pins this against a
   hand-computed value.
3. A small MLP ([`models/heads.py`](../src/sparc/models/heads.py)) maps each
   region vector to the embedding the loss compares.

## Shapes, for one batch

| Stage | Shape (B = 64, ResNet-18, 224 px) |
|---|---|
| Views q, k | 64 × 3 × 224 × 224 |
| Masks q, k | 64 × 224 × 224, integer labels |
| Encoder feature map | 64 × 512 × 7 × 7 |
| Global: pool → MLP | 64 × 512 → 64 × 128 |
| Region: pool → MLP | 64 × 16 × 512 → 64 × 16 × 128, plus a 64 × 16 validity mask |
| L_global | scalar, against a 4096 × 128 bank |
| L_region | scalar, over all valid (image, region) pairs in the batch |

## Hyper-parameters the paper uses

`configs/pretrain/sparc_coco_r18.yaml`: λ = 0.5, region temperature 0.2, global
temperature 0.5, projection dim 128, memory bank 4,096, 16 regions per image
pooled on a 7×7 grid, Adam 1e-4, weight decay 1e-4, cosine schedule, batch 64,
100 epochs, momentum 0.996 → 1.0, AMP. The **global temperature of 0.5** is worth
noting: it is lightly's default rather than the 0.1–0.2 usually quoted for
MoCo v2, and every published number was produced with it.

## Baselines

[`methods/moco.py`](../src/sparc/methods/moco.py) is MoCo v2 in substance — MLP
head, v2 augmentations, cosine momentum — with two deviations to state when
citing it: Adam rather than SGD, and no shuffling BatchNorm (single GPU).
[`methods/densecl.py`](../src/sparc/methods/densecl.py) is DenseCL; its own
global-vs-dense mix `dense_lambda` is the companion sweep. All three see the
same augmentation distribution, parameter for parameter.

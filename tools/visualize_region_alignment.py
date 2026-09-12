#!/usr/bin/env python
"""See what SPARC sees: two augmented views, their superpixels, and which
regions were matched across the pair.

    python tools/visualize_region_alignment.py \
        --images /path/to/coco/train2017 --masks /path/to/superpixel_masks/slic_n100 \
        --out /tmp/alignment --num-images 8

Each output panel has four tiles: view 1 and view 2 with superpixel boundaries
drawn on, then the same two views with every *matched* region -- a superpixel
visible in both views, hence a positive pair -- painted in one colour per id.
Unmatched pixels are left grey. This is the mechanism of the region loss made
visible, and it is also the fastest way to notice an image/mask misalignment:
boundaries that do not follow the photo's edges mean the two are out of step.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.morphology import dilation, disk
from skimage.segmentation import find_boundaries
from skimage.util import img_as_ubyte

from sparc.data.mask_dataset import IMAGE_SUFFIXES
from sparc.data.transforms import TwoCropsTransformWithMask
from sparc.models import match_region_ids

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def boundary_overlay(
    image: Image.Image,
    mask: np.ndarray,
    color=(1.0, 1.0, 1.0),
    halo_color=(0.0, 0.0, 0.0),
    line_radius: int = 1,
    halo_radius: int = 2,
) -> Image.Image:
    """Superpixel boundaries as a bright line with a dark halo.

    A thin single-colour line disappears against same-hued regions of a photo
    (red lines on red food). The dark halo guarantees contrast on any
    background; white reads as an annotation rather than a neon overlay.

    halo_radius must stay strictly greater than line_radius: the halo is painted
    first and the line on top, so at equal radii the line's footprint exactly
    covers the halo and the dark ring vanishes.
    """
    rgb = np.asarray(image.convert("RGB")).astype(np.float64) / 255.0
    boundaries = find_boundaries(mask, mode="thick")
    halo = dilation(boundaries, footprint=disk(halo_radius))
    line = dilation(boundaries, footprint=disk(line_radius))
    out = rgb.copy()
    out[halo] = halo_color
    out[line] = color
    return Image.fromarray(img_as_ubyte(out.clip(0, 1)))


def matched_overlay(image: Image.Image, mask: np.ndarray, matched_ids: list[int]) -> Image.Image:
    """Paint matched regions with a stable per-id colour; leave the rest grey."""
    rgb = np.asarray(image.convert("RGB")).astype(np.float64) / 255.0
    grey = rgb.mean(axis=-1, keepdims=True).repeat(3, axis=-1) * 0.55 + 0.25
    out = grey.copy()
    rng = np.random.default_rng(0)
    palette = rng.uniform(0.25, 1.0, size=(max(matched_ids, default=0) + 1, 3))
    for rid in matched_ids:
        sel = mask == rid
        out[sel] = 0.35 * rgb[sel] + 0.65 * palette[rid]
    return Image.fromarray(img_as_ubyte(out.clip(0, 1)))


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    arr = t.permute(1, 2, 0).numpy() * STD + MEAN
    return Image.fromarray(img_as_ubyte(arr.clip(0, 1)))


def iter_images(root: Path) -> list[Path]:
    return sorted(
        Path(d) / f
        for d, _, fs in os.walk(root)
        for f in fs
        if Path(f).suffix.lower() in IMAGE_SUFFIXES
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--masks", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--num-images", type=int, default=8)
    ap.add_argument("--input-size", type=int, default=224)
    ap.add_argument("--max-regions", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    transform = TwoCropsTransformWithMask(input_size=args.input_size)

    saved = 0
    for image_path in iter_images(args.images):
        if saved >= args.num_images:
            break
        mask_path = args.masks / image_path.relative_to(args.images).with_suffix(".npy")
        if not mask_path.exists():
            continue
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        mask = np.load(mask_path).astype(np.int64, copy=False)
        if mask.shape != (image.height, image.width):
            continue

        v1, v2, m1, m2 = transform(image, mask)
        ids_q, _, valid_q, _ = match_region_ids(m1[None], m2[None], max_regions=args.max_regions)
        matched = sorted(ids_q[0][valid_q[0]].tolist())

        img1, img2 = tensor_to_image(v1), tensor_to_image(v2)
        n1, n2 = m1.numpy(), m2.numpy()
        tiles = [
            boundary_overlay(img1, n1),
            boundary_overlay(img2, n2),
            matched_overlay(img1, n1, matched),
            matched_overlay(img2, n2, matched),
        ]
        size = args.input_size
        panel = Image.new("RGB", (4 * size + 3 * 4, size), (255, 255, 255))
        for i, tile in enumerate(tiles):
            panel.paste(tile, (i * (size + 4), 0))
        dest = args.out / f"{image_path.stem}.png"
        panel.save(dest)
        visible = f"{len(np.unique(n1))}/{len(np.unique(n2))}"
        print(f"{dest}: {len(matched)} matched regions of {visible} visible")
        saved += 1

    if saved == 0:
        raise SystemExit("No image/mask pairs found; check --images and --masks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

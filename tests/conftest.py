"""Shared fixtures.

Everything here is synthetic and tiny: the fast suite must run on CPU with no
datasets, no network and no GPU, so that CI actually gets run.
"""

from __future__ import annotations

import os

# Must precede the torch import: the tensors in this suite are tiny, and letting
# BLAS/OMP fan out across every core spends far more time on thread handoff than
# on arithmetic. On a 64-core login node this took a 2x3x64x64 resnet18 forward
# from 22s to well under a second.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pytest
import torch

torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def _deterministic():
    """Seed every RNG the library can touch, before every test."""
    torch.manual_seed(0)
    np.random.seed(0)
    import random

    random.seed(0)


@pytest.fixture
def image_mask_pair():
    """Factory for an RGB image that encodes its own mask labels.

    Exposed as a fixture rather than a module-level import: `tests` is a common
    top-level package name (ultralytics installs one), so `from tests.conftest
    import ...` resolves to whatever is first on sys.path rather than to this
    file.
    """
    return _make_image_mask_pair


@pytest.fixture
def two_region_mask():
    """[1, 8, 8] mask: top half label 0, bottom half label 1.

    Label 0 is deliberately used -- it is a *valid* region id in this codebase,
    not a background/ignore marker, and that is easy to regress.
    """
    mask = torch.zeros(1, 8, 8, dtype=torch.long)
    mask[:, 4:, :] = 1
    return mask


def _make_image_mask_pair(size: int = 64, n_labels: int = 2):
    """An RGB image whose content encodes its own mask label.

    Left half is black / label 0, right half is white / label 1. Any transform
    that moves the image without moving the mask identically will break the
    correspondence, which is what the alignment tests look for.
    """
    from PIL import Image

    arr = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.int32)
    arr[:, size // 2 :, :] = 255
    mask[:, size // 2 :] = 1
    if n_labels > 2:  # add horizontal banding for a harder case
        for i in range(2, n_labels):
            lo = (i - 1) * size // n_labels
            hi = i * size // n_labels
            arr[lo:hi, :, :] = i * (255 // n_labels)
            mask[lo:hi, :] = i
    return Image.fromarray(arr), mask


@pytest.fixture
def synthetic_corpus(tmp_path):
    """A tiny image+mask corpus that `sparc-pretrain` can actually train on.

    Four regions per image (quadrants), so region matching has something real to
    do rather than degenerating to a single region.
    """
    from PIL import Image

    from sparc.data.superpixel import write_meta

    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    rng = np.random.default_rng(0)
    h = w = 48
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(8):
        arr = np.stack([yy * 4 % 256, xx * 4 % 256, (yy + xx) * 2 % 256], -1).astype(np.uint8)
        arr = np.clip(arr + rng.normal(0, 10, arr.shape), 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(images / f"img{i:03d}.png")
        mask = np.zeros((h, w), dtype=np.int32)
        mask[h // 2 :, :] = 1
        mask[:, w // 2 :] += 2
        np.save(masks / f"img{i:03d}.npy", mask)
    write_meta(masks, {"mode": "precomputed_npy", "method": "synthetic", "n_segments": 4})
    return images, masks

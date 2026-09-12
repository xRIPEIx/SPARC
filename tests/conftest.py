"""Shared fixtures.

Everything here is synthetic and tiny: the fast suite must run on CPU with no
datasets, no network and no GPU, so that CI actually gets run.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


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

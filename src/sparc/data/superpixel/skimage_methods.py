"""Superpixel generators that run in-process via scikit-image.

Every method honours one contract:

    (image_rgb: np.ndarray[H, W, 3], n_segments: int, **kwargs) -> np.ndarray[H, W] of int

`sparc.cli.masks` dispatches to these by name. SLIC is the method used in the
paper; felzenszwalb and compact_watershed are kept as worked examples showing
that the contract accommodates algorithms with no direct segment-count knob.

To add your own, see docs/extending.md -- it is one function plus one YAML file.

NOTE ON PINNING: scikit-image is pinned exactly in pyproject.toml because
slic() output can change between releases, which would silently change every
downstream number.
"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, watershed
from skimage.util import regular_grid

METHOD_CHOICES = ("slic", "felzenszwalb", "compact_watershed")


def compute_slic_mask(
    image_rgb: np.ndarray,
    n_segments: int,
    compactness: float = 10.0,
    sigma: float = 1.0,
) -> np.ndarray:
    return slic(
        image_rgb,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        channel_axis=-1,
    )


def compute_felzenszwalb_mask(
    image_rgb: np.ndarray,
    n_segments: int,
    sigma: float = 1.0,
    min_size_floor: int = 5,
) -> np.ndarray:
    """Felzenszwalb-Huttenlocher graph-based segmentation.

    Felzenszwalb has no direct segment-count knob (it exposes `scale`, which
    trades off region count non-monotonically with image content). We binary
    search `scale` per image so the output segment count lands close to
    `n_segments`, keeping the same `--n-segments`-driven CLI contract as SLIC.
    """
    image_h, image_w = image_rgb.shape[:2]
    min_size = max(min_size_floor, int((image_h * image_w) / max(n_segments, 1) / 4))

    scale_low, scale_high = 1.0, 5000.0
    best_mask = None
    best_diff = None
    for _ in range(12):
        scale = (scale_low + scale_high) / 2
        mask = felzenszwalb(image_rgb, scale=scale, sigma=sigma, min_size=min_size)
        count = int(mask.max()) + 1
        diff = abs(count - n_segments)
        if best_diff is None or diff < best_diff:
            best_diff, best_mask = diff, mask
        if count > n_segments:
            scale_low = scale
        else:
            scale_high = scale

    return best_mask


def compute_compact_watershed_mask(
    image_rgb: np.ndarray,
    n_segments: int,
    compactness: float = 0.001,
) -> np.ndarray:
    """Compact watershed (Neubert & Protzel): SLIC-style regular-grid seeds,
    flooded on a gradient image with a compactness penalty.

    Unlike Felzenszwalb/Quickshift, seed count directly controls output
    label count (up to merging away unreachable seeds), so no search over a
    proxy parameter is needed to hit the requested `n_segments` -- one
    `watershed` call per image, same cost profile as SLIC.
    """
    image_h, image_w = image_rgb.shape[:2]
    grid = regular_grid((image_h, image_w), n_segments)
    markers = np.zeros((image_h, image_w), dtype=np.int64)
    grid_view = markers[grid]
    markers[grid] = np.arange(1, grid_view.size + 1).reshape(grid_view.shape)

    gradient = sobel(rgb2gray(image_rgb))
    labels = watershed(gradient, markers=markers, compactness=compactness)
    return labels - 1  # markers start at 1; shift to match start_label=0 elsewhere


def compute_mask(method: str, image_rgb: np.ndarray, n_segments: int, **kwargs) -> np.ndarray:
    if method == "slic":
        return compute_slic_mask(image_rgb, n_segments, **kwargs)
    if method == "felzenszwalb":
        return compute_felzenszwalb_mask(image_rgb, n_segments, **kwargs)
    if method == "compact_watershed":
        return compute_compact_watershed_mask(image_rgb, n_segments, **kwargs)
    raise ValueError(
        f"Unknown superpixel method {method!r}. Available: {METHOD_CHOICES}. "
        "To add one, see docs/extending.md."
    )

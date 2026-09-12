"""Superpixel method registry.

Every method honours one contract:

    (image_rgb: np.ndarray[H, W, 3], n_segments: int, **kwargs) -> np.ndarray[H, W] of int

Labels must start at 0 and be dense (no gaps), because downstream code treats
label 0 as a real region and max()+1 as the region count.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import numpy as np

SuperpixelFn = Callable[..., np.ndarray]
_METHODS: dict[str, SuperpixelFn] = {}


def register_superpixel(name: str) -> Callable[[SuperpixelFn], SuperpixelFn]:
    def decorator(fn: SuperpixelFn) -> SuperpixelFn:
        if name in _METHODS:
            raise ValueError(f"Superpixel method {name!r} is already registered")
        _METHODS[name] = fn
        return fn

    return decorator


def list_superpixel_methods() -> list[str]:
    return sorted(_METHODS)


def get_superpixel_method(name: str) -> SuperpixelFn:
    try:
        return _METHODS[name]
    except KeyError:
        raise ValueError(
            f"Unknown superpixel method {name!r}. Available: {list_superpixel_methods()}. "
            f"To add one, see docs/extending.md."
        ) from None


def accepted_kwargs(name: str, candidates: dict) -> dict:
    """Keep only the keyword arguments the method's signature accepts.

    Lets one config schema list every method's tunables (compactness, sigma,
    ...) while each method takes only its own.
    """
    params = inspect.signature(get_superpixel_method(name)).parameters
    return {k: v for k, v in candidates.items() if k in params and v is not None}


def compute_mask(method: str, image_rgb: np.ndarray, n_segments: int, **kwargs) -> np.ndarray:
    mask = get_superpixel_method(method)(image_rgb, n_segments, **kwargs)
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.shape != image_rgb.shape[:2]:
        raise ValueError(
            f"{method}: expected an [H, W] label map matching the image, "
            f"got {mask.shape} for image {image_rgb.shape[:2]}"
        )
    if not np.issubdtype(mask.dtype, np.integer):
        raise ValueError(f"{method}: label map must be integer, got {mask.dtype}")
    return mask

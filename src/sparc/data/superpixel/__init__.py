"""Superpixel generation: registry, built-in methods, mask-directory metadata."""

# Importing the built-ins fires their registration decorators.
from sparc.data.superpixel import skimage_methods  # noqa: E402,F401  (side effect)
from sparc.data.superpixel.meta import (
    FORMAT_ID,
    META_FILENAME,
    is_synthetic_constant,
    load_meta,
    write_meta,
)
from sparc.data.superpixel.registry import (
    accepted_kwargs,
    compute_mask,
    get_superpixel_method,
    list_superpixel_methods,
    register_superpixel,
)

#: Kept for callers that enumerate methods; the registry is the source of truth.
METHOD_CHOICES = tuple(list_superpixel_methods())

__all__ = [
    "FORMAT_ID",
    "METHOD_CHOICES",
    "META_FILENAME",
    "accepted_kwargs",
    "compute_mask",
    "get_superpixel_method",
    "is_synthetic_constant",
    "list_superpixel_methods",
    "load_meta",
    "register_superpixel",
    "write_meta",
]

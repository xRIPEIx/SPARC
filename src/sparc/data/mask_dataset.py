"""Image dataset backed by precomputed or synthetic superpixel masks."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from sparc.data.superpixel.meta import is_synthetic_constant, load_meta


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


class RegionMaskDataset(Dataset):
    """Pairs RGB images with superpixel label maps.

    Mask roots may contain either per-image ``.npy`` files or a
    ``superpixel_mask_meta.json`` with ``mode: synthetic_constant`` (e.g. n_segments=1),
    in which case a constant label map is synthesized to match each image size.
    """

    def __init__(self, image_root: str | Path, mask_root: str | Path, transform):
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)
        self.transform = transform
        self.meta = load_meta(self.mask_root)
        self.synthetic = is_synthetic_constant(self.meta)
        self.constant_label = int((self.meta or {}).get("constant_label", 0))

        if not self.image_root.exists():
            raise FileNotFoundError(f"Image root does not exist: {self.image_root}")
        if not self.mask_root.exists():
            raise FileNotFoundError(f"Superpixel mask root does not exist: {self.mask_root}")

        self.image_paths = sorted(
            Path(dirpath) / filename
            for dirpath, _dirnames, filenames in os.walk(self.image_root)
            for filename in filenames
            if Path(filename).suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.image_paths:
            raise RuntimeError(f"No supported images found under: {self.image_root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
            if self.synthetic:
                mask = np.full(
                    (image_rgb.height, image_rgb.width),
                    self.constant_label,
                    dtype=np.int32,
                )
            else:
                mask_path = self.mask_path_for(image_path)
                if not mask_path.exists():
                    raise FileNotFoundError(
                        f"Missing superpixel mask for {image_path}: {mask_path}"
                    )
                mask = np.load(mask_path).astype(np.int32, copy=False)

                if mask.shape != (image_rgb.height, image_rgb.width):
                    raise RuntimeError(
                        f"Mask shape {mask.shape} does not match image shape "
                        f"{(image_rgb.height, image_rgb.width)} for {image_path}"
                    )

            return self.transform(image_rgb, mask)

    def mask_path_for(self, image_path: Path) -> Path:
        relative_path = image_path.relative_to(self.image_root)
        return self.mask_root / relative_path.with_suffix(".npy")

"""Datasets and dataloaders for SSL pretraining."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from sparc.data.mask_dataset import IMAGE_SUFFIXES, RegionMaskDataset
from sparc.data.transforms import TwoCropsTransformWithMask
from sparc.methods.base import SSLBatch
from sparc.utils.seed import dataloader_generator, worker_init_fn


class ImageDataset(Dataset):
    """Images only, for methods that do not use region masks.

    Mirrors RegionMaskDataset's traversal and ordering so that a run differs
    between methods only in the objective, not in which images it sees or in
    what order.
    """

    def __init__(self, image_root: str | Path, transform):
        self.image_root = Path(image_root)
        self.transform = transform
        if not self.image_root.exists():
            raise FileNotFoundError(f"Image root does not exist: {self.image_root}")
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
        with Image.open(self.image_paths[index]) as image:
            return self.transform(image.convert("RGB"))


class _Subset(Dataset):
    """First-n view, used only by `data.limit` for smoke tests."""

    def __init__(self, base: Dataset, limit: int):
        self.base = base
        self.limit = min(limit, len(base))  # type: ignore[arg-type]

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int):
        return self.base[index]


def build_transform(*, requires_masks: bool, method_name: str, input_size: int):
    """Augmentation pipeline.

    Methods that need masks use the paired transform, which shares crop and flip
    parameters between image (bicubic) and mask (nearest). Mask-free methods use
    lightly's reference transforms.

    These agree parameter for parameter -- min_scale 0.2, colour jitter
    0.4/0.4/0.4/0.1 at p=0.8, grayscale 0.2, blur 0.5, hflip 0.5, ImageNet
    normalisation -- so every method sees the same augmentation distribution and
    a difference in results cannot be attributed to a difference in augmentation.
    """
    if requires_masks:
        return TwoCropsTransformWithMask(input_size=input_size)
    if method_name == "densecl":
        from lightly.transforms.densecl_transform import DenseCLTransform

        return DenseCLTransform(input_size=input_size)
    from lightly.transforms.moco_transform import MoCoV2Transform

    return MoCoV2Transform(input_size=input_size)


def collate_with_masks(samples) -> SSLBatch:
    import torch

    v1, v2, m1, m2 = zip(*samples, strict=True)
    return SSLBatch(torch.stack(v1), torch.stack(v2), torch.stack(m1), torch.stack(m2))


def collate_images(samples) -> SSLBatch:
    import torch

    v1, v2 = zip(*[(s[0], s[1]) for s in samples], strict=True)
    return SSLBatch(torch.stack(v1), torch.stack(v2))


def build_pretrain_dataloader(config, *, requires_masks: bool) -> DataLoader:
    """Construct the pretraining dataloader from a resolved config."""
    transform = build_transform(
        requires_masks=requires_masks,
        method_name=config.method.name,
        input_size=config.data.input_size,
    )

    if requires_masks:
        if not config.data.masks:
            raise ValueError(
                f"Method {config.method.name!r} needs region masks but data.masks is unset. "
                f"Generate them with `sparc-masks`, then set data.masks to that directory."
            )
        dataset: Dataset = RegionMaskDataset(config.data.images, config.data.masks, transform)
        collate = collate_with_masks
    else:
        dataset = ImageDataset(config.data.images, transform)
        collate = collate_images

    if config.data.limit:
        dataset = _Subset(dataset, int(config.data.limit))

    return DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        drop_last=True,
        # Only meaningful with a CUDA target; warns otherwise.
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
        # Explicit, rather than relying on PyTorch's happens-to-be-deterministic
        # worker seeding. See sparc.utils.seed.
        generator=dataloader_generator(config.train.seed),
        worker_init_fn=worker_init_fn,
        persistent_workers=config.data.num_workers > 0,
    )

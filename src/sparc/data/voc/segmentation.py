"""VOC segmentation datasets, transforms and batching."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from sparc.data.voc._retry import retry_read
from sparc.data.voc.classes import IGNORE_INDEX

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SegTrainTransform:
    """Random scale, pad, crop and flip, applied identically to image and mask.

    The mask is resized NEAREST and padded with IGNORE_INDEX so that padded
    pixels contribute to neither the loss nor mIoU.
    """

    def __init__(
        self,
        crop_size: int = 512,
        min_scale: float = 0.5,
        max_scale: float = 2.0,
        hflip_prob: float = 0.5,
    ):
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.hflip_prob = hflip_prob

    def __call__(self, image: Image.Image, mask: Image.Image):
        scale = random.uniform(self.min_scale, self.max_scale)
        w, h = image.size
        new_h, new_w = int(h * scale), int(w * scale)

        image = TF.resize(image, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [new_h, new_w], interpolation=TF.InterpolationMode.NEAREST)

        if new_h < self.crop_size or new_w < self.crop_size:
            pad_h = max(self.crop_size - new_h, 0)
            pad_w = max(self.crop_size - new_w, 0)
            image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)
            mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=IGNORE_INDEX)

        i, j, h_crop, w_crop = torchvision.transforms.RandomCrop.get_params(
            image, output_size=(self.crop_size, self.crop_size)
        )
        image = TF.crop(image, i, j, h_crop, w_crop)
        mask = TF.crop(mask, i, j, h_crop, w_crop)

        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = TF.normalize(TF.to_tensor(image), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return image, torch.as_tensor(np.array(mask), dtype=torch.long)


class SegEvalTransform:
    """Resize the shorter side; no cropping, so evaluation sees whole images."""

    def __init__(self, resize_size: int = 520):
        self.resize_size = resize_size

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.resize(image, [self.resize_size], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.resize_size], interpolation=TF.InterpolationMode.NEAREST)
        image = TF.normalize(TF.to_tensor(image), mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return image, torch.as_tensor(np.array(mask), dtype=torch.long)


class SegWrapper(Dataset):
    """Applies a paired transform to a torchvision segmentation dataset."""

    def __init__(self, base_dataset: Dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, mask = retry_read(
            lambda: self.base_dataset[index], what=f"segmentation sample at index {index}"
        )
        return self.transform(image, mask)


def collate_segmentation(batch):
    """Pad a batch to a common size.

    Images pad with 0, masks with IGNORE_INDEX so padding affects neither the
    loss nor the confusion matrix.
    """
    images, masks = zip(*batch, strict=True)
    max_h = max(img.shape[-2] for img in images)
    max_w = max(img.shape[-1] for img in images)

    padded_images, padded_masks = [], []
    for img, mask in zip(images, masks, strict=True):
        h, w = img.shape[-2:]
        pad = (0, max_w - w, 0, max_h - h)  # (left, right, top, bottom)
        padded_images.append(F.pad(img, pad, value=0.0))
        padded_masks.append(F.pad(mask, pad, value=IGNORE_INDEX))

    return torch.stack(padded_images, dim=0), torch.stack(padded_masks, dim=0)


def build_segmentation_datasets(
    voc_root: str,
    *,
    sbd_root: str | None = None,
    train_source: str = "voc2012",
    crop_size: int = 512,
    eval_size: int = 520,
    download: bool = False,
):
    """Train from VOC2012 train or SBD train_noval; always validate on VOC2012 val."""
    if train_source == "sbd":
        if not sbd_root:
            raise ValueError("train_source='sbd' requires sbd_root")
        train_base = torchvision.datasets.SBDataset(
            root=sbd_root, image_set="train_noval", mode="segmentation", download=download
        )
    elif train_source == "voc2012":
        train_base = torchvision.datasets.VOCSegmentation(
            root=voc_root, year="2012", image_set="train", download=download
        )
    else:
        raise ValueError(f"Unknown train_source {train_source!r}; expected 'voc2012' or 'sbd'")

    val_base = torchvision.datasets.VOCSegmentation(
        root=voc_root, year="2012", image_set="val", download=download
    )
    return (
        SegWrapper(train_base, SegTrainTransform(crop_size=crop_size)),
        SegWrapper(val_base, SegEvalTransform(resize_size=eval_size)),
    )

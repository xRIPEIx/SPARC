"""Paired SSL transforms for RGB images and precomputed region masks."""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms


class GaussianBlur:
    def __init__(self, sigma: tuple[float, float] = (0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, image: Image.Image) -> Image.Image:
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))


class TwoCropsTransformWithMask:
    """Create two SSL RGB views with spatially aligned mask views."""

    def __init__(
        self,
        input_size: int = 224,
        scale: tuple[float, float] = (0.2, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        hflip_prob: float = 0.5,
        color_jitter_prob: float = 0.8,
        grayscale_prob: float = 0.2,
        blur_prob: float = 0.5,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.size = (input_size, input_size)
        self.scale = scale
        self.ratio = ratio
        self.hflip_prob = hflip_prob
        self.color_jitter_prob = color_jitter_prob
        self.grayscale_prob = grayscale_prob
        self.blur_prob = blur_prob

        self.color_jitter = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
        )
        self.blur = GaussianBlur()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(
        self, image: Image.Image, mask: Image.Image | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        view1, mask1 = self.transform_one_view(image, mask)
        view2, mask2 = self.transform_one_view(image, mask)
        return view1, view2, mask1, mask2

    def transform_one_view(
        self, image: Image.Image, mask: Image.Image | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(mask, Image.Image):
            mask_array = np.asarray(mask)
            if np.issubdtype(mask_array.dtype, np.integer):
                mask_array = mask_array.astype(np.int32, copy=False)
            mask = Image.fromarray(mask_array)

        # Pass the PIL image straight through: get_params only reads the
        # dimensions (via F.get_dimensions) and accepts PIL or tensor alike,
        # so the previous self.to_tensor(image) was materializing a
        # full-resolution float32 copy (~3.7MB for a 640x480 COCO image, plus
        # the uint8->float divide) purely to be measured and discarded --
        # twice per sample, since transform_one_view runs once per view.
        # Output is unchanged: (C, H, W) is identical either way and the
        # random draws don't depend on the input type, so the seed stream is
        # untouched and results stay comparable with earlier runs.
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image,
            scale=list(self.scale),
            ratio=list(self.ratio),
        )
        image = image.crop((j, i, j + w, i + h)).resize(
            self.size, resample=Image.Resampling.BICUBIC
        )
        mask = mask.crop((j, i, j + w, i + h)).resize(self.size, resample=Image.Resampling.NEAREST)

        if random.random() < self.hflip_prob:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        if random.random() < self.color_jitter_prob:
            image = self.color_jitter(image)

        if random.random() < self.grayscale_prob:
            image = ImageOps.grayscale(image).convert("RGB")

        if random.random() < self.blur_prob:
            image = self.blur(image)

        image_tensor = self.normalize(self.to_tensor(image))
        mask_tensor = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image_tensor, mask_tensor

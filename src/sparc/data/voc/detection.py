"""VOC detection dataset, transform and batching."""

from __future__ import annotations

import random
from typing import Any

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from sparc.data.voc._retry import retry_read
from sparc.data.voc.classes import VOC_CLASS_TO_IDX


class VOCDetectionTransform:
    """Tensor conversion plus horizontal flip, with boxes flipped to match."""

    def __init__(self, train: bool = True, hflip_prob: float = 0.5):
        self.train = train
        self.hflip_prob = hflip_prob

    def __call__(self, image: Image.Image, target: dict[str, torch.Tensor]):
        image = TF.to_tensor(image)

        if self.train and random.random() < self.hflip_prob:
            _, _h, w = image.shape
            image = TF.hflip(image)
            boxes = target["boxes"].clone()
            if boxes.numel() > 0:
                x_min = boxes[:, 0].clone()
                x_max = boxes[:, 2].clone()
                boxes[:, 0] = w - x_max
                boxes[:, 2] = w - x_min
                target["boxes"] = boxes

        return image, target


class VOCDetectionDataset(Dataset):
    def __init__(
        self,
        root: str,
        year: str = "2012",
        image_set: str = "train",
        train: bool = True,
        download: bool = False,
        ignore_difficult: bool = True,
    ):
        self.dataset = torchvision.datasets.VOCDetection(
            root=root, year=year, image_set=image_set, download=download
        )
        self.transform = VOCDetectionTransform(train=train)
        self.ignore_difficult = ignore_difficult
        self.year = year
        self.image_set = image_set

    def __len__(self) -> int:
        return len(self.dataset)

    def _parse_objects(self, annotation: dict[str, Any]):
        objects = annotation["annotation"].get("object", []) or []
        if isinstance(objects, dict):
            objects = [objects]

        boxes, labels, iscrowd = [], [], []
        for obj in objects:
            difficult = int(obj.get("difficult", 0))
            if self.ignore_difficult and difficult == 1:
                continue
            name = obj["name"]
            if name not in VOC_CLASS_TO_IDX:
                continue

            bbox = obj["bndbox"]
            # VOC annotations are 1-indexed; convert to 0-indexed pixel coords.
            xmin = float(bbox["xmin"]) - 1.0
            ymin = float(bbox["ymin"]) - 1.0
            xmax = float(bbox["xmax"]) - 1.0
            ymax = float(bbox["ymax"]) - 1.0
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(VOC_CLASS_TO_IDX[name])
            iscrowd.append(difficult)

        if not boxes:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
                torch.zeros((0,), dtype=torch.int64),
            )
        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.int64),
            torch.tensor(iscrowd, dtype=torch.int64),
        )

    def __getitem__(self, index: int):
        image, raw_target = retry_read(
            lambda: self.dataset[index], what=f"VOC annotation at index {index}"
        )
        boxes, labels, iscrowd = self._parse_objects(raw_target)

        area = torch.zeros((boxes.shape[0],), dtype=torch.float32)
        if boxes.numel() > 0:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }
        return self.transform(image, target)


def collate_detection(batch):
    """Detection images vary in size and are passed as a list, not a tensor."""
    return tuple(zip(*batch, strict=True))


def build_detection_datasets(
    voc_root: str, *, download: bool = False, ignore_difficult: bool = True
):
    """VOC2012 train -> VOC2012 val.

    NOTE: VOC2012's Main/val.txt (detection) and Segmentation/val.txt are
    different, only partially overlapping image subsets. Some
    segmentation-annotated images carry no bounding boxes at all, so a fixed
    sample list from one task must never be reused for the other.
    """
    train = VOCDetectionDataset(
        root=voc_root,
        image_set="train",
        train=True,
        download=download,
        ignore_difficult=ignore_difficult,
    )
    val = VOCDetectionDataset(
        root=voc_root,
        image_set="val",
        train=False,
        download=download,
        ignore_difficult=ignore_difficult,
    )
    return train, val

"""PASCAL VOC class definitions."""

from __future__ import annotations

VOC_CLASSES = [
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

VOC_CLASS_TO_IDX = {name: idx for idx, name in enumerate(VOC_CLASSES)}
NUM_VOC_CLASSES = 21

#: Segmentation label for pixels that must not contribute to loss or mIoU.
#: Also used as the padding value when batching variable-size masks.
IGNORE_INDEX = 255

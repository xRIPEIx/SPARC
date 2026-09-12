"""Downstream metrics.

Segmentation metrics are accumulated in a confusion matrix so that mIoU is
computed over the whole validation set at once, rather than averaged per batch
(which would weight small images the same as large ones).
"""

from __future__ import annotations

import torch

from sparc.data.voc.classes import IGNORE_INDEX, NUM_VOC_CLASSES


def update_confusion_matrix(
    confmat: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_VOC_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    pred = pred.view(-1)
    target = target.view(-1)

    keep = target != ignore_index
    pred, target = pred[keep], target[keep]
    keep = (target >= 0) & (target < num_classes)
    pred, target = pred[keep], target[keep]

    indices = num_classes * target + pred
    confmat += torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
    return confmat


def compute_miou_from_confmat(confmat: torch.Tensor) -> dict[str, float]:
    """mIoU averaged over classes that appear, plus overall pixel accuracy.

    Classes with an empty union are excluded rather than counted as zero: a
    class absent from both predictions and ground truth is undefined, not wrong,
    and counting it would drag mIoU down by a fixed amount.
    """
    confmat = confmat.float()
    tp = torch.diag(confmat)
    union = confmat.sum(dim=1) + confmat.sum(dim=0) - tp
    valid = union > 0

    iou = tp / torch.clamp(union, min=1.0)
    pixel_acc = tp.sum() / torch.clamp(confmat.sum(), min=1.0)
    return {
        "mIoU": float(iou[valid].mean().item()) if valid.any() else 0.0,
        "pixel_acc": float(pixel_acc.item()),
    }


def build_map_metric():
    """torchmetrics mean average precision, imported lazily.

    The previous implementation swallowed this ImportError at module import and
    set the symbol to None, so a missing dependency surfaced only once
    evaluation reached the metric -- potentially hours into a job.
    """
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "Detection AP needs torchmetrics and pycocotools:\n"
            "  pip install 'sparc-ssl[dev]'  (or: pip install torchmetrics pycocotools)"
        ) from exc
    return MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=False)

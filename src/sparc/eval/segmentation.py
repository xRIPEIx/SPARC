"""VOC semantic segmentation fine-tuning and evaluation."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from sparc.data.voc.classes import IGNORE_INDEX, NUM_VOC_CLASSES
from sparc.eval.metrics import compute_miou_from_confmat, update_confusion_matrix
from sparc.utils.amp import autocast


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    *,
    scaler: torch.amp.GradScaler,
    amp: bool = False,
    print_freq: int = 50,
) -> float:
    model.train()
    total_loss = 0.0
    began = time.time()

    for step, (images, masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp):
            logits = model(images)["out"]
            loss = F.cross_entropy(logits, masks, ignore_index=IGNORE_INDEX)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach())

        if print_freq and step % print_freq == 0:
            print(
                f"[seg] epoch={epoch} step={step}/{len(loader)} loss={total_loss / step:.4f}",
                flush=True,
            )

    avg = total_loss / max(len(loader), 1)
    print(f"[seg] epoch={epoch} train_loss={avg:.4f} time={time.time() - began:.1f}s", flush=True)
    return avg


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    confmat = torch.zeros((NUM_VOC_CLASSES, NUM_VOC_CLASSES), dtype=torch.int64)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)["out"]
        # The FCN head outputs at stride 8/32; upsample to label resolution
        # rather than downsampling the labels, which would discard thin classes.
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
        confmat = update_confusion_matrix(confmat, logits.argmax(dim=1).cpu(), masks)

    results = compute_miou_from_confmat(confmat)
    print(f"[seg eval] {results}", flush=True)
    return results

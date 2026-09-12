"""VOC object detection fine-tuning and evaluation."""

from __future__ import annotations

import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from sparc.eval.metrics import build_map_metric
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

    for step, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp):
            # In train mode torchvision detection models return a dict of losses
            # rather than predictions.
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach())

        if print_freq and step % print_freq == 0:
            parts = {k: round(float(v.detach()), 4) for k, v in loss_dict.items()}
            print(
                f"[det] epoch={epoch} step={step}/{len(loader)} "
                f"loss={total_loss / step:.4f} parts={parts}",
                flush=True,
            )

    avg = total_loss / max(len(loader), 1)
    print(f"[det] epoch={epoch} train_loss={avg:.4f} time={time.time() - began:.1f}s", flush=True)
    return avg


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    metric = build_map_metric()

    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)
        preds = [
            {
                "boxes": o["boxes"].detach().cpu(),
                "scores": o["scores"].detach().cpu(),
                "labels": o["labels"].detach().cpu(),
            }
            for o in outputs
        ]
        gts = [
            {"boxes": t["boxes"].detach().cpu(), "labels": t["labels"].detach().cpu()}
            for t in targets
        ]
        metric.update(preds, gts)

    out = metric.compute()
    results = {
        "AP": float(out["map"]),
        "AP50": float(out["map_50"]),
        "AP75": float(out["map_75"]),
    }
    print(f"[det eval] {results}", flush=True)
    return results

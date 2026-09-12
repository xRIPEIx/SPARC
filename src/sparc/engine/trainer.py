"""The pretraining loop.

Method-agnostic by construction: it schedules the momentum coefficient, calls
`update_momentum` then `training_step`, and handles AMP, stepping, logging and
checkpointing. It does not know which objective is running, and adding a method
does not change this file.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from lightly.utils.scheduler import cosine_schedule

from sparc.methods.base import SSLBatch, SSLMethod
from sparc.utils.amp import autocast


@dataclass
class EpochResult:
    epoch: int
    avg_loss: float
    lr: float
    seconds: float
    metrics: dict[str, float]


def train_one_epoch(
    method: SSLMethod,
    loader: Iterable[SSLBatch],
    *,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    momentum: float,
    amp: bool = False,
    log_every: int = 0,
) -> tuple[float, dict[str, float]]:
    method.train()
    total = 0.0
    sums: dict[str, float] = {}
    steps = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        # Applied once per step, before the forward pass, so the key encoder the
        # queries are contrasted against is the updated one.
        method.update_momentum(momentum)

        with autocast(enabled=amp):
            output = method.training_step(batch)

        scaler.scale(output.loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += float(output.loss.detach())
        for key, value in output.metrics.items():
            sums[key] = sums.get(key, 0.0) + value
        steps += 1

        if log_every and step % log_every == 0:
            print(f"    step {step:05d} loss={float(output.loss.detach()):.5f}", flush=True)

    if steps == 0:
        raise RuntimeError(
            "The dataloader produced no batches. With drop_last=True this happens "
            "when the dataset is smaller than one batch."
        )
    return total / steps, {k: v / steps for k, v in sums.items()}


def train(
    method: SSLMethod,
    loader: Iterable[SSLBatch],
    *,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    start_epoch: int = 1,
    amp: bool = False,
    momentum_start: float = 0.996,
    momentum_end: float = 1.0,
    log_every: int = 0,
    on_epoch_end: Callable[[EpochResult], None] | None = None,
) -> list[EpochResult]:
    history: list[EpochResult] = []

    for epoch in range(start_epoch, epochs + 1):
        began = time.time()
        # Indexed from epoch-1 so the first epoch starts at momentum_start.
        momentum = cosine_schedule(epoch - 1, epochs, momentum_start, momentum_end)

        avg_loss, metrics = train_one_epoch(
            method,
            loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            momentum=momentum,
            amp=amp,
            log_every=log_every,
        )
        scheduler.step()

        result = EpochResult(
            epoch=epoch,
            avg_loss=avg_loss,
            lr=float(scheduler.get_last_lr()[0]),
            seconds=time.time() - began,
            metrics=metrics,
        )
        history.append(result)

        extra = " ".join(f"{k}={v:.5f}" for k, v in metrics.items() if k != "loss")
        print(
            f"epoch={epoch:04d}/{epochs} loss={avg_loss:.5f} lr={result.lr:.6g} "
            f"momentum={momentum:.5f} {extra} ({result.seconds:.1f}s)",
            flush=True,
        )

        if on_epoch_end is not None:
            on_epoch_end(result)

    return history

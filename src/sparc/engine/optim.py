"""Optimiser and schedule construction."""

from __future__ import annotations

import torch
from torch import nn


def build_optimizer(
    model: nn.Module, *, name: str = "adam", lr: float = 3e-4, weight_decay: float = 1e-4
) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name!r}; expected one of: adam, adamw")


def build_scheduler(optimizer: torch.optim.Optimizer, *, name: str = "cosine", epochs: int = 100):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "none":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)
    raise ValueError(f"Unknown scheduler {name!r}; expected one of: cosine, none")

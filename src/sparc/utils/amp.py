"""Mixed-precision helpers."""

from __future__ import annotations

import contextlib

import torch


def build_grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=enabled)


def autocast(enabled: bool):
    """Autocast on CUDA; a no-op elsewhere so the same code path runs on CPU."""
    if enabled and torch.cuda.is_available():
        return torch.amp.autocast("cuda", enabled=True)
    return contextlib.nullcontext()

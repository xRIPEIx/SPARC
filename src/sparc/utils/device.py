"""Device selection."""

from __future__ import annotations

import torch


def resolve_device(device: str = "auto", gpu: int | None = None) -> torch.device:
    """Resolve a device string.

    "auto" picks CUDA when available, else CPU. `gpu` selects an index.
    """
    if gpu is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(f"--gpu {gpu} requested but CUDA is not available")
        return torch.device(f"cuda:{gpu}")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

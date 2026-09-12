"""Reproducible seeding."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dataloader_generator(seed: int) -> torch.Generator:
    """Generator for DataLoader shuffling.

    The previous implementation seeded the global RNGs and then built the
    DataLoader with neither `generator=` nor `worker_init_fn`. That happens to
    be reproducible on current PyTorch, because workers are seeded
    deterministically from the base generator -- but it is reproducible by
    accident, via a version-dependent implementation detail, underneath a claim
    that rests on 5-seed replication. Passing both explicitly makes it a
    property of this code instead.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def worker_init_fn(worker_id: int) -> None:
    """Seed each worker's Python and NumPy RNGs from torch's per-worker seed.

    torch seeds its own per-worker RNG, but `random` and `numpy` inside a worker
    are not covered -- and the augmentation pipeline draws from `random`.
    """
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)

"""Projection heads used during SSL pretraining."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RegionProjectionHead(nn.Module):
    """MLP projection head for superpixel pooled region features.

    This head is for SSL pretraining losses only. Downstream detection and
    segmentation evaluation should transfer the pretrained backbone, not this
    projection head.
    """

    def __init__(self, in_dim: int = 2048, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, region_feat: torch.Tensor) -> torch.Tensor:
        z = self.mlp(region_feat)
        return F.normalize(z, dim=-1)

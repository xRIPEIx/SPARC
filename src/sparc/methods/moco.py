"""MoCo v2.

This is MoCo v2 in substance -- MLP projection head, v2 augmentations, cosine
momentum schedule -- with two deviations worth stating plainly in any paper that
cites it: the optimiser is Adam rather than SGD with momentum, and there is no
shuffling BatchNorm, because training is single-GPU. lightly's NTXentLoss memory
bank stands in for the official queue.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from lightly.loss import NTXentLoss
from lightly.models import utils
from lightly.models.modules import MoCoProjectionHead

from sparc.methods.base import SSLBatch, SSLMethod, StepOutput
from sparc.methods.registry import register_method
from sparc.models.backbones import BackboneSpec


@register_method("moco")
class MoCo(SSLMethod):
    def __init__(
        self,
        spec: BackboneSpec,
        *,
        proj_dim: int = 128,
        memory_bank_size: int = 4096,
        temperature: float = 0.1,
    ):
        super().__init__(spec)
        self.backbone = spec.final_extractor()
        self.stage = spec.stage_names[-1]
        self.projection_head = MoCoProjectionHead(spec.feature_dim, spec.feature_dim, proj_dim)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NTXentLoss(
            temperature=temperature, memory_bank_size=(memory_bank_size, proj_dim)
        )

    def _pool(self, feature_map: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(feature_map, (1, 1)).flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_head(self._pool(self.backbone(x)[self.stage]))

    @torch.no_grad()
    def forward_momentum(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self._pool(self.backbone_momentum(x)[self.stage])
        return self.projection_head_momentum(pooled).detach()

    def training_step(self, batch: SSLBatch) -> StepOutput:
        query = self(batch.view_q)
        key = self.forward_momentum(batch.view_k)
        loss = self.criterion(query, key)
        return StepOutput(loss=loss, metrics={"loss": float(loss.detach())})

    def update_momentum(self, momentum: float) -> None:
        utils.update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        utils.update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)

    def encoder_state_dict(self) -> dict[str, torch.Tensor]:
        return self.backbone.state_dict()

    def criteria_state(self) -> dict[str, torch.nn.Module]:
        return {"criterion": self.criterion}

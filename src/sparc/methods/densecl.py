"""DenseCL: a global image-level term plus a dense pixel-level term.

`dense_lambda` mixes the two. It is DenseCL's own hyper-parameter, not anything
SPARC adds, and sweeping it is the companion ablation to sweeping SPARC's
lambda_region: both interpolate from the same MoCo-global objective at lambda=0.

Note that dense_lambda defaults to 0.5, so the configuration `densecl_lambda_0p5`
IS the DenseCL baseline. They must never be reported as independent evidence.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from lightly.loss import NTXentLoss
from lightly.models import utils
from lightly.models.modules import DenseCLProjectionHead

from sparc.methods.base import SSLBatch, SSLMethod, StepOutput
from sparc.methods.registry import register_method
from sparc.models.backbones import BackboneSpec


@register_method("densecl")
class DenseCL(SSLMethod):
    def __init__(
        self,
        spec: BackboneSpec,
        *,
        proj_dim: int = 128,
        memory_bank_size: int = 4096,
        temperature: float = 0.1,
        dense_lambda: float = 0.5,
    ):
        super().__init__(spec)
        if not 0.0 <= dense_lambda <= 1.0:
            raise ValueError(f"dense_lambda must be in [0, 1], got {dense_lambda}")
        self.dense_lambda = dense_lambda
        self.backbone = spec.final_extractor()
        self.stage = spec.stage_names[-1]

        self.projection_head_global = DenseCLProjectionHead(
            spec.feature_dim, spec.feature_dim, proj_dim
        )
        self.projection_head_local = DenseCLProjectionHead(
            spec.feature_dim, spec.feature_dim, proj_dim
        )

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_global_momentum = copy.deepcopy(self.projection_head_global)
        self.projection_head_local_momentum = copy.deepcopy(self.projection_head_local)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_global_momentum)
        utils.deactivate_requires_grad(self.projection_head_local_momentum)

        self.criterion_global = NTXentLoss(
            temperature=temperature, memory_bank_size=(memory_bank_size, proj_dim)
        )
        self.criterion_local = NTXentLoss(
            temperature=temperature, memory_bank_size=(memory_bank_size, proj_dim)
        )

    def _heads(self, x, backbone, head_global, head_local):
        features = backbone(x)[self.stage]  # B, C, H, W
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).flatten(start_dim=1)
        z_global = head_global(pooled)
        local_features = features.flatten(start_dim=2).permute(0, 2, 1)  # B, HW, C
        z_local = head_local(local_features)
        return local_features, z_global, z_local

    def forward(self, x: torch.Tensor):
        return self._heads(
            x, self.backbone, self.projection_head_global, self.projection_head_local
        )

    @torch.no_grad()
    def forward_momentum(self, x: torch.Tensor):
        return self._heads(
            x,
            self.backbone_momentum,
            self.projection_head_global_momentum,
            self.projection_head_local_momentum,
        )

    def training_step(self, batch: SSLBatch) -> StepOutput:
        q_feat, q_global, q_local = self(batch.view_q)
        k_feat, k_global, k_local = self.forward_momentum(batch.view_k)

        # Pair each query location with its most similar key location, which is
        # what makes the dense term correspondence-aware rather than positional.
        k_local = utils.select_most_similar(q_feat, k_feat, k_local)

        loss_global = self.criterion_global(q_global, k_global)
        loss_local = self.criterion_local(q_local.flatten(end_dim=1), k_local.flatten(end_dim=1))
        loss = (1.0 - self.dense_lambda) * loss_global + self.dense_lambda * loss_local
        return StepOutput(
            loss=loss,
            metrics={
                "loss": float(loss.detach()),
                "loss_global": float(loss_global.detach()),
                "loss_local": float(loss_local.detach()),
            },
        )

    def update_momentum(self, momentum: float) -> None:
        utils.update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        utils.update_momentum(
            self.projection_head_global, self.projection_head_global_momentum, m=momentum
        )
        utils.update_momentum(
            self.projection_head_local, self.projection_head_local_momentum, m=momentum
        )

    def encoder_state_dict(self) -> dict[str, torch.Tensor]:
        return self.backbone.state_dict()

    def criteria_state(self) -> dict[str, torch.nn.Module]:
        return {"criterion_global": self.criterion_global, "criterion_local": self.criterion_local}

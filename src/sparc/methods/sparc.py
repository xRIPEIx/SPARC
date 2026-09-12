"""SPARC: SuperPixel-Aware Region Contrastive learning.

The objective is a convex combination of a global image-level term and a
region-level term:

    L = (1 - lambda_region) * L_global + lambda_region * L_region

L_global is NT-Xent over pooled image embeddings against a momentum queue.
L_region is a DetCon-style contrastive loss over superpixel region embeddings:
a positive pair shares the same image AND the same original superpixel id across
the two augmented views, and every other valid region in the candidate view is a
negative.

Superpixel masks are used to pool features *after* the encoder. They are never
fed into the CNN as an extra input channel -- the encoder sees ordinary RGB, so
a SPARC-pretrained backbone transfers to downstream tasks with no mask
dependency at all.

lambda_region = 0 reduces this to the MoCo-global objective, which is what makes
the lambda sweep meet the DenseCL sweep and the MoCo baseline at a single shared
point. That three-way agreement is the cheapest validity check in the study.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from lightly.loss import NTXentLoss
from lightly.models import utils
from lightly.models.modules import DenseCLProjectionHead

from sparc.losses import region_contrastive_loss
from sparc.methods.base import SSLBatch, SSLMethod, StepOutput
from sparc.methods.registry import register_method
from sparc.models import RegionProjectionHead, SuperpixelRegionPool
from sparc.models.backbones import BackboneSpec


@register_method("sparc")
class SPARC(SSLMethod):
    requires_region_masks = True

    def __init__(
        self,
        spec: BackboneSpec,
        *,
        proj_dim: int = 128,
        memory_bank_size: int = 4096,
        temperature: float = 0.1,
        lambda_region: float = 0.5,
        region_temperature: float = 0.2,
        max_regions: int = 16,
        region_pool_size: int = 7,
        region_min_area: float = 1.0,
    ):
        super().__init__(spec)
        if not 0.0 <= lambda_region <= 1.0:
            raise ValueError(f"lambda_region must be in [0, 1], got {lambda_region}")
        self.lambda_region = lambda_region
        self.region_temperature = region_temperature

        self.backbone = spec.final_extractor()
        self.stage = spec.stage_names[-1]

        self.projection_head_global = DenseCLProjectionHead(
            spec.feature_dim, spec.feature_dim, proj_dim
        )
        self.region_pool = SuperpixelRegionPool(
            output_size=(region_pool_size, region_pool_size),
            max_regions=max_regions,
            min_area=region_min_area,
        )
        self.region_projector = RegionProjectionHead(
            in_dim=spec.feature_dim, hidden_dim=spec.feature_dim, out_dim=proj_dim
        )

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_global_momentum = copy.deepcopy(self.projection_head_global)
        self.region_projector_momentum = copy.deepcopy(self.region_projector)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_global_momentum)
        utils.deactivate_requires_grad(self.region_projector_momentum)

        self.criterion_global = NTXentLoss(
            temperature=temperature, memory_bank_size=(memory_bank_size, proj_dim)
        )

    def forward_query(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)[self.stage]
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).flatten(start_dim=1)
        return features, self.projection_head_global(pooled)

    @torch.no_grad()
    def forward_momentum(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone_momentum(x)[self.stage]
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).flatten(start_dim=1)
        return features.detach(), self.projection_head_global_momentum(pooled).detach()

    def training_step(self, batch: SSLBatch) -> StepOutput:
        if batch.mask_q is None or batch.mask_k is None:
            raise ValueError(
                "SPARC requires region masks. Point data.masks at a mask directory, "
                "or generate one with `sparc-masks`."
            )

        fq, q_global = self.forward_query(batch.view_q)
        fk, k_global = self.forward_momentum(batch.view_k)

        hq, hk, ids_q, ids_k, valid_q, valid_k = self.region_pool.forward_pair(
            fq, batch.mask_q, fk, batch.mask_k
        )
        zq = self.region_projector(hq)
        with torch.no_grad():
            zk = self.region_projector_momentum(hk).detach()

        loss_global = self.criterion_global(q_global, k_global)
        loss_region = region_contrastive_loss(
            zq, zk, ids_q, ids_k, valid_q, valid_k, temperature=self.region_temperature
        )
        loss = (1.0 - self.lambda_region) * loss_global + self.lambda_region * loss_region

        return StepOutput(
            loss=loss,
            metrics={
                "loss": float(loss.detach()),
                "loss_global": float(loss_global.detach()),
                "loss_region": float(loss_region.detach()),
                # Regions matched in both views. If this collapses toward zero
                # the region term is training on almost nothing, which is worth
                # seeing in the log rather than inferring from a flat loss.
                "num_regions": float((valid_q & valid_k).sum().detach()),
            },
        )

    def update_momentum(self, momentum: float) -> None:
        utils.update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        utils.update_momentum(
            self.projection_head_global, self.projection_head_global_momentum, m=momentum
        )
        utils.update_momentum(self.region_projector, self.region_projector_momentum, m=momentum)

    def encoder_state_dict(self) -> dict[str, torch.Tensor]:
        return self.backbone.state_dict()

    def criteria_state(self) -> dict[str, torch.nn.Module]:
        return {"criterion_global": self.criterion_global}

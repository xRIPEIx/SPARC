"""Superpixel region pooling for dense SSL features."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def match_region_ids(
    mask_q: torch.Tensor,
    mask_k: torch.Tensor,
    max_regions: int = 16,
    min_area: float = 1.0,
    ignore_index: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample superpixel IDs visible in both transformed views.

    Returns region_ids_q, region_ids_k, valid_q, and valid_k, each with shape
    [B, R]. The two ID tensors contain the same sampled original superpixel IDs.
    Superpixel ID 0 is valid; only ignore_index is removed. Areas are measured in
    full-resolution transformed mask pixels.
    """
    if mask_q.ndim != 3 or mask_k.ndim != 3:
        raise ValueError(
            f"mask_q and mask_k must have shape [B, H, W], got {mask_q.shape} and {mask_k.shape}"
        )
    if mask_q.shape != mask_k.shape:
        raise ValueError(
            f"mask_q and mask_k must have the same shape, got {mask_q.shape} and {mask_k.shape}"
        )

    mask_q = mask_q.to(dtype=torch.long)
    mask_k = mask_k.to(device=mask_q.device, dtype=torch.long)

    region_ids_q = torch.full(
        (mask_q.shape[0], max_regions),
        fill_value=ignore_index,
        dtype=torch.long,
        device=mask_q.device,
    )
    region_ids_k = torch.full_like(region_ids_q, fill_value=ignore_index)
    valid_q = torch.zeros(
        (mask_q.shape[0], max_regions),
        dtype=torch.bool,
        device=mask_q.device,
    )
    valid_k = torch.zeros_like(valid_q)

    for batch_idx in range(mask_q.shape[0]):
        ids_q, counts_q = torch.unique(mask_q[batch_idx], return_counts=True)
        ids_k, counts_k = torch.unique(mask_k[batch_idx], return_counts=True)

        keep_q = (ids_q != ignore_index) & (counts_q.to(torch.float32) >= min_area)
        keep_k = (ids_k != ignore_index) & (counts_k.to(torch.float32) >= min_area)
        ids_q = ids_q[keep_q]
        ids_k = ids_k[keep_k]

        if ids_q.numel() == 0 or ids_k.numel() == 0:
            continue

        in_both = (ids_q[:, None] == ids_k[None, :]).any(dim=1)
        ids_common = ids_q[in_both]
        if ids_common.numel() == 0:
            continue

        if ids_common.numel() > max_regions:
            selection = torch.randperm(ids_common.numel(), device=ids_common.device)[:max_regions]
            ids_common = ids_common[selection]

        region_ids_q[batch_idx, : ids_common.numel()] = ids_common
        region_ids_k[batch_idx, : ids_common.numel()] = ids_common
        valid_q[batch_idx, : ids_common.numel()] = True
        valid_k[batch_idx, : ids_common.numel()] = True

    return region_ids_q, region_ids_k, valid_q, valid_k


class SuperpixelRegionPool(nn.Module):
    """Pools CNN feature maps over aligned superpixel regions."""

    def __init__(
        self,
        output_size: tuple[int, int] = (7, 7),
        max_regions: int = 16,
        min_area: float = 1.0,
        ignore_index: int = -1,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.output_size = output_size
        self.max_regions = max_regions
        self.min_area = min_area
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(
        self,
        feat: torch.Tensor,
        mask: torch.Tensor,
        region_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool one feature map batch using one transformed superpixel mask batch.

        Args:
            feat: Tensor of shape [B, C, Hf, Wf].
            mask: Integer tensor of shape [B, H, W].
            region_ids: Optional tensor of shape [B, R]. If provided, these IDs
                are pooled in order and invalid/missing regions are marked false.

        Returns:
            region_feat: Tensor of shape [B, R, C].
            region_ids: Long tensor of shape [B, R].
            region_valid: Boolean tensor of shape [B, R].
        """
        self._validate_inputs(feat, mask)
        mask = mask.to(device=feat.device)
        batch_size, channels, _feat_h, _feat_w = feat.shape
        output_size = self._get_output_size()
        feat = self._resize_feature_map(feat, output_size)

        if region_ids is None:
            region_ids = self._sample_batch_region_ids(mask, output_size)
        else:
            region_ids = self._normalize_region_ids(region_ids, batch_size, feat.device)

        region_feat = feat.new_zeros((batch_size, self.max_regions, channels))
        region_valid = torch.zeros(
            (batch_size, self.max_regions), dtype=torch.bool, device=feat.device
        )

        for batch_idx in range(batch_size):
            slot_valid = region_ids[batch_idx] != self.ignore_index
            if not slot_valid.any():
                continue

            slot_indices = torch.nonzero(slot_valid, as_tuple=False).flatten()
            selected_ids = region_ids[batch_idx, slot_indices]
            binary_low = self._build_and_downsample_region_mask_stack(
                mask[batch_idx],
                selected_ids,
                output_size,
                dtype=feat.dtype,
            )
            areas = binary_low.sum(dim=(1, 2))
            pooled_valid = areas > self.eps
            if not pooled_valid.any():
                continue

            pooled = torch.einsum("chw,rhw->rc", feat[batch_idx], binary_low) / (
                areas[:, None] + self.eps
            )
            pooled = pooled.to(dtype=feat.dtype)
            valid_slots = slot_indices[pooled_valid]
            region_feat[batch_idx, valid_slots] = pooled[pooled_valid]
            region_valid[batch_idx, valid_slots] = True
        return region_feat, region_ids, region_valid

    def forward_pair(
        self,
        feat_q: torch.Tensor,
        mask_q: torch.Tensor,
        feat_k: torch.Tensor,
        mask_k: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Pool two views using the same sampled region IDs per image.

        Region IDs are sampled from the intersection of IDs visible in both
        transformed masks, so hq[b, r] and hk[b, r] share the same original
        superpixel ID when both valid.

        Returns:
            hq: Tensor of shape [B, R, C].
            hk: Tensor of shape [B, R, C].
            region_ids_q: Long tensor of shape [B, R].
            region_ids_k: Long tensor of shape [B, R].
            valid_q: Boolean tensor of shape [B, R].
            valid_k: Boolean tensor of shape [B, R].
        """
        self._validate_inputs(feat_q, mask_q)
        self._validate_inputs(feat_k, mask_k)
        if feat_q.shape != feat_k.shape:
            raise ValueError(
                f"feat_q and feat_k must have the same shape, got {feat_q.shape} and {feat_k.shape}"
            )
        if feat_q.device != feat_k.device:
            raise ValueError(
                "feat_q and feat_k must be on the same device, "
                f"got {feat_q.device} and {feat_k.device}"
            )
        if mask_q.shape != mask_k.shape:
            raise ValueError(
                f"mask_q and mask_k must have the same shape, got {mask_q.shape} and {mask_k.shape}"
            )

        mask_q = mask_q.to(device=feat_q.device)
        mask_k = mask_k.to(device=feat_q.device)
        region_ids_q, region_ids_k, matched_valid_q, matched_valid_k = match_region_ids(
            mask_q,
            mask_k,
            max_regions=self.max_regions,
            min_area=self.min_area,
            ignore_index=self.ignore_index,
        )
        hq, region_ids_q, pooled_valid_q = self.forward(
            feat_q,
            mask_q,
            region_ids=region_ids_q,
        )
        hk, region_ids_k, pooled_valid_k = self.forward(
            feat_k,
            mask_k,
            region_ids=region_ids_k,
        )
        valid_q = matched_valid_q & pooled_valid_q
        valid_k = matched_valid_k & pooled_valid_k
        return hq, hk, region_ids_q, region_ids_k, valid_q, valid_k

    def _validate_inputs(self, feat: torch.Tensor, mask: torch.Tensor) -> None:
        if feat.ndim != 4:
            raise ValueError(f"feat must have shape [B, C, Hf, Wf], got {feat.shape}")
        if mask.ndim != 3:
            raise ValueError(f"mask must have shape [B, H, W], got {mask.shape}")
        if feat.shape[0] != mask.shape[0]:
            raise ValueError(
                f"feat and mask batch sizes must match, got {feat.shape[0]} and {mask.shape[0]}"
            )

    def _get_output_size(self) -> tuple[int, int]:
        output_size = self.output_size
        if output_size[0] <= 0 or output_size[1] <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")
        return output_size

    def _resize_feature_map(self, feat: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        if feat.shape[-2:] == output_size:
            return feat
        if output_size[0] <= feat.shape[-2] and output_size[1] <= feat.shape[-1]:
            return F.adaptive_avg_pool2d(feat, output_size=output_size)
        return F.interpolate(feat, size=output_size, mode="bilinear", align_corners=False)

    def _normalize_region_ids(
        self, region_ids: torch.Tensor, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if region_ids.ndim != 2:
            raise ValueError(f"region_ids must have shape [B, R], got {region_ids.shape}")
        if region_ids.shape[0] != batch_size:
            raise ValueError(
                f"region_ids batch size must be {batch_size}, got {region_ids.shape[0]}"
            )
        if region_ids.shape[1] != self.max_regions:
            raise ValueError(
                f"region_ids must have {self.max_regions} columns, got {region_ids.shape[1]}"
            )
        return region_ids.to(device=device, dtype=torch.long)

    def _sample_batch_region_ids(
        self, mask: torch.Tensor, output_size: tuple[int, int]
    ) -> torch.Tensor:
        region_ids = torch.full(
            (mask.shape[0], self.max_regions),
            fill_value=self.ignore_index,
            dtype=torch.long,
            device=mask.device,
        )
        for batch_idx in range(mask.shape[0]):
            ids = self._valid_region_ids(mask[batch_idx], output_size)
            ids = self._sample_ids(ids)
            if ids.numel() > 0:
                region_ids[batch_idx, : ids.numel()] = ids
        return region_ids

    def _valid_region_ids(self, mask: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        ids = torch.unique(mask)
        ids = ids[ids != self.ignore_index]
        valid_ids = []
        for region_id in ids:
            area = (mask == region_id).sum()
            if area.to(torch.float32) >= self.min_area:
                valid_ids.append(region_id)

        if not valid_ids:
            return torch.empty(0, dtype=torch.long, device=mask.device)
        return torch.stack(valid_ids).to(dtype=torch.long)

    def _sample_ids(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.numel() <= self.max_regions:
            return ids
        selection = torch.randperm(ids.numel(), device=ids.device)[: self.max_regions]
        return ids[selection]

    def _build_and_downsample_region_mask_stack(
        self,
        mask: torch.Tensor,
        region_ids: torch.Tensor,
        output_size: tuple[int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert integer region IDs to a binary stack, then downsample.

        The integer superpixel mask is not resized directly because averaging label
        IDs would create meaningless fractional labels. Instead this builds one
        binary mask per selected region:

            [H, W] + [R] -> [R, H, W] -> [R, Hf, Wf]

        The downsampled masks are soft occupancy masks. A value of 0.25 means
        that roughly a quarter of that feature cell belongs to the region.
        """
        binary = (mask[None] == region_ids[:, None, None]).to(dtype=dtype)
        # adaptive_avg_pool2d can promote half inputs to float on CUDA; cast back
        # so einsum(feat, binary_low) stays in feat.dtype for AMP index-puts.
        binary_low = F.adaptive_avg_pool2d(binary[:, None], output_size=output_size)
        return binary_low[:, 0].to(dtype=dtype)

"""DetCon-style batch contrastive loss for matched region embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def region_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    ids1: torch.Tensor,
    ids2: torch.Tensor,
    valid1: torch.Tensor,
    valid2: torch.Tensor,
    temperature: float = 0.2,
    symmetric: bool = True,
) -> torch.Tensor:
    """Compute DetCon-style in-batch region contrastive loss.

    A positive pair has the same image index and the same original superpixel region
    ID. Every other valid region embedding in the candidate view is a negative.
    No region queue is used.
    """
    _validate_region_loss_inputs(z1, z2, ids1, ids2, valid1, valid2, temperature)

    loss = _one_way_region_loss(
        query_regions=z1,
        key_regions=z2,
        query_ids=ids1,
        key_ids=ids2,
        query_valid=valid1,
        key_valid=valid2,
        temperature=temperature,
    )
    if symmetric:
        loss = 0.5 * (
            loss
            + _one_way_region_loss(
                query_regions=z2,
                key_regions=z1,
                query_ids=ids2,
                key_ids=ids1,
                query_valid=valid2,
                key_valid=valid1,
                temperature=temperature,
            )
        )
    return loss


def _one_way_region_loss(
    query_regions: torch.Tensor,
    key_regions: torch.Tensor,
    query_ids: torch.Tensor,
    key_ids: torch.Tensor,
    query_valid: torch.Tensor,
    key_valid: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    device = query_regions.device
    query_valid = query_valid.to(device=device, dtype=torch.bool)
    key_valid = key_valid.to(device=device, dtype=torch.bool)
    query_ids = query_ids.to(device=device, dtype=torch.long)
    key_ids = key_ids.to(device=device, dtype=torch.long)

    batch_size, num_regions = query_valid.shape
    image_ids = torch.arange(batch_size, device=device)[:, None].expand(
        batch_size,
        num_regions,
    )

    q = query_regions[query_valid]
    k = key_regions[key_valid]
    if q.numel() == 0 or k.numel() == 0:
        return (query_regions.sum() + key_regions.sum()) * 0.0

    q_ids = query_ids[query_valid]
    k_ids = key_ids[key_valid]
    q_images = image_ids[query_valid]
    k_images = image_ids[key_valid]

    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    logits = torch.einsum("nd,md->nm", q, k) / temperature

    positive_mask = (q_images[:, None] == k_images[None, :]) & (q_ids[:, None] == k_ids[None, :])
    has_positive = positive_mask.any(dim=1)
    if not has_positive.any():
        return (query_regions.sum() + key_regions.sum()) * 0.0

    logits = logits[has_positive]
    positive_mask = positive_mask[has_positive]
    negative_inf = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positive_mask, negative_inf)

    log_denominator = torch.logsumexp(logits, dim=1)
    log_numerator = torch.logsumexp(positive_logits, dim=1)
    return -(log_numerator - log_denominator).mean()


def _validate_region_loss_inputs(
    z1: torch.Tensor,
    z2: torch.Tensor,
    ids1: torch.Tensor,
    ids2: torch.Tensor,
    valid1: torch.Tensor,
    valid2: torch.Tensor,
    temperature: float,
) -> None:
    if z1.ndim != 3:
        raise ValueError(f"z1 must have shape [B, R, D], got {z1.shape}")
    if z2.shape != z1.shape:
        raise ValueError(f"z2 must match z1 shape, got {z2.shape} and {z1.shape}")
    expected_region_shape = z1.shape[:2]
    for name, value in (
        ("ids1", ids1),
        ("ids2", ids2),
        ("valid1", valid1),
        ("valid2", valid2),
    ):
        if value.shape != expected_region_shape:
            raise ValueError(f"{name} must have shape {expected_region_shape}, got {value.shape}")
    if z2.device != z1.device:
        raise ValueError(f"z1 and z2 must be on the same device, got {z1.device} and {z2.device}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")

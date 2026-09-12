"""Tests for SuperpixelRegionPool.

The occupancy-weighting test is the important one here. Region masks are
downsampled to the feature grid as *soft* occupancy (a value of 0.25 means a
quarter of that feature cell belongs to the region), and pooling is a weighted
mean by that occupancy. Nothing in the previous implementation tested it.
"""

from __future__ import annotations

import pytest
import torch

from sparc.models import SuperpixelRegionPool


class TestOccupancyWeighting:
    def test_partial_cell_occupancy_is_weighted_not_binary(self):
        """Hand-computed. Feature grid is 1x2 after pooling, values [4, 8].

        Region 1 covers all of cell 0 (8/8 px -> occupancy 1.0) and a quarter of
        cell 1 (2/8 px -> occupancy 0.25):

            pooled = (4 * 1.0 + 8 * 0.25) / (1.0 + 0.25) = 6 / 1.25 = 4.8

        If occupancy were rounded to binary the answer would be (4 + 8) / 2 = 6.0,
        so this test distinguishes weighted from unweighted pooling.

        Region 0 holds the remaining 6/8 px of cell 1 (occupancy 0.75) and none
        of cell 0, so it pools to exactly 8.0.
        """
        feat = torch.zeros(1, 1, 2, 8)
        feat[0, 0, :, 0:4] = 4.0
        feat[0, 0, :, 4:8] = 8.0

        mask = torch.zeros(1, 2, 8, dtype=torch.long)
        mask[0, :, 0:4] = 1  # all of output cell 0
        mask[0, 0, 4:6] = 1  # 2 of the 8 pixels in output cell 1

        pool = SuperpixelRegionPool(output_size=(1, 2), max_regions=2)
        region_feat, ids, valid = pool(feat, mask, region_ids=torch.tensor([[0, 1]]))

        assert valid.all()
        assert region_feat[0, 0, 0].item() == pytest.approx(8.0, abs=1e-4)
        assert region_feat[0, 1, 0].item() == pytest.approx(4.8, abs=1e-4)
        assert region_feat[0, 1, 0].item() != pytest.approx(6.0, abs=1e-2)

    def test_constant_features_pool_to_the_constant(self):
        """Occupancy weights are normalised, so a constant feature map must pool
        to that constant regardless of region shape."""
        feat = torch.full((1, 3, 8, 8), 2.5)
        mask = torch.zeros(1, 8, 8, dtype=torch.long)
        mask[0, 5:, 3:] = 1  # deliberately ragged
        pool = SuperpixelRegionPool(output_size=(4, 4), max_regions=2)
        region_feat, _, valid = pool(feat, mask, region_ids=torch.tensor([[0, 1]]))
        assert valid.all()
        assert torch.allclose(region_feat, torch.full_like(region_feat, 2.5), atol=1e-5)


class TestRegionIdSemantics:
    def test_label_zero_is_a_valid_region(self, two_region_mask):
        """Label 0 is a real superpixel here, not background. Dropping it would
        quietly discard one region per image."""
        feat = torch.randn(1, 4, 8, 8)
        pool = SuperpixelRegionPool(output_size=(4, 4), max_regions=2)
        _, ids, valid = pool(feat, two_region_mask, region_ids=torch.tensor([[0, 1]]))
        assert valid.all(), "region id 0 was treated as invalid"
        assert set(ids[0].tolist()) == {0, 1}

    def test_region_id_absent_from_mask_is_marked_invalid(self):
        """An id with no pixels has zero occupancy everywhere; it must come back
        invalid rather than as a zero vector that looks like a real embedding."""
        feat = torch.randn(1, 4, 8, 8)
        mask = torch.zeros(1, 8, 8, dtype=torch.long)  # only label 0 exists
        pool = SuperpixelRegionPool(output_size=(4, 4), max_regions=2)
        _, _, valid = pool(feat, mask, region_ids=torch.tensor([[0, 99]]))
        assert valid[0, 0].item() is True or bool(valid[0, 0])
        assert not bool(valid[0, 1]), "absent region id was reported valid"

    def test_ignore_index_slots_are_invalid(self):
        feat = torch.randn(1, 4, 8, 8)
        mask = torch.zeros(1, 8, 8, dtype=torch.long)
        pool = SuperpixelRegionPool(output_size=(4, 4), max_regions=3, ignore_index=-1)
        _, _, valid = pool(feat, mask, region_ids=torch.tensor([[0, -1, -1]]))
        assert bool(valid[0, 0]) and not bool(valid[0, 1]) and not bool(valid[0, 2])


class TestFeatureResize:
    @pytest.mark.parametrize("out", [(1, 1), (3, 3), (7, 7)])
    def test_downsampling_path(self, out):
        """output <= feature size uses adaptive_avg_pool2d."""
        feat = torch.randn(2, 5, 7, 7)
        mask = torch.zeros(2, 28, 28, dtype=torch.long)
        mask[:, 14:, :] = 1
        pool = SuperpixelRegionPool(output_size=out, max_regions=2)
        region_feat, _, _ = pool(feat, mask, region_ids=torch.tensor([[0, 1], [0, 1]]))
        assert region_feat.shape == (2, 2, 5)

    def test_upsampling_path(self):
        """region_pool_size 9 against a 7x7 feature map takes the bilinear
        F.interpolate branch. This was a real swept configuration."""
        feat = torch.randn(2, 5, 7, 7)
        mask = torch.zeros(2, 36, 36, dtype=torch.long)
        mask[:, 18:, :] = 1
        pool = SuperpixelRegionPool(output_size=(9, 9), max_regions=2)
        region_feat, _, valid = pool(feat, mask, region_ids=torch.tensor([[0, 1], [0, 1]]))
        assert region_feat.shape == (2, 2, 5)
        assert valid.all()
        assert torch.isfinite(region_feat).all()


class TestForwardPair:
    def test_paired_views_share_region_ids(self):
        """hq[b, r] and hk[b, r] must describe the same original superpixel."""
        feat_q, feat_k = torch.randn(2, 6, 7, 7), torch.randn(2, 6, 7, 7)
        mask = torch.zeros(2, 28, 28, dtype=torch.long)
        mask[:, 14:, :] = 1
        pool = SuperpixelRegionPool(output_size=(7, 7), max_regions=4)
        hq, hk, ids_q, ids_k, vq, vk = pool.forward_pair(feat_q, mask, feat_k, mask.clone())
        both = vq & vk
        assert both.any()
        assert torch.equal(ids_q[both], ids_k[both])
        assert hq.shape == hk.shape == (2, 4, 6)

    def test_gradients_flow_to_features(self):
        feat = torch.randn(1, 4, 7, 7, requires_grad=True)
        mask = torch.zeros(1, 28, 28, dtype=torch.long)
        mask[:, 14:, :] = 1
        pool = SuperpixelRegionPool(output_size=(7, 7), max_regions=2)
        hq, _, _, _, _, _ = pool.forward_pair(feat, mask, feat, mask.clone())
        hq.sum().backward()
        assert feat.grad is not None and torch.isfinite(feat.grad).all()


class TestValidation:
    def test_rejects_wrong_feature_rank(self):
        pool = SuperpixelRegionPool()
        with pytest.raises(ValueError, match=r"\[B, C, Hf, Wf\]"):
            pool(torch.randn(4, 7, 7), torch.zeros(1, 8, 8, dtype=torch.long))

    def test_rejects_wrong_mask_rank(self):
        pool = SuperpixelRegionPool()
        with pytest.raises(ValueError, match=r"\[B, H, W\]"):
            pool(torch.randn(1, 4, 7, 7), torch.zeros(8, 8, dtype=torch.long))

    def test_rejects_batch_size_mismatch(self):
        pool = SuperpixelRegionPool()
        with pytest.raises(ValueError, match="batch sizes"):
            pool(torch.randn(2, 4, 7, 7), torch.zeros(3, 8, 8, dtype=torch.long))

    def test_rejects_region_ids_with_wrong_width(self):
        pool = SuperpixelRegionPool(max_regions=4)
        with pytest.raises(ValueError, match="columns"):
            pool(
                torch.randn(1, 4, 7, 7),
                torch.zeros(1, 8, 8, dtype=torch.long),
                region_ids=torch.tensor([[0, 1]]),
            )

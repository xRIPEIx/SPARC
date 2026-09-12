"""Tests for the region contrastive loss.

The known-value test below is the regression anchor for the whole method: if
the loss changes, every published number changes with it, and at sigma ~= 0.3
mIoU that drift would be statistically invisible in downstream results.
"""

from __future__ import annotations

import math

import pytest
import torch

from sparc.losses import region_contrastive_loss


def _ids(*rows):
    return torch.tensor(rows, dtype=torch.long)


def _valid(*rows):
    return torch.tensor(rows, dtype=torch.bool)


class TestKnownValues:
    def test_orthonormal_pair_matches_hand_computation(self):
        """Hand-derived anchor. Do not update this number to make a change pass.

        Setup: B=1, R=2, D=2, z2 == z1, rows already unit-norm and orthogonal:

            z = [[1, 0],
                 [0, 1]]        ids = [0, 1]     tau = 0.5

        logits = (z @ z.T) / tau = [[2, 0],
                                    [0, 2]]

        Positives are (same image) AND (same region id), so the positive mask is
        the identity. For row 0:

            log_numerator   = 2
            log_denominator = logsumexp([2, 0]) = 2 + log(1 + e^-2)
            row loss        = -(2 - 2 - log(1 + e^-2)) = log(1 + e^-2)

        Row 1 is symmetric, so the mean is the same, and because z2 == z1 the
        symmetric average over both directions does not change it either.
        """
        z = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        loss = region_contrastive_loss(
            z,
            z.clone(),
            _ids([0, 1]),
            _ids([0, 1]),
            _valid([True, True]),
            _valid([True, True]),
            temperature=0.5,
        )
        expected = math.log1p(math.exp(-2.0))  # 0.12692801...
        assert loss.item() == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("batch_size", [2, 3, 5])
    def test_identical_embeddings_give_log_b(self, batch_size):
        """With one region per image and every embedding identical, all logits
        are equal, so the loss collapses to log(B) for any temperature."""
        z = torch.ones(batch_size, 1, 4)
        ids = torch.zeros(batch_size, 1, dtype=torch.long)
        valid = torch.ones(batch_size, 1, dtype=torch.bool)
        loss = region_contrastive_loss(z, z.clone(), ids, ids, valid, valid, temperature=0.7)
        assert loss.item() == pytest.approx(math.log(batch_size), abs=1e-6)


class TestDegenerateCases:
    """These must return 0 *and stay attached to the graph*.

    The implementation returns `(z1.sum() + z2.sum()) * 0.0` rather than a bare
    scalar precisely so that .backward() does not blow up on a batch where no
    region survived matching. A "simplification" to torch.tensor(0.0) would
    pass an equality check and then crash training.
    """

    def test_no_valid_regions(self):
        z = torch.randn(2, 3, 4, requires_grad=True)
        valid = torch.zeros(2, 3, dtype=torch.bool)
        loss = region_contrastive_loss(
            z, z, _ids([0, 1, 2], [0, 1, 2]), _ids([0, 1, 2], [0, 1, 2]), valid, valid
        )
        assert loss.item() == 0.0
        assert loss.requires_grad
        loss.backward()  # must not raise

    def test_valid_regions_but_no_positives(self):
        """Valid on both sides, but no (image, id) pair matches."""
        z = torch.randn(2, 1, 4, requires_grad=True)
        valid = torch.ones(2, 1, dtype=torch.bool)
        loss = region_contrastive_loss(z, z, _ids([0], [0]), _ids([7], [7]), valid, valid)
        assert loss.item() == 0.0
        assert loss.requires_grad
        loss.backward()


class TestSemantics:
    def test_symmetric_is_mean_of_both_directions(self):
        z1, z2 = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
        ids = _ids([0, 1, 2], [0, 1, 2])
        valid = torch.ones(2, 3, dtype=torch.bool)
        kw = dict(temperature=0.2)
        both = region_contrastive_loss(z1, z2, ids, ids, valid, valid, symmetric=True, **kw)
        fwd = region_contrastive_loss(z1, z2, ids, ids, valid, valid, symmetric=False, **kw)
        bwd = region_contrastive_loss(z2, z1, ids, ids, valid, valid, symmetric=False, **kw)
        assert both.item() == pytest.approx(0.5 * (fwd.item() + bwd.item()), abs=1e-6)

    def test_positives_require_same_image_not_just_same_id(self):
        """Region id 0 in image 0 and region id 0 in image 1 are different
        regions. Treating ids as globally unique would silently create false
        positives across the batch."""
        z = torch.randn(2, 1, 8)
        ids, valid = _ids([0], [0]), torch.ones(2, 1, dtype=torch.bool)
        matched = region_contrastive_loss(z, z.clone(), ids, ids, valid, valid, temperature=0.2)
        # Each row has exactly one positive (itself); with 2 candidates the loss
        # is strictly positive rather than 0, proving the off-image pair was not
        # counted as a positive.
        assert matched.item() > 0.0

    def test_invalid_entries_are_excluded(self):
        """Padding slots must not influence the result."""
        z = torch.randn(1, 2, 8)
        z_pad = torch.cat([z, torch.randn(1, 3, 8) * 1e3], dim=1)
        ids, ids_pad = _ids([0, 1]), _ids([0, 1, 5, 6, 7])
        v = _valid([True, True])
        v_pad = _valid([True, True, False, False, False])
        a = region_contrastive_loss(z, z, ids, ids, v, v, temperature=0.2)
        b = region_contrastive_loss(z_pad, z_pad, ids_pad, ids_pad, v_pad, v_pad, temperature=0.2)
        assert a.item() == pytest.approx(b.item(), abs=1e-6)


class TestValidation:
    def test_rejects_non_positive_temperature(self):
        z = torch.randn(1, 2, 4)
        ids, v = _ids([0, 1]), _valid([True, True])
        with pytest.raises(ValueError, match="temperature"):
            region_contrastive_loss(z, z, ids, ids, v, v, temperature=0.0)

    def test_rejects_shape_mismatch(self):
        z = torch.randn(1, 2, 4)
        with pytest.raises(ValueError):
            region_contrastive_loss(
                z, z, _ids([0]), _ids([0, 1]), _valid([True]), _valid([True, True])
            )

    def test_rejects_wrong_rank(self):
        with pytest.raises(ValueError, match=r"\[B, R, D\]"):
            z = torch.randn(2, 4)
            region_contrastive_loss(
                z, z, _ids([0, 1]), _ids([0, 1]), _valid([True, True]), _valid([True, True])
            )

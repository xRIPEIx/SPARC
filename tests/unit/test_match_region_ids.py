"""Tests for match_region_ids.

This is what makes a positive pair a positive pair: only superpixels visible in
*both* augmented views can be contrasted, so the sampled ids are the
intersection of the two transformed masks.
"""

from __future__ import annotations

import pytest
import torch

from sparc.models import match_region_ids


def _mask(rows, h=4, w=4):
    """Build a [1, h, w] mask from a list of (label, row_slice) pairs."""
    m = torch.zeros(1, h, w, dtype=torch.long)
    for label, sl in rows:
        m[0, sl, :] = label
    return m


class TestIntersection:
    def test_only_ids_present_in_both_views_are_sampled(self):
        q = _mask([(0, slice(0, 2)), (1, slice(2, 3)), (2, slice(3, 4))])
        k = _mask([(0, slice(0, 2)), (1, slice(2, 4))])  # label 2 cropped away
        ids_q, ids_k, vq, vk = match_region_ids(q, k, max_regions=4)
        sampled = set(ids_q[0][vq[0]].tolist())
        assert sampled == {0, 1}, f"expected the intersection {{0, 1}}, got {sampled}"
        assert 2 not in sampled

    def test_both_views_receive_the_same_ids_in_the_same_slots(self):
        q = _mask([(0, slice(0, 2)), (1, slice(2, 4))])
        ids_q, ids_k, vq, vk = match_region_ids(q, q.clone(), max_regions=4)
        assert torch.equal(ids_q, ids_k)
        assert torch.equal(vq, vk)

    def test_disjoint_masks_yield_no_valid_regions(self):
        q = _mask([(0, slice(0, 4))])
        k = _mask([(7, slice(0, 4))])
        _, _, vq, vk = match_region_ids(q, k, max_regions=4)
        assert not vq.any() and not vk.any()


class TestLabelZero:
    def test_label_zero_is_kept(self):
        """Only ignore_index is removed. Label 0 is a real superpixel."""
        q = _mask([(0, slice(0, 4))])
        ids, _, valid, _ = match_region_ids(q, q.clone(), max_regions=4)
        assert bool(valid[0, 0])
        assert ids[0, 0].item() == 0

    def test_ignore_index_is_removed(self):
        q = torch.full((1, 4, 4), -1, dtype=torch.long)
        q[0, 0, :] = 5
        ids, _, valid, _ = match_region_ids(q, q.clone(), max_regions=4, ignore_index=-1)
        assert set(ids[0][valid[0]].tolist()) == {5}


class TestFiltering:
    def test_min_area_filters_small_regions(self):
        """Areas are measured in full-resolution transformed mask pixels."""
        q = _mask([(0, slice(0, 3)), (1, slice(3, 4))])  # label 1 has 4 px
        _, _, valid_keep, _ = match_region_ids(q, q.clone(), max_regions=4, min_area=4.0)
        assert valid_keep.sum().item() == 2
        _, _, valid_drop, _ = match_region_ids(q, q.clone(), max_regions=4, min_area=5.0)
        ids, _, v, _ = match_region_ids(q, q.clone(), max_regions=4, min_area=5.0)
        assert set(ids[0][v[0]].tolist()) == {0}, "the 4-pixel region should be filtered out"

    def test_truncates_to_max_regions(self):
        q = _mask([(i, slice(i, i + 1)) for i in range(8)], h=8, w=4)
        ids, _, valid, _ = match_region_ids(q, q.clone(), max_regions=3)
        assert valid.sum().item() == 3
        assert set(ids[0][valid[0]].tolist()).issubset(set(range(8)))

    def test_truncation_is_random_not_the_first_n(self):
        """Selection uses torch.randperm. Pinning this matters: the draw order is
        part of the RNG stream, which is why vectorising this loop would change
        published numbers."""
        q = _mask([(i, slice(i, i + 1)) for i in range(8)], h=8, w=4)
        seen = set()
        for seed in range(12):
            torch.manual_seed(seed)
            ids, _, valid, _ = match_region_ids(q, q.clone(), max_regions=3)
            seen.add(tuple(sorted(ids[0][valid[0]].tolist())))
        assert len(seen) > 1, "selection appears deterministic; expected random sampling"


class TestValidation:
    def test_rejects_wrong_rank(self):
        with pytest.raises(ValueError, match=r"\[B, H, W\]"):
            match_region_ids(
                torch.zeros(4, 4, dtype=torch.long), torch.zeros(4, 4, dtype=torch.long)
            )

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            match_region_ids(
                torch.zeros(1, 4, 4, dtype=torch.long), torch.zeros(1, 8, 8, dtype=torch.long)
            )

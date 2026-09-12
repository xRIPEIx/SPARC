"""Tests for the paired image/mask transform.

This is the highest-consequence invariant in the data path: the image is
resized BICUBIC and the mask NEAREST, and they are cropped and flipped using
shared parameters. If those ever desynchronise, every region embedding is
pooled over the wrong pixels -- and training would still converge to a
plausible-looking number, so nothing downstream would flag it.

The fixture image encodes its own labels (dark <-> label 0, bright <-> label 1),
which turns "are they still aligned?" into a checkable property.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from sparc.data import TwoCropsTransformWithMask


def _identity_norm_transform(**kw):
    """Transform with normalisation disabled so pixel values stay in [0, 1],
    and colour ops off unless a test turns them on."""
    defaults = dict(
        input_size=64,
        scale=(1.0, 1.0),
        ratio=(1.0, 1.0),
        hflip_prob=0.0,
        color_jitter_prob=0.0,
        grayscale_prob=0.0,
        blur_prob=0.0,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    defaults.update(kw)
    return TwoCropsTransformWithMask(**defaults)


class TestSpatialAlignment:
    def test_image_content_and_mask_labels_stay_in_correspondence(self, image_mask_pair):
        """The core invariant: bright pixels carry label 1, dark pixels label 0,
        whatever geometric transform was applied."""
        image, mask = image_mask_pair(size=64)
        tf = _identity_norm_transform()
        img_t, mask_t = tf.transform_one_view(image, mask)

        bright = img_t[:, mask_t == 1].mean().item()
        dark = img_t[:, mask_t == 0].mean().item()
        assert bright > 0.9, f"label-1 pixels should be bright, got {bright:.3f}"
        assert dark < 0.1, f"label-0 pixels should be dark, got {dark:.3f}"

    def test_horizontal_flip_moves_image_and_mask_together(self, image_mask_pair):
        """hflip_prob=1.0 with a full-image crop makes this exact: the left half
        must end up bright AND labelled 1. Flipping only one of the two would
        leave the means looking fine on average but invert the pairing."""
        image, mask = image_mask_pair(size=64)
        tf = _identity_norm_transform(hflip_prob=1.0)
        img_t, mask_t = tf.transform_one_view(image, mask)

        assert (mask_t[:, :32] == 1).all(), "mask was not flipped"
        assert img_t[:, :, :32].mean().item() > 0.9, "image was not flipped"
        assert (mask_t[:, 32:] == 0).all()
        assert img_t[:, :, 32:].mean().item() < 0.1

    @pytest.mark.parametrize("trial", range(8))
    def test_alignment_survives_random_crops(self, trial, image_mask_pair):
        """Same invariant under the real augmentation: random resized crops plus
        random flips. Boundary pixels get bicubic ringing, so this compares
        means with a margin rather than per-pixel equality."""
        random.seed(trial)
        torch.manual_seed(trial)
        image, mask = image_mask_pair(size=64)
        tf = _identity_norm_transform(scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3), hflip_prob=0.5)
        img_t, mask_t = tf.transform_one_view(image, mask)

        if (mask_t == 1).any() and (mask_t == 0).any():
            assert img_t[:, mask_t == 1].mean() > img_t[:, mask_t == 0].mean() + 0.5


class TestMaskIntegrity:
    def test_mask_labels_are_never_interpolated(self, image_mask_pair):
        """NEAREST resampling must not invent labels that were not in the input.
        BICUBIC on the mask would produce fractional ids, which would silently
        fragment regions."""
        image, mask = image_mask_pair(size=64, n_labels=4)
        original = set(np.unique(mask).tolist())
        tf = _identity_norm_transform(input_size=37, scale=(0.3, 1.0))  # forces a real resize
        for _ in range(10):
            _, mask_t = tf.transform_one_view(image, mask)
            produced = set(mask_t.unique().tolist())
            assert produced.issubset(original), f"invented labels: {produced - original}"

    def test_mask_dtype_is_long(self, image_mask_pair):
        image, mask = image_mask_pair(size=64)
        _, mask_t = _identity_norm_transform().transform_one_view(image, mask)
        assert mask_t.dtype == torch.long

    def test_colour_ops_do_not_touch_the_mask(self, image_mask_pair):
        """Jitter, grayscale and blur apply to the image only."""
        image, mask = image_mask_pair(size=64)
        tf = _identity_norm_transform(color_jitter_prob=1.0, grayscale_prob=1.0, blur_prob=1.0)
        _, mask_t = tf.transform_one_view(image, mask)
        assert set(mask_t.unique().tolist()).issubset({0, 1})
        assert (mask_t[:, :32] == 0).all() and (mask_t[:, 32:] == 1).all()


class TestTwoCrops:
    def test_call_returns_two_independent_views_with_their_masks(self, image_mask_pair):
        image, mask = image_mask_pair(size=64)
        tf = _identity_norm_transform(scale=(0.2, 1.0), hflip_prob=0.5)
        v1, v2, m1, m2 = tf(image, mask)
        assert v1.shape == v2.shape == (3, 64, 64)
        assert m1.shape == m2.shape == (64, 64)
        assert not torch.equal(m1, m2), "two views should be independently sampled"


class TestDeterminism:
    def test_same_seed_reproduces_the_same_pair(self, image_mask_pair):
        """Pins the RNG draw *order*, not just the outputs. The transform draws
        crop params, then flip, then each colour op; reordering those would keep
        every test above passing while changing the augmentation stream and so
        every trained model."""
        image, mask = image_mask_pair(size=64)
        tf = TwoCropsTransformWithMask(input_size=32)

        random.seed(1234)
        torch.manual_seed(1234)
        a_img, a_mask = tf.transform_one_view(image, mask)

        random.seed(1234)
        torch.manual_seed(1234)
        b_img, b_mask = tf.transform_one_view(image, mask)

        assert torch.equal(a_img, b_img)
        assert torch.equal(a_mask, b_mask)

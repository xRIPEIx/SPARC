"""Tests for the in-process superpixel generators.

Every method honours one contract:
    (image_rgb[H, W, 3], n_segments) -> integer label map [H, W]

These tests are what a contributor adding a new method runs to know their
function is wired up correctly (see docs/extending.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from sparc.data.superpixel import METHOD_CHOICES, compute_mask


@pytest.fixture
def photo_like():
    """Smooth gradients plus blobs. Pure uniform noise is a poor test image:
    region-growing methods degenerate on it and land nowhere near the requested
    segment count."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:96, 0:96]
    base = (yy * 2 + xx).astype(np.float64)
    base += 60 * np.sin(xx / 9.0) + 60 * np.cos(yy / 7.0)
    img = np.stack([base, base * 0.7 + 40, base * 0.4 + 80], axis=-1)
    img += rng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


class TestContract:
    @pytest.mark.parametrize("method", METHOD_CHOICES)
    def test_returns_integer_label_map_of_matching_shape(self, method, photo_like):
        mask = compute_mask(method, photo_like, n_segments=50)
        assert mask.shape == photo_like.shape[:2]
        assert np.issubdtype(mask.dtype, np.integer)

    @pytest.mark.parametrize("method", METHOD_CHOICES)
    def test_labels_start_at_zero(self, method, photo_like):
        """Label 0 is a valid region downstream, and match_region_ids removes
        only ignore_index, so a method emitting 1-based labels would quietly
        shift every id."""
        mask = compute_mask(method, photo_like, n_segments=50)
        assert mask.min() == 0, f"{method} produced labels starting at {mask.min()}"

    @pytest.mark.parametrize("method", METHOD_CHOICES)
    def test_labels_are_dense(self, method, photo_like):
        """No gaps in the label range -- downstream code treats max()+1 as the
        region count."""
        mask = compute_mask(method, photo_like, n_segments=50)
        assert set(np.unique(mask).tolist()) == set(range(int(mask.max()) + 1))

    @pytest.mark.parametrize("method", METHOD_CHOICES)
    def test_segment_count_is_in_the_right_ballpark(self, method, photo_like):
        """Only slic and compact_watershed control the count directly;
        felzenszwalb binary-searches a proxy parameter, so the tolerance is
        deliberately loose."""
        mask = compute_mask(method, photo_like, n_segments=50)
        count = int(mask.max()) + 1
        assert 10 <= count <= 200, f"{method} returned {count} segments for a target of 50"


class TestDispatcher:
    def test_unknown_method_raises_with_a_useful_message(self):
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        with pytest.raises(ValueError) as exc:
            compute_mask("not_a_method", img, n_segments=10)
        message = str(exc.value)
        assert "not_a_method" in message
        assert "slic" in message, "the error should list what is available"

    def test_method_choices_matches_the_dispatcher(self):
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        for method in METHOD_CHOICES:
            compute_mask(method, img, n_segments=4)  # must not raise

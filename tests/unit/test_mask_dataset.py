"""Tests for RegionMaskDataset and the mask-directory metadata.

The dataset pairs an image with its precomputed mask purely by relative path.
That contract is what lets any image root be combined with any mask root, and
it is the thing a user hits first when pointing SPARC at their own data.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from sparc.data import RegionMaskDataset
from sparc.data.superpixel import FORMAT_ID, META_FILENAME, load_meta, write_meta
from sparc.data.superpixel.meta import MODE_SYNTHETIC, is_synthetic_constant


def _identity_transform(image, mask):
    return image, mask


def _build_corpus(tmp_path, n=3, size=(12, 10), with_masks=True, subdir=""):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    (images / subdir).mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.fromarray(np.full((*size, 3), i * 20, dtype=np.uint8)).save(
            images / subdir / f"img{i}.png"
        )
        if with_masks:
            (masks / subdir).mkdir(parents=True, exist_ok=True)
            np.save(masks / subdir / f"img{i}.npy", np.zeros(size, dtype=np.int32))
    return images, masks


class TestPairing:
    def test_pairs_images_with_masks_by_relative_path(self, tmp_path):
        images, masks = _build_corpus(tmp_path)
        ds = RegionMaskDataset(images, masks, _identity_transform)
        assert len(ds) == 3
        image, mask = ds[0]
        assert isinstance(image, Image.Image)
        assert mask.shape == (12, 10)

    def test_preserves_subdirectory_structure(self, tmp_path):
        images, masks = _build_corpus(tmp_path, subdir="train2017")
        ds = RegionMaskDataset(images, masks, _identity_transform)
        assert len(ds) == 3
        expected = masks / "train2017" / "img0.npy"
        assert ds.mask_path_for(ds.image_paths[0]) == expected

    def test_ordering_is_stable(self, tmp_path):
        """Sorted paths, so a run_id maps to the same sample ordering on any
        filesystem. os.walk order alone is not stable across systems."""
        images, masks = _build_corpus(tmp_path, n=5)
        a = RegionMaskDataset(images, masks, _identity_transform).image_paths
        b = RegionMaskDataset(images, masks, _identity_transform).image_paths
        assert a == b == sorted(a)


class TestFailureModes:
    def test_missing_mask_raises_rather_than_skipping(self, tmp_path):
        """A precompute job once under-covered the corpus by 287 masks and it
        was only caught when training hit this error. Silently skipping would
        have produced a quietly smaller dataset instead."""
        images, masks = _build_corpus(tmp_path)
        (masks / "img1.npy").unlink()
        ds = RegionMaskDataset(images, masks, _identity_transform)
        with pytest.raises(FileNotFoundError, match="img1"):
            for i in range(len(ds)):
                ds[i]

    def test_shape_mismatch_raises(self, tmp_path):
        images, masks = _build_corpus(tmp_path)
        np.save(masks / "img0.npy", np.zeros((99, 99), dtype=np.int32))
        ds = RegionMaskDataset(images, masks, _identity_transform)
        with pytest.raises(RuntimeError, match="does not match"):
            ds[0]

    def test_missing_roots_raise_immediately(self, tmp_path):
        images, masks = _build_corpus(tmp_path)
        with pytest.raises(FileNotFoundError):
            RegionMaskDataset(tmp_path / "nope", masks, _identity_transform)
        with pytest.raises(FileNotFoundError):
            RegionMaskDataset(images, tmp_path / "nope", _identity_transform)

    def test_empty_image_root_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        (tmp_path / "masks").mkdir()
        with pytest.raises(RuntimeError, match="No supported images"):
            RegionMaskDataset(tmp_path / "empty", tmp_path / "masks", _identity_transform)


class TestSyntheticMode:
    def test_synthetic_constant_mode_needs_no_mask_files(self, tmp_path):
        """The n_segments=1 configuration: one region covering the whole image,
        synthesised per sample rather than stored 118k times."""
        images, masks = _build_corpus(tmp_path, with_masks=False)
        masks.mkdir(exist_ok=True)
        write_meta(masks, {"mode": MODE_SYNTHETIC, "constant_label": 0})
        ds = RegionMaskDataset(images, masks, _identity_transform)
        _, mask = ds[0]
        assert mask.shape == (12, 10)
        assert np.unique(mask).tolist() == [0]


class TestMeta:
    def test_write_then_load_round_trips_and_stamps_the_format(self, tmp_path):
        root = tmp_path / "masks"
        write_meta(root, {"mode": "precomputed_npy", "method": "slic", "n_segments": 100})
        assert (root / META_FILENAME).is_file()
        meta = load_meta(root)
        assert meta["format"] == FORMAT_ID
        assert meta["method"] == "slic"
        assert not is_synthetic_constant(meta)

    def test_load_meta_returns_none_when_absent(self, tmp_path):
        (tmp_path / "masks").mkdir()
        assert load_meta(tmp_path / "masks") is None
